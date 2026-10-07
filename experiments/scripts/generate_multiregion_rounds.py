#!/usr/bin/env python3
"""Generate disjoint graph-local query regions from a DiskANN disk index.

Each round's region is collected by BFS over the on-disk graph from a start node
chosen to be far (in vector space) from previous starts; regions never overlap.
Queries are exact base-vector self-queries, so ground truth is the node itself.
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
from collections import deque
from pathlib import Path

import numpy as np

SECTOR_SIZE = 4096


def map_u8bin(path: Path) -> tuple[np.memmap, int, int]:
    with path.open("rb") as source:
        count, dim = struct.unpack("<II", source.read(8))
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(count, dim)), count, dim


class DiskGraph:
    def __init__(self, index_path: Path, dim: int) -> None:
        self._file = index_path.open("rb")
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        rows, cols = struct.unpack_from("<ii", self._mapping, 0)
        if cols != 1 or rows < 5:
            raise SystemExit(f"invalid metadata shape: {rows}x{cols}")
        num_points, ndims, medoid, max_node_len, nodes_per_sector = struct.unpack_from("<QQQQQ", self._mapping, 8)
        if nodes_per_sector == 0:
            raise SystemExit("multi-sector nodes are not supported")
        if ndims != dim:
            raise SystemExit(f"expected dimension {dim}, got {ndims}")
        self.num_points = num_points
        self.medoid = medoid
        self._max_node_len = max_node_len
        self._nodes_per_sector = nodes_per_sector
        self._coord_bytes = dim
        self._max_degree = (max_node_len - dim - 4) // 4

    def neighbors(self, node_id: int) -> np.ndarray:
        sector = 1 + node_id // self._nodes_per_sector
        offset = sector * SECTOR_SIZE + (node_id % self._nodes_per_sector) * self._max_node_len + self._coord_bytes
        (degree,) = struct.unpack_from("<I", self._mapping, offset)
        if degree > self._max_degree:
            raise RuntimeError(f"invalid degree {degree} for node {node_id}")
        return np.frombuffer(self._mapping, dtype="<u4", count=degree, offset=offset + 4)

    def close(self) -> None:
        self._mapping.close()
        self._file.close()


def choose_spaced_start(base: np.memmap, claimed: np.ndarray, previous_starts: list[int],
                        candidate_count: int, rng: np.random.Generator) -> int:
    candidates = rng.integers(0, base.shape[0], size=candidate_count * 4)
    candidates = np.unique(candidates[~claimed[candidates]])[:candidate_count]
    if not previous_starts:
        return int(candidates[0])
    candidate_vectors = np.asarray(base[np.sort(candidates)], dtype=np.int32)
    candidates = np.sort(candidates)
    start_vectors = np.asarray(base[sorted(previous_starts)], dtype=np.int32)
    min_distances = np.full(candidates.size, np.iinfo(np.int64).max, dtype=np.int64)
    for start_vector in start_vectors:
        difference = candidate_vectors - start_vector
        min_distances = np.minimum(min_distances, np.einsum("ij,ij->i", difference, difference).astype(np.int64))
    return int(candidates[int(np.argmax(min_distances))])


def bfs_disjoint_region(graph: DiskGraph, start: int, target: int, claimed: np.ndarray) -> np.ndarray:
    region = []
    queued = {start}
    queue = deque([start])
    while queue and len(region) < target:
        node = queue.popleft()
        if claimed[node]:
            continue
        claimed[node] = True
        region.append(node)
        for neighbor in graph.neighbors(node):
            neighbor = int(neighbor)
            if not claimed[neighbor] and neighbor not in queued:
                queued.add(neighbor)
                queue.append(neighbor)
    if len(region) < target:
        raise SystemExit(f"region from node {start} exhausted at {len(region)} nodes")
    return np.asarray(region, dtype=np.uint32)


def save_self_gt(path: Path, node_ids: np.ndarray) -> None:
    with path.open("wb") as output:
        output.write(struct.pack("<II", node_ids.size, 1))
        node_ids.astype("<u4").tofile(output)
        np.zeros(node_ids.size, dtype=np.float32).tofile(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--disk-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="SIFT1B")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--region-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-candidates", type=int, default=2000)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    base, base_count, dim = map_u8bin(args.base)
    graph = DiskGraph(args.disk_index, dim)
    if graph.num_points != base_count:
        raise SystemExit(f"base/index mismatch: base={base_count}, index={graph.num_points}")
    if args.rounds * args.region_size > base_count:
        raise SystemExit("requested regions exceed the dataset")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_tag or f"small_regions_{args.rounds}x{args.region_size // 1000}k"
    query_path = args.out_dir / f"{stem}.u8bin"
    gt_path = args.out_dir / f"{stem}_gt.bin"
    node_ids_path = args.out_dir / f"{stem}_node_ids.bin"
    manifest_path = args.out_dir / f"{stem}.json"

    rng = np.random.default_rng(args.seed)
    claimed = np.zeros(base_count, dtype=np.bool_)
    starts: list[int] = []
    all_ids = []
    for round_index in range(args.rounds):
        start = choose_spaced_start(base, claimed, starts, args.start_candidates, rng)
        region = bfs_disjoint_region(graph, start, args.region_size, claimed)
        starts.append(start)
        all_ids.append(region)
        print(f"round {round_index + 1}: start={start}, region={region.size}", flush=True)
    graph.close()

    combined_ids = np.concatenate(all_ids)
    with query_path.open("wb") as query_output:
        query_output.write(struct.pack("<II", combined_ids.size, dim))
        for region in all_ids:
            order = np.argsort(region)
            vectors = np.empty((region.size, dim), dtype=np.uint8)
            vectors[order] = base[region[order]]
            vectors.tofile(query_output)
    combined_ids.astype("<u4").tofile(node_ids_path)
    save_self_gt(gt_path, combined_ids)
    manifest = {
        "dataset": args.dataset,
        "kind": "disjoint graph-local regions; exact base-vector self-queries",
        "query_file": str(query_path),
        "gt_file": str(gt_path),
        "node_ids_file": str(node_ids_path),
        "rounds": args.rounds,
        "region_size": args.region_size,
        "regions_disjoint": True,
        "seed": args.seed,
        "round_details": [
            {"round": i + 1, "start_node": starts[i], "begin": i * args.region_size,
             "end": (i + 1) * args.region_size}
            for i in range(args.rounds)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
