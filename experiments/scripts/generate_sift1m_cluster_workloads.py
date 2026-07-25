#!/usr/bin/env python3
"""Generate clustered-region query workloads for SIFT1M (paper-style, 10K queries each).

Regions: small X/Y (10K nodes), large X/Y (100K nodes) from BFS on the Vamana graph.
Queries: base vectors at sampled graph node ids (same trace order for temporal workloads).
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import deque
from pathlib import Path

import numpy as np

SMALL_SIZE = 10_000
LARGE_SIZE = 100_000
NUM_QUERIES = 10_000
DIM = 128
SEED = 42


def load_vamana_graph(mem_index: Path) -> tuple[int, int, list[list[int]]]:
    data = mem_index.read_bytes()
    off = 0
    file_size = struct.unpack_from("<Q", data, off)[0]
    off += 8
    assert file_size == len(data), f"size mismatch {file_size} vs {len(data)}"
    width, medoid = struct.unpack_from("<II", data, off)
    off += 8
    frozen_num = struct.unpack_from("<Q", data, off)[0]
    off += 8
    _ = width, frozen_num

    adj: list[list[int]] = []
    while off < len(data):
        if off + 4 > len(data):
            break
        (nnbrs,) = struct.unpack_from("<I", data, off)
        off += 4
        if nnbrs == 0 and off >= len(data):
            break
        nbrs = list(struct.unpack_from(f"<{nnbrs}I", data, off))
        off += 4 * nnbrs
        if nnbrs > width:
            off += 4 * (nnbrs - width)
        adj.append(nbrs[:width])
    return medoid, len(adj), adj


def bfs_region(adj: list[list[int]], start: int, target: int) -> set[int]:
    n = len(adj)
    if start < 0 or start >= n:
        start = 0
    seen: set[int] = set()
    q: deque[int] = deque([start])
    while q and len(seen) < target:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        for v in adj[u]:
            if 0 <= v < n and v not in seen:
                q.append(v)
    return seen


def pick_far_start(adj: list[list[int]], avoid: set[int], medoid: int) -> int:
    """BFS layering from medoid; pick the furthest reachable node outside avoid."""
    n = len(adj)
    dist = [-1] * n
    dist[medoid] = 0
    q: deque[int] = deque([medoid])
    best = medoid
    best_d = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
                if v not in avoid and dist[v] >= best_d:
                    best_d = dist[v]
                    best = v
    for i in range(n - 1, -1, -1):
        if i not in avoid:
            return i
    return best


def load_fbin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        arr = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)
    return arr


def save_fbin(path: Path, arr: np.ndarray) -> None:
    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(arr.astype(np.float32, copy=False).tobytes())


def save_gt(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    n, k = ids.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, k))
        f.write(ids.astype(np.uint32, copy=False).tobytes())
        f.write(dists.astype(np.float32, copy=False).tobytes())


def compute_gt(queries: np.ndarray, base: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k GT via ||q-b||^2 = ||q||^2 + ||b||^2 - 2 q·b (batched)."""
    nq = queries.shape[0]
    nb = base.shape[0]
    base_norm = np.sum(base * base, axis=1)
    ids = np.empty((nq, k), dtype=np.uint32)
    dists = np.empty((nq, k), dtype=np.float32)
    qbatch = 128
    for i in range(0, nq, qbatch):
        q = queries[i : i + qbatch]
        q_norm = np.sum(q * q, axis=1, keepdims=True)
        d2 = q_norm + base_norm[None, :] - 2.0 * (q @ base.T)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        row = np.arange(idx.shape[0])[:, None]
        order = np.argsort(d2[row, idx], axis=1)
        top = idx[row, order]
        ids[i : i + qbatch] = top.astype(np.uint32)
        dists[i : i + qbatch] = np.take_along_axis(d2, top, axis=1).astype(np.float32)
    return ids, dists


def sample_node_ids(rng: np.random.Generator, pool: set[int], count: int) -> np.ndarray:
    pool_arr = np.fromiter(pool, dtype=np.uint32)
    if len(pool_arr) >= count:
        return rng.choice(pool_arr, size=count, replace=False)
    return rng.choice(pool_arr, size=count, replace=True)


def build_workloads(
    base: np.ndarray,
    base_file: Path,
    regions: dict[str, set[int]],
    out_dir: Path,
    k: int,
    gt_tool: Path | None,
) -> dict:
    rng = np.random.default_rng(SEED)
    n_base = base.shape[0]
    meta: dict = {"num_queries": NUM_QUERIES, "dim": DIM, "workloads": {}}

    specs = {
        "uniform": lambda: rng.integers(0, n_base, size=NUM_QUERIES, dtype=np.uint32),
        "small_x": lambda: sample_node_ids(rng, regions["small_x"], NUM_QUERIES),
        "small_y": lambda: sample_node_ids(rng, regions["small_y"], NUM_QUERIES),
        "large_x": lambda: sample_node_ids(rng, regions["large_x"], NUM_QUERIES),
        "large_y": lambda: sample_node_ids(rng, regions["large_y"], NUM_QUERIES),
        "x_only": lambda: sample_node_ids(rng, regions["large_x"], NUM_QUERIES),
        "x_then_y": lambda: np.concatenate(
            [
                sample_node_ids(rng, regions["large_x"], NUM_QUERIES // 2),
                sample_node_ids(rng, regions["large_y"], NUM_QUERIES // 2),
            ]
        ),
        "y_only": lambda: sample_node_ids(rng, regions["large_y"], NUM_QUERIES),
    }

    for name, pick in specs.items():
        node_ids = pick()
        queries = base[node_ids]
        qpath = out_dir / f"{name}_10k.fbin"
        gtpath = out_dir / f"{name}_10k_gt.bin"
        save_fbin(qpath, queries)
        if gt_tool is not None:
            import subprocess

            subprocess.run(
                [
                    str(gt_tool),
                    "--data_type",
                    "float",
                    "--dist_fn",
                    "l2",
                    "--base_file",
                    str(base_file),
                    "--query_file",
                    str(qpath),
                    "--gt_file",
                    str(gtpath),
                    "--K",
                    str(k),
                ],
                check=True,
            )
        else:
            gt_ids, gt_dists = compute_gt(queries, base, k=k)
            save_gt(gtpath, gt_ids, gt_dists)
        meta["workloads"][name] = {
            "query_file": str(qpath),
            "gt_file": str(gtpath),
            "node_ids_file": str(out_dir / f"{name}_10k_node_ids.bin"),
        }
        (out_dir / f"{name}_10k_node_ids.bin").write_bytes(node_ids.astype(np.uint32).tobytes())
        print(f"  {name}: queries -> {qpath.name}, gt@{k} -> {gtpath.name}")

    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--recall-at", type=int, default=1)
    ap.add_argument(
        "--gt-tool",
        type=Path,
        default=None,
        help="compute_groundtruth binary; if set, skip slow Python GT and call this per workload",
    )
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = (args.out_dir or data_dir / "workloads").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mem_index = data_dir / "sift1m_index_mem.index"
    base_path = data_dir / "sift_base.fbin"
    medoid, num_points, adj = load_vamana_graph(mem_index)
    print(f"graph: {num_points} points, medoid={medoid}")

    small_x = bfs_region(adj, medoid, SMALL_SIZE)
    y_start = pick_far_start(adj, small_x, medoid)
    small_y = bfs_region(adj, y_start, SMALL_SIZE)

    large_x = bfs_region(adj, medoid, LARGE_SIZE)
    y2_start = pick_far_start(adj, large_x, medoid)
    large_y = bfs_region(adj, y2_start, LARGE_SIZE)

    overlap_small = len(small_x & small_y)
    overlap_large = len(large_x & large_y)

    regions = {
        "small_x": small_x,
        "small_y": small_y,
        "large_x": large_x,
        "large_y": large_y,
    }

    region_meta = {
        "dataset": "sift1m",
        "num_points": num_points,
        "medoid": medoid,
        "small_size": SMALL_SIZE,
        "large_size": LARGE_SIZE,
        "small_y_start": y_start,
        "large_y_start": y2_start,
        "overlap_small_xy": overlap_small,
        "overlap_large_xy": overlap_large,
        "overlap_large_pct": round(100.0 * overlap_large / LARGE_SIZE, 4),
        "region_files": {},
    }
    for name, nodes in regions.items():
        p = out_dir / f"region_{name}.bin"
        arr = np.array(sorted(nodes), dtype=np.uint32)
        p.write_bytes(arr.tobytes())
        region_meta["region_files"][name] = {"path": str(p), "count": len(arr)}

    print(f"regions: small overlap={overlap_small}, large overlap={overlap_large} ({region_meta['overlap_large_pct']}%)")

    print("loading base vectors...")
    base = load_fbin(base_path)
    assert base.shape[0] == num_points

    print("building workloads (10K queries each)...")
    wl_meta = build_workloads(base, base_path, regions, out_dir, k=args.recall_at, gt_tool=args.gt_tool)

    meta = {**region_meta, **wl_meta}
    meta_path = out_dir / "workloads_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
