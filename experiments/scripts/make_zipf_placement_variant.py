#!/usr/bin/env python3
"""Remap an existing finite-Zipf self-query workload to a different hot-item placement.

The Zipf rank of every query is kept; only the node assigned to each rank changes:
  bfs     ranks follow BFS discovery order inside the graph-local region (original)
  shuffle the same region, ranks randomly permuted
  global  100K hot items drawn uniformly from the whole dataset
Vectors are read from the DiskANN disk index, so no separate base file is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

SECTOR = 4096


def read_vectors(index: Path, ids: np.ndarray, ndims: int, node_len: int, nnps: int) -> np.ndarray:
    out = np.empty((ids.size, ndims), dtype=np.uint8)
    fd = os.open(index, os.O_RDONLY)
    try:
        for i, nid in enumerate(ids.tolist()):
            off = (1 + nid // nnps) * SECTOR + (nid % nnps) * node_len
            out[i] = np.frombuffer(os.pread(fd, ndims, off), dtype=np.uint8)
    finally:
        os.close(fd)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--region-node-ids", type=Path, required=True)
    ap.add_argument("--query-node-ids", type=Path, required=True)
    ap.add_argument("--mode", choices=["bfs", "shuffle", "global"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--region-size", type=int, default=100_000)
    ap.add_argument("--npts", type=int, default=1_000_000_000)
    ap.add_argument("--ndims", type=int, default=128)
    ap.add_argument("--node_len", type=int, default=388)
    ap.add_argument("--nnps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    region = np.fromfile(args.region_node_ids, dtype="<u4")[: args.region_size].astype(np.int64)
    qids = np.fromfile(args.query_node_ids, dtype="<u4")[: args.n].astype(np.int64)
    rank_of = {int(nid): r for r, nid in enumerate(region.tolist())}
    ranks = np.fromiter((rank_of[int(x)] for x in qids), dtype=np.int64, count=qids.size)

    rng = np.random.default_rng(args.seed)
    if args.mode == "bfs":
        items = region
    elif args.mode == "shuffle":
        items = region[rng.permutation(region.size)]
    else:
        items = np.unique(rng.integers(0, args.npts, size=args.region_size * 2))
        items = rng.permutation(items)[: args.region_size]
    new_ids = items[ranks]

    uniq, inverse = np.unique(new_ids, return_inverse=True)
    vecs = read_vectors(args.index, uniq, args.ndims, args.node_len, args.nnps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(struct.pack("<II", new_ids.size, args.ndims))
        vecs[inverse].tofile(f)
    new_ids.astype("<u4").tofile(args.out.with_name(args.out.stem + "_node_ids.bin"))
    counts = np.bincount(ranks, minlength=args.region_size)
    top = np.sort(counts)[::-1]
    meta = {
        "mode": args.mode,
        "seed": args.seed,
        "n": int(new_ids.size),
        "region_size": args.region_size,
        "unique_queries": int(uniq.size),
        "top_10_share": float(top[:10].sum() / new_ids.size),
        "top_1_percent_share": float(top[: args.region_size // 100].sum() / new_ids.size),
        "source_query_node_ids": str(args.query_node_ids),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
