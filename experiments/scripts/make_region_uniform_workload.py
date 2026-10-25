#!/usr/bin/env python3
"""Queries spread uniformly over a graph-local region instead of concentrated on hot items.

Each query picks a node uniformly at random from the first R nodes (BFS order) of a region
and adds independent Gaussian noise, so there is no popular item and no two queries are
identical; only the region as a whole is hot. Also writes a uniform sample of query
positions on which exact ground truth is computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from make_zipf_placement_variant import read_vectors


def write_u8bin(path: Path, x: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array(x.shape, dtype=np.int32).tofile(f)
        np.ascontiguousarray(x, dtype=np.uint8).tofile(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--region-node-ids", type=Path, required=True)
    ap.add_argument("--region-size", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--sigma", type=float, default=24.0)
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--ndims", type=int, default=128)
    ap.add_argument("--node_len", type=int, default=388)
    ap.add_argument("--nnps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()

    region = np.fromfile(args.region_node_ids, dtype="<u4")[: args.region_size].astype(np.int64)
    if region.size < args.region_size:
        raise SystemExit(f"region file has only {region.size} IDs")
    vecs = read_vectors(args.index, region, args.ndims, args.node_len, args.nnps)

    rng = np.random.default_rng(args.seed)
    pick = rng.integers(0, region.size, size=args.n)
    out = np.empty((args.n, args.ndims), dtype=np.uint8)
    for b in range(0, args.n, 200000):
        e = min(args.n, b + 200000)
        x = vecs[pick[b:e]].astype(np.float32) + rng.normal(0.0, args.sigma, (e - b, args.ndims)).astype(np.float32)
        out[b:e] = np.clip(np.rint(x), 0, 255).astype(np.uint8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_u8bin(args.out.with_suffix(".u8bin"), out)
    np.save(args.out.parent / (args.out.name + "_node_ids.npy"), region[pick].astype(np.uint32))
    pos = np.sort(rng.choice(args.n, size=min(args.sample, args.n), replace=False)).astype(np.int64)
    np.save(args.out.parent / (args.out.name + "_sample_pos.npy"), pos)
    write_u8bin(args.out.parent / (args.out.name + "_sample.u8bin"), out[pos])

    _, counts = np.unique(pick, return_counts=True)
    meta = {"kind": "region-uniform", "region_size": args.region_size, "sigma": args.sigma,
            "seed": args.seed, "n": args.n, "distinct_items": int(counts.size),
            "max_item_share": float(counts.max() / args.n), "sample": int(pos.size)}
    (args.out.parent / (args.out.name + ".json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
