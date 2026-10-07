#!/usr/bin/env python3
"""Add independent Gaussian noise to every query of an existing workload.

The popularity of each source vector (and thus the hotspot) is unchanged, but no two
queries are identical, so a search cannot be served by repeating an earlier query.
Also writes a uniform sample of query positions on which exact ground truth is computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_u8bin(path: Path) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(int(n), int(d)))


def write_u8bin(path: Path, x: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array(x.shape, dtype=np.int32).tofile(f)
        np.ascontiguousarray(x, dtype=np.uint8).tofile(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--chunk", type=int, default=200000)
    args = ap.parse_args()

    src = read_u8bin(args.src)
    n, d = src.shape
    rng = np.random.default_rng(args.seed)
    out = np.empty((n, d), dtype=np.uint8)
    for b in range(0, n, args.chunk):
        e = min(n, b + args.chunk)
        x = src[b:e].astype(np.float32) + rng.normal(0.0, args.sigma, (e - b, d)).astype(np.float32)
        out[b:e] = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    write_u8bin(args.out.with_suffix(".u8bin"), out)

    pos = np.sort(rng.choice(n, size=min(args.sample, n), replace=False)).astype(np.int64)
    np.save(args.out.parent / (args.out.name + "_sample_pos.npy"), pos)
    write_u8bin(args.out.parent / (args.out.name + "_sample.u8bin"), out[pos])

    uniq = len(np.unique(out.view(np.dtype((np.void, d)))))
    meta = {"src": str(args.src), "sigma": args.sigma, "seed": args.seed, "n": n,
            "unique_queries": int(uniq), "sample": int(len(pos))}
    (args.out.parent / (args.out.name + ".json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
