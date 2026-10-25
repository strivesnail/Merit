#!/usr/bin/env python3
"""Fig.1 workload: dataset query pool, each repeated R times, fully shuffled."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_vec_bin(path: Path):
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
        body = f.read()
    if len(body) == n * d:
        dtype = np.uint8
    elif len(body) == n * d * 4:
        dtype = np.float32
    else:
        raise SystemExit(f"bad bin {path}: n={n} d={d} body={len(body)}")
    return n, d, dtype, np.frombuffer(body, dtype=dtype).reshape(n, d).copy()


def read_gt(path: Path, nq: int, k: int = 10):
    with path.open("rb") as f:
        n, kk = struct.unpack("ii", f.read(8))
        ids = np.fromfile(f, dtype=np.int32, count=n * kk).reshape(n, kk)
    if n < nq:
        raise SystemExit(f"gt {path} has n={n} < pool {nq}")
    return ids[:nq, :k].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--max-pool", type=int, default=10000)
    ap.add_argument("--gt-k", type=int, default=10)
    args = ap.parse_args()

    n, d, dtype, q = read_vec_bin(args.query)
    pool = min(n, args.max_pool)
    q = q[:pool]
    gt = read_gt(args.gt, pool, k=args.gt_k)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(np.repeat(np.arange(pool), args.repeat))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.tag}_x{args.repeat}_shuffled"
    # keep extension hint for runners
    ext = "u8bin" if dtype == np.uint8 else "fbin"
    qpath = out / f"{tag}.{ext}"
    gpath = out / f"{tag}_gt{args.gt_k}.ibin"
    with qpath.open("wb") as f:
        np.array([len(order), d], dtype=np.int32).tofile(f)
        q[order].astype(dtype, copy=False).tofile(f)
    with gpath.open("wb") as f:
        np.array([len(order), args.gt_k], dtype=np.int32).tofile(f)
        gt[order].astype(np.int32).tofile(f)
    meta = {
        "tag": tag,
        "pool": pool,
        "dim": d,
        "dtype": str(dtype),
        "repeat": args.repeat,
        "n": int(len(order)),
        "seed": args.seed,
        "query_src": str(args.query),
        "gt_src": str(args.gt),
        "query_out": str(qpath),
        "gt_out": str(gpath),
    }
    (out / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
