#!/usr/bin/env python3
"""SIFT100M public 10K queries, each repeated R times, fully shuffled.

With --zipf S: the 10K queries once each plus --zipf-n extra accesses drawn from the same 10K with
Zipf(S) popularity (random rank assignment), all shuffled together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from make_zipf_hotspot_workload import WLD, read_gt, read_u8bin, write_gt

DATA = Path("/home/jianz/workload/data/sift100m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--zipf", type=float, default=None)
    ap.add_argument("--zipf-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--query", type=Path, default=DATA / "query.public.10K.u8bin")
    ap.add_argument("--gt", type=Path, default=DATA / "bigann-100M")
    ap.add_argument("--out-dir", type=Path, default=WLD)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    q = np.asarray(read_u8bin(args.query))
    gt_ids, gt_d = read_gt(args.gt)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    n = len(q)
    if args.zipf is None:
        order = rng.permutation(np.repeat(np.arange(n), args.repeat))
        tag = f"pub10kx{args.repeat}_shuffled"
    else:
        p = 1.0 / np.arange(1, n + 1) ** args.zipf
        rank_to_query = rng.permutation(n)
        extra = rank_to_query[rng.choice(n, size=args.zipf_n, p=p / p.sum())]
        order = rng.permutation(np.concatenate([np.arange(n), extra]))
        tag = f"pub10k_zipf{str(args.zipf).replace('.', 'p')}_{args.zipf_n // 1000}k_shuffled"
        counts = np.bincount(extra, minlength=n)
        print("extra distinct", int((counts > 0).sum()), "top1 share", round(counts.max() / args.zipf_n, 4),
              "top1% share", round(np.sort(counts)[::-1][: n // 100].sum() / args.zipf_n, 4))
    with (out / f"{tag}.u8bin").open("wb") as f:
        np.array([len(order), q.shape[1]], dtype=np.int32).tofile(f)
        q[order].tofile(f)
    write_gt(out / f"{tag}_gt10.bin", gt_ids[order, :10], gt_d[order, :10])
    meta = {"distinct": n, "repeat": args.repeat if args.zipf is None else None, "zipf": args.zipf,
            "zipf_extra": args.zipf_n if args.zipf is not None else 0, "n": int(len(order)), "shuffled": True,
            "seed": args.seed}
    (out / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    print(tag, json.dumps(meta))


if __name__ == "__main__":
    main()
