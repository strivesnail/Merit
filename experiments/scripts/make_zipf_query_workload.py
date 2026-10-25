#!/usr/bin/env python3
"""Zipfian repetition of the knn2000_x71 hot queries.

Keeps the 10K original queries and replaces the 7000 distinct hot queries of knn2000_x71 with 7000
draws (with replacement) from them under Zipf(s) popularity over a random rank order. Repeated
queries are exact copies, so ground truth rows are copied from knn2000_x71.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WLD = Path("/home/jianz/workload/data/sift100m/workloads/cluster_hotspot")
SRC = "knn2000_x71_block_s4_pct"
N_ORIG = 10000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zipf", type=float, nargs="+", default=[0.0, 0.5, 0.8, 1.0, 1.2, 1.5])
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, d = np.fromfile(WLD / f"{SRC}.u8bin", dtype=np.int32, count=2)
    q = np.fromfile(WLD / f"{SRC}.u8bin", dtype=np.uint8, offset=8).reshape(n, d)
    gn, gk = np.fromfile(WLD / f"{SRC}_gt10.bin", dtype=np.int32, count=2)
    gt_ids = np.fromfile(WLD / f"{SRC}_gt10.bin", dtype=np.uint32, count=gn * gk, offset=8).reshape(gn, gk)
    gt_d = np.fromfile(WLD / f"{SRC}_gt10.bin", dtype=np.float32, count=gn * gk, offset=8 + 4 * gn * gk).reshape(gn, gk)
    hot = np.arange(N_ORIG, n)

    for s in args.zipf:
        rank = rng.permutation(len(hot))
        p = 1.0 / np.arange(1, len(hot) + 1) ** s
        p /= p.sum()
        rows = np.concatenate([np.arange(N_ORIG), hot[rank[rng.choice(len(hot), len(hot), p=p)]]])
        tag = f"knn2000_x71q{s:g}_block_s4_pct"
        with (WLD / f"{tag}.u8bin").open("wb") as f:
            np.array([len(rows), d], dtype=np.int32).tofile(f)
            q[rows].tofile(f)
        with (WLD / f"{tag}_gt10.bin").open("wb") as f:
            np.array([len(rows), gk], dtype=np.int32).tofile(f)
            gt_ids[rows].tofile(f)
            gt_d[rows].tofile(f)
        counts = np.sort(np.bincount(rows[N_ORIG:] - N_ORIG, minlength=len(hot)))[::-1]
        meta = {"zipf_s": s, "source": SRC, "hot_draws": int(len(hot)),
                "distinct_hot_queries": int((counts > 0).sum()),
                "top1_share": float(counts[0] / len(hot)), "top10_share": float(counts[:10].sum() / len(hot)),
                "top100_share": float(counts[:100].sum() / len(hot)), "n": int(len(rows))}
        (WLD / f"{tag}.json").write_text(json.dumps(meta, indent=2))
        print(tag, json.dumps(meta))


if __name__ == "__main__":
    main()
