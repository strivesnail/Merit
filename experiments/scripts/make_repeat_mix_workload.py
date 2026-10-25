#!/usr/bin/env python3
"""Mix 10K uniform queries with repeated hot queries.

Picks NQ random queries from the hot block of knn2000_x71, repeats each REPEAT times, shuffles the
repeated accesses and inserts them at random positions among the 10K uniform queries (whose relative
order is kept). Ground truth rows are copied from knn2000_x71.
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
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=70)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, d = np.fromfile(WLD / f"{SRC}.u8bin", dtype=np.int32, count=2)
    q = np.fromfile(WLD / f"{SRC}.u8bin", dtype=np.uint8, offset=8).reshape(n, d)
    gn, gk = np.fromfile(WLD / f"{SRC}_gt10.bin", dtype=np.int32, count=2)
    gt_ids = np.fromfile(WLD / f"{SRC}_gt10.bin", dtype=np.uint32, count=gn * gk, offset=8).reshape(gn, gk)
    gt_d = np.fromfile(WLD / f"{SRC}_gt10.bin", dtype=np.float32, count=gn * gk, offset=8 + 4 * gn * gk).reshape(gn, gk)

    picked = rng.choice(np.arange(N_ORIG, n), args.nq, replace=False)
    hot = rng.permutation(np.repeat(picked, args.repeat))
    total = N_ORIG + len(hot)
    is_hot = np.zeros(total, dtype=bool)
    is_hot[rng.choice(total, len(hot), replace=False)] = True
    rows = np.empty(total, dtype=np.int64)
    rows[~is_hot] = np.arange(N_ORIG)
    rows[is_hot] = hot

    tag = f"knn2000_x71r{args.nq}x{args.repeat}_block_s4_pct"
    with (WLD / f"{tag}.u8bin").open("wb") as f:
        np.array([total, d], dtype=np.int32).tofile(f)
        q[rows].tofile(f)
    with (WLD / f"{tag}_gt10.bin").open("wb") as f:
        np.array([total, gk], dtype=np.int32).tofile(f)
        gt_ids[rows].tofile(f)
        gt_d[rows].tofile(f)
    np.save(WLD / f"{tag}_is_region.npy", is_hot)
    meta = {"source": SRC, "uniform": N_ORIG, "hot_queries": int(args.nq), "repeat": int(args.repeat),
            "hot_accesses": int(len(hot)), "n": int(total), "interleaved": True, "seed": args.seed}
    (WLD / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    print(tag, json.dumps(meta))


if __name__ == "__main__":
    main()
