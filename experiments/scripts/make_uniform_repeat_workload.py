#!/usr/bin/env python3
"""The 10K uniform knn2000 queries, each accessed R times (R = 1..10).

The original 10K queries keep their order; the (R-1)*10K extra accesses are shuffled and inserted at
random positions.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from make_zipf_hotspot_workload import WLD, read_gt, read_u8bin, write_gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    q1 = np.asarray(read_u8bin(WLD / "knn2000_x1_block_s4_pct.u8bin"))
    gt_ids, gt_d = read_gt(WLD / "knn2000_x1_block_s4_pct_gt10.bin")
    n = len(q1)
    for r in args.levels:
        extra = rng.permutation(np.repeat(np.arange(n), r - 1))
        total = n * r
        is_extra = np.zeros(total, dtype=bool)
        is_extra[rng.choice(total, len(extra), replace=False)] = True
        order = np.empty(total, dtype=np.int64)
        order[~is_extra] = np.arange(n)
        order[is_extra] = extra
        tag = f"knn2000_u10kx{r}_block_s4_pct"
        with (WLD / f"{tag}.u8bin").open("wb") as f:
            np.array([total, q1.shape[1]], dtype=np.int32).tofile(f)
            q1[order].tofile(f)
        write_gt(WLD / f"{tag}_gt10.bin", gt_ids[order, :10], gt_d[order, :10])
        meta = {"distinct": n, "repeat": r, "n": int(total), "original_order_kept": True, "seed": args.seed}
        (WLD / f"{tag}.json").write_text(json.dumps(meta, indent=2))
        print(tag, json.dumps(meta))


if __name__ == "__main__":
    main()
