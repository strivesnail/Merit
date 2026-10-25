#!/usr/bin/env python3
"""10K uniform queries mixed with hot accesses at different repetition levels.

Default: 5000 hot accesses at every level. With --grow, all HOT distinct hot queries are used at every
level, so level xR has HOT*R hot accesses.

The hot region is the REGION base vectors nearest to the centroid of the knn2000 hot region (recovered
from the knn2000_x71 hot-block ground truth). HOT distinct hot queries are region members plus Gaussian
noise. By default level xR uses HOT / R of them, each accessed R times; the accesses are shuffled and inserted at
random positions among the 10K uniform queries, whose order is kept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from make_zipf_hotspot_workload import BASE, WLD, brute_force_gt, read_gt, read_u8bin, write_gt

N_ORIG = 10000


def nearest_base(center: np.ndarray, k: int, chunk: int = 2_000_000) -> np.ndarray:
    base = read_u8bin(BASE)
    c = torch.tensor(center[None], dtype=torch.float32, device="cuda")
    best_d = torch.full((k,), float("inf"), device="cuda")
    best_i = torch.zeros(k, dtype=torch.int64, device="cuda")
    for s in range(0, base.shape[0], chunk):
        x = torch.tensor(np.asarray(base[s:s + chunk]), dtype=torch.float32, device="cuda")
        dv, di = torch.topk(((x - c) ** 2).sum(1), k, largest=False)
        best_d, sel = torch.topk(torch.cat([best_d, dv]), k, largest=False)
        best_i = torch.cat([best_i, di + s])[sel]
    return np.sort(best_i.cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", type=int, default=3000)
    ap.add_argument("--hot", type=int, default=5000)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 5, 10, 50])
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--grow", action="store_true",
                    help="keep all HOT distinct hot queries at every level, so level xR adds HOT*R hot accesses")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    q1 = np.asarray(read_u8bin(WLD / "knn2000_x1_block_s4_pct.u8bin"))
    gt1_ids, gt1_d = read_gt(WLD / "knn2000_x1_block_s4_pct_gt10.bin")
    gt71_ids, _ = read_gt(WLD / "knn2000_x71_block_s4_pct_gt10.bin")
    old_members = np.asarray(read_u8bin(BASE)[np.unique(gt71_ids[N_ORIG:, 0])]).astype(np.float32)
    member_ids = nearest_base(old_members.mean(0), args.region)
    members = np.asarray(read_u8bin(BASE)[member_ids]).astype(np.float32)
    print(f"region: {len(member_ids)} base vectors")

    pick = rng.integers(len(members), size=args.hot)
    hot_q = np.clip(np.rint(members[pick] + rng.normal(0.0, args.sigma, (args.hot, members.shape[1]))), 0, 255)
    hot_q = hot_q.astype(np.uint8)
    hot_ids, hot_d = brute_force_gt(hot_q, 10)
    print(f"distinct hot queries: {len(np.unique(hot_q, axis=0))}")

    for r in args.levels:
        nq = args.hot if args.grow else args.hot // r
        chosen = rng.choice(args.hot, nq, replace=False)
        acc = rng.permutation(np.repeat(chosen, r))
        total = N_ORIG + len(acc)
        is_hot = np.zeros(total, dtype=bool)
        is_hot[rng.choice(total, len(acc), replace=False)] = True
        allq = np.empty((total, q1.shape[1]), dtype=np.uint8)
        allq[~is_hot] = q1
        allq[is_hot] = hot_q[acc]
        ids = np.empty((total, 10), dtype=np.uint32)
        dist = np.empty((total, 10), dtype=np.float32)
        ids[~is_hot], dist[~is_hot] = gt1_ids[:, :10], gt1_d[:, :10]
        ids[is_hot], dist[is_hot] = hot_ids[acc], hot_d[acc]
        tag = f"knn{args.region}_h{args.hot}{'g' if args.grow else 'x'}{r}_block_s4_pct"
        with (WLD / f"{tag}.u8bin").open("wb") as f:
            np.array(allq.shape, dtype=np.int32).tofile(f)
            allq.tofile(f)
        write_gt(WLD / f"{tag}_gt10.bin", ids, dist)
        np.save(WLD / f"{tag}_is_region.npy", is_hot)
        meta = {"region": args.region, "hot_distinct_pool": args.hot, "repeat": r, "hot_queries": int(nq),
                "hot_accesses": int(len(acc)), "uniform": N_ORIG, "n": int(total), "interleaved": True,
                "sigma": args.sigma, "seed": args.seed}
        (WLD / f"{tag}.json").write_text(json.dumps(meta, indent=2))
        print(tag, json.dumps(meta))


if __name__ == "__main__":
    main()
