#!/usr/bin/env python3
"""Zipfian variants of the knn2000 hotspot workload.

Keeps the 10K original queries of knn2000_x1 and appends HOT region queries as one block, like
knn2000_x51 (5000) / knn2000_x71 (7000). Region members are recovered as the top-1 ground-truth base vectors of the hot block of
knn2000_x71. Each hot query picks a member with Zipf(s) popularity (random rank order) and adds
Gaussian noise (sigma 4), so s = 0 matches the uniform knn2000_x51 construction.
Ground truth for the new queries is exact top-10 squared L2 over the 100M base (GPU brute force).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

WLD = Path("/home/jianz/workload/data/sift100m/workloads/cluster_hotspot")
BASE = Path("/home/jianz/workload/data/sift100m/base.1B.u8bin.crop_nb_100000000")


def read_u8bin(path: Path) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(int(n), int(d)))


def read_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    n, k = np.fromfile(path, dtype=np.int32, count=2)
    ids = np.fromfile(path, dtype=np.uint32, count=n * k, offset=8).reshape(n, k)
    dist = np.fromfile(path, dtype=np.float32, count=n * k, offset=8 + 4 * n * k).reshape(n, k)
    return ids, dist


def write_gt(path: Path, ids: np.ndarray, dist: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array(ids.shape, dtype=np.int32).tofile(f)
        ids.astype(np.uint32).tofile(f)
        dist.astype(np.float32).tofile(f)


def brute_force_gt(q: np.ndarray, k: int, cand: int = 32, chunk: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    base = read_u8bin(BASE)
    dev = "cuda"
    qt = torch.tensor(q, dtype=torch.float32, device=dev)
    qn = (qt * qt).sum(1, keepdim=True)
    best_d = torch.full((len(q), cand), float("inf"), device=dev)
    best_i = torch.zeros((len(q), cand), dtype=torch.int64, device=dev)
    for s in range(0, base.shape[0], chunk):
        x = torch.tensor(np.asarray(base[s:s + chunk]), dtype=torch.float32, device=dev)
        d = qn - 2.0 * qt @ x.T + (x * x).sum(1)[None]
        dv, di = torch.topk(d, cand, dim=1, largest=False)
        best_d, sel = torch.topk(torch.cat([best_d, dv], 1), cand, dim=1, largest=False)
        best_i = torch.gather(torch.cat([best_i, di + s], 1), 1, sel)
    ids = best_i.cpu().numpy()
    # Exact integer distances on the candidates, then the final top-k.
    vec = np.asarray(base[np.sort(np.unique(ids))])
    pos = {v: j for j, v in enumerate(np.sort(np.unique(ids)))}
    cv = vec[np.vectorize(pos.get)(ids)].astype(np.int64)
    exact = ((cv - q[:, None, :].astype(np.int64)) ** 2).sum(2)
    order = np.argsort(exact, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(ids, order, 1), np.take_along_axis(exact, order, 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zipf", type=float, nargs="+", default=[0.5, 0.8, 1.0, 1.2, 1.5])
    ap.add_argument("--hot", type=int, default=5000)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    q1 = np.asarray(read_u8bin(WLD / "knn2000_x1_block_s4_pct.u8bin"))
    gt1_ids, gt1_d = read_gt(WLD / "knn2000_x1_block_s4_pct_gt10.bin")
    gt71_ids, _ = read_gt(WLD / "knn2000_x71_block_s4_pct_gt10.bin")
    member_ids = np.unique(gt71_ids[len(q1):, 0])
    members = np.asarray(read_u8bin(BASE)[member_ids]).astype(np.float32)
    print(f"region members recovered: {len(member_ids)}")

    hot_all, metas = [], []
    for s in args.zipf:
        rank = rng.permutation(len(members))
        p = 1.0 / np.arange(1, len(members) + 1) ** s
        p /= p.sum()
        pick = rank[rng.choice(len(members), args.hot, p=p)]
        hot = np.clip(np.rint(members[pick] + rng.normal(0.0, args.sigma, (args.hot, members.shape[1]))), 0, 255)
        hot_all.append(hot.astype(np.uint8))
        counts = np.sort(np.bincount(pick, minlength=len(members)))[::-1]
        metas.append({"zipf_s": s, "hot_queries": args.hot, "region_members": int(len(members)),
                      "distinct_members_used": int((counts > 0).sum()),
                      "top1_member_share": float(counts[0] / args.hot),
                      "top10_member_share": float(counts[:10].sum() / args.hot),
                      "top100_member_share": float(counts[:100].sum() / args.hot),
                      "sigma": args.sigma, "n": int(len(q1) + args.hot)})

    check_q = np.asarray(read_u8bin(WLD / "knn2000_x51_block_s4_pct.u8bin")[len(q1):len(q1) + 200])
    check_ids, _ = read_gt(WLD / "knn2000_x51_block_s4_pct_gt10.bin")
    ids, dist = brute_force_gt(np.concatenate(hot_all + [check_q]), 10)
    got = ids[-200:]
    agree = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(got, check_ids[len(q1):len(q1) + 200, :10])])
    print(f"GT check vs existing knn2000_x51 GT: {agree:.4f}")
    ids, dist = ids[:-200], dist[:-200]
    for j, (s, hot, meta) in enumerate(zip(args.zipf, hot_all, metas)):
        tag = f"knn2000_x{args.hot // 100 + 1}z{s:g}_block_s4_pct"
        allq = np.vstack([q1, hot])
        with (WLD / f"{tag}.u8bin").open("wb") as f:
            np.array(allq.shape, dtype=np.int32).tofile(f)
            allq.tofile(f)
        sl = slice(j * args.hot, (j + 1) * args.hot)
        write_gt(WLD / f"{tag}_gt10.bin", np.vstack([gt1_ids[:, :10], ids[sl]]), np.vstack([gt1_d[:, :10], dist[sl]]))
        np.save(WLD / f"{tag}_is_region.npy", np.concatenate([np.zeros(len(q1), bool), np.ones(args.hot, bool)]))
        (WLD / f"{tag}.json").write_text(json.dumps(meta, indent=2))
        print(tag, json.dumps(meta))


if __name__ == "__main__":
    main()
