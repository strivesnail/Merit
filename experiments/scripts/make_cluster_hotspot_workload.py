#!/usr/bin/env python3
"""Add a spatial hotspot to an existing query set without repeating any query.

The base vectors are partitioned into k-means cells (centroids fitted on a base sample). One cell
is chosen as the hot region; extra queries are drawn from base vectors inside that cell plus
Gaussian noise, so every added query is distinct and lies in the region. The extra queries are
shuffled together with the original ones.

Factor f > 1 raises the region's query density to f times its original share; f = 0 makes the
region hold half of all queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read_u8bin(path: Path) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(int(n), int(d)))


def write_u8bin(path: Path, x: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array(x.shape, dtype=np.int32).tofile(f)
        np.ascontiguousarray(x, dtype=np.uint8).tofile(f)


def kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


def assign(x: np.ndarray, c: torch.Tensor, batch: int = 0) -> np.ndarray:
    batch = batch or max(10000, int(2e8 // len(c)))
    out = []
    for b in range(0, len(x), batch):
        t = torch.tensor(np.asarray(x[b:b + batch], dtype=np.float32), device=c.device)
        out.append(torch.cdist(t, c).argmin(1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--factors", type=float, nargs="+", default=[2, 10, 50, 0])
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--fit-sample", type=int, default=100000)
    ap.add_argument("--pool-sample", type=int, default=4000000, help="base vectors scanned for region members")
    ap.add_argument("--sigma", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--keep-total", action="store_true",
                    help="drop as many random non-region original queries as are added")
    ap.add_argument("--append-block", action="store_true",
                    help="keep all original queries in order and append the added queries as one contiguous block")
    ap.add_argument("--add-unit", type=int, default=0,
                    help="factor f adds (f-1)*add_unit queries instead of (f-1)*original region count")
    ap.add_argument("--knn-region", type=int, default=0,
                    help="region = this many pool vectors nearest to a random pool vector (instead of a k-means cell)")
    ap.add_argument("--tag-suffix", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = read_u8bin(args.base)
    q = np.asarray(read_u8bin(args.queries))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    fit_idx = np.sort(rng.choice(base.shape[0], args.fit_sample, replace=False))
    c = kmeans(torch.tensor(np.asarray(base[fit_idx], dtype=np.float32), device=dev), args.k, 30, args.seed)

    qa = assign(q, c)
    counts = np.bincount(qa, minlength=args.k)
    # Pick the cell whose original query count is closest to the median, i.e. a typical region.
    region = int(np.argsort(np.abs(counts - np.median(counts)))[0])
    orig = int(counts[region])

    pool_idx = np.sort(rng.choice(base.shape[0], args.pool_sample, replace=False))
    pool = np.asarray(base[pool_idx])
    members = pool[assign(pool, c) == region]
    name = f"cluster{args.k}"
    if args.knn_region:
        center = torch.tensor(pool[rng.integers(len(pool))].astype(np.float32), device=dev)[None]
        d = np.concatenate([torch.cdist(torch.tensor(pool[b:b + 1000000].astype(np.float32), device=dev),
                                        center)[:, 0].cpu().numpy() for b in range(0, len(pool), 1000000)])
        members = pool[np.argsort(d)[:args.knn_region]]
        qd = torch.cdist(torch.tensor(q.astype(np.float32), device=dev), center)[:, 0].cpu().numpy()
        qa = np.where(qd <= np.sort(d)[args.knn_region - 1], region, -1)
        orig = int((qa == region).sum())
        name = f"knn{args.knn_region}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_all = []
    for f in args.factors:
        if f == 0:
            add = int(round(len(q) / 2 - orig)) if args.keep_total else int(round(len(q) - 2 * orig))
        else:
            add = int(round((f - 1) * (args.add_unit or orig)))
        pick = members[rng.choice(len(members), add, replace=add > len(members))].astype(np.float32)
        extra = np.clip(np.rint(pick + rng.normal(0.0, args.sigma, pick.shape)), 0, 255).astype(np.uint8)
        keep = np.ones(len(q), dtype=bool)
        if args.keep_total and not args.append_block:
            outside = np.flatnonzero(qa != region)
            keep[rng.choice(outside, add, replace=False)] = False
        allq = np.vstack([q[keep], extra])
        is_region = np.concatenate([(qa == region)[keep], np.ones(add, dtype=bool)])
        if not args.append_block:
            perm = rng.permutation(len(allq))
            allq, is_region = allq[perm], is_region[perm]
        tag = ("half" if f == 0 else f"x{int(f)}") + args.tag_suffix
        prefix = args.out_dir / f"{name}_{tag}"
        write_u8bin(prefix.with_suffix(".u8bin"), allq)
        np.save(prefix.parent / (prefix.name + "_is_region.npy"), is_region)
        uniq = len(np.unique(allq.view(np.dtype((np.void, allq.shape[1])))))
        meta = {"factor": f, "k": args.k, "region": region, "orig_region_queries": orig,
                "added": add, "removed": int((~keep).sum()), "n": int(len(allq)), "region_share": float(is_region.mean()),
                "unique_queries": int(uniq), "region_pool_members": int(len(members)), "sigma": args.sigma}
        (prefix.parent / (prefix.name + ".json")).write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta))
        meta_all.append(meta)


if __name__ == "__main__":
    main()
