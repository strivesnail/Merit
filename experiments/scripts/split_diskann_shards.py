#!/usr/bin/env python3
"""Split oversized DiskANN build shards so each fits in RAM.

A shard with more than --max-pts points is clustered with k-means (k = ceil(n / --target-pts)); every point goes
to its nearest sub-centroid and also to the second nearest when it is near the boundary (squared distance ratio
>= --overlap), mirroring DiskANN's own 2-way overlapping partition. Sub-shard 0 replaces the original id file
(kept as .orig); the rest are appended as new shard numbers, so a DISKANN_BUILD_RESUME build picks them up.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np


def read_ids(path: Path) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    assert d == 1
    return np.fromfile(path, dtype=np.uint32, count=n, offset=8)


def write_ids(path: Path, ids: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array([len(ids), 1], dtype=np.int32).tofile(f)
        np.sort(ids).astype(np.uint32).tofile(f)


def kmeans(x: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        lab = sq_dists(x, c).argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    return c


def sq_dists(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (x * x).sum(1, keepdims=True) - 2 * x @ c.T + (c * c).sum(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--prefix", type=Path, required=True, help="..._mem.index_tempFiles")
    ap.add_argument("--max-pts", type=int, default=95_000_000)
    ap.add_argument("--target-pts", type=int, default=60_000_000)
    ap.add_argument("--overlap", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shard = lambda p: Path(f"{args.prefix}_subshard-{p}_ids_uint32.bin")
    num = 0
    while shard(num).exists():
        num += 1
    npts, dim = np.fromfile(args.base, dtype=np.int32, count=2)
    base = np.memmap(args.base, dtype=np.int8, mode="r", offset=8, shape=(int(npts), int(dim)))
    rng = np.random.default_rng(args.seed)
    next_id = num
    for p in range(num):
        if Path(f"{args.prefix}_subshard-{p}_mem.index").exists():
            continue
        ids = read_ids(shard(p))
        if len(ids) <= args.max_pts:
            continue
        k = math.ceil(len(ids) / args.target_pts)
        ids = np.sort(ids)
        sample = np.sort(rng.choice(ids, min(len(ids), 500_000), replace=False))
        cent = kmeans(base[sample].astype(np.float32), k, 25, rng)
        parts: list[list[np.ndarray]] = [[] for _ in range(k)]
        for s in range(0, len(ids), 4_000_000):
            chunk = ids[s:s + 4_000_000]
            d = sq_dists(base[chunk].astype(np.float32), cent)
            order = d.argsort(1)
            first, second = order[:, 0], order[:, 1]
            r = np.take_along_axis(d, first[:, None], 1)[:, 0] / np.maximum(
                np.take_along_axis(d, second[:, None], 1)[:, 0], 1e-9)
            for j in range(k):
                parts[j].append(chunk[(first == j) | ((second == j) & (r >= args.overlap))])
        parts_ids = [np.concatenate(x) for x in parts]
        sizes = [len(x) for x in parts_ids]
        assert max(sizes) <= args.max_pts, f"shard {p}: parts {sizes} still too large, lower --overlap"
        os.replace(shard(p), str(shard(p)) + ".orig")
        write_ids(shard(p), parts_ids[0])
        out = [p]
        for x in parts_ids[1:]:
            write_ids(shard(next_id), x)
            out.append(next_id)
            next_id += 1
        stale = Path(f"{args.prefix}_subshard-{p}.bin")
        if stale.exists():
            stale.unlink()
        print(f"shard {p}: {len(ids)} pts -> shards {out} sizes {sizes} (overlap pts {sum(sizes) - len(ids)})",
              flush=True)


if __name__ == "__main__":
    main()
