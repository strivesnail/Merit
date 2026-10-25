#!/usr/bin/env python3
"""Precompute SPACEV1B workloads from the Bing query log + official queries.

Steps (all resumable via marker files under --out):
  1. Deduplicate query_log.bin -> unique queries + inverse map
  2. GPU brute-force GT for unique queries (DiskANN gt format, K=10)
  3. Expand GT to the full log order; also emit official query symlink
  4. Train/test split of the log (first half profile / second half search)
  5. Skewness summary JSON for the paper

Uses CUDA device --gpu (default 1) so it can run beside DiskANN build on the CPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def read_i8bin(path: Path) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.memmap(path, dtype=np.int8, mode="r", offset=8, shape=(int(n), int(d)))


def write_i8bin(path: Path, x: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array(x.shape, dtype=np.int32).tofile(f)
        np.asarray(x, dtype=np.int8).tofile(f)


def write_gt(path: Path, ids: np.ndarray, dist: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array(ids.shape, dtype=np.int32).tofile(f)
        ids.astype(np.uint32).tofile(f)
        dist.astype(np.float32).tofile(f)


def read_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    n, k = np.fromfile(path, dtype=np.int32, count=2)
    ids = np.fromfile(path, dtype=np.uint32, count=n * k, offset=8).reshape(n, k)
    dist = np.fromfile(path, dtype=np.float32, count=n * k, offset=8 + 4 * n * k).reshape(n, k)
    return ids, dist


def brute_force_gt(base: np.ndarray, q: np.ndarray, k: int, device: str,
                   cand: int = 64, chunk: int = 100_000, qbatch: int = 2048
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k via one-pass GPU streaming L2; queries batched per base chunk."""
    q = np.asarray(q, dtype=np.int8)
    nq = len(q)
    best_d = torch.full((nq, cand), float("inf"), device=device)
    best_i = torch.zeros((nq, cand), dtype=torch.int64, device=device)
    nbase = base.shape[0]
    for s in range(0, nbase, chunk):
        xb = np.asarray(base[s:s + chunk], dtype=np.float32)
        x = torch.tensor(xb, device=device)
        xn = (x * x).sum(1)
        nk = min(cand, x.shape[0])
        for b0 in range(0, nq, qbatch):
            qb = torch.tensor(q[b0:b0 + qbatch].astype(np.float32), device=device)
            d = (qb * qb).sum(1, keepdim=True) - 2.0 * qb @ x.T + xn[None]
            dv, di = torch.topk(d, nk, dim=1, largest=False)
            bd = best_d[b0:b0 + len(qb)]
            bi = best_i[b0:b0 + len(qb)]
            bd, sel = torch.topk(torch.cat([bd, dv], 1), cand, dim=1, largest=False)
            bi = torch.gather(torch.cat([bi, di + s], 1), 1, sel)
            best_d[b0:b0 + len(qb)] = bd
            best_i[b0:b0 + len(qb)] = bi
        if (s // chunk) % 50 == 0:
            print(f"  gt {min(s + chunk, nbase)}/{nbase}", flush=True)
        del x, xn
    ids = best_i.cpu().numpy()
    # Exact integer L2 on candidates (handles float ties).
    uniq = np.unique(ids)
    vec = np.asarray(base[uniq], dtype=np.int16)
    cv = vec[np.searchsorted(uniq, ids)]
    exact = ((cv.astype(np.int32) - q[:, None, :].astype(np.int32)) ** 2).sum(2)
    order = np.argsort(exact, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(ids, order, 1).astype(np.uint32), np.take_along_axis(exact, order, 1).astype(np.float32)


def skew_stats(order: np.ndarray) -> dict:
    counts = np.bincount(order)
    counts = counts[counts > 0]
    ranked = np.sort(counts)[::-1]
    n = int(order.size)
    n_unique = int(len(counts))
    top1 = int(ranked[0]) if n_unique else 0
    top1pct = int(ranked[: max(1, n_unique // 100)].sum()) if n_unique else 0
    return {
        "n_access": n,
        "n_unique": n_unique,
        "repeat_frac": round(1.0 - n_unique / n, 4) if n else 0.0,
        "hottest_count": top1,
        "hottest_share": round(top1 / n, 4) if n else 0.0,
        "top1pct_share": round(top1pct / n, 4) if n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("/mnt/raid5/spacev1b/base.1B.i8bin"))
    ap.add_argument("--src", type=Path, default=Path("/mnt/hdd/merit_park/spacev1b"))
    ap.add_argument("--out", type=Path, default=Path("/home/jianz/workload/data/spacev1b/workloads"))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--skip-gt", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"
    status = args.out / "prep_status.txt"

    def log(msg: str) -> None:
        line = msg
        print(line, flush=True)
        with status.open("a") as f:
            f.write(line + "\n")

    log_q = read_i8bin(args.src / "query_log.bin")
    off_q = read_i8bin(args.src / "query.bin")
    # --- 1. unique queries ---
    uniq_path = args.out / "query_log_unique.i8bin"
    map_path = args.out / "query_log_to_unique.bin"
    if not (args.out / "unique.done").exists():
        log(f"dedup log n={len(log_q)}")
        # view as rows of bytes for hashing
        view = np.asarray(log_q).view(np.dtype((np.void, log_q.shape[1]))).ravel()
        uniq_view, inv = np.unique(view, return_inverse=True)
        uniq = uniq_view.view(np.int8).reshape(-1, log_q.shape[1])
        write_i8bin(uniq_path, uniq)
        inv.astype(np.int32).tofile(map_path)
        (args.out / "unique.done").write_text(f"{len(uniq)} {len(log_q)}\n")
        log(f"unique={len(uniq)} / {len(log_q)}")
    else:
        uniq = np.asarray(read_i8bin(uniq_path))
        inv = np.fromfile(map_path, dtype=np.int32)
        log(f"reuse unique={len(uniq)}")

    # --- 2. GT for unique ---
    uniq_gt = args.out / "query_log_unique_gt10.bin"
    if not args.skip_gt and not (args.out / "unique_gt.done").exists():
        log(f"gpu gt start device={args.gpu} nq={len(uniq)} base={args.base}")
        base = read_i8bin(args.base)
        ids, dist = brute_force_gt(base, uniq, args.k, device, chunk=args.chunk)
        write_gt(uniq_gt, ids, dist)
        (args.out / "unique_gt.done").write_text("ok\n")
        log("unique gt done")
    elif uniq_gt.exists():
        log("reuse unique gt")

    # --- 3. full log order + official ---
    full_q = args.out / "query_log.i8bin"
    full_gt = args.out / "query_log_gt10.bin"
    if not (args.out / "full_log.done").exists() and uniq_gt.exists():
        write_i8bin(full_q, np.asarray(log_q))
        uids, ud = read_gt(uniq_gt)
        write_gt(full_gt, uids[inv], ud[inv])
        # official
        write_i8bin(args.out / "query_public.i8bin", np.asarray(off_q))
        # truth.bin is already DiskANN format
        if not (args.out / "query_public_gt10.bin").exists():
            os.symlink(args.src / "truth.bin", args.out / "query_public_gt10.bin")
        (args.out / "full_log.done").write_text("ok\n")
        log("full log + official ready")

    # --- 4. train / test split (first / second half of accesses) ---
    if uniq_gt.exists() and not (args.out / "split.done").exists():
        n = len(log_q)
        mid = n // 2
        for tag, sl in [("log_train", slice(0, mid)), ("log_test", slice(mid, n))]:
            write_i8bin(args.out / f"{tag}.i8bin", np.asarray(log_q[sl]))
            uids, ud = read_gt(uniq_gt)
            write_gt(args.out / f"{tag}_gt10.bin", uids[inv[sl]], ud[inv[sl]])
        (args.out / "split.done").write_text(f"train=0:{mid} test={mid}:{n}\n")
        log(f"split train=0:{mid} test={mid}:{n}")

    # --- 5. skew summary ---
    stats = {
        "log": skew_stats(inv),
        "train": skew_stats(inv[: len(inv) // 2]) if len(inv) else {},
        "test": skew_stats(inv[len(inv) // 2 :]) if len(inv) else {},
        "note": "Bing SPACEV1B query_log.bin; no timestamps; file order treated as arrival order",
    }
    (args.out / "skew_stats.json").write_text(json.dumps(stats, indent=2))
    log(json.dumps(stats["log"]))
    log("PREP DONE")


if __name__ == "__main__":
    main()
