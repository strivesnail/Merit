#!/usr/bin/env python3
"""Exact top-k L2 ground truth by scanning the vectors stored in a DiskANN disk index.

Each 4 KB sector after the header holds `nnps` fixed-size node slots whose first `ndims`
bytes are the uint8 vector. Distances of uint8 vectors are exact in FP32 (TF32 disabled).
Writes one DiskANN-format ground-truth file per input query file.
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

SECTOR = 4096


def read_u8bin(path: Path) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.fromfile(path, dtype=np.uint8, offset=8).reshape(int(n), int(d))


def reader(path: Path, npts: int, nnps: int, node_len: int, ndims: int,
           chunk_sectors: int, outs: list[queue.Queue]) -> None:
    total_sectors = (npts + nnps - 1) // nnps
    fd = os.open(path, os.O_RDONLY)
    try:
        for s0 in range(0, total_sectors, chunk_sectors):
            s1 = min(total_sectors, s0 + chunk_sectors)
            buf = os.pread(fd, (s1 - s0) * SECTOR, (1 + s0) * SECTOR)
            sec = np.frombuffer(buf, dtype=np.uint8).reshape(s1 - s0, SECTOR)
            slots = sec[:, :nnps * node_len].reshape(s1 - s0, nnps, node_len)[:, :, :ndims]
            base = s0 * nnps
            n = min(npts - base, (s1 - s0) * nnps)
            item = (base, np.ascontiguousarray(slots.reshape(-1, ndims)[:n]))
            for out in outs:
                out.put(item)
    finally:
        os.close(fd)
        for out in outs:
            out.put(None)


def scan(dev: torch.device, q_np: np.ndarray, k: int, tile: int, npts: int,
         qu: queue.Queue, res: dict, tag: str) -> None:
    q = torch.from_numpy(q_np).to(dev, torch.float32)
    qn = (q * q).sum(1, keepdim=True)
    best_d = torch.full((q.shape[0], k), float("inf"), device=dev)
    best_i = torch.full((q.shape[0], k), -1, dtype=torch.int64, device=dev)
    t0 = time.time()
    while (item := qu.get()) is not None:
        base, x_np = item
        x_all = torch.from_numpy(x_np).to(dev)
        for b in range(0, x_all.shape[0], tile):
            x = x_all[b:b + tile].to(torch.float32)
            d = q @ x.T
            d.mul_(-2.0).add_((x * x).sum(1)[None, :]).add_(qn)
            td, ti = torch.topk(d, k, dim=1, largest=False)
            cd = torch.cat([best_d, td], 1)
            ci = torch.cat([best_i, ti + (base + b)], 1)
            best_d, o = torch.topk(cd, k, dim=1, largest=False)
            best_i = torch.gather(ci, 1, o)
        done = base + x_np.shape[0]
        el = time.time() - t0
        print(f"[{tag}] {done/1e6:8.1f}M / {npts/1e6:.0f}M  {el:7.0f}s  eta {el*(npts/done-1):7.0f}s",
              flush=True)
    res[tag] = (best_d.cpu().numpy(), best_i.cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--queries", type=Path, nargs="+", required=True)
    ap.add_argument("--npts", type=int, default=1_000_000_000)
    ap.add_argument("--ndims", type=int, default=128)
    ap.add_argument("--node_len", type=int, default=388)
    ap.add_argument("--nnps", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tile", type=int, default=65536)
    ap.add_argument("--chunk_sectors", type=int, default=262144)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    k = args.k
    qs = [read_u8bin(p) for p in args.queries]
    q_all = np.concatenate(qs)
    ngpu = torch.cuda.device_count()
    parts = np.array_split(q_all, ngpu)
    qus = [queue.Queue(maxsize=3) for _ in range(ngpu)]
    res: dict = {}
    workers = [threading.Thread(target=scan, args=(torch.device(f"cuda:{g}"), parts[g], k, args.tile,
                                                   args.npts, qus[g], res, str(g))) for g in range(ngpu)]
    for w in workers:
        w.start()
    reader(args.index, args.npts, args.nnps, args.node_len, args.ndims, args.chunk_sectors, qus)
    for w in workers:
        w.join()
    bd = np.concatenate([res[str(g)][0] for g in range(ngpu)])
    bi = np.concatenate([res[str(g)][1] for g in range(ngpu)]).astype(np.uint32)
    off = 0
    for p, qa in zip(args.queries, qs):
        n = qa.shape[0]
        out = p.parent / (p.stem + f"_gt{k}.bin")
        with out.open("wb") as f:
            np.array([n, k], dtype=np.int32).tofile(f)
            bi[off:off + n].tofile(f)
            bd[off:off + n].astype(np.float32).tofile(f)
        off += n
        print("wrote", out)


if __name__ == "__main__":
    main()
