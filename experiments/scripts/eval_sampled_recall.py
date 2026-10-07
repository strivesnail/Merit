#!/usr/bin/env python3
"""Recall@k of a DiskANN result file, evaluated on the sampled query positions that have exact GT."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_bin_u32(path: Path) -> np.ndarray:
    n, k = np.fromfile(path, dtype=np.int32, count=2)
    return np.fromfile(path, dtype=np.uint32, offset=8, count=int(n) * int(k)).reshape(int(n), int(k))


def recall(result: Path, pos: np.ndarray, gt: np.ndarray, k: int) -> float:
    res = read_bin_u32(result)[pos, :k]
    g = gt[:, :k]
    return float(np.mean([len(np.intersect1d(a, b)) / k for a, b in zip(res, g)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--pos", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    pos = np.load(args.pos)
    gt = read_bin_u32(args.gt)
    assert gt.shape[0] == len(pos)
    for r in args.results:
        print(f"{r}\trecall@{args.k}={recall(r, pos, gt, args.k):.4f}")


if __name__ == "__main__":
    main()
