#!/usr/bin/env python3
"""Collect QPS, mean IOs, mean/P99 latency, and sampled Recall@k of a run_1b_perturbed.sh sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eval_sampled_recall import read_bin_u32, recall


def parse_row(run_out: Path, L: int, W: int) -> dict | None:
    for line in run_out.read_text(errors="ignore").splitlines():
        f = line.split()
        if len(f) >= 7 and f[0] == str(L) and f[1] == str(W) and re.match(r"^[\d.]+$", f[2]):
            return {"qps": float(f[2]), "mean_lat_us": float(f[3]), "p99_lat_us": float(f[4]),
                    "mean_ios": float(f[5])}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--pos", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--W", type=int, default=4)
    args = ap.parse_args()
    pos = np.load(args.pos)
    gt = read_bin_u32(args.gt)
    rows = []
    for d in sorted(args.root.iterdir()):
        m = re.match(r"^(.+)_L(\d+)$", d.name)
        if not m or not (d / "run.out").exists():
            continue
        sysname, L = m.group(1), int(m.group(2))
        row = parse_row(d / "run.out", L, args.W)
        res = d / f"result_{L}_idx_uint32.bin"
        if row is None or not res.exists():
            continue
        row.update(system=sysname, L=L, recall=recall(res, pos, gt, args.k))
        rows.append(row)
    rows.sort(key=lambda r: (r["system"], r["L"]))
    for r in rows:
        print(f"{r['system']:10s} L={r['L']:4d} recall@{args.k}={r['recall']:.4f} QPS={r['qps']:8.1f} "
              f"IOs={r['mean_ios']:6.1f} lat={r['mean_lat_us']:7.0f}us p99={r['p99_lat_us']:7.0f}us")
    (args.root / "recall_qps.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
