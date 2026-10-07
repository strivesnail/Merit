#!/usr/bin/env python3
"""Collect QPS, latency, mean IOs, and sampled Recall@k of run_1b_pipeann.sh outputs.

Each <root>/<mode>/run.out holds one PipeANN result row per L
(L, width, QPS, avg lat, P99 lat, mean hops, mean IOs[, recall]) and
<root>/<mode>/result_<L>_idx_uint32.bin holds the returned ids.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eval_sampled_recall import read_bin_u32, recall


ROW = re.compile(r"(?<![\d.])(\d+)\s+(\d+)" + r"\s+(\d+\.\d\d)" * 5)


def parse_rows(run_out: Path) -> list[dict]:
    # Without a GT file PipeANN prints no newline after a row, so rows may share a line.
    return [{"L": int(f[0]), "width": int(f[1]), "qps": float(f[2]), "mean_lat_us": float(f[3]),
             "p99_lat_us": float(f[4]), "mean_ios": float(f[6])}
            for f in ROW.findall(run_out.read_text(errors="ignore"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--pos", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    pos = np.load(args.pos)
    gt = read_bin_u32(args.gt)
    out = []
    for d in sorted(p for p in args.root.iterdir() if (p / "run.out").exists()):
        rss = re.search(r"RSS_KB=(\d+)", (d / "run.out").read_text(errors="ignore"))
        for r in parse_rows(d / "run.out"):
            res = d / f"result_{r['L']}_idx_uint32.bin"
            if not res.exists():
                continue
            r.update(system=d.name, recall=recall(res, pos, gt, args.k),
                     rss_gb=int(rss.group(1)) / 2**20 if rss else None)
            out.append(r)
    for r in out:
        print(f"{r['system']:12s} L={r['L']:4d} W={r['width']:3d} recall@{args.k}={r['recall']:.4f} "
              f"QPS={r['qps']:8.1f} IOs={r['mean_ios']:6.1f} lat={r['mean_lat_us']:7.0f}us "
              f"p99={r['p99_lat_us']:7.0f}us rss={r['rss_gb'] or 0:.2f}GB")
    (args.root / "recall_qps.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
