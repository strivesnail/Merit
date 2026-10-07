#!/usr/bin/env python3
"""Collect the SIFT10M equal-memory comparison (run_10m_compare.sh): MERIT, DiskANN BFS, Starling, MARGO.

Per (workload, system, L): sampled Recall@10, QPS, mean latency, mean SSD page reads per query, peak RSS.
Writes <runs>/compare_10m.json and the bar figure docs/figures/report/fig_10m_starling_margo.{png,pdf}.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eval_sampled_recall import read_bin_u32, recall

WORKLOADS = ["zipf_a1p2_1m_s24", "zipf_a1p0_1m_s24", "zipf_a0p8_1m_s24", "zipf_a0p6_1m_s24", "uniform_1m_s24",
             "union_a1p2_uniform_2m_s24"]
LABEL = {"zipf_a1p2_1m_s24": "Zipf 1.2", "zipf_a1p0_1m_s24": "Zipf 1.0", "zipf_a0p8_1m_s24": "Zipf 0.8",
         "zipf_a0p6_1m_s24": "Zipf 0.6", "uniform_1m_s24": "Uniform", "union_a1p2_uniform_2m_s24": "Mixed"}
SYSTEMS = [("bfs", "DiskANN"), ("starling", "Starling"), ("margo", "MARGO"), ("merit", "MERIT")]
NUM = r"(\d+(?:\.\d+)?)"


def diskann_row(run_out: Path, L: int) -> dict | None:
    text = run_out.read_text(errors="ignore")
    m = re.search(rf"^\s+{L}\s+\d+\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}", text, re.M)
    hwm = re.findall(r"after_search rss=\s*\d+ kB hwm=\s*(\d+) kB", text)
    if not m:
        return None
    return {"qps": float(m[1]), "mean_lat_us": float(m[2]), "mean_ios": float(m[4]),
            "rss_gb": int(hwm[-1]) / 2**20 if hwm else None}


def pagesearch_rows(run_out: Path) -> dict[int, dict]:
    text = run_out.read_text(errors="ignore")
    rss = re.search(r"RSS_KB=(\d+)", text)
    rows = {}
    for m in re.finditer(rf"^\s+(\d+)\s+\d+\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+\d+\s+\d+\s+\d+", text, re.M):
        rows[int(m[1])] = {"qps": float(m[2]), "mean_lat_us": float(m[3]), "mean_ios": float(m[5]),
                           "rss_gb": int(rss[1]) / 2**20 if rss else None}
    return rows


def collect(runs: Path, wl: Path, Ls: list[int]) -> list[dict]:
    out = []
    for w in WORKLOADS:
        pos = np.load(wl / f"{w}_sample_pos.npy")
        gt = read_bin_u32(wl / f"{w}_sample_gt10.bin")
        for s, _ in SYSTEMS:
            for L in Ls:
                if s in ("merit", "bfs"):
                    d = runs / "diskann" / w / f"{'bfs_eqmem' if s == 'bfs' else 'merit'}_L{L}"
                    r = diskann_row(d / "run.out", L) if (d / "run.out").exists() else None
                else:
                    d, r = None, None
                    for cand in sorted((runs / s / w).glob("c*_m*")) if (runs / s / w).exists() else []:
                        if (cand / "run.out").exists() and L in (rows := pagesearch_rows(cand / "run.out")):
                            d, r = cand, rows[L]
                res = d / f"result_{L}_idx_uint32.bin" if d else None
                if r is None or res is None or not res.exists():
                    continue
                r.update(workload=w, system=s, L=L, recall=recall(res, pos, gt, 10))
                out.append(r)
    return out


def at_recall(rows: list[dict], target: float) -> list[dict]:
    """Per (workload, system): QPS (log-linear) and SSD reads (linear) interpolated at a target recall."""
    out = []
    for w in WORKLOADS:
        for s, _ in SYSTEMS:
            pts = sorted((r for r in rows if r["workload"] == w and r["system"] == s), key=lambda r: r["recall"])
            for a, b in zip(pts, pts[1:]):
                if a["recall"] <= target <= b["recall"] and b["recall"] > a["recall"]:
                    t = (target - a["recall"]) / (b["recall"] - a["recall"])
                    out.append({"workload": w, "system": s, "L": f"{a['L']}-{b['L']}",
                                "qps": float(np.exp(np.log(a["qps"]) + t * (np.log(b["qps"]) - np.log(a["qps"])))),
                                "mean_ios": a["mean_ios"] + t * (b["mean_ios"] - a["mean_ios"])})
                    break
    return out


def figure(rows: list[dict], title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = [w for w in WORKLOADS if any(r["workload"] == w for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    width = 0.2
    colors = {"bfs": "#9e9e9e", "starling": "#6baed6", "margo": "#fd8d3c", "merit": "#d62728"}
    for ax, key, ylab in ((axes[0], "qps", "QPS"), (axes[1], "mean_ios", "SSD reads / query")):
        for i, (s, name) in enumerate(SYSTEMS):
            vals = [next((r[key] for r in rows if r["workload"] == w and r["system"] == s), 0) for w in ws]
            ax.bar(np.arange(len(ws)) + (i - 1.5) * width, vals, width, label=name, color=colors[s])
        ax.set_xticks(np.arange(len(ws)))
        ax.set_xticklabels([LABEL[w] for w in ws], fontsize=8)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=4, loc="upper right")
    fig.suptitle(f"SIFT10M, equal memory, 20 threads, W=4, {title}", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("/mnt/graid_single/sift10m/runs"))
    ap.add_argument("--wl", type=Path, default=Path("/home/jianz/workload/data/sift10m/workloads/core"))
    ap.add_argument("--Ls", type=int, nargs="+", default=[15, 20, 30, 50, 100, 150])
    ap.add_argument("--targets", type=float, nargs="+", default=[0.99])
    ap.add_argument("--fig", type=Path,
                    default=Path(__file__).resolve().parents[2] / "docs/figures/report/fig_10m_starling_margo")
    args = ap.parse_args()
    rows = collect(args.runs, args.wl, args.Ls)
    for r in rows:
        print(f"{r['workload']:28s} {r['system']:9s} L={r['L']:3d} recall={r['recall']:.4f} QPS={r['qps']:8.1f} "
              f"IOs={r['mean_ios']:6.2f} lat={r['mean_lat_us']:7.0f}us rss={r['rss_gb'] or 0:.3f}GB")
    matched = {}
    for tgt in args.targets:
        matched[str(tgt)] = m = at_recall(rows, tgt)
        print(f"--- interpolated at Recall@10 = {tgt}")
        for r in m:
            print(f"{r['workload']:28s} {r['system']:9s} L={r['L']:7s} QPS={r['qps']:8.1f} IOs={r['mean_ios']:6.2f}")
        if m:
            figure(m, f"Recall@10={tgt}", args.fig.with_name(args.fig.name + f"_r{str(tgt)[2:]}"))
    (args.runs / "compare_10m.json").write_text(json.dumps({"rows": rows, "at_recall": matched}, indent=2))


if __name__ == "__main__":
    main()
