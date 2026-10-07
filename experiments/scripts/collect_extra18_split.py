#!/usr/bin/env python3
"""Collect and plot the SIFT10M Zipf-1.2 extra-18-MiB cache split sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("/mnt/graid_single/sift10m/runs/extra18_split_a1p2"))
parser.add_argument(
    "--contrib-root", type=Path, default=Path("/mnt/graid_single/sift10m/runs/nd_contribution_a1p2")
)
parser.add_argument(
    "--curve-root", type=Path, default=Path("/mnt/graid_single/sift10m/runs/nd_contribution_curve_a1p2")
)
parser.add_argument("--n-curve-root", type=Path)
parser.add_argument("--d-curve-root", type=Path)
parser.add_argument(
    "--out",
    type=Path,
    default=Path(__file__).resolve().parents[2] / "docs/figures/report/fig_extra18_split_a1p2",
)
args = parser.parse_args()

ROOT = args.root
CONTRIB_ROOT = args.contrib_root
CURVE_ROOT = args.curve_root
N_CURVE_ROOT = args.n_curve_root or CURVE_ROOT
D_CURVE_ROOT = args.d_curve_root or CURVE_ROOT
OUT = args.out
BASE_N = 1173
BASE_D = 4693


def last_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


rows = []
for point in sorted(ROOT.glob("n[0-9][0-9][0-9]")):
    result = point / "recall_qps.json"
    run = point / "merit_L100/run.out"
    if not result.exists() or not run.exists():
        continue
    perf = next(r for r in json.loads(result.read_text()) if r["system"] == "merit" and r["L"] == 100)
    text = run.read_text(errors="ignore")
    pct_n = int(point.name[1:])
    ncap = last_int(r"MERIT N-cache exact node cap: (\d+)", text)
    dcap = last_int(r"d-cache pages<=(\d+)", text)
    mcap = last_int(r"m-cache cap=(\d+)", text)
    after_load = last_int(r"after_load rss=\s*(\d+) kB", text)
    after_search = last_int(r"after_search rss=\s*(\d+) kB", text)
    hwm = last_int(r"after_search rss=\s*\d+ kB hwm=\s*(\d+) kB", text)
    perf.update(
        point=point.name,
        pct_n=pct_n,
        pct_md=100 - pct_n,
        ncap=ncap,
        dpages=dcap,
        mcap=mcap,
        rss_after_load_kb=after_load,
        rss_after_search_kb=after_search,
        hwm_kb=hwm,
        index_extra_mib=(after_search - after_load) / 1024 if after_search and after_load else None,
        added_n=(ncap - BASE_N) if ncap is not None else None,
        added_d=(dcap - BASE_D) if dcap is not None else None,
    )
    rows.append(perf)

ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "split_results.json").write_text(json.dumps(rows, indent=2))

q = {}
for tag in ("q00_none", "q10_n", "q01_d", "q11_both"):
    p = CONTRIB_ROOT / tag / "recall_qps.json"
    if p.exists():
        q[tag] = json.loads(p.read_text())[0]
contrib = None
if len(q) == 4:
    q00, q10, q01, q11 = (q[k]["qps"] for k in ("q00_none", "q10_n", "q01_d", "q11_both"))
    n_gain = ((q10 - q00) + (q11 - q01)) / 2
    d_gain = ((q01 - q00) + (q11 - q10)) / 2
    total_gain = q11 - q00
    contrib = {
        "q00": q00,
        "q10": q10,
        "q01": q01,
        "q11": q11,
        "n_gain": n_gain,
        "d_gain": d_gain,
        "total_gain": total_gain,
        "n_share": n_gain / total_gain,
        "d_share": d_gain / total_gain,
    }
    (ROOT / "nd_contribution.json").write_text(json.dumps(contrib, indent=2))

point_contrib = []
if "q00_none" in q:
    q00 = q["q00_none"]["qps"]
    for r in rows:
        n_path = N_CURVE_ROOT / r["point"] / "n_only/recall_qps.json"
        d_path = D_CURVE_ROOT / r["point"] / "d_only/recall_qps.json"
        if not n_path.exists() or not d_path.exists():
            continue
        q10 = json.loads(n_path.read_text())[0]["qps"]
        q01 = json.loads(d_path.read_text())[0]["qps"]
        q11 = r["qps"]
        n_gain = ((q10 - q00) + (q11 - q01)) / 2
        d_gain = ((q01 - q00) + (q11 - q10)) / 2
        point_contrib.append(
            {
                "point": r["point"],
                "pct_n": r["pct_n"],
                "ncap": r["ncap"],
                "dpages": r["dpages"],
                "q00": q00,
                "q10": q10,
                "q01": q01,
                "q11": q11,
                "n_gain": n_gain,
                "d_gain": d_gain,
                "total_gain": q11 - q00,
            }
        )
if point_contrib:
    (ROOT / "pointwise_contribution.json").write_text(json.dumps(point_contrib, indent=2))

print(
    f"{'% N':>5s} {'N cap':>8s} {'D pages':>8s} {'M cap':>8s} "
    f"{'QPS':>9s} {'reads':>7s} {'recall':>7s} {'index MiB':>10s}"
)
for r in rows:
    print(
        f"{r['pct_n']:5d} {r['ncap']:8d} {r['dpages']:8d} {r['mcap']:8d} "
        f"{r['qps']:9.1f} {r['mean_ios']:7.2f} {r['recall']:7.4f} {r['index_extra_mib']:10.1f}"
    )

if rows:
    best = max(rows, key=lambda r: r["qps"])
    print(
        f"best: {best['pct_n']}% to N, N={best['ncap']}, D={best['dpages']}, "
        f"M={best['mcap']}, QPS={best['qps']:.1f}, reads={best['mean_ios']:.2f}"
    )

    x = [r["pct_n"] for r in rows]
    ncols = 3 if point_contrib else 2
    fig, axes = plt.subplots(1, ncols, figsize=(12.0 if point_contrib else 8.2, 3.1))
    ax, ax2 = axes[0], axes[1]
    ax.plot(x, [r["qps"] for r in rows], "o-", color="#d62728", lw=1.8)
    ax.set_ylabel("QPS")
    ax2.plot(x, [r["mean_ios"] for r in rows], "o-", color="#1f77b4", lw=1.8)
    ax2.set_ylabel("SSD reads / query")
    for a in (ax, ax2):
        a.set_xlabel("Extra 18 MiB assigned to n-cache (%)")
        a.set_xticks(x)
        a.grid(alpha=0.3)
    if point_contrib:
        ax3 = axes[2]
        ax3.plot(
            [r["pct_n"] for r in point_contrib],
            [r["n_gain"] for r in point_contrib],
            "o-",
            color="#d62728",
            label="n-cache",
        )
        ax3.plot(
            [r["pct_n"] for r in point_contrib],
            [r["d_gain"] for r in point_contrib],
            "o-",
            color="#1f77b4",
            label="d-cache",
        )
        ax3.set_ylabel("QPS improvement")
        ax3.set_xlabel("Extra 18 MiB assigned to n-cache (%)")
        ax3.set_xticks([r["pct_n"] for r in point_contrib])
        ax3.set_title("Pointwise Shapley QPS contribution", fontsize=9)
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
    fig.suptitle("SIFT10M Zipf 1.2, L=100, W=4, 20 threads", fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    fig.savefig(OUT.with_suffix(".pdf"))
