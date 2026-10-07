#!/usr/bin/env python3
"""Summarize MERIT vs. equal-memory DiskANN BFS across the SIFT1B workload studies."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/mnt/graid_single/sift1b/runs")
OUT = Path(__file__).resolve().parents[2] / "docs/figures/report/fig_workload_summary"


def qps(run: str, system: str) -> float:
    for row in json.load(open(R / run / "recall_qps.json")):
        if row["system"] == system and row["L"] == 100:
            return row["qps"]
    raise KeyError((run, system))


BFS_A12 = qps("perturbed_s24_recall_qps", "bfs_eqmem")
MERIT_A12 = qps("wgate_zipf_a1p2_1m_s24_L100", "merit")
BFS_UNI = qps("uniform_s24_L100", "bfs_eqmem")
MERIT_UNI = qps("wgate_uniform_1m_s24_L100", "merit")

panels = [
    ("(a) Skew (Zipf $\\alpha$)", ["1.2", "1.0", "0.8", "0.6", "uniform"], [
        (BFS_A12, MERIT_A12),
        (qps("skew_a1p0_s24_L100", "bfs_eqmem"), qps("wgate_zipf_a1p0_1m_s24_L100", "merit")),
        (qps("skew_a0p8_s24_L100", "bfs_eqmem"), qps("wgate_zipf_a0p8_1m_s24_L100", "merit")),
        (qps("skew_a0p6_s24_L100", "bfs_eqmem"), qps("wgate_zipf_a0p6_1m_s24_L100", "merit")),
        (BFS_UNI, MERIT_UNI),
    ]),
    ("(b) Hot-item placement ($\\alpha$=1.2)", ["BFS order", "shuffled\nin region", "global\nrandom"], [
        (BFS_A12, MERIT_A12),
        (qps("placement_shuffle_s24_L100", "bfs_eqmem"), qps("placement_shuffle_s24_L100", "merit")),
        (qps("placement_global_s24_L100", "bfs_eqmem"), qps("placement_global_s24_L100", "merit")),
    ]),
    ("(c) Uniform fraction $p$ (1M queries)", ["0", "0.25", "0.5", "0.75", "1"], [
        (BFS_A12, MERIT_A12),
        (qps("mix_a1p2_u0p25_s24_L100", "bfs_eqmem"), qps("mix_a1p2_u0p25_s24_L100", "merit")),
        (qps("mix_a1p2_u0p5_s24_L100", "bfs_eqmem"), qps("mix_a1p2_u0p5_s24_L100", "merit")),
        (qps("mix_a1p2_u0p75_s24_L100", "bfs_eqmem"), qps("mix_a1p2_u0p75_s24_L100", "merit")),
        (BFS_UNI, MERIT_UNI),
    ]),
    ("(d) 1M Zipf + 1M uniform (2M queries)", ["fixed\ncapacity", "capacity scaled\nto 1.05M keys"], [
        (qps("union_a1p2_uniform_2m_s24_L100", "bfs_eqmem"), qps("union_a1p2_uniform_2m_s24_L100", "merit")),
        (qps("union_a1p2_uniform_2m_s24_L100", "bfs_eqmem"),
         qps("union_a1p2_uniform_2m_s24_scaled_L100", "merit")),
    ]),
]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [5, 3, 5, 2.4]})
for ax, (title, labels, pairs) in zip(axes, panels):
    xs = range(len(labels))
    w = 0.38
    bfs = [p[0] for p in pairs]
    mer = [p[1] for p in pairs]
    ax.bar([x - w / 2 for x in xs], bfs, w, color="#9e9e9e", label="DiskANN BFS (equal memory)")
    ax.bar([x + w / 2 for x in xs], mer, w, color="#1f77b4", label="MERIT")
    for x, b, m in zip(xs, bfs, mer):
        r = m / b
        ax.text(x + w / 2, m + 60, f"{r:.2f}$\\times$", ha="center", va="bottom", fontsize=8,
                color="#c62828" if r < 1 else "black")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 5600)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Throughput (QPS), L=100")
axes[0].legend(fontsize=8, loc="lower left")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=150)
fig.savefig(OUT.with_suffix(".pdf"))
print(OUT.with_suffix(".png"))
