#!/usr/bin/env python3
"""DiskANN BFS vs. MERIT vs. PipeANN on the SIFT1B core workloads, compared at MERIT's recall.

PipeANN pipelined search (L=90/100) is linearly interpolated in recall to MERIT's Recall@10;
PipeANN coroutine search runs DiskANN beam search at L=100 and matches the recall directly.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/mnt/graid_single/sift1b/runs")
OUT = Path(__file__).resolve().parents[2] / "docs/figures/report/fig_pipeann_compare"

WORKLOADS = [  # label, BFS run, MERIT run, PipeANN run
    ("Zipf 1.2", "perturbed_s24_recall_qps", "wgate_zipf_a1p2_1m_s24_L100", "pipeann/zipf_a1p2"),
    ("Zipf 1.0", "skew_a1p0_s24_L100", "wgate_zipf_a1p0_1m_s24_L100", "pipeann/zipf_a1p0"),
    ("Zipf 0.8", "skew_a0p8_s24_L100", "wgate_zipf_a0p8_1m_s24_L100", "pipeann/zipf_a0p8"),
    ("Zipf 0.6", "skew_a0p6_s24_L100", "wgate_zipf_a0p6_1m_s24_L100", "pipeann/zipf_a0p6"),
    ("uniform", "uniform_s24_L100", "wgate_uniform_1m_s24_L100", "pipeann/uniform"),
    ("1M Zipf 1.2\n+1M uniform", "union_a1p2_uniform_2m_s24_L100", "union_a1p2_uniform_2m_s24_L100",
     "pipeann/union"),
]


def row(run: str, system: str, L: int = 100) -> dict:
    for r in json.load(open(R / run / "recall_qps.json")):
        if r["system"] == system and r["L"] == L:
            return r
    raise KeyError((run, system, L))


def at_recall(run: str, system: str, target: float) -> tuple[float, float]:
    """Linear interpolation (or extrapolation from the two nearest L) of QPS and IOs in recall."""
    rows = sorted((r for r in json.load(open(R / run / "recall_qps.json")) if r["system"] == system),
                  key=lambda r: r["recall"])
    a, b = rows[0], rows[-1]
    t = (target - a["recall"]) / (b["recall"] - a["recall"])
    return a["qps"] + t * (b["qps"] - a["qps"]), a["mean_ios"] + t * (b["mean_ios"] - a["mean_ios"])


table = []
for label, bfs_run, merit_run, pa_run in WORKLOADS:
    bfs, merit, coro = row(bfs_run, "bfs_eqmem"), row(merit_run, "merit"), row(pa_run, "coro")
    pipe_qps, pipe_ios = at_recall(pa_run, "pipe", merit["recall"])
    table.append(dict(label=label, recall=merit["recall"], bfs=bfs["qps"], merit=merit["qps"], pipe=pipe_qps,
                      coro=coro["qps"], coro_recall=coro["recall"], bfs_ios=bfs["mean_ios"],
                      merit_ios=merit["mean_ios"], pipe_ios=pipe_ios))

print(f"{'workload':24s} {'recall':>6s} {'BFS':>6s} {'MERIT':>6s} {'Pipe':>6s} {'Coro':>6s}  "
      f"{'IO BFS':>6s} {'IO MER':>6s} {'IO Pipe':>7s}")
for t in table:
    print(f"{t['label'].replace(chr(10), ' '):24s} {t['recall']:6.4f} {t['bfs']:6.0f} {t['merit']:6.0f} "
          f"{t['pipe']:6.0f} {t['coro']:6.0f}  {t['bfs_ios']:6.1f} {t['merit_ios']:6.1f} {t['pipe_ios']:7.1f}"
          f"   coro recall {t['coro_recall']:.4f}")

systems = [("bfs", "DiskANN BFS", "#9e9e9e"), ("merit", "MERIT", "#1f77b4"),
           ("pipe", "PipeANN (pipelined)", "#ff7f0e"), ("coro", "PipeANN (coroutine)", "#2ca02c")]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [3, 2]})
xs = range(len(table))
w = 0.2
for i, (key, name, color) in enumerate(systems):
    ax.bar([x + (i - 1.5) * w for x in xs], [t[key] for t in table], w, color=color, label=name)
ax.set_xticks(list(xs))
ax.set_xticklabels([t["label"] for t in table], fontsize=8)
ax.set_ylabel("Throughput (QPS) at MERIT's recall")
ax.set_title("(a) Throughput, equal memory (~31 GB RSS), 20 threads", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))

for i, (key, name, color) in enumerate(systems[:3]):
    ax2.bar([x + (i - 1) * 0.27 for x in xs], [t[f"{key}_ios"] for t in table], 0.27, color=color, label=name)
ax2.set_xticks(list(xs))
ax2.set_xticklabels([t["label"] for t in table], fontsize=7)
ax2.set_ylabel("Mean SSD reads per query")
ax2.set_title("(b) SSD reads per query", fontsize=9)
ax2.grid(axis="y", alpha=0.3)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
print(OUT.with_suffix(".png"))
