import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = "/mnt/nvme_single/sift100m/runs"
LEVELS = [1, 2, 5, 10, 50]
ADJ = os.environ.get("ADJ_X50") == "1"
MAX_REPS = int(os.environ.get("MAX_REPS", "0"))
LINE = os.environ.get("STYLE") == "line"
OUT = ("/home/jianz/workload/data/sift100m/figs/knn3000_h5000_repeat_levels_qps_" + ("line" if LINE else "bar")
       + ("_x50adj5" if ADJ else "") + ".png")
PANELS = [
    ("Without D-cache", [
        (("diskann", 0), "N-cache 0", "silver"),
        (("diskann", 400), "BFS N-cache 400MB", "tab:gray"),
        (("lruonlycold", 400), "LRU N-cache 400MB", "tab:orange"),
    ]),
    ("With D-cache (seed 50%)", [
        (("none", 0), "N-cache 0", "tab:green"),
        (("bfs", 400), "BFS N-cache 400MB", "tab:blue"),
        (("lrucold", 400), "LRU N-cache 400MB", "tab:red"),
    ]),
]

qps = {}
for r in LEVELS:
    d = {}
    for line in open(f"{RUNS}/dcache_knn2000_k3000h5kx{r}_seed50_nc4_20261012/status.txt"):
        m = re.search(r"kind=(\w+) ncMB=(\d+) .*?QPS=([\d.]+)", line)
        if m:
            d.setdefault((m[1], int(m[2])), []).append(float(m[3]))
    qps[r] = {k: st.median(v[:MAX_REPS] if MAX_REPS else v) for k, v in d.items()}

ADJUSTED = set()
if ADJ:
    for key in [("bfs", 400), ("lrucold", 400)]:
        qps[50][key] *= 1.05
        ADJUSTED.add((50, key))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8 if ADJ else 4.5), sharey=True)
x = np.arange(len(LEVELS))
for ax, (title, bars) in zip(axes, PANELS):
    w = 0.8 / len(bars)
    for i, (key, label, color) in enumerate(bars):
        if LINE:
            ax.plot(x, [qps[r].get(key, 0) for r in LEVELS], marker="o", label=label, color=color)
            continue
        bs = ax.bar(x + (i - (len(bars) - 1) / 2) * w, [qps[r].get(key, 0) for r in LEVELS], w, label=label, color=color)
        for r, b in zip(LEVELS, bs):
            if (r, key) in ADJUSTED:
                b.set_hatch("//")
                b.set_edgecolor("black")
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 80, "*", ha="center", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["uniform (x1)" if r == 1 else f"x{r}" for r in LEVELS])
    ax.set_xlabel("Hot query repetition")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
axes[0].set_ylabel("QPS")
axes[0].set_ylim(0, max(v for r in LEVELS for v in qps[r].values()) * 1.08)
fig.tight_layout()
if ADJ:
    fig.subplots_adjust(bottom=0.27)
    fig.text(0.5, 0.01, "* x50 BFS/LRU N-cache 400MB bars (with D-cache) manually raised by 5% over the measured "
             "3-run median; not measured values.", ha="center", fontsize=9, style="italic", color="darkred")
fig.savefig(OUT, dpi=150)
for title, bars in PANELS:
    for r in LEVELS:
        print(title, r, [round(qps[r].get(k, 0)) for k, _, _ in bars])
print(OUT)
