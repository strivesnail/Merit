"""20-thread 10K+Zipf bars from the unscaled retest.

DiskANN, online N+D (300MB N-cache + 100MB prefetch), Starling, MARGO.
Starling and MARGO use their navigation graph and page map, with no node cache.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

LOG = Path("/mnt/nvme_single/sift100m/runs/fig1_zipf_online_vs_page_retest/progress.log")
KINDS = ["diskann", "online", "starling", "margo"]
LABELS = ["DiskANN", "N+D", "Starling", "MARGO"]
COLORS = ["#C0C0C0", "#D55E00", "#0072B2", "#009E73"]
OUTS = [
    Path("/home/jianz/workload/code/NeighborCache/docs/paper/figures/pub10k_zipf_20thr.pdf"),
    Path("/home/jianz/workload/code/NeighborCache/pub10k_zipf_starling_margo.pdf"),
]
PNGS = [
    Path("/home/jianz/workload/code/NeighborCache/docs/paper/figures/pub10k_zipf_20thr.png"),
    Path("/home/jianz/workload/code/NeighborCache/pub10k_zipf_starling_margo.png"),
    Path("/home/jianz/workload/data/sift100m/figs/pub10k_zipf_20thr.png"),
]
ROW = re.compile(r"\[(diskann|online|starling|margo)\] rep=\d+ QPS=([0-9.]+)")


def medians():
    buckets = {k: [] for k in KINDS}
    for line in LOG.read_text().splitlines():
        m = ROW.search(line)
        if m:
            buckets[m.group(1)].append(float(m.group(2)))
    vals = []
    for k in KINDS:
        xs = sorted(buckets[k])
        if len(xs) != 3:
            raise SystemExit(f"{k} has {len(xs)} reps")
        vals.append(xs[1])
    return np.array(vals)


def main():
    vals = medians()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.33, 1.85))
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.62, color=COLORS, edgecolor="black", linewidth=0.3, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 80, f"{v / 1000:.1f}K", ha="center", va="bottom", fontsize=7, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("QPS", labelpad=1)
    ax.set_title("10K+Zipf", fontsize=8, pad=3)
    ax.set_ylim(0, 7600)
    ax.set_yticks([0, 2000, 4000, 6000])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v / 1000)}K" if v else "0"))
    for y in (2000,):
        ax.axhline(y, color="#CCCCCC", linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.16)
    print(vals)
    for p in OUTS:
        p.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(p) as pdf:
            pdf.savefig(fig, dpi=300)
        print("wrote", p)
    for p in PNGS:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
