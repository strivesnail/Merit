"""20-thread grouped QPS: DiskANN, N+D, Starling, MARGO on three workloads.

Same session as fig1_three_vs_baselines. N+D is write-on-evict NeighborCache
with a 300MB N-cache and a 100MB prefetch. Starling and MARGO use their
navigation graph and no node cache. L=128, W=4. Medians of 3 interleaved reps.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

LOG = Path("/mnt/nvme_single/sift100m/runs/fig1_three_vs_baselines/progress.log")
WORKLOADS = ["10K", "10K+Zipf", "10K×5"]
TAGS = ["pub10k", "zipf", "x5"]
KINDS = ["diskann", "online", "starling", "margo"]
CONFIGS = [
    ("DiskANN", "#C0C0C0"),
    ("N+D", "#D55E00"),
    ("Starling", "#0072B2"),
    ("MARGO", "#009E73"),
]
COL_IN = 3.33
HEIGHT_IN = 1.85
OUTS = [
    Path("/home/jianz/workload/code/NeighborCache/docs/paper/figures/pub10k_three_vs_baselines_20thr.pdf"),
    Path("/home/jianz/workload/code/NeighborCache/pub10k_three_vs_baselines.pdf"),
]
PNGS = [
    Path("/home/jianz/workload/code/NeighborCache/docs/paper/figures/pub10k_three_vs_baselines_20thr.png"),
    Path("/home/jianz/workload/code/NeighborCache/pub10k_three_vs_baselines.png"),
    Path("/home/jianz/workload/data/sift100m/figs/pub10k_three_vs_baselines_20thr.png"),
]
ROW = re.compile(
    r"\[(pub10k|zipf|x5) (diskann|online|starling|margo)\] rep=\d+ QPS=([0-9.]+)"
)


def medians():
    buckets = {(t, k): [] for t in TAGS for k in KINDS}
    for line in LOG.read_text().splitlines():
        m = ROW.search(line)
        if m:
            buckets[(m.group(1), m.group(2))].append(float(m.group(3)))
    qps = np.zeros((len(TAGS), len(KINDS)))
    for i, tag in enumerate(TAGS):
        for j, kind in enumerate(KINDS):
            xs = sorted(buckets[(tag, kind)])
            if len(xs) != 3:
                raise SystemExit(f"{tag} {kind} has {len(xs)} reps")
            qps[i, j] = xs[1]
    return qps


def main():
    qps = medians()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.major.size": 1.6,
            "ytick.major.size": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(COL_IN, HEIGHT_IN))
    x = np.arange(len(WORKLOADS))
    n = len(CONFIGS)
    w = 0.18
    for j, (label, color) in enumerate(CONFIGS):
        xs = x + (j - (n - 1) / 2) * w
        ax.bar(
            xs,
            qps[:, j],
            w * 0.92,
            color=color,
            edgecolor="black",
            linewidth=0.25,
            label=label,
            zorder=3,
        )
        for xi, v in zip(xs, qps[:, j]):
            ax.text(
                xi,
                v + 90,
                f"{v / 1000:.1f}K",
                ha="center",
                va="bottom",
                fontsize=5.6,
                color="black",
                clip_on=False,
                zorder=4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS)
    ax.set_ylabel("QPS", labelpad=1)
    ax.set_ylim(0, 8000)
    ax.set_yticks([0, 2000, 4000, 6000])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v / 1000)}K" if v else "0")
    )
    for y in (2000, 4000, 6000):
        ax.axhline(y, color="#CCCCCC", linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.55, 1.0),
        frameon=False,
        ncol=4,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.7,
        borderaxespad=0.0,
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.13, right=0.995, top=0.84, bottom=0.16)
    print(np.array2string(qps, precision=2))
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
