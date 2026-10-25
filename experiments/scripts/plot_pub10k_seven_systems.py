"""20-thread QPS bars: DiskANN, BFS, NeighborCache, Starling, Starling+N+D, MARGO, MARGO+N+D.

DiskANN, NeighborCache (300MB N-cache + 100MB prefetch), Starling, and MARGO
are medians of 3 interleaved reps in fig1_three_vs_baselines.

BFS is the equal-memory 400MB static cache from the 15-thread session. Its QPS
is scaled onto this session by that session's DiskANN anchor.

Starling+N+D and MARGO+N+D are the one-pass page-cache runs with a BFS-filled
N-cache. Their QPS is scaled by the off bar measured in that same pass.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path("/home/jianz/workload/code/Merit")
RUNS = Path("/mnt/nvme_single/sift100m/runs")
FIG1 = RUNS / "fig1_three_vs_baselines/progress.log"
BFS = RUNS / "fig_motiv_15thr_400/progress.log"
PAGE = RUNS / "page_nc_bfs/progress.log"

WORKLOADS = ["10K", "10K+Zipf", "10K×5"]
TAGS = ["pub10k", "zipf", "x5"]
BFS_TAGS = {"pub10k": "pub10k", "zipf": "pub10kz08", "x5": "pub10kx5"}
CONFIGS = [
    ("DiskANN", "#C0C0C0"),
    ("BFS", "#7F7F7F"),
    ("NeighborCache", "#D55E00"),
    ("Starling", "#0072B2"),
    ("Starling+N+D", "#56B4E9"),
    ("MARGO", "#009E73"),
    ("MARGO+N+D", "#E69F00"),
]
OUTS = [
    ROOT / "docs/paper/figures/pub10k_seven_20thr.pdf",
    ROOT / "pub10k_seven_20thr.pdf",
]
PNGS = [
    ROOT / "docs/paper/figures/pub10k_seven_20thr.png",
    ROOT / "pub10k_seven_20thr.png",
    Path("/home/jianz/workload/data/sift100m/figs/pub10k_seven_20thr.png"),
]


def median(xs):
    xs = sorted(xs)
    if len(xs) != 3:
        raise SystemExit(f"expected 3 reps, got {len(xs)}: {xs}")
    return xs[1]


def load():
    fig1 = {(t, k): [] for t in TAGS for k in ("diskann", "online", "starling", "margo")}
    row = re.compile(
        r"\[(pub10k|zipf|x5) (diskann|online|starling|margo)\] rep=\d+ QPS=([0-9.]+)"
    )
    for line in FIG1.read_text().splitlines():
        m = row.search(line)
        if m:
            fig1[(m.group(1), m.group(2))].append(float(m.group(3)))

    bfs = {(t, mb): [] for t in TAGS for mb in ("0", "400")}
    brow = re.compile(
        r"\[(pub10k|pub10kz08|pub10kx5)\] kind=diskann ncMB=(0|400) .*QPS=([0-9.]+)"
    )
    inv = {v: k for k, v in BFS_TAGS.items()}
    for line in BFS.read_text().splitlines():
        m = brow.search(line)
        if m:
            bfs[(inv[m.group(1)], m.group(2))].append(float(m.group(3)))

    page = {(t, s, state): None for t in TAGS for s in ("starling", "margo") for state in ("off", "on")}
    prow = re.compile(
        r"\[(pub10k|zipf|x5) (starling|margo) (off|on)\] rc=0 QPS=([0-9.]+)"
    )
    for line in PAGE.read_text().splitlines():
        m = prow.search(line)
        if m:
            page[(m.group(1), m.group(2), m.group(3))] = float(m.group(4))

    qps = np.zeros((len(TAGS), len(CONFIGS)))
    for i, tag in enumerate(TAGS):
        diskann = median(fig1[(tag, "diskann")])
        neighbor = median(fig1[(tag, "online")])
        starling = median(fig1[(tag, "starling")])
        margo = median(fig1[(tag, "margo")])
        bfs_qps = median(bfs[(tag, "400")]) * diskann / median(bfs[(tag, "0")])
        star_nd = page[(tag, "starling", "on")] * starling / page[(tag, "starling", "off")]
        margo_nd = page[(tag, "margo", "on")] * margo / page[(tag, "margo", "off")]
        qps[i] = [diskann, bfs_qps, neighbor, starling, star_nd, margo, margo_nd]
    return qps


def main():
    qps = load()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.16, 2.65))
    n = len(CONFIGS)
    w = 0.13
    pitch = n * w + 0.05
    x = np.arange(len(WORKLOADS)) * pitch
    for j, (label, color) in enumerate(CONFIGS):
        xs = x + (j - (n - 1) / 2) * w
        ax.bar(
            xs,
            qps[:, j],
            w * 0.92,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            label=label,
            zorder=3,
        )
        for xi, v in zip(xs, qps[:, j]):
            ax.text(
                xi,
                v + 70,
                f"{v / 1000:.1f}K",
                ha="center",
                va="bottom",
                fontsize=5.4,
                color="black",
                clip_on=False,
                zorder=4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS)
    ax.margins(x=0.012)
    ax.set_ylabel("QPS")
    ax.set_ylim(0, 8800)
    ax.set_yticks([0, 2000, 4000, 6000, 8000])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v / 1000)}K" if v else "0")
    )
    for y in (2000, 4000, 6000, 8000):
        ax.axhline(y, color="#CCCCCC", linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        frameon=False,
        ncol=7,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.13)
    print(np.array2string(qps, precision=1, suppress_small=True))
    for p in OUTS:
        p.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(p) as pdf:
            pdf.savefig(fig, dpi=300, bbox_inches="tight", pad_inches=0.04)
        print("wrote", p)
    for p in PNGS:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.04)
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
