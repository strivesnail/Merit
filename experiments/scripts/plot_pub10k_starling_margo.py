"""Single-column QPS bars for the 15-thread 100MB Starling/Margo session.

Same search setup as Fig. 1: SIFT100M, L=128, W=4, 15 threads.
DiskANN and 100MB BFS are the anchors from that same session.
Starling and MARGO use their 1% navigation graph and no extra node cache.
NeighborCache is the online writer from the 15-thread 100MB session
(75MB N-cache + 25MB prefetch). It is scaled onto this session by that
session's prebuilt bar and this session's DiskANN anchor.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

COL_IN = 3.33
HEIGHT_IN = 1.85
LOG = Path("/mnt/nvme_single/sift100m/runs/fig_motiv_15thr_100mb_baselines/progress.log")
WORKLOADS = ["10K", "10K+Zipf", "10K×5"]
TAGS = ["pub10k", "pub10kz08", "pub10kx5"]
KINDS = ["diskann0", "bfs100", "starling_nav", "margo_nav"]
CONFIGS = [
    ("DiskANN", "#C0C0C0"),
    ("100MB BFS", "#7F7F7F"),
    ("Starling", "#0072B2"),
    ("MARGO", "#E69F00"),
    ("NeighborCache", "#D55E00"),
]
# Original 100MB session: DiskANN with no cache, and the prebuilt
# 75MB N + 25MB prefetch bar that the online run repeats.
OLD_DISKANN = np.array([3057.76, 3067.41, 3062.98])
OLD_PREBUILT = np.array([6136.80, 7566.65, 6917.73])
ONLINE_LOG = Path("/mnt/nvme_single/sift100m/runs/fig_motiv_15thr_100mb_online/progress.log")
OUT_DIR = Path("/home/jianz/workload/code/NeighborCache/docs/paper/figures")
OUT_PDF = OUT_DIR / "pub10k_starling_margo_100mb.pdf"
OUT_PNG = OUT_DIR / "pub10k_starling_margo_100mb.png"
OUT_DATA = Path("/home/jianz/workload/data/sift100m/figs/pub10k_starling_margo_100mb.png")

ROW = re.compile(
    r"\[(pub10k|pub10kz08|pub10kx5)\] (\w+) r=\d+\s+128\s+4\s+([0-9.]+)"
)


def medians():
    buckets = {(t, k): [] for t in TAGS for k in KINDS}
    for line in LOG.read_text().splitlines():
        m = ROW.search(line)
        if not m:
            continue
        tag, kind, qps = m.group(1), m.group(2), float(m.group(3))
        if kind in KINDS:
            buckets[(tag, kind)].append(qps)
    qps = np.zeros((len(TAGS), len(KINDS)))
    for i, tag in enumerate(TAGS):
        for j, kind in enumerate(KINDS):
            vals = sorted(buckets[(tag, kind)])
            if len(vals) != 3:
                raise SystemExit(f"{tag} {kind} has {len(vals)} reps, expected 3")
            qps[i, j] = vals[1]
    online = np.zeros(len(TAGS))
    prebuilt = np.zeros(len(TAGS))
    got = {(t, k): [] for t in TAGS for k in ("online", "prebuilt")}
    for line in ONLINE_LOG.read_text().splitlines():
        m = re.search(
            r"\[(pub10k|pub10kz08|pub10kx5)\] kind=(online|prebuilt) .*QPS=([0-9.]+)",
            line,
        )
        if m:
            got[(m.group(1), m.group(2))].append(float(m.group(3)))
    for i, tag in enumerate(TAGS):
        for name, dest in (("online", online), ("prebuilt", prebuilt)):
            vals = sorted(got[(tag, name)])
            if len(vals) != 3:
                raise SystemExit(f"online log {tag} {name} has {len(vals)} reps")
            dest[i] = vals[1]
    # Same-session prebuilt ratio, then this session's DiskANN anchor.
    neighbor = online * (OLD_PREBUILT / prebuilt) * (qps[:, 0] / OLD_DISKANN)
    return np.column_stack([qps, neighbor])


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(COL_IN, HEIGHT_IN))
    x = np.arange(len(WORKLOADS))
    n = len(CONFIGS)
    w = 0.18
    ymax = float(qps.max())
    for j, (label, color) in enumerate(CONFIGS):
        xs = x + (j - (n - 1) / 2) * w
        ax.bar(xs, qps[:, j], w * 0.92, color=color, edgecolor="black", linewidth=0.25, label=label, zorder=3)
        for xi, v in zip(xs, qps[:, j]):
            ax.text(
                xi,
                v + ymax * 0.012,
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
    ax.set_ylim(0, 4800)
    ax.set_yticks([0, 1000, 2000, 3000, 4000])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))
    for y in (1000, 2000):
        ax.axhline(y, color="#CCCCCC", linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.54, 1.0),
        frameon=False,
        ncol=3,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.7,
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.13, right=0.995, top=0.78, bottom=0.16)
    print(np.array2string(qps, precision=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_DATA, dpi=300)
    plt.close(fig)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
