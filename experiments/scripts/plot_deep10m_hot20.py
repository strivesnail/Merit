"""DEEP10M QPS and Mean IOs. +N is a cold 100MB N-cache. +D is that N-cache plus the LP D-cache."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

WORKLOADS = ["10K", "10K+Zipf", "10K×5"]
CONFIGS = [
    ("Starling", "#0072B2", [4842.21, 4785.57, 4843.62]),
    ("Starling+N", "#56B4E9", [4764.96, 5861.89, 5195.77]),
    ("Starling+N+D", "#A6CEE3", [5751.69, 7170.52, 6262.04]),
    ("MARGO", "#009E73", [5043.44, 5003.97, 5058.01]),
    ("MARGO+N", "#E69F00", [4953.09, 6097.33, 5376.57]),
    ("MARGO+N+D", "#F6C945", [5847.07, 7274.29, 6375.59]),
]
IOS = np.array(
    [
        [107.69, 108.29, 107.69],
        [104.78, 81.75, 94.08],
        [96.11, 75.37, 86.44],
        [102.74, 103.20, 102.74],
        [100.29, 78.14, 89.96],
        [94.06, 73.34, 84.54],
    ]
)
OUT = Path("/mnt/nvme1/deep10m/runs/node_hot20")


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)


def grouped(ax, values, ylabel, ylim, yticks, label_fmt, dy):
    n = len(CONFIGS)
    w = 0.13
    pitch = n * w + 0.08
    x = np.arange(len(WORKLOADS)) * pitch
    for j, (label, color, _) in enumerate(CONFIGS):
        xs = x + (j - (n - 1) / 2) * w
        ax.bar(xs, values[j], w * 0.92, color=color, edgecolor="black", linewidth=0.4, label=label, zorder=3)
        for xi, v in zip(xs, values[j]):
            ax.text(xi, v + dy, label_fmt(v), ha="center", va="bottom", fontsize=5.2, color="black", clip_on=False, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    for y in yticks[1:]:
        ax.axhline(y, color="#CCCCCC", linewidth=0.4, zorder=0)
    style_ax(ax)
    return ax


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    qps = np.array([row for _, _, row in CONFIGS])
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 5.4))
    grouped(
        axes[0],
        qps,
        "QPS",
        (0, 8000),
        [0, 2000, 4000, 6000, 8000],
        lambda v: f"{v / 1000:.1f}K",
        80,
    )
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v / 1000)}K" if v else "0"))
    grouped(axes[1], IOS, "Mean IOs", (0, 150), [0, 40, 80, 120], lambda v: f"{v:.1f}", 1.6)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        frameon=False,
        ncol=6,
        handlelength=1.0,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.06, hspace=0.32)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "deep10m_hot20.png"
    pdf = OUT / "deep10m_hot20.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.04)
    with PdfPages(pdf) as pp:
        pp.savefig(fig, dpi=300, bbox_inches="tight", pad_inches=0.04)
    print("wrote", png)
    print("wrote", pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
