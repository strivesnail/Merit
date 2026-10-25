"""Single-column QPS bars sized like the paper's other column figs (~3.0x1.46in).

Fig. 1, 15 threads, L=128, W=4. Percent is the share of the 1,338,445 visited
nodes held in the node cache. DF is DiskANN's layout. LP is the prebuilt
full-neighborhood layout. The two LP bars also keep a 25MB prefetch.

Bars 1--3 are the original 100MB-session medians (100MB is 20.19% of visited).
20%LRU+LP is the later N-sweep at exact 20% (267,689 nodes), scaled onto this
figure by that sweep's 15% point over the original 15% N+D bar.
5%LRU+LP is the n5 session scaled the same way onto that bar.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ACM sigconf column is about 3.33in. Value labels are 5.6pt and sit on
# one line above each bar. A little overlap between neighbors is fine.
COL_IN = 3.33
HEIGHT_IN = 1.85

WORKLOADS = ["10K", "10K+Zipf", "10K×5"]

# Rows: 10K, 10K+Zipf, 10K×5.
# Columns: 0%+DF, 20%BFS+DF, 20%LRU+DF, 20%LRU+LP, 5%LRU+LP.
QPS = np.array(
    [
        [3057.76, 3307.65, 3171.71, 6253.73 * 6136.80 / 6215.84, 6135.94],
        [3067.41, 3314.35, 4197.56, 7813.05 * 7566.65 / 7598.75, 7002.17],
        [3062.98, 3304.08, 3702.48, 7173.51 * 6917.73 / 6955.22, 6527.09],
    ]
)

# Light bars take black digits; dark bars take white.
CONFIGS = [
    ("0%+DF_Layout", "#C0C0C0", "black"),
    ("20%BFS+DF_Layout", "#7F7F7F", "white"),
    ("20%LRU+DF_Layout", "#E69F00", "black"),
    ("20%LRU+LP_Layout", "#D55E00", "white"),
    ("5%LRU+LP_Layout", "#0072B2", "white"),
]

OUT_DIR = Path("/home/jianz/workload/code/NeighborCache/docs/paper/figures")
OUT_PDF = OUT_DIR / "pub10k_workloads_grouped_qps.pdf"
OUT_PNG = OUT_DIR / "pub10k_workloads_grouped_qps.png"
OUT_DATA = Path("/home/jianz/workload/data/sift100m/figs/pub10k_workloads_grouped_qps.png")


def fmt_qps(v: float) -> str:
    return f"{v / 1000:.1f}K"


def apply_rc():
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


def plot_column(qps):
    fig, ax = plt.subplots(figsize=(COL_IN, HEIGHT_IN))
    x = np.arange(len(WORKLOADS))
    n = len(CONFIGS)
    w = 0.192
    ymax = float(qps.max())

    for j, (label, color, _ink) in enumerate(CONFIGS):
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
                v + ymax * 0.015,
                fmt_qps(v),
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
    ax.set_ylim(0, 8800)
    ax.set_yticks([0, 2000, 4000, 6000, 8000])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}K" if v >= 1000 else f"{int(v)}")
    )
    for y in (2000, 4000, 6000):
        ax.axhline(y, color="#CCCCCC", linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.54, 1.0),
        frameon=False,
        ncol=2,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.7,
        labelspacing=0.2,
        borderaxespad=0.0,
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.12, right=0.995, top=0.78, bottom=0.16)
    return fig


def main():
    apply_rc()
    print(np.array2string(QPS, precision=2))
    fig = plot_column(QPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_DATA, dpi=300)
    plt.close(fig)

    from pypdf import PdfReader

    box = PdfReader(str(OUT_PDF)).pages[0].mediabox
    print(f"ours:     {float(box.width)/72:.3f}x{float(box.height)/72:.3f}in")
    ref = PdfReader(
        "/home/jianz/workload/code/NeighborCache/docs/figures/report/fig2_latency_breakdown.pdf"
    ).pages[0].mediabox
    print(f"ref small:{float(ref.width)/72:.3f}x{float(ref.height)/72:.3f}in")
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
