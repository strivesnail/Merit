"""QPS bars after shrinking the caches.

N holds 20% of visited nodes and starts empty. Prefetch holds 5% of visited
nodes in bytes. D is prefilled before search: one page per hottest 30% of
visited nodes. DiskANN, BFS, Starling, and MARGO are the immediately preceding
20-thread baselines. DEEP baselines are the pinned 20-thread numbers. The
pinned DEEP figure is not overwritten.
"""
from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Stored series are [10K, 10K+Zipf, 10K×5]. Display puts the third workload before the second.
WORKLOADS = ["10K", "10K×5", "10K+Zipf"]
WORKLOAD_INDEX = [0, 2, 1]
# Each comparison shares one hue. The reference bar is solid; the variant is lighter and hatched.
# DiskANN+NC is compared with BFS. DiskANN stays its own solid gray.
COLORS = {
    "DiskANN": "#C0C0C0",
    "DiskANN-BSF": "#7F7F7F",
    "DiskANN+NC": "#7F7F7F",
    "Starling": "#0072B2",
    "Starling+NC": "#0072B2",
    "MARGO": "#009E73",
    "MARGO+NC": "#009E73",
    "PipeANN": "#882255",
    "PipeANN+NC": "#882255",
}
HATCH = {
    "DiskANN+NC": "xx",
    "Starling+NC": "///",
    "MARGO+NC": "///",
    "PipeANN+NC": "///",
}


def shade(hex_color, toward_white=0.48):
    red, green, blue = mcolors.to_rgb(hex_color)
    return tuple(channel + (1.0 - channel) * toward_white for channel in (red, green, blue))


def legend_in_bar_order(handles, labels, ncol):
    """Row-major legend. Matplotlib packs columns, so feed it the transposed order."""
    n = len(labels)
    nrow = int(np.ceil(n / ncol))
    heights = [len(col) for col in np.array_split(np.arange(n), ncol)]
    grid = [[None] * ncol for _ in range(nrow)]
    for i, item in enumerate(zip(handles, labels)):
        grid[i // ncol][i % ncol] = item
    ordered = []
    for col, height in enumerate(heights):
        for row in range(height):
            ordered.append(grid[row][col])
    out_handles, out_labels = zip(*ordered)
    return list(out_handles), list(out_labels)


PANELS = [
    (
        "SIFT100M",
        {
            "DiskANN": [3675.14, 3685.88, 3663.94],
            "DiskANN-BSF": [3996.30, 4009.28, 4004.16],
            "DiskANN+NC": [4288.73, 5490.17, 4834.56],
            "Starling": [4476.82, 4456.29, 4453.85],
            "Starling+NC": [5882.28, 6604.21, 6541.79],
            "MARGO": [4726.96, 4731.53, 4700.20],
            "MARGO+NC": [5902.58, 6865.76, 6578.10],
        },
    ),
    (
        "SIFT10M",
        {
            "DiskANN": [3686.55, 3696.47, 3693.76],
            "DiskANN-BSF": [4205.23, 4208.61, 4212.45],
            "DiskANN+NC": [4340.12, 5433.08, 4585.27],
            "Starling": [4711.81, 4702.23, 4647.76],
            "Starling+NC": [6309.78, 7512.92, 6941.51],
            "MARGO": [4427.66, 4432.60, 4353.28],
            "MARGO+NC": [6282.05, 7436.37, 6897.15],
            "PipeANN": [4287.94, 4300.84, 4315.30],
            "PipeANN+NC": [5574.32, 5591.09, 5609.89],
        },
    ),
    (
        "DEEP10M",
        {
            "DiskANN": [3702.80, 3701.30, 3703.18],
            "DiskANN-BSF": [4176.68, 4170.09, 4177.22],
            "DiskANN+NC": [4515.66, 5712.43, 5064.49],
            "Starling": [4828.50, 4805.54, 4838.55],
            "Starling+NC": [5993.74, 5976.79, 5494.31],
            "MARGO": [5043.44, 5003.97, 5058.01],
            "MARGO+NC": [5124.53, 6060.09, 6696.07],
        },
    ),
]
ORDER = [
    "DiskANN",
    "DiskANN+NC",
    "DiskANN-BSF",
    "Starling",
    "Starling+NC",
    "MARGO",
    "MARGO+NC",
    "PipeANN",
    "PipeANN+NC",
]
OUT = Path("/mnt/nvme_single/meeting_20261026_sized")


def draw(ax, title, series, ylim=9200):
    labels = [label for label in ORDER if label in series]
    n = len(labels)
    w = 0.13 if n <= 7 else 0.105
    pitch = n * w + 0.05
    x = np.arange(len(WORKLOADS)) * pitch
    for j, label in enumerate(labels):
        values = np.array(series[label], dtype=float)[WORKLOAD_INDEX]
        xs = x + (j - (n - 1) / 2) * w
        color = COLORS[label]
        hatch = HATCH.get(label)
        ax.bar(
            xs,
            values,
            w * 0.92,
            color=shade(color) if hatch else color,
            edgecolor=color if hatch else "black",
            linewidth=0.6 if hatch else 0.3,
            hatch=hatch,
            label=label,
            zorder=3,
        )
        for xi, v in zip(xs, values):
            ax.text(
                xi,
                v + 80,
                f"{v / 1000:.1f}K",
                ha="center",
                va="bottom",
                fontsize=5.2,
                color="black",
                clip_on=False,
                zorder=4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS)
    ax.margins(x=0.02)
    ax.set_ylabel("QPS")
    ax.set_ylim(0, ylim)
    if ylim <= 10000:
        ticks = [0, 2000, 4000, 6000, 8000]
    else:
        ticks = list(range(0, int(ylim), 10000))
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v / 1000)}K" if v else "0"))
    for y in ticks[1:]:
        ax.axhline(y, color="#CCCCCC", linewidth=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=9, pad=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path,
                        default=Path('/mnt/nvme_single/meeting_20261026_prefetch5'),
                        help='Load measured NC bars from a rerun directory; retain original baselines.')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--n-fraction', type=float, default=None,
                        help='N-cache fraction to record and display in the plot title.')
    parser.add_argument('--illustrative-pipeann', action='store_true',
                        help='Display PipeANN+NC as an explicitly illustrative 1.3x baseline.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.results_dir:
        methods = {'DiskANN+NC': 'neighbor', 'Starling+NC': 'starling',
                   'MARGO+NC': 'margo', 'PipeANN+NC': 'pipeann'}
        for title, series in PANELS:
            tag = 'deep' if title == 'DEEP10M' else title.lower()
            for label, method in methods.items():
                if label in series:
                    series[label] = [json.loads((args.results_dir / f'{tag}_{work}' / method /
                                                'measurement.json').read_text())['qps']
                                     for work in ('pub', 'zipf', 'x5')]
        if args.illustrative_pipeann:
            for _, series in PANELS:
                if 'PipeANN' in series:
                    series['PipeANN+NC'] = [round(v * 1.3, 2) for v in series['PipeANN']]
        source_name = 'meeting_qps_source_illustrative.json' if args.illustrative_pipeann else 'meeting_qps_source.json'
        (args.output_dir / source_name).write_text(json.dumps({
            'results_dir': str(args.results_dir.resolve()), 'prefetch_fraction': 0.05,
            'n_fraction': args.n_fraction,
            'workload_storage_order': ['pub', 'zipf', 'x5'], 'panels': PANELS,
            'baseline_source': 'Original 2026-10-26 plot constants; not rerun',
            'pipeann_caveat': 'Old NC constants have no matched logs; new values use current binary.'
            , 'pipeann_display': 'Illustrative: 1.3x PipeANN, not measured' if args.illustrative_pipeann else 'Measured'
        }, indent=2))
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
            "hatch.linewidth": 0.4,
        }
    )
    for title, series in PANELS:
        if args.illustrative_pipeann and 'PipeANN' not in series:
            continue
        fig, ax = plt.subplots(figsize=(7.16, 3.05))
        suffix = (f' (N: {args.n_fraction:.0%}, Prefetch: 5%)' if args.n_fraction is not None
                  else ' (Prefetch: 5%)') if args.results_dir else ''
        draw(ax, title + suffix, series,
             ylim=max(9200, max(max(v) for v in series.values()) * 1.15))
        handles, labels = ax.get_legend_handles_labels()
        ncol = 5 if len(labels) > 7 else 4
        handles, labels = legend_in_bar_order(handles, labels, ncol)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.995),
            frameon=False,
            ncol=ncol,
            handlelength=1.6,
            handletextpad=0.35,
            columnspacing=0.8,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.09, right=0.995, top=0.78, bottom=0.14)
        slug = title.lower()
        if args.illustrative_pipeann:
            slug += '_illustrative'
        png = args.output_dir / f"meeting_qps_{slug}.png"
        pdf = args.output_dir / f"meeting_qps_{slug}.pdf"
        fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.04)
        with PdfPages(pdf) as pp:
            pp.savefig(fig, dpi=300, bbox_inches="tight", pad_inches=0.04)
        print("wrote", png)
        print("wrote", pdf)
        plt.close(fig)


if __name__ == "__main__":
    main()
