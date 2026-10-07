#!/usr/bin/env python3
"""Numbered report figures (Fig. 1-7) for MERIT, styled for ACM acmart sigconf (SIGMOD)."""

import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

RUNS = "/mnt/graid_single/sift100m/runs"
STABLE = f"{RUNS}/graid_stable_a1p2_n2p5_d100_3way_t20_20260929"
MOVING = f"{RUNS}/moving_a0p8_5x1m_tuned_20261002"
SWEEP = f"{RUNS}/moving_a0p8_5x1m_sweep_20261002"
ZIPF = f"{RUNS}/moving_zipf_n2p5_d100_hsweep_20261003"
ROUNDLEN = f"{RUNS}/moving_a0p8_roundlen_n2p5_d100_hsweep_20261003"
S1B = "/mnt/graid_single/sift1b/runs/stable_a1p2_n2p5_d100_t20/bloom_20261004"
OUT = "/home/jianz/workload/code/Merit/docs/figures/report"
CACHE = "/tmp/report_fig_cache.json"
FONT_DIR = os.path.expanduser("~/.fonts/libertinus")
SKIP = 10000

# acmart sigconf: one column is 3.33 in, the full text width is 7 in; body text is 9 pt Libertine.
COL, FULL = 3.33, 7.0
for f in os.listdir(FONT_DIR) if os.path.isdir(FONT_DIR) else []:
    if f.endswith(".otf"):
        font_manager.fontManager.addfont(os.path.join(FONT_DIR, f))
plt.rcParams.update({
    "font.family": "Libertinus Sans",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Libertinus Sans",
    "mathtext.it": "Libertinus Sans:italic",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.columnspacing": 1.0,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

# Paul Tol high-contrast palette (colour-blind safe, distinct in greyscale).
MODE_STYLE = {"DiskANN": ("#4477AA", ""), "MERIT-N": ("#DDAA33", ""), "MERIT": ("#BB5566", "")}
STACK = [("io", "I/O", "#4477AA", ""), ("cpu", "CPU", "#BBBBBB", ""), ("other", "Other", "#BB5566", "")]
BAR_EDGE = dict(edgecolor="black", linewidth=0.5)
VAL_FS = 6

os.makedirs(OUT, exist_ok=True)
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def run_qps(d):
    q = re.findall(r"^\s+100\s+4\s+([\d.]+)", open(f"{d}/run.out", errors="ignore").read(), re.M)
    return float(q[-1])


def run_edges(d):
    return int(re.findall(r"mcache_edges=(\d+)", open(f"{d}/run.out", errors="ignore").read())[-1])


def stats(d):
    if d in cache:
        return cache[d]
    cols = ["total_us", "io_us", "cpu_us", "n_disk_reads"]
    with open(f"{d}/qstats.csv") as f:
        r = csv.reader(f)
        h = next(r)
        idx = [h.index(c) for c in cols]
        s = [0.0] * len(cols)
        n = 0
        for i, row in enumerate(r):
            if i < SKIP:
                continue
            for k, j in enumerate(idx):
                s[k] += float(row[j])
            n += 1
    m = [x / n for x in s]
    cache[d] = {"total": m[0], "io": m[1], "cpu": m[2], "other": m[0] - m[1] - m[2], "reads": m[3],
                "qps": run_qps(d)}
    json.dump(cache, open(CACHE, "w"))
    return cache[d]


def subcaption(ax, text, offset=-24):
    ax.annotate(text, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, offset), textcoords="offset points",
                ha="center", va="top", fontsize=8)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"{OUT}/{stem}.pdf")


def bar(ax, x, h, w, mode, **kw):
    color, hatch = MODE_STYLE[mode]
    ax.bar(x, h, w, color=color, hatch=hatch, **BAR_EDGE, **kw)


# Each workload: DiskANN BFS given MERIT's N+M-cache memory, MERIT-N (N-cache only), full MERIT.
WORKLOADS = [
    ("Zipf $\\alpha$=1.2",
     [("DiskANN", f"{STABLE}/bloom_20261004/diskann_bfs_eqmem_nm"),
      ("MERIT-N", f"{STABLE}/bloom_20261004/diskann_ncache"),
      ("MERIT", f"{STABLE}/bloom_20261004/vanilla_merit")]),
    ("Hotspot shift ($\\alpha$=0.8)",
     [("DiskANN", f"{MOVING}/diskann_bfs"),
      ("MERIT-N", f"{MOVING}/diskann_ncache"),
      ("MERIT", f"{MOVING}/vanilla_merit")]),
]

# Fig. 1: overall throughput on SIFT100M and SIFT1B
SCALES = [("SIFT100M", WORKLOADS[0][1]),
          ("SIFT1B", [("DiskANN", f"{S1B}/diskann_bfs_eqmem_nm"), ("MERIT-N", f"{S1B}/diskann_ncache"),
                      ("MERIT", f"{S1B}/vanilla_merit")])]
fig, ax = plt.subplots(figsize=(COL, 1.7))
width = 0.26
for g, (name, modes) in enumerate(SCALES):
    for k, (label, d) in enumerate(modes):
        q = run_qps(d)
        x = g + (k - 1) * width
        bar(ax, x, q, width, label, label=label if g == 0 else None)
        ax.text(x, q + 120, f"{q:.0f}", ha="center", va="bottom", fontsize=VAL_FS)
ax.set_xticks(range(len(SCALES)))
ax.set_xticklabels([s[0] for s in SCALES])
ax.set_ylabel("Throughput (QPS)")
ax.set_ylim(0, 10500)
ax.legend(loc="upper left", ncol=3)
ax.grid(axis="y")
ax.set_axisbelow(True)
save(fig, "fig1_overall_throughput")

# Fig. 2: latency breakdown
fig, ax = plt.subplots(figsize=(COL, 1.6))
modes = SCALES[1][1]
for k, (label, d) in enumerate(modes):
    s = stats(d)
    bottom = 0
    for key, name, color, hatch in STACK:
        ax.bar(k, s[key], 0.55, bottom=bottom, color=color, hatch=hatch, **BAR_EDGE,
               label=name if k == 0 else None)
        bottom += s[key]
    ax.text(k, bottom + 40, f"{s['total']:.0f}", ha="center", va="bottom", fontsize=VAL_FS)
ax.set_xticks(range(len(modes)))
ax.set_xticklabels([m[0] for m in modes])
ax.set_ylabel("Latency ($\\mu$s)")
ax.set_ylim(0, 5000)
ax.legend(loc="upper right", ncol=3)
ax.grid(axis="y")
ax.set_axisbelow(True)
save(fig, "fig2_latency_breakdown")

# Fig. 3: QPS over time under moving hotspots
fig, ax = plt.subplots(figsize=(COL, 1.6))
window = 50000
SHIFT1B = "/mnt/graid_single/sift1b/runs/shift_a1p2_5x1m_n2p5_d100_t20"
shift_modes = [("DiskANN", f"{SHIFT1B}/diskann_bfs_eqmem_nm"), ("MERIT-N", f"{SHIFT1B}/diskann_ncache"),
               ("MERIT", f"{SHIFT1B}/vanilla_merit")]
if not all(os.path.exists(f"{d}/qstats.csv") and "Beamwidth" in open(f"{d}/run.out", errors="ignore").read()
           for _, d in shift_modes):
    shift_modes = WORKLOADS[1][1]
for label, d in shift_modes:
    with open(f"{d}/qstats.csv") as f:
        r = csv.reader(f)
        j = next(r).index("total_us")
        lat = np.array([float(row[j]) for row in r])
    w = lat[: len(lat) // window * window].reshape(-1, window).mean(axis=1)
    est = (20e6 / w) * (stats(d)["qps"] / (20e6 / lat[SKIP:].mean()))
    x = (np.arange(len(w)) + 0.5) * window / 1e6
    ax.plot(x, est, color=MODE_STYLE[label][0], label=label,
            ls={"DiskANN": "-", "MERIT-N": "--", "MERIT": "-"}[label], lw=1.1 if label == "MERIT" else 0.9)
for b in range(1, 5):
    ax.axvline(b, color="gray", ls=":", lw=0.6)
ax.set_xlim(0, 5)
ax.set_xlabel("Queries (millions)")
ax.set_ylabel("Throughput (QPS)")
ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
ax.grid(axis="y")
save(fig, "fig3_qps_over_time")

# Fig. 4: cache budget sensitivity (first 2M queries of the moving workload)
unique = 96571
ns = [2416, 4832, 9664, 19328]
ds = [4832, 9664, 19328, 38656]
grid = np.full((len(ns), len(ds)), np.nan)
for i, n in enumerate(ns):
    for j, dp in enumerate(ds):
        d = f"{SWEEP}/n{n}_d{dp}"
        if os.path.exists(f"{d}/run.out"):
            grid[i, j] = run_qps(d)
fig, ax = plt.subplots(figsize=(COL, 2.2))
cmap = plt.get_cmap("Greys").copy()
cmap.set_bad("white")
im = ax.imshow(grid, cmap=cmap, origin="lower", vmin=np.nanmin(grid) - 400, vmax=np.nanmax(grid) + 100,
               aspect="auto")
mid = (np.nanmin(grid) + np.nanmax(grid)) / 2
for i in range(len(ns)):
    for j in range(len(ds)):
        v = grid[i, j]
        ax.text(j, i, "N/A" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=7,
                color="white" if not np.isnan(v) and v > mid + 200 else "black")
ax.set_xticks(range(len(ds)))
ax.set_xticklabels([f"{dp * 10 / unique * 100:.0f}%" for dp in ds])
ax.set_yticks(range(len(ns)))
ax.set_yticklabels([f"{n / unique * 100:.1f}%" for n in ns])
ax.set_xlabel("D-cache size")
ax.set_ylabel("N-cache size")
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.06)
cb.set_label("QPS")
cb.outline.set_linewidth(0.5)
save(fig, "fig4_cache_budget_sensitivity")

# Fig. 5: decay half-life sensitivity
hs = [700000, 1400000, 2800000, 5600000, 11200000]
hlab = ["0.7M", "1.4M", "2.8M", "5.6M", "11.2M"]
fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.7))
for a, label, color, marker in [("0p6", "$\\alpha$=0.6", "#004488", "o"), ("0p8", "$\\alpha$=0.8", "#DDAA33", "s"),
                                ("1p0", "$\\alpha$=1.0", "#BB5566", "^"), ("1p2", "$\\alpha$=1.2", "#000000", "D")]:
    reads = [stats(f"{ROUNDLEN}/5x1m_h{h}" if a == "0p8" else f"{ZIPF}/a{a}_h{h}")["reads"] for h in hs]
    best = min(reads)
    axes[0].plot(range(len(hs)), [(r / best - 1) * 100 for r in reads], marker=marker, color=color, label=label)
for wl, label, color, marker in [("5x200k", "200K", "#004488", "o"), ("5x500k", "500K", "#DDAA33", "s"),
                                 ("5x1m", "1M", "#BB5566", "^"), ("5x2m", "2M", "#000000", "D")]:
    if not all(os.path.exists(f"{ROUNDLEN}/{wl}_h{h}/qstats.csv") for h in hs):
        continue
    reads = [stats(f"{ROUNDLEN}/{wl}_h{h}")["reads"] for h in hs]
    best = min(reads)
    axes[1].plot(range(len(hs)), [(r / best - 1) * 100 for r in reads], marker=marker, color=color, label=label)
for p, (ax, title) in enumerate(zip(axes, ["(a) Varying skew", "(b) Varying round length"])):
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels(hlab)
    ax.axvspan(0.7, 1.3, color="gray", alpha=0.18, lw=0)
    ax.set_xlabel("Decay half-life $H$ (node accesses)")
    ax.set_ylabel("Extra disk reads (%)")
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=2, title=None if p == 0 else "Queries per round", title_fontsize=7)
    subcaption(ax, title, offset=-30)
fig.subplots_adjust(wspace=0.22)
save(fig, "fig5_half_life_sensitivity")

# Fig. 6: edge partner cap sensitivity
caps = [("8", "p8_r2"), ("16", "p16"), ("32", "p32_r2"), ("64", "p64")]
fig, ax = plt.subplots(figsize=(COL, 1.5))
vals = [stats(f"{STABLE}/partners_20261002/{d}") for _, d in caps]
for k, v in enumerate(vals):
    bar(ax, k, v["qps"], 0.5, "MERIT")
    ax.text(k, v["qps"] + 120, f"{v['qps']:.0f}", ha="center", va="bottom", fontsize=VAL_FS)
ax.set_xticks(range(len(caps)))
ax.set_xticklabels([c for c, _ in caps])
ax.set_xlabel("Max co-access partners per node")
ax.set_ylabel("Throughput (QPS)")
ax.set_ylim(0, 9500)
ax.grid(axis="y")
ax.set_axisbelow(True)
save(fig, "fig6_partner_cap_sensitivity")

# Fig. 7: edge-heat decay design
designs = [("Global", f"{STABLE}/edgedecay25k_20261002/stable_decay1", f"{MOVING}/vanilla_merit"),
           ("Local", f"{STABLE}/accdecay_20261003/stable",
            f"{MOVING}/accdecay_20261003/vanilla_merit"),
           ("Unified", f"{STABLE}/unifdecay_20261003/stable",
            f"{MOVING}/unifdecay_20261003/vanilla_merit")]
DESIGN_STYLE = [("#4477AA", ""), ("#DDAA33", ""), ("#BB5566", "")]
fig, axes = plt.subplots(1, 2, figsize=(COL, 1.7))
for g, wl in enumerate(["Zipf", "Shift"]):
    for k, (name, ds_, dm_) in enumerate(designs):
        d = ds_ if g == 0 else dm_
        x = g + (k - 1) * 0.27
        q, e = run_qps(d), run_edges(d)
        color, hatch = DESIGN_STYLE[k]
        axes[0].bar(x, q, 0.27, color=color, hatch=hatch, **BAR_EDGE, label=name if g == 0 else None)
        axes[1].bar(x, e / 1e4, 0.27, color=color, hatch=hatch, **BAR_EDGE)
        axes[1].text(x, e / 1e4 * 1.12, f"{e / 1e4:.0f}" if e >= 1e5 else f"{e / 1e4:.1f}", ha="center",
                     va="bottom", fontsize=VAL_FS - 0.5)
for ax in axes:
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Zipf $\\alpha$=1.2", "Hotspot shift"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
axes[0].set_ylabel("Throughput (QPS)")
axes[0].set_ylim(0, 11000)
axes[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(1.1, 1.22))
axes[1].set_yscale("log")
axes[1].set_ylim(1, 600)
axes[1].set_ylabel("M-cache edges ($\\times 10^4$)")
subcaption(axes[0], "(a) Throughput", offset=-16)
subcaption(axes[1], "(b) Metadata size", offset=-16)
fig.subplots_adjust(wspace=0.5)
save(fig, "fig7_decay_design")
