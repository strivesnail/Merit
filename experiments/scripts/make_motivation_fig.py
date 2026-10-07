#!/usr/bin/env python3
"""Section 2 motivation figure (SIFT1B, Zipf 1.2), styled for ACM acmart sigconf."""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

TRACE = "/mnt/graid_single/sift1b/runs/motivation_trace_a1p2/motivation.json"
CURVE = "/mnt/graid_single/sift1b/runs/motivation_bfs_curve_a1p2"
SIM = "/mnt/graid_single/sift1b/runs/motivation_trace_a1p2/dcache_sim.json"
PARENT = "/mnt/graid_single/sift1b/runs/motivation_trace_a1p2/motivation_parent.json"
MEM_NODES = 1173
OUT = "/home/jianz/workload/code/Merit/docs/figures/report"
FONT_DIR = os.path.expanduser("~/.fonts/libertinus")
COL = 3.33

# DiskANN disk reads per query on SIFT1B Zipf 1.2 measured in earlier runs
# (runs/stable_a1p2_n2p5_d100_t20: diskann_bfs_0, bloom_20261004/diskann_bfs_eqmem_nm, diskann_bfs_eqmem_nm).
BFS = {0: 119.04, 40928: 111.46, 345505: 110.01}

for f in os.listdir(FONT_DIR) if os.path.isdir(FONT_DIR) else []:
    if f.endswith(".otf"):
        font_manager.fontManager.addfont(os.path.join(FONT_DIR, f))
plt.rcParams.update({
    "font.family": "Libertinus Sans", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.linewidth": 0.4, "grid.alpha": 0.4, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})
BAR_EDGE = dict(edgecolor="black", linewidth=0.5)

for d in os.listdir(CURVE) if os.path.isdir(CURVE) else []:
    m = re.fullmatch(r"bfs_(\d+)", d)
    if not m:
        continue
    rows = re.findall(r"^\s+100\s+4\s+\S+\s+\S+\s+\S+\s+([\d.]+)", open(f"{CURVE}/{d}/run.out", errors="ignore").read(), re.M)
    if rows:
        BFS[int(m.group(1))] = float(rows[-1])

r = json.load(open(TRACE))
par = json.load(open(PARENT))
base = r["reads_per_query"]
fig, axes = plt.subplots(1, 2, figsize=(COL, 1.45), gridspec_kw={"wspace": 0.55, "width_ratios": [1.75, 1]})

ax = axes[0]
sim = json.load(open(SIM))
row = next(x for x in sim["rows"] if x["mem_nodes"] == MEM_NODES)
names = ["No\ncache", "Mem.", "+Nbr.\npages", "+Dist.\npages", "+Co-read\npages"]
prow = next(x for x in par["sim"]["rows"] if x["mem_nodes"] == MEM_NODES)
vals = [base, row["memory_only"], row["neighbor_pages"], row["vector_pages"], prow["coread_parent_pages"]]
colors = ["#BBBBBB", "#DDAA33", "#4477AA", "#228833", "#BB5566"]
ax.bar(range(5), vals, 0.6, color=colors, **BAR_EDGE)
for x, v in enumerate(vals):
    ax.text(x, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5)
ax.set_xticks(range(5), names, fontsize=5.5)
ax.set_ylabel("Disk reads / query")
ax.set_ylim(0, 135)
ax.grid(axis="y")
ax.set_title("(a) Disk reads", fontsize=8, y=-0.62)

ax = axes[1]
ph = r["page_hits"]
names = ["ID", "Nbr.", "Dist.", "Co-\nread"]
vals = [ph["id_layout"], ph["graph_neighbors"], ph["vector_nearest"],
        par["page_hits"]["coread_parent_learned_earlier"]]
colors = ["#BBBBBB", "#4477AA", "#228833", "#BB5566"]
ax.bar(range(4), vals, 0.6, color=colors, **BAR_EDGE)
for x, v in enumerate(vals):
    ax.text(x, v + 0.06, f"{v:.2f}" if v >= 0.01 else "<0.01", ha="center", va="bottom", fontsize=5.5)
ax.set_xticks(range(4), names, fontsize=5.5)
ax.set_ylabel("Useful nodes / page")
ax.set_ylim(0, 3)
ax.grid(axis="y")
ax.set_title("(b) Page layout", fontsize=8, y=-0.62)

os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_motivation.{ext}", dpi=300)
print(f"{OUT}/fig_motivation.pdf", BFS)
