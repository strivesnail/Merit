"""Per-query node accesses broken down by the cache/storage level that served them (stacked bars)."""
import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATUS = os.environ.get(
    "STATUS", "/mnt/nvme_single/sift100m/runs/dcache_knn2000_pub10k_seed50_nc4_fp20261014/status.txt")
OUT = os.environ.get("OUT", "/home/jianz/workload/data/sift100m/figs/pub10k_hierarchy_bars.png")
TITLE = os.environ.get("TITLE", "SIFT100M, 10K public queries")
CONFIGS = [
    (("diskann", 0), "DiskANN"),
    (("diskann", 400), "DiskANN\n(BFS 400MB)"),
    (("lruonlycold", 400), "LRU N-cache only\n(400MB)"),
    (("lrucold", 400), "LRU N-cache 400MB\n+ D-cache"),
]
FIELDS = ["QPS", "base_reads", "dcache_reads", "dcache_scratch_hits", "ncache_hits", "beam_lookups"]

rows = {}
for line in open(STATUS):
    m = re.search(r"kind=(\w+) ncMB=(\d+) ", line)
    if not m:
        continue
    vals = {f: float(v) for f, v in re.findall(r"(\w+)=([\d.]+)", line) if f in FIELDS}
    if len(vals) == len(FIELDS):
        rows.setdefault((m[1], int(m[2])), []).append(vals)

med = {k: {f: st.median(r[f] for r in v) for f in FIELDS} for k, v in rows.items()}
levels = [
    ("N-cache (DRAM)", "tab:red"),
    ("D-cache page in DRAM", "tab:green"),
    ("D-cache (SSD)", "tab:olive"),
    ("Base index (SSD)", "tab:gray"),
]
stack = []
for key, _ in CONFIGS:
    d = med[key]
    ssd_dc = max(d["beam_lookups"] - d["ncache_hits"] - d["dcache_scratch_hits"] - d["base_reads"], 0.0)
    stack.append([d["ncache_hits"], d["dcache_scratch_hits"], ssd_dc, d["base_reads"]])
stack = np.array(stack)

if os.environ.get("MODE", "qps") == "qps":
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CONFIGS))
    q = [med[key]["QPS"] for key, _ in CONFIGS]
    ax.bar(x, q, 0.55, color=["silver", "tab:gray", "tab:orange", "tab:red"])
    for i in range(len(CONFIGS)):
        ax.text(x[i], q[i] + max(q) * 0.01, f"{q[i]:,.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in CONFIGS])
    ax.set_ylabel("QPS")
    ax.set_ylim(0, max(q) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(TITLE)
    fig.tight_layout()
    OUT = OUT.replace(".png", "_qps.png")
    fig.savefig(OUT, dpi=150)
    print([round(v) for v in q], OUT)
    raise SystemExit

fig, ax = plt.subplots(figsize=(6.5, 4.8))
x = np.arange(len(CONFIGS))
bottom = np.zeros(len(CONFIGS))
for j, (name, color) in enumerate(levels):
    ax.bar(x, stack[:, j], 0.55, bottom=bottom, color=color, label=name)
    bottom += stack[:, j]
for i, (key, _) in enumerate(CONFIGS):
    ax.text(x[i], bottom[i] + 2, f"{med[key]['QPS']:,.0f} QPS", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([c[1] for c in CONFIGS])
ax.set_ylabel("Node accesses per query")
ax.set_ylim(0, bottom.max() * 1.15)
ax.grid(axis="y", alpha=0.3)
ax.set_title(TITLE)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
for (key, label), s in zip(CONFIGS, stack):
    print(label.replace("\n", " "), "QPS", round(med[key]["QPS"]), "levels", [round(v, 1) for v in s],
          "reads", round(med[key]["base_reads"] + med[key]["dcache_reads"], 1), "n_reps", len(rows[key]))
print(OUT)
