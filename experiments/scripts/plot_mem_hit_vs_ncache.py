"""Memory hit rate (node accesses served without an SSD read) vs. N-cache / BFS cache size."""
import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATUS = os.environ.get(
    "STATUS", "/mnt/nvme_single/sift100m/runs/dcache_knn2000_pub10kz08_seed50_nc4_fpsize20261014/status.txt")
OUT = os.environ.get("OUT", "/home/jianz/workload/data/sift100m/figs/pub10k_zipf0p8_mem_hit_vs_ncache.png")
TITLE = os.environ.get("TITLE", "SIFT100M, 10K public queries + 10K Zipf(0.8) (shuffled)")
LINES = [
    ("lrucold", "LRU N-cache + D-cache", "tab:red", "-"),
    ("lruonlycold", "LRU N-cache only", "tab:orange", "--"),
    ("diskann", "DiskANN (BFS)", "tab:gray", "--"),
]
FIELDS = ["dcache_scratch_hits", "ncache_hits", "beam_lookups"]

# At 0MB the N-cache disappears: LRU + D-cache becomes D-cache only, the others plain DiskANN.
ZERO_MB_KIND = {"lrucold": "none", "lruonlycold": "diskann", "diskann": "diskann"}
SIZES = [int(s) for s in os.environ.get("SIZES", "0,100,200,300,400").split(",")]

hit = {}
for status in STATUS.split(","):
    for line in open(status):
        m = re.search(r"kind=(\w+) ncMB=(\d+) ", line)
        if not m:
            continue
        vals = {f: float(v) for f, v in re.findall(r"(\w+)=([\d.]+)", line) if f in FIELDS}
        if len(vals) == len(FIELDS):
            rate = 100.0 * (vals["ncache_hits"] + vals["dcache_scratch_hits"]) / vals["beam_lookups"]
            hit.setdefault((m[1], int(m[2])), []).append(rate)
for kind, base in ZERO_MB_KIND.items():
    if (base, 0) in hit:
        hit[(kind, 0)] = hit[(base, 0)]

sizes = [mb for mb in SIZES if any((k, mb) in hit for k, *_ in LINES)]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
x = np.arange(len(sizes))
for kind, label, color, ls in LINES:
    y = [st.median(hit[(kind, mb)]) if (kind, mb) in hit else np.nan for mb in sizes]
    ax.plot(x, y, marker="o", ls=ls, color=color, label=label)
    print(f"{label:24}", [round(v, 1) for v in y])
ax.set_xticks(x)
ax.set_xticklabels([f"{mb}MB" for mb in sizes])
ax.set_xlabel("N-cache size (BFS cache size for DiskANN)")
ax.set_ylabel("Memory hit rate (%)")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
ax.set_title(TITLE, fontsize=11)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(OUT)
