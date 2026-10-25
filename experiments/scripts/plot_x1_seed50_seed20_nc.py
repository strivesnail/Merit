import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/mnt/nvme_single/sift100m/runs"
S50 = f"{RUNS}/dcache_knn2000_x1_seed50_nc_vs_bfs_20261012/status.txt"
S20 = f"{RUNS}/dcache_knn2000_x1_seed0.2_nc4_20261013/status.txt"
OUT = "/home/jianz/workload/data/sift100m/figs/knn2000_x1_seed50_seed20_ncache_qps.png"


def load(path):
    d = {}
    for line in open(path):
        m = re.search(r"kind=(\w+) ncMB=(\d+) .*?QPS=([\d.]+).*?base_reads=([\d.]+) dcache_reads=([\d.]+)", line)
        if m:
            kind, mb, q, b, dr = m.groups()
            d.setdefault((kind, int(mb)), []).append((float(q), float(b) + float(dr)))
    return d


d50, d20 = load(S50), load(S20)
mbs = sorted({mb for k, mb in d50})
xs = list(range(len(mbs)))
lines = [
    (d50, "lrucold", "LRU N-cache + D-cache (seed 50%)", "tab:red", "-"),
    (d50, "bfs", "BFS N-cache + D-cache (seed 50%)", "tab:blue", "-"),
    (d20, "lrucold", "LRU N-cache + D-cache (seed 20%)", "tab:red", "--"),
    (d20, "bfs", "BFS N-cache + D-cache (seed 20%)", "tab:blue", "--"),
    (d50, "lruonlycold", "LRU N-cache without D-cache", "tab:orange", "-"),
    (d50, "diskann", "BFS N-cache without D-cache (DiskANN)", "tab:gray", "-"),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for d, kind, label, color, ls in lines:
    zero = ("diskann", 0) if kind in ("diskann", "lruonlycold") else ("none", 0)
    keys = [zero if mb == 0 else (kind, mb) for mb in mbs]
    for i, ax in zip((1, 0), axes):
        ax.plot(xs, [st.median(v[i] for v in d[k]) for k in keys], marker="o", linestyle=ls, color=color, label=label)
for ax, yl in zip(axes, ["Disk reads / query", "QPS"]):
    ax.set_xticks(xs)
    ax.set_xticklabels([str(mb) for mb in mbs])
    ax.set_xlabel("N-cache size (MB)")
    ax.set_ylabel(yl)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
fig.suptitle("Uniform")
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(OUT, dpi=150)
print(OUT)
