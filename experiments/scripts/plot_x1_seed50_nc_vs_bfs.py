import re
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = sys.argv[1] if len(sys.argv) > 1 else "x1"
HOT = {"x1": "0%", "x31": "30%", "x51": "50%", "x71": "70%"}.get(TAG)
if HOT is None and "r" in TAG:
    nq, rep = TAG.split("r")[1].split("x")
    HOT = f"10K uniform + {nq} hot queries x {rep}"
elif HOT is None:
    HOT = f"70% hot queries Zipf s={TAG.split('q')[1]}"
if TAG == "x1":
    STATUS = "/mnt/nvme_single/sift100m/runs/dcache_knn2000_x1_seed50_nc_vs_bfs_20261012/status.txt"
else:
    STATUS = f"/mnt/nvme_single/sift100m/runs/dcache_knn2000_{TAG}_seed50_nc4_20261012/status.txt"
SEED = sys.argv[2] if len(sys.argv) > 2 else "0.5"
SEED_PCT = f"{float(SEED) * 100:g}"
OUT = f"/home/jianz/workload/data/sift100m/figs/knn2000_{TAG}_seed{SEED_PCT}_ncache_vs_bfs_qps.png"
DC_KINDS = ("none", "bfs", "lrucold")
SOURCES = [(STATUS, None)]
if SEED != "0.5":
    seed_status = f"/mnt/nvme_single/sift100m/runs/dcache_knn2000_{TAG}_seed{SEED}_nc4_20261013/status.txt"
    SOURCES = [(STATUS, False), (seed_status, True)]

d = {}
for path, dc_only in SOURCES:
    for line in open(path):
        m = re.search(r"kind=(\w+) ncMB=(\d+) .*?QPS=([\d.]+).*?base_reads=([\d.]+) dcache_reads=([\d.]+)", line)
        if m:
            kind, mb, q, b, dr = m.groups()
            if dc_only is not None and (kind in DC_KINDS) != dc_only:
                continue
            d.setdefault((kind, int(mb)), []).append((float(q), float(b) + float(dr)))

mbs = sorted({mb for k, mb in d})
xs = list(range(len(mbs)))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for kind, label, color in [
    ("lrucold", "D-cache + LRU N-cache", "tab:red"),
    ("bfs", "D-cache + BFS N-cache", "tab:blue"),
    ("lruonlycold", "LRU N-cache without D-cache", "tab:orange"),
    ("diskann", "DiskANN (BFS N-cache)", "tab:gray"),
]:
    zero = ("diskann", 0) if kind in ("diskann", "lruonlycold") else ("none", 0)
    keys = [zero if mb == 0 else (kind, mb) for mb in mbs]
    if any(k not in d for k in keys):
        continue
    for i, ax in enumerate(axes):
        ax.plot(xs, [st.median(v[i] for v in d[k]) for k in keys], "o-", color=color, label=label)
for ax, yl in zip(axes, ["QPS", "Disk reads / query"]):
    ax.set_xticks(xs)
    ax.set_xticklabels([str(mb) for mb in mbs])
    ax.set_xlabel("N-cache size (MB)")
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle(f"{HOT}, seed {SEED_PCT}%" if "uniform" in HOT else f"{HOT} hotspot, seed {SEED_PCT}%")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(OUT)
