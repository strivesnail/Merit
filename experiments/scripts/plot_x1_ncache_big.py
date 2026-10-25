import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATUS = "/mnt/nvme_single/sift100m/runs/dcache_knn2000_x1_ncache_big_20261012/status.txt"
OUT = "/home/jianz/workload/data/sift100m/figs/knn2000_x1_seed100_ncache_size_qps.png"

d = {}
for line in open(STATUS):
    m = re.search(r"ncMB=(\d+) .*?QPS=([\d.]+).*?base_reads=([\d.]+) dcache_reads=([\d.]+)", line)
    if m:
        mb, q, b, dr = m.groups()
        d.setdefault(int(mb), []).append((float(q), float(b) + float(dr)))

mbs = sorted(mb for mb in d if mb <= 1600)
qps = [st.median(v[0] for v in d[mb]) for mb in mbs]
reads = [st.median(v[1] for v in d[mb]) for mb in mbs]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
xs = list(range(len(mbs)))
axes[0].plot(xs, qps, "o-", color="tab:blue")
axes[1].plot(xs, reads, "o-", color="tab:blue")
for ax, yl in zip(axes, ["QPS", "Disk reads / query"]):
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{mb * 1048576 // 388 / 1e6:.2f}M" if mb else "0" for mb in mbs], rotation=30)
    ax.set_xlabel("N-cache capacity (nodes)")
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
fig.suptitle("0% hotspot, seed 100%, LRU N-cache")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
for mb, q, r in zip(mbs, qps, reads):
    print(mb, round(q), round(r, 2), len(d[mb]))
print(OUT)
