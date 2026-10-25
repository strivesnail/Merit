import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATUS = "/mnt/nvme_single/sift100m/runs/dcache_knn2000_seedfrac_cap2_20261012/status.txt"
TAGS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["x1", "x31", "x51"]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/jianz/workload/data/sift100m/figs/knn2000_seedfrac_2page_hotspots.png"
LABEL = {"x1": "0% hotspot", "x31": "30% hotspot", "x51": "50% hotspot", "x71": "70% hotspot"}
COLOR = {"x1": "tab:blue", "x31": "tab:green", "x51": "tab:orange", "x71": "tab:red"}
TOTAL_SECTORS = 1e7

pat = re.compile(r"\[(x\d+)\] (nocache|seedfrac=([\d.]+)( cap=(\d+))?) .*?QPS=([\d.]+).*?base_reads=([\d.]+) dcache_reads=([\d.]+)")
sec_pat = re.compile(r"packed file sectors=(\d+)")
data = {}
nocache = {}
for line in open(STATUS):
    m = pat.search(line)
    if not m:
        continue
    tag, kind, frac, _, cap, qps, br, dr = m.groups()
    qps, reads = float(qps), float(br) + float(dr)
    if kind == "nocache":
        nocache[tag] = (qps, reads)
        continue
    s = sec_pat.search(line)
    cap = cap or "5000"
    data.setdefault((tag, cap), {})[float(frac)] = (int(s.group(1)), qps, reads)

CAPS = {"5000": "shared cache 20MB", "60000": "shared cache 240MB"}
caps = sys.argv[3].split(",") if len(sys.argv) > 3 else list(CAPS)
fig, axes = plt.subplots(len(caps), 2, figsize=(11, 4 * len(caps)), squeeze=False)
for row, (cap, cname) in enumerate((c, CAPS[c]) for c in caps):
    for tag in TAGS:
        pts = sorted(data.get((tag, cap), {}).values())
        if not pts:
            continue
        x = [p[0] / TOTAL_SECTORS * 100 for p in pts]
        axes[row][0].plot(x, [p[1] for p in pts], "o-", color=COLOR[tag], label=LABEL[tag])
        axes[row][1].plot(x, [p[2] for p in pts], "o-", color=COLOR[tag], label=LABEL[tag])
    if "x1" in nocache:
        axes[row][0].axhline(nocache["x1"][0], color="gray", label="DiskANN")
        axes[row][1].axhline(nocache["x1"][1], color="gray", label="DiskANN")
    axes[row][0].set_ylabel("QPS")
    axes[row][1].set_ylabel("Disk reads / query")
    for ax in axes[row]:
        ax.set_title(f"2-page seed group, {cname}")
        ax.set_xlabel("D-cache size (% of base index)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(OUT)
