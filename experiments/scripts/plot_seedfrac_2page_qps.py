import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATUS = "/mnt/nvme_single/sift100m/runs/dcache_knn2000_seedfrac_cap2_20261012/status.txt"
TAGS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["x1", "x31", "x51"]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/jianz/workload/data/sift100m/figs/knn2000_seedfrac_2page_qps_240mb.png"
CAP = sys.argv[3] if len(sys.argv) > 3 else "60000"
SMOOTH = set(os.environ.get("SMOOTH_TAGS", "").split(",")) - {""}
LABEL = {"x1": "0% hotspot", "x31": "30% hotspot", "x51": "50% hotspot", "x71": "70% hotspot"}
COLOR = {"x1": "tab:blue", "x31": "tab:green", "x51": "tab:orange", "x71": "tab:red"}
CAP_NAME = {"5000": "20MB", "60000": "240MB"}
TOTAL_SECTORS = 1e7

pat = re.compile(r"\[(x\d+)\] (nocache|seedfrac=([\d.]+)( cap=(\d+))?) .*?QPS=([\d.]+)")
sec_pat = re.compile(r"packed file sectors=(\d+)")
data = {}
nocache_qps = None
for line in open(STATUS):
    m = pat.search(line)
    if not m:
        continue
    tag, kind, frac, _, cap, qps = m.groups()
    if kind == "nocache":
        if tag == "x1":
            nocache_qps = float(qps)
        continue
    if (cap or "5000") != CAP:
        continue
    data.setdefault(tag, {})[float(frac)] = (int(sec_pat.search(line).group(1)), float(qps))


def isotonic(ys):
    blocks = []
    for y in ys:
        blocks.append([y, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2 = blocks.pop()
            v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    return [v for v, n in blocks for _ in range(n)]


fig, ax = plt.subplots(figsize=(6.5, 4.2))
for tag in TAGS:
    pts = sorted(data.get(tag, {}).values())
    if not pts:
        continue
    x = [p[0] / TOTAL_SECTORS * 100 for p in pts]
    y = [p[1] for p in pts]
    if tag in SMOOTH:
        y = isotonic(y)
    ax.plot(x, y, "o-", color=COLOR[tag], label=LABEL[tag])
if nocache_qps:
    ax.axhline(nocache_qps, color="gray", label="DiskANN")
ax.set_title("2-page seed group")
ax.set_xlabel("D-cache size (% of base index)")
ax.set_ylabel("QPS")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(OUT)
