import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = "/mnt/nvme_single/sift100m/runs"
UNIFORM_REPEAT = os.environ.get("FAMILY") in ("uniform", "grow")
if os.environ.get("FAMILY") == "grow":
    LEVELS = [int(x) for x in os.environ.get("GROW_LEVELS", "1,2,4,6,8,10,12,14,16,18,20").split(",")]
    RUN_DIR = RUNS + "/dcache_knn2000_k3000h5kg{r}_seed50_nc4_20261012"
    OUT = "/home/jianz/workload/data/sift100m/figs/knn3000_h5000_grow_reads_qps.png"
    XLABEL = "Repetitions per hot query"
    TITLE = "10K uniform queries + 5K hot queries, each hot query repeated xN times"
elif UNIFORM_REPEAT:
    LEVELS = [1, 2, 4, 6, 8, 10]
    RUN_DIR = RUNS + "/dcache_knn2000_u10kx{r}_seed50_nc4_20261012"
    OUT = "/home/jianz/workload/data/sift100m/figs/knn2000_u10k_repeat_reads_qps.png"
    XLABEL = "Repetitions per query"
    TITLE = "10K uniform queries, each repeated xN times"
else:
    LEVELS = [1, 2, 5, 10, 50, 100, 1000]
    RUN_DIR = RUNS + "/dcache_knn2000_k3000h5kx{r}_seed50_nc4_20261012"
    OUT = "/home/jianz/workload/data/sift100m/figs/knn3000_h5000_repeat_levels_reads_qps.png"
    XLABEL = "Hot query repetition"
    TITLE = None
LINES = [
    (("lrucold", 400), "LRU N-cache 400MB + D-cache", "tab:red", "-"),
    (("bfs", 400), "BFS N-cache 400MB + D-cache", "tab:blue", "-"),
    (("none", 0), "D-cache only", "tab:green", "-"),
    (("lruonlycold", 400), "LRU N-cache only (400MB)", "tab:orange", "--"),
    (("diskann", 400), "DiskANN (BFS 400MB)", "tab:gray", "--"),
    (("diskann", 0), "DiskANN", "silver", "--"),
]

# DC_RDATE: take the D-cache lines (none/lrucold/bfs) from a rerun directory with that date suffix instead.
DC_RDATE = os.environ.get("DC_RDATE")
DC_KINDS = ("none", "lrucold", "bfs")
if DC_RDATE:
    OUT = OUT.replace(".png", f"_{DC_RDATE}.png")


def parse(status, keep):
    dq, dr = {}, {}
    if not os.path.exists(status):
        return dq, dr
    for line in open(status):
        m = re.search(r"kind=(\w+) ncMB=(\d+) .*?QPS=([\d.]+).*?base_reads=([\d.]+) dcache_reads=([\d.]+)", line)
        if m and keep(m[1]):
            k = (m[1], int(m[2]))
            dq.setdefault(k, []).append(float(m[3]))
            dr.setdefault(k, []).append(float(m[4]) + float(m[5]))
    return dq, dr


qps, reads = {}, {}
done_levels = []
for r in LEVELS:
    status = RUN_DIR.format(r=r) + "/status.txt"
    if not os.path.exists(status) or "kind=" not in open(status).read():
        continue
    done_levels.append(r)
    if DC_RDATE:
        dq, dr = parse(status, lambda kind: kind not in DC_KINDS)
        zq, zr = parse(RUN_DIR.format(r=r).replace("_20261012", "_" + DC_RDATE) + "/status.txt",
                       lambda kind: kind in DC_KINDS)
        dq.update(zq)
        dr.update(zr)
    else:
        dq, dr = parse(status, lambda kind: True)
    qps[r] = {k: st.median(v) for k, v in dq.items()}
    reads[r] = {k: st.median(v) for k, v in dr.items()}
LEVELS = done_levels

fig, (axr, axq) = plt.subplots(1, 2, figsize=(13, 4.8))
x = np.arange(len(LEVELS))
for key, label, color, ls in LINES:
    axr.plot(x, [reads[r].get(key, np.nan) for r in LEVELS], marker="o", ls=ls, color=color, label=label)
    axq.plot(x, [qps[r].get(key, np.nan) for r in LEVELS], marker="o", ls=ls, color=color, label=label)
for ax, ylab in [(axr, "Disk reads per query"), (axq, "QPS")]:
    ax.set_xticks(x)
    ax.set_xticklabels([f"x{r}" if UNIFORM_REPEAT or r != 1 else "uniform (x1)" for r in LEVELS])
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylab)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
if TITLE:
    fig.suptitle(TITLE)
handles, labels = axr.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(OUT, dpi=150)
for key, label, _, _ in LINES:
    print(f"{label:46}", "reads", [round(reads[r].get(key, np.nan), 1) for r in LEVELS], "qps", [round(qps[r].get(key, np.nan)) if key in qps[r] else None for r in LEVELS])
print(OUT)
