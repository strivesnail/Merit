#!/usr/bin/env python3
"""Extend SIFT1M mem0.01GB no-exclude seed/nbrTop sweep with Top5/10/70/80/90%.

Resume-safe: reuses existing Top20–60 results under mem001_noexclude_sweep.
nbrTop: among children of selected seeds, rank by seed→child edge heat.
Regenerates the two plots:
  seed_nbrtop_disk_reads_mem001_noexclude.png
  seed_nbrtop_vs_base_mem001_noexclude.png
"""
import csv
import json
import os
import shutil
import struct
import statistics as st
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["LD_LIBRARY_PATH"] = (
    os.environ.get("HOME", "") + "/miniconda3/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MERIT_USE_RAMFS"] = "0"
os.environ["MERIT_RECORD_DRIVEN_SEED_ACCESS"] = "1"
os.environ["MERIT_ADAPTIVE_PARENT_OR_SELF"] = "1"
for k in [
    "MERIT_ADAPTIVE_EXTENT",
    "MERIT_DISABLE_MULTIREAD",
    "MERIT_PARENT_SINGLE_PAGE",
    "MERIT_CACHE_REPLICA_FALLBACK",
]:
    os.environ.pop(k, None)

SEARCH = Path("/home/jianz/Merit/diskann/build/apps/search_disk_index")
DATA = Path("/home/jianz/Merit/data/sift1m")
PROF_SRC = DATA / "workloads/profiles/uniform_10k"
FULL_PROF = str(PROF_SRC)
IDX = (DATA / "sift1m_index_disk.index").stat().st_size
OUT = DATA / "runs/adaptive_reduction_sweep/uniform/mem001_noexclude_sweep"
PLOT_DIR = DATA / "runs/adaptive_reduction_sweep"
MEM_GB = 0.01
ADAPT_RATIO = 0.906
SWEEP_RATIO = 0.90

seed_pcts = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
nbr_pcts = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
colors = {
    5: "#8c564b",
    10: "#17becf",
    20: "#1f77b4",
    30: "#ff7f0e",
    40: "#2ca02c",
    50: "#d62728",
    60: "#9467bd",
    70: "#e377c2",
    80: "#bcbd22",
    90: "#7f7f7f",
}
env = os.environ.copy()


def read_u32(p):
    with open(p, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return list(struct.unpack(f"<{n}I", f.read(4 * n)))


def read_u64(p):
    with open(p, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return list(struct.unpack(f"<{n}Q", f.read(8 * n)))


def write_prof(prefix, fu, fv, fc):
    for suffix, arr, t in [
        ("_edge_u.bin", fu, "I"),
        ("_edge_v.bin", fv, "I"),
        ("_edge_count.bin", fc, "Q"),
    ]:
        with open(str(prefix) + suffix, "wb") as f:
            f.write(struct.pack("<II", len(arr), 1))
            if arr:
                f.write(struct.pack(f"<{len(arr)}{t}", *arr))
    shutil.copy2(str(PROF_SRC) + "_node_expand.bin", str(prefix) + "_node_expand.bin")


def stats_from_qstats(p):
    rows = list(csv.DictReader(open(p)))
    return {
        "reads": st.mean(float(r["n_disk_reads"]) for r in rows),
        "mem_hits": st.mean(float(r.get("n_mem_hits", 0) or 0) for r in rows),
        "n_queries": len(rows),
    }


def run_case(tag, profile, ratio, layout="directed_seed_replica_budget_pct100"):
    case = OUT / tag
    case.mkdir(parents=True, exist_ok=True)
    qstats = case / "qstats.csv"
    dc = case / "run_merit_dc.data"
    if qstats.exists() and qstats.stat().st_size > 1000 and (ratio == 0 or dc.exists()):
        try:
            s = stats_from_qstats(qstats)
            if s["n_queries"] >= 9000:
                mb = dc.stat().st_size / 1024 / 1024 if dc.exists() else 0
                r = {
                    **s,
                    "cache_mb": mb,
                    "vs_base": (dc.stat().st_size / IDX if dc.exists() else 0),
                }
                print(
                    f"reuse {tag}: reads={r['reads']:.2f} cache={mb:.0f}MB n={r['n_queries']}",
                    flush=True,
                )
                return r
        except Exception as e:
            print(f"reuse fail {tag}: {e}, rerun", flush=True)

    for p in case.glob("run*"):
        try:
            p.unlink()
        except OSError:
            pass
    if qstats.exists():
        try:
            qstats.unlink()
        except OSError:
            pass

    cmd = [
        str(SEARCH),
        "--data_type",
        "float",
        "--dist_fn",
        "l2",
        "--index_path_prefix",
        str(DATA / "sift1m_index"),
        "--query_file",
        str(DATA / "workloads/uniform_10k.fbin"),
        "--gt_file",
        str(DATA / "workloads/uniform_10k_gt.bin"),
        "--result_path",
        str(case / "run"),
        "--recall_at",
        "1",
        "--search_list",
        "50",
        "--beamwidth",
        "4",
        "--num_threads",
        "8",
        "--merit_memory_gb",
        str(MEM_GB),
        "--merit_profile_prefix",
        profile,
        "--merit_disk_cache_exclude_memory",
        "false",
        "--dump_query_stats",
        str(qstats),
    ]
    if ratio > 0:
        cmd += [
            "--merit_disk_cache_layout",
            layout,
            "--merit_disk_cache_ratio",
            str(ratio),
        ]

    print(f"RUN {tag} ...", flush=True)
    with open(case / "run.out", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        print(f"FAIL {tag} rc={rc}", flush=True)
        raise RuntimeError(f"{tag} failed rc={rc}")

    dc = case / "run_merit_dc.data"
    mb = dc.stat().st_size / 1024 / 1024 if dc.exists() else 0
    s = stats_from_qstats(qstats)
    r = {
        **s,
        "cache_mb": mb,
        "vs_base": (dc.stat().st_size / IDX if dc.exists() else 0),
    }
    print(
        f"RESULT {tag}: reads={r['reads']:.2f} mem={r['mem_hits']:.1f} "
        f"cache={mb:.0f}MB vsbase={r['vs_base']:.3f}x",
        flush=True,
    )
    return r


def plot_disk(baselines, all_results):
    fig, ax = plt.subplots(figsize=(13.5, 8.2), dpi=140)
    pts = {}
    for sp in seed_pcts:
        series = all_results[str(sp)]
        xs = [r["nbr_pct"] for r in series]
        ys = [r["reads"] for r in series]
        ax.plot(xs, ys, "-o", color=colors[sp], lw=2.0, ms=5.5, label=f"Top{sp}%")
        for r in series:
            pts.setdefault(r["nbr_pct"], []).append((sp, r["reads"]))
    gap = 2.4
    for x, items in pts.items():
        items = sorted(items, key=lambda t: -t[1])
        ys = [y for _, y in items]
        top = 0.5 * (ys[0] + ys[-1]) + (len(items) - 1) * gap / 2 + 1.2
        for i, (sp, y) in enumerate(items):
            ax.annotate(
                f"{y:.1f}",
                (x, y),
                xytext=(x, top - i * gap),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=10,
                color=colors[sp],
                arrowprops=dict(arrowstyle="-", color=colors[sp], lw=0.5, alpha=0.4),
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
            )
    db = baselines["diskann"]["reads"]
    ab = baselines["adaptive"]["reads"]
    ax.axhline(db, color="#111111", ls="--", lw=2.2, label=f"DiskANN+mem ({db:.1f})")
    ax.axhline(ab, color="#555555", ls=":", lw=2.2, label=f"Adaptive+mem ({ab:.1f})")
    ax.set_xlabel("nbrTop (%)")
    ax.set_ylabel("Disk reads")
    ax.set_title(f"Seed Top5–90%: Disk reads (mem={MEM_GB}GB)")
    ax.set_xlim(108, 2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    all_y = [r["reads"] for sp in seed_pcts for r in all_results[str(sp)]] + [db, ab]
    ax.set_ylim(min(all_y) - 2, max(all_y) + (len(seed_pcts) - 1) * gap / 2 + 4)
    fig.tight_layout()
    p = OUT / "disk_reads.png"
    fig.savefig(p)
    plt.close(fig)
    shutil.copy2(p, PLOT_DIR / "seed_nbrtop_disk_reads_mem001_noexclude.png")


def plot_vsbase(all_results):
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=140)
    for sp in seed_pcts:
        series = all_results[str(sp)]
        ax.plot(
            [r["nbr_pct"] for r in series],
            [r["vs_base"] for r in series],
            "-o",
            color=colors[sp],
            lw=2,
            ms=5,
            label=f"Top{sp}%",
        )
        for r in series:
            ax.annotate(
                f"{r['vs_base']:.2f}x",
                (r["nbr_pct"], r["vs_base"]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=9,
                color=colors[sp],
            )
    ax.set_xlabel("nbrTop (%)")
    ax.set_ylabel("Disk cache / base")
    ax.set_title(f"Seed Top5–90%: Disk cache vs base (mem={MEM_GB}GB)")
    ax.set_xlim(105, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    p = OUT / "vs_base.png"
    fig.savefig(p)
    plt.close(fig)
    shutil.copy2(p, PLOT_DIR / "seed_nbrtop_vs_base_mem001_noexclude.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    us = read_u32(str(PROF_SRC) + "_edge_u.bin")
    vs = read_u32(str(PROF_SRC) + "_edge_v.bin")
    cs = read_u64(str(PROF_SRC) + "_edge_count.bin")
    heat = defaultdict(int)
    out = defaultdict(list)
    for u, v, c in zip(us, vs, cs):
        if c:
            heat[u] += c
            out[u].append((v, c))
    parents = sorted(heat.keys(), key=lambda p: (-heat[p], p))
    print(
        f"[mem001_noexclude_sweep] parents={len(parents)} edges={len(us)} "
        f"mem={MEM_GB}GB nbrTop_by=edge_heat",
        flush=True,
    )

    baselines = {
        "diskann": run_case("diskann", FULL_PROF, 0.0),
        "adaptive": run_case(
            "adaptive", FULL_PROF, ADAPT_RATIO, layout="directed_seed_replica_pct100"
        ),
    }
    all_results = {}
    for sp in seed_pcts:
        seeds = parents[: max(1, int(len(parents) * sp / 100))]
        edge_score = defaultdict(int)
        for p in seeds:
            for v, c in out[p]:
                edge_score[v] += c
        nbr_list = sorted(
            edge_score.keys(),
            key=lambda n: (-edge_score[n], n),
        )
        series = []
        for npct in nbr_pcts:
            tag = f"top{sp}_nbrTop{npct}"
            k = max(1, int(len(nbr_list) * npct / 100))
            kept = set(nbr_list[:k])
            prof = DATA / f"workloads/profiles/mem001_noex_top{sp}_nbr{npct}"
            fu, fv, fc = [], [], []
            for p in seeds:
                for v, c in out[p]:
                    if v in kept:
                        fu.append(p)
                        fv.append(v)
                        fc.append(c)
            write_prof(prof, fu, fv, fc)
            r = run_case(tag, str(prof), SWEEP_RATIO)
            r["seed_pct"] = sp
            r["nbr_pct"] = npct
            series.append(r)
        all_results[str(sp)] = series

    payload = {
        "mem_gb": MEM_GB,
        "exclude_memory": False,
        "seed_pcts": seed_pcts,
        "nbr_pcts": nbr_pcts,
        "baselines": baselines,
        "sweep": all_results,
    }
    (OUT / "sweep.json").write_text(json.dumps(payload, indent=2))
    plot_disk(baselines, all_results)
    plot_vsbase(all_results)
    print("plots updated:", flush=True)
    print(" ", PLOT_DIR / "seed_nbrtop_disk_reads_mem001_noexclude.png", flush=True)
    print(" ", PLOT_DIR / "seed_nbrtop_vs_base_mem001_noexclude.png", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
