#!/usr/bin/env python3
"""SIFT100M uniform_10k seed/nbrTop sweep: disk-only then mem=0.5GB. Resume-safe.

nbrTop ranks children of selected seeds by seed→child edge heat.
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
os.environ["OMP_NUM_THREADS"] = "16"
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
DATA = Path("/home/jianz/Merit/data/sift100m")
PROF_SRC = DATA / "workloads/profiles/uniform_10k"
QUERY = DATA / "workloads/uniform_10k.u8bin"
GT = DATA / "workloads/uniform_10k_gt.bin"
IDX = (DATA / "sift100m_index_disk.index").stat().st_size
FULL_PROF = str(PROF_SRC)
PLOT_DIR = DATA / "runs/adaptive_reduction_sweep"
DISK_OUT = DATA / "runs/adaptive_reduction_sweep/uniform/seed_nbrtop_multi_sweep_10k"
MEM_OUT = DATA / "runs/adaptive_reduction_sweep/uniform/mem050_sweep_10k"
MEM_GB = 0.5
NUM_THREADS = "16"

seed_pcts = [20, 30, 40, 50, 60]
nbr_pcts = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
colors = {20: "#4C78A8", 30: "#F58518", 40: "#54A24B", 50: "#E45756", 60: "#B279A2"}
ADAPT_RATIO = 0.906
SWEEP_RATIO = 0.90
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
        "n_ios": st.mean(float(r["n_ios"]) for r in rows),
        "mem_hits": st.mean(float(r.get("n_mem_hits", 0) or 0) for r in rows),
        "merit_hits": st.mean(float(r.get("n_merit_hits", 0) or 0) for r in rows),
        "n_queries": len(rows),
    }


def run_case(
    out_root,
    tag,
    profile,
    ratio,
    mem_gb=0.0,
    exclude_mem=False,
    layout="directed_seed_replica_budget_pct100",
):
    case = out_root / tag
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
                    f"reuse {out_root.name}/{tag}: ios={r['n_ios']:.1f} "
                    f"reads={r['reads']:.2f} n={r['n_queries']}",
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
        "uint8",
        "--dist_fn",
        "l2",
        "--index_path_prefix",
        str(DATA / "sift100m_index"),
        "--query_file",
        str(QUERY),
        "--gt_file",
        str(GT),
        "--result_path",
        str(case / "run"),
        "--recall_at",
        "1",
        "--search_list",
        "50",
        "--beamwidth",
        "4",
        "--num_threads",
        NUM_THREADS,
        "--dump_query_stats",
        str(qstats),
    ]
    if mem_gb > 0:
        cmd += [
            "--merit_memory_gb",
            str(mem_gb),
            "--merit_profile_prefix",
            profile,
            "--merit_disk_cache_exclude_memory",
            "true" if exclude_mem else "false",
        ]
    elif ratio > 0:
        cmd += ["--merit_profile_prefix", profile]
    if ratio > 0:
        cmd += [
            "--merit_disk_cache_layout",
            layout,
            "--merit_disk_cache_ratio",
            str(ratio),
        ]

    print(f"RUN {out_root.name}/{tag} ...", flush=True)
    with open(case / "run.out", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        print(f"FAIL {out_root.name}/{tag} rc={rc}", flush=True)
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
        f"RESULT {out_root.name}/{tag}: ios={r['n_ios']:.1f} reads={r['reads']:.2f} "
        f"mem={r['mem_hits']:.1f} cache={mb:.0f}MB",
        flush=True,
    )
    return r


def run_sweep(out_root, mem_gb=0.0, prof_prefix="sweep10k"):
    out_root.mkdir(parents=True, exist_ok=True)
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
        f"[{out_root.name}] parents={len(parents)} edges={len(us)} mem={mem_gb}GB "
        f"nbrTop_by=edge_heat",
        flush=True,
    )
    baselines = {
        "diskann": run_case(
            out_root, "diskann", FULL_PROF, 0.0, mem_gb=mem_gb, exclude_mem=mem_gb > 0
        ),
        "adaptive": run_case(
            out_root,
            "adaptive",
            FULL_PROF,
            ADAPT_RATIO,
            mem_gb=mem_gb,
            exclude_mem=mem_gb > 0,
            layout="directed_seed_replica_pct100",
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
            prof = DATA / f"workloads/profiles/{prof_prefix}_top{sp}_nbr{npct}"
            fu, fv, fc = [], [], []
            for p in seeds:
                for v, c in out[p]:
                    if v in kept:
                        fu.append(p)
                        fv.append(v)
                        fc.append(c)
            write_prof(prof, fu, fv, fc)
            r = run_case(
                out_root, tag, str(prof), SWEEP_RATIO, mem_gb=mem_gb, exclude_mem=mem_gb > 0
            )
            r["seed_pct"] = sp
            r["nbr_pct"] = npct
            series.append(r)
        all_results[str(sp)] = series
    payload = {
        "workload": "uniform_10k",
        "n_queries": 10000,
        "mem_gb": mem_gb,
        "baselines": baselines,
        "sweep": all_results,
    }
    (out_root / "multi_sweep.json").write_text(json.dumps(payload, indent=2))
    return baselines, all_results


def plot_set(baselines, all_results, tag_suffix, y_key="reads", ylabel="Disk reads"):
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(13.5, 8.2), dpi=140)
    pts = {}
    for sp in seed_pcts:
        xs = [r["nbr_pct"] for r in all_results[str(sp)]]
        ys = [r[y_key] for r in all_results[str(sp)]]
        ax.plot(xs, ys, "-o", color=colors[sp], lw=2.4, ms=7, label=f"Top{sp}%")
        for r in all_results[str(sp)]:
            pts.setdefault(r["nbr_pct"], []).append((sp, r[y_key]))
    gap = 3.6
    for x, items in pts.items():
        items = sorted(items, key=lambda t: -t[1])
        ys = [y for _, y in items]
        top = 0.5 * (ys[0] + ys[-1]) + (len(items) - 1) * gap / 2 + 1.8
        for i, (sp, y) in enumerate(items):
            ax.annotate(
                f"{y:.1f}",
                (x, y),
                xytext=(x, top - i * gap),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=12,
                color=colors[sp],
                arrowprops=dict(arrowstyle="-", color=colors[sp], lw=0.7, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8),
            )
    db = baselines["diskann"][y_key]
    ab = baselines["adaptive"][y_key]
    suffix = f"+mem {MEM_GB}GB" if "mem" in tag_suffix else "disk-only"
    ax.axhline(db, color="#D62728", ls="--", lw=2.2, label=f"DiskANN ({db:.1f})")
    ax.axhline(ab, color="#2CA02C", ls="--", lw=2.2, label=f"Adaptive ({ab:.1f})")
    ax.set_xlabel("nbrTop (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"SIFT100M uniform 10K — {ylabel} ({suffix})")
    ax.set_xlim(108, 2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=13, framealpha=0.92)
    all_y = [r[y_key] for sp in seed_pcts for r in all_results[str(sp)]] + [db, ab]
    ax.set_ylim(min(all_y) - 2, max(all_y) + (len(seed_pcts) - 1) * gap / 2 + 4)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"seed_nbrtop_{tag_suffix}_10k.png")
    plt.close(fig)


def plot_vsbase(all_results, tag_suffix):
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=140)
    for sp in seed_pcts:
        series = all_results[str(sp)]
        ax.plot(
            [r["nbr_pct"] for r in series],
            [r["vs_base"] for r in series],
            "-o",
            color=colors[sp],
            lw=2.2,
            ms=6.5,
            label=f"Top{sp}%",
        )
        for r in series:
            ax.annotate(
                f"{r['vs_base']:.2f}x",
                (r["nbr_pct"], r["vs_base"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=11,
                color=colors[sp],
            )
    ax.set_xlabel("nbrTop (%)")
    ax.set_ylabel("Disk cache / base")
    ax.set_title(f"SIFT100M uniform 10K — disk cache vs base ({tag_suffix})")
    ax.set_xlim(105, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=13, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"seed_nbrtop_vs_base_{tag_suffix}_10k.png")
    plt.close(fig)


def main():
    print("=== PHASE 1: disk-only ===", flush=True)
    db, dr = run_sweep(DISK_OUT, mem_gb=0.0, prof_prefix="disk10k")
    plot_set(db, dr, "disk_reads", "reads", "Disk reads")
    plot_set(db, dr, "n_ios", "n_ios", "n_ios (visited fetches / query)")
    plot_vsbase(dr, "disk")
    print("=== PHASE 1 plots done ===", flush=True)

    print(f"=== PHASE 2: mem {MEM_GB}GB ===", flush=True)
    mb, mr = run_sweep(MEM_OUT, mem_gb=MEM_GB, prof_prefix="mem050_10k")
    plot_set(mb, mr, "disk_reads_mem050", "reads", "Disk reads")
    plot_set(mb, mr, "n_ios_mem050", "n_ios", "n_ios (visited fetches / query)")
    plot_vsbase(mr, "mem050")
    print("=== summary disk-only vs mem ===", flush=True)
    for k in ["n_ios", "reads"]:
        print(
            f"{k:8s} DiskANN {db['diskann'][k]:.2f} -> {mb['diskann'][k]:.2f}  "
            f"Adaptive {db['adaptive'][k]:.2f} -> {mb['adaptive'][k]:.2f}",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
