#!/usr/bin/env python3
"""Are hot nodes around the medoid? Graph-hop and L2 vs expand heat.

Uses the in-memory Vamana graph for shortest-path hops from the search start
medoid, then overlays existing (or newly saved) access profiles.
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SECTOR_LEN = 4096


def load_vamana_offsets(mem_index: Path) -> tuple[int, np.ndarray, mmap.mmap]:
    f = open(mem_index, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    file_size, width, medoid = struct.unpack_from("<QII", mm, 0)
    _ = width
    assert file_size == len(mm), f"size mismatch {file_size} vs {len(mm)}"
    off = 24
    starts = []
    while off + 4 <= len(mm):
        (nnbrs,) = struct.unpack_from("<I", mm, off)
        if nnbrs == 0 and off + 4 >= len(mm):
            break
        starts.append(off)
        off += 4 + 4 * nnbrs
    node_off = np.fromiter(starts, dtype=np.int64)
    return medoid, node_off, mm


def bfs_hops_mmap(medoid: int, node_off: np.ndarray, mm: mmap.mmap) -> np.ndarray:
    n = int(len(node_off))
    hops = np.full(n, 255, dtype=np.uint8)
    hops[medoid] = 0
    q = deque([int(medoid)])
    while q:
        u = q.popleft()
        hu = int(hops[u])
        off = int(node_off[u])
        (nnbrs,) = struct.unpack_from("<I", mm, off)
        nbrs = np.frombuffer(mm, dtype=np.uint32, count=nnbrs, offset=off + 4)
        for v in nbrs:
            v = int(v)
            if v >= n or hops[v] != 255:
                continue
            hops[v] = hu + 1
            q.append(v)
    return hops


def load_u64_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        arr = np.frombuffer(f.read(n * d * 8), dtype=np.uint64)
    return np.asarray(arr, dtype=np.float64)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr = x[mask].argsort().argsort().astype(np.float64)
    yr = y[mask].argsort().argsort().astype(np.float64)
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def load_base_vectors(path: Path, n: int) -> np.ndarray | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        nb, d = struct.unpack("<II", f.read(8))
        if nb != n:
            return None
        if path.suffix == ".fbin":
            return np.frombuffer(f.read(), dtype=np.float32).reshape(nb, d)
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(nb, d).astype(np.float32)


def analyze_workload(name: str, expand: np.ndarray, hops: np.ndarray, l2: np.ndarray | None) -> dict:
    n = len(expand)
    reached = hops < 255
    hot = expand > 0
    total = float(expand.sum()) or 1.0
    max_h = int(hops[reached].max()) if reached.any() else 0
    per_hop = []
    for h in range(max_h + 1):
        m = hops == h
        cnt = int(m.sum())
        heat = float(expand[m].sum())
        per_hop.append(
            {
                "hop": h,
                "nodes": cnt,
                "touched": int(((expand > 0) & m).sum()),
                "mean_expand": float(expand[m].mean()) if cnt else 0.0,
                "heat_share": heat / total,
                "max_expand": float(expand[m].max()) if cnt else 0.0,
            }
        )
    k = max(1, n // 100)
    top = np.argpartition(expand, -k)[-k:]
    top_h = hops[top]
    top_h = top_h[top_h < 255]
    out = {
        "workload": name,
        "nodes": n,
        "touched": int(hot.sum()),
        "total_expand": total,
        "spearman_hop_vs_expand": spearman(hops.astype(np.float64), expand),
        "spearman_hop_vs_expand_touched": spearman(hops[hot].astype(np.float64), expand[hot]),
        "top1pct_median_hop": float(np.median(top_h)) if len(top_h) else float("nan"),
        "top1pct_mean_hop": float(np.mean(top_h)) if len(top_h) else float("nan"),
        "top1pct_within_hop3": float(np.mean(top_h <= 3)) if len(top_h) else float("nan"),
        "top1pct_within_hop5": float(np.mean(top_h <= 5)) if len(top_h) else float("nan"),
        "per_hop": per_hop,
    }
    if l2 is not None:
        out["spearman_l2_vs_expand"] = spearman(l2, expand)
        out["spearman_l2_vs_expand_touched"] = spearman(l2[hot], expand[hot])
        # 10 quantile bins of L2
        qs = np.quantile(l2, np.linspace(0, 1, 11))
        bins = []
        for i in range(10):
            lo, hi = qs[i], qs[i + 1]
            m = (l2 >= lo) & (l2 <= hi if i == 9 else l2 < hi)
            bins.append(
                {
                    "bin": i,
                    "l2_lo": float(lo),
                    "l2_hi": float(hi),
                    "nodes": int(m.sum()),
                    "mean_expand": float(expand[m].mean()) if m.any() else 0.0,
                    "heat_share": float(expand[m].sum()) / total,
                }
            )
        out["l2_bins"] = bins
    return out


def plot_results(results: list[dict], out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for r in results:
        hops = [p["hop"] for p in r["per_hop"]]
        axes[0].plot(hops, [p["mean_expand"] for p in r["per_hop"]], marker="o", label=r["workload"])
        axes[1].plot(hops, [p["heat_share"] for p in r["per_hop"]], marker="o", label=r["workload"])
    axes[0].set_title("Mean node_expand vs graph hop from medoid")
    axes[0].set_xlabel("Hop from medoid")
    axes[0].set_ylabel("Mean expand count")
    axes[1].set_title("Share of total expand mass vs hop")
    axes[1].set_xlabel("Hop from medoid")
    axes[1].set_ylabel("Heat share")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "hop_vs_heat.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [r["workload"] for r in results]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["top1pct_within_hop3"] for r in results], 0.4, label="top 1% within hop<=3")
    ax.bar(x + 0.2, [r["top1pct_within_hop5"] for r in results], 0.4, label="top 1% within hop<=5")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of top-1% hottest nodes")
    ax.set_title("Do the hottest nodes sit near the medoid?")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "top_hot_near_medoid.png", dpi=140)
    plt.close(fig)

    if any("l2_bins" in r for r in results):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for r in results:
            if "l2_bins" not in r:
                continue
            ax.plot(range(10), [b["mean_expand"] for b in r["l2_bins"]], marker="o", label=r["workload"])
        ax.set_title("Mean expand vs L2-to-medoid quantile bin")
        ax.set_xlabel("L2-to-medoid quantile bin (0=closest)")
        ax.set_ylabel("Mean expand count")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "l2_vs_heat.png", dpi=140)
        plt.close(fig)


def default_profiles(data_dir: Path) -> dict[str, Path]:
    prof = data_dir / "workloads" / "profiles"
    found = {}
    for p in sorted(prof.glob("*_node_expand.bin")):
        name = p.name[: -len("_node_expand.bin")]
        # skip derived sweep leftovers if any remain
        if "top" in name and "nbr" in name:
            continue
        found[name] = p
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--index-prefix", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    prefix = args.index_prefix or {
        "sift1m": "sift1m_index",
        "sift10m": "sift10m_index",
        "sift100m": "sift100m_index",
    }.get(data_dir.name, data_dir.name + "_index")
    mem_index = data_dir / f"{prefix}_mem.index"
    hops_path = data_dir / "workloads" / "medoid_hops.u8.bin"
    out_dir = args.out_dir or (data_dir / "runs" / "medoid_hop_heat")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if hops_path.exists():
        hops = np.fromfile(hops_path, dtype=np.uint8)
        medoid = int((hops == 0).nonzero()[0][0])
        print(f"reuse hops {hops_path} medoid={medoid} max={int(hops[hops<255].max())}", flush=True)
    else:
        print(f"loading graph {mem_index} ...", flush=True)
        medoid, node_off, mm = load_vamana_offsets(mem_index)
        print(f"graph n={len(node_off)} medoid={medoid}", flush=True)
        hops = bfs_hops_mmap(medoid, node_off, mm)
        mm.close()
        hops.tofile(hops_path)
        print(f"hops -> {hops_path} unreachable={(hops==255).sum()}", flush=True)

    base = data_dir / "sift_base.fbin"
    if not base.exists():
        base = data_dir / "sift_base.u8bin"
    vecs = None if len(hops) > 20_000_000 else load_base_vectors(base, len(hops))
    l2 = None
    if vecs is not None:
        mv = vecs[medoid]
        l2 = np.sqrt(np.sum((vecs - mv) ** 2, axis=1)).astype(np.float32)
        print("computed L2 to medoid", flush=True)

    profiles = default_profiles(data_dir)
    if args.only:
        profiles = {k: v for k, v in profiles.items() if k in set(args.only)}
    if not profiles:
        raise SystemExit(f"no *_node_expand.bin under {data_dir}/workloads/profiles")
    print("profiles:", ", ".join(profiles), flush=True)

    results = []
    for name, path in profiles.items():
        expand = load_u64_bin(path)
        if len(expand) != len(hops):
            print(f"skip {name}: expand n={len(expand)} hops n={len(hops)}", flush=True)
            continue
        r = analyze_workload(name, expand, hops, l2)
        results.append(r)
        print(
            f"  {name}: spearman_hop={r['spearman_hop_vs_expand']:.3f} "
            f"top1% hop<=3={r['top1pct_within_hop3']:.3f} "
            f"median_hop={r['top1pct_median_hop']:.1f}",
            flush=True,
        )

    plot_results(results, out_dir, f"{data_dir.name}: heat vs distance to medoid")
    summary = {
        "dataset": data_dir.name,
        "medoid": int((hops == 0).nonzero()[0][0]),
        "n": int(len(hops)),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
