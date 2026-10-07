#!/usr/bin/env python3
"""Collect the focused MERIT D-cache optimization ablation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/graid_single/sift10m/runs/dcache_optimization_a1p2")
OUT = Path(__file__).resolve().parents[2] / "docs/figures/report/fig_dcache_optimization_a1p2"
ORDER = ("n_only", "legacy", "async_w1", "async_w2", "async_w4", "forced_on")


def mean_qstats(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    fields = (
        "total_us",
        "io_us",
        "cpu_us",
        "n_disk_reads",
        "n_merit_dyn_disk_reads",
        "n_merit_base_pages_avoided",
    )
    return {
        field: sum(float(row[field]) for row in rows) / len(rows)
        for field in fields
        if field in rows[0]
    }


rows = []
for tag in ORDER:
    point = (
        Path("/mnt/graid_single/sift10m/runs/extra18_split_a1p2/n012")
        if tag == "legacy" and not (ROOT / tag / "recall_qps.json").exists()
        else (
            Path("/mnt/graid_single/sift10m/runs/dcache_optimization_forced_on")
            if tag == "forced_on"
            else ROOT / tag
        )
    )
    result = point / "recall_qps.json"
    if not result.exists():
        continue
    perf = json.loads(result.read_text())[0]
    system = perf["system"]
    run = point / f"{system}_L100/run.out"
    qstats = point / f"{system}_L100/qstats.csv"
    text = run.read_text(errors="ignore")
    perf.update(tag=tag, **mean_qstats(qstats))
    async_line = re.findall(r"MERIT async_commit:.*", text)
    gate_lines = re.findall(r"MERIT dcache net gate:.*", text)
    perf["async_stats"] = async_line[-1] if async_line else ""
    perf["gate_transitions"] = gate_lines
    rows.append(perf)

(ROOT / "optimization_results.json").write_text(json.dumps(rows, indent=2))

for row in rows:
    print(
        f"{row['tag']:10s} QPS={row['qps']:9.1f} recall={row['recall']:.4f} "
        f"reads={row.get('n_disk_reads', row['mean_ios']):6.2f} "
        f"cpu_us={row.get('cpu_us', 0):7.1f} io_us={row.get('io_us', 0):7.1f}"
    )

if rows:
    labels = [row["tag"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    axes[0].bar(labels, [row["qps"] for row in rows])
    axes[0].set_ylabel("QPS")
    axes[1].bar(labels, [row.get("n_disk_reads", row["mean_ios"]) for row in rows])
    axes[1].set_ylabel("SSD reads / query")
    axes[2].bar(labels, [row.get("cpu_us", 0) for row in rows])
    axes[2].set_ylabel("CPU µs / query")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("SIFT10M Zipf 1.2, N=6,835, D=6,721, L=100, W=4, 20 threads", fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    fig.savefig(OUT.with_suffix(".pdf"))
