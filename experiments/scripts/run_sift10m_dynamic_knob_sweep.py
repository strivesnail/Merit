#!/usr/bin/env python3
"""Run resumable full-stream SIFT10M dynamic 3-cache knob sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    tag: str
    ncache_gb: float = 0.0381469727
    dcache_pages: int = 200_001
    mcache_nodes: int = 800_002
    half_life_queries: int = 25_000
    seed_t: int = 2
    replacement_margin_pct: float = 0.0


def single_configs() -> list[Config]:
    base = Config("single_base")
    configs = [base]
    factors = {
        "ncache": ("ncache_gb", [0.0190734863, 0.0381469727, 0.0762939454]),
        "dcache": ("dcache_pages", [100_001, 200_001, 400_001]),
        "mcache": ("mcache_nodes", [400_002, 800_002, 1_600_004]),
        "half": ("half_life_queries", [10_000, 25_000, 50_000, 100_000]),
        "seedt": ("seed_t", [1, 2, 3, 4]),
        "margin": ("replacement_margin_pct", [0.0, 5.0, 10.0, 20.0]),
    }
    base_values = asdict(base)
    for factor, (field, values) in factors.items():
        for value in values:
            if value == base_values[field]:
                continue
            changed = dict(base_values)
            changed[field] = value
            if isinstance(value, float):
                value_tag = str(value).replace(".", "p")
            else:
                value_tag = str(value)
            changed["tag"] = f"single_{factor}_{value_tag}"
            configs.append(Config(**changed))
    return configs


def summarize(qstats: Path, cfg: Config) -> dict:
    official_latency_us: list[float] = []
    official_reads: list[float] = []
    official_base_reads: list[float] = []
    official_dyn_reads: list[float] = []
    official_dyn_hits: list[float] = []
    official_flushes: list[float] = []
    round_latency: list[list[float]] = [[] for _ in range(10)]
    round_reads: list[list[float]] = [[] for _ in range(10)]
    first_full: int | None = None
    rows = 0

    with qstats.open(newline="") as src:
        for row in csv.DictReader(src):
            query_id = int(row["query_id"])
            rows += 1
            if first_full is None and int(row["n_merit_dyn_pages"]) >= cfg.dcache_pages:
                first_full = query_id
            if 500_000 <= query_id < 600_000:
                latency = float(row["total_us"])
                reads = float(row["n_disk_reads"])
                official_latency_us.append(latency)
                official_reads.append(reads)
                official_base_reads.append(float(row["n_base_pages"]))
                official_dyn_reads.append(float(row["n_merit_dyn_disk_reads"]))
                official_dyn_hits.append(float(row["n_merit_dyn_hits"]))
                official_flushes.append(float(row["n_merit_dyn_flushes"]))
                round_id = (query_id - 500_000) // 10_000
                round_latency[round_id].append(latency)
                round_reads[round_id].append(reads)

    if rows != 600_000:
        raise RuntimeError(f"{qstats}: expected 600000 rows, got {rows}")
    if first_full is None or first_full > 500_000:
        raise RuntimeError(
            f"{cfg.tag}: d-cache did not fill before official rounds; first_full={first_full}"
        )

    sorted_latency = sorted(official_latency_us)
    p99_index = max(0, min(len(sorted_latency) - 1, int(0.99 * len(sorted_latency))))
    return {
        "config": asdict(cfg),
        "rows": rows,
        "first_full_query": first_full,
        "official": {
            "mean_latency_ms": statistics.fmean(official_latency_us) / 1000.0,
            "p99_latency_ms": sorted_latency[p99_index] / 1000.0,
            "estimated_qps": 16_000_000.0 / statistics.fmean(official_latency_us),
            "mean_physical_reads": statistics.fmean(official_reads),
            "mean_base_reads": statistics.fmean(official_base_reads),
            "mean_dynamic_reads": statistics.fmean(official_dyn_reads),
            "mean_dynamic_hits": statistics.fmean(official_dyn_hits),
            "mean_flushes": statistics.fmean(official_flushes),
        },
        "rounds": [
            {
                "round": i + 1,
                "mean_latency_ms": statistics.fmean(round_latency[i]) / 1000.0,
                "mean_physical_reads": statistics.fmean(round_reads[i]),
            }
            for i in range(10)
        ],
    }


def run_one(cfg: Config, args: argparse.Namespace) -> dict:
    prefix = args.output_dir / cfg.tag
    qstats = Path(f"{prefix}_qstats.csv")
    result_json = Path(f"{prefix}.json")
    log_path = Path(f"{prefix}.out")

    if result_json.exists() and qstats.exists() and not args.force:
        result = json.loads(result_json.read_text())
        if result.get("rows") == 600_000 and result.get("first_full_query", 600_001) <= 500_000:
            print(f"SKIP {cfg.tag}: valid completed result", flush=True)
            return result

    env = os.environ.copy()
    for name in (
        "MERIT_RECORD_DRIVEN_SEED_ACCESS",
        "MERIT_ADAPTIVE_PARENT_OR_SELF",
        "MERIT_ADAPTIVE_EXTENT",
        "MERIT_PARENT_SINGLE_PAGE",
        "MERIT_CACHE_REPLICA_FALLBACK",
        "MERIT_DISABLE_MULTIREAD",
        "MERIT_MCACHE_EDGE_K",
        "MERIT_STASH_CAP",
    ):
        env.pop(name, None)
    env.update(
        {
            "OMP_NUM_THREADS": "16",
            "MERIT_USE_RAMFS": "0",
            "MERIT_DYNAMIC_3CACHE": "1",
            "MERIT_DCACHE_CAP": str(cfg.dcache_pages),
            "MERIT_MCACHE_CAP": str(cfg.mcache_nodes),
            "MERIT_SCORE_HALF_LIFE_QUERIES": str(cfg.half_life_queries),
            "MERIT_SCORE_STAGE_QUERIES": "1000",
            "MERIT_SEED_T": str(cfg.seed_t),
            "MERIT_REPLACEMENT_MARGIN_PCT": str(cfg.replacement_margin_pct),
            "LD_LIBRARY_PATH": f"{Path.home()}/miniconda3/lib:{env.get('LD_LIBRARY_PATH', '')}",
        }
    )
    command = [
        "/usr/bin/time",
        "-v",
        str(args.search),
        "--data_type",
        "uint8",
        "--dist_fn",
        "l2",
        "--index_path_prefix",
        str(args.data_dir / "sift10m_index"),
        "--query_file",
        str(args.data_dir / "workloads/fill_500k_official_10x.u8bin"),
        "--gt_file",
        str(args.data_dir / "workloads/fill_500k_official_10x_gt.bin"),
        "--result_path",
        str(prefix),
        "--recall_at",
        "1",
        "--search_list",
        "50",
        "--beamwidth",
        "4",
        "--num_threads",
        "16",
        "--num_nodes_to_cache",
        "0",
        "--merit_memory_gb",
        str(cfg.ncache_gb),
        "--merit_memory_runtime_admit",
        "true",
        "--dump_query_stats",
        str(qstats),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"RUN {cfg.tag}: {json.dumps(asdict(cfg), sort_keys=True)}", flush=True)
    started = time.time()
    with log_path.open("w") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{cfg.tag}: search failed with exit code {completed.returncode}; see {log_path}")

    result = summarize(qstats, cfg)
    result["wall_seconds_observed"] = time.time() - started
    result["log_path"] = str(log_path)
    result_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"DONE {cfg.tag}: full={result['first_full_query']} "
        f"reads={result['official']['mean_physical_reads']:.4f} "
        f"mean={result['official']['mean_latency_ms']:.3f}ms "
        f"p99={result['official']['p99_latency_ms']:.3f}ms",
        flush=True,
    )
    return result


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=repo / "data/sift10m")
    parser.add_argument("--search", type=Path, default=repo / "diskann/build/apps/search_disk_index")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "data/sift10m/runs/dynamic_knob_sweep",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.search = args.search.resolve()
    args.output_dir = args.output_dir.resolve()

    results = []
    for cfg in single_configs():
        results.append(run_one(cfg, args))
        (args.output_dir / "single_summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
