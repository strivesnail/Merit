#!/usr/bin/env python3
"""Run access profiling for each query file under data/workloads."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

SEARCH = Path("/home/jianz/Merit/diskann/build/apps/search_disk_index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--index-prefix", type=str, default=None)
    ap.add_argument("--data-type", type=str, required=True, choices=["float", "uint8", "int8"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="workload name prefixes to profile")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    prefix = args.index_prefix or {
        "sift1m": "sift1m_index",
        "sift10m": "sift10m_index",
        "sift100m": "sift100m_index",
    }.get(data_dir.name)
    if not prefix:
        raise SystemExit("pass --index-prefix")
    wl_dir = data_dir / "workloads"
    prof_dir = wl_dir / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    out_dir = data_dir / "runs" / "medoid_hop_heat" / "profile_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(Path.home() / "miniconda3/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = str(args.threads)

    queries = sorted(wl_dir.glob("*.u8bin")) + sorted(wl_dir.glob("*.fbin"))
    queries = [q for q in queries if "gt" not in q.name and q.stat().st_size > 16]
    for q in queries:
        name = q.stem
        if args.only and not any(name.startswith(x) for x in args.only):
            continue
        dest = prof_dir / f"{name}_node_expand.bin"
        if dest.exists() and dest.stat().st_size > 16:
            print(f"skip existing {name}", flush=True)
            continue
        print(f"PROFILE {name} <- {q}", flush=True)
        cmd = [
            str(SEARCH),
            "--data_type", args.data_type,
            "--dist_fn", "l2",
            "--index_path_prefix", str(data_dir / prefix),
            "--query_file", str(q),
            "--gt_file", "null",
            "--result_path", str(out_dir / name),
            "--recall_at", "1",
            "--search_list", "50",
            "--beamwidth", "4",
            "--num_threads", str(args.threads),
            "--num_nodes_to_cache", "0",
            "--enable_access_profile",
            "--access_profile_prefix", str(prof_dir / name),
        ]
        subprocess.run(cmd, check=True, env=env)
        print(f"done {name}", flush=True)


if __name__ == "__main__":
    main()
