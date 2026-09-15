#!/usr/bin/env python3
"""Build a long SIFT10M stream: distinct self-query warmup, then official queries."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


DIM = 128


def u8bin_memmap(path: Path) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        n, d = struct.unpack("<II", f.read(8))
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(n, d)), n, d


def load_gt_first(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        n, k = struct.unpack("<II", f.read(8))
        ids = np.fromfile(f, dtype=np.uint32, count=n * k).reshape(n, k)
        dists = np.fromfile(f, dtype=np.float32, count=n * k).reshape(n, k)
    return ids[:, 0].copy(), dists[:, 0].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--warmup-queries", type=int, default=3_000_000)
    ap.add_argument("--official-repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--output-prefix", type=Path, default=None)
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    base_path = data_dir / "sift_base.u8bin"
    if not base_path.exists():
        base_path = data_dir / "base.1B.u8bin.crop_nb_10000000"
    official_query_path = data_dir / "sift_query.u8bin"
    official_gt_path = data_dir / "sift_groundtruth.bin"

    base, n_base, dim = u8bin_memmap(base_path)
    official, n_official, official_dim = u8bin_memmap(official_query_path)
    if dim != DIM or official_dim != DIM:
        raise SystemExit(f"expected dim={DIM}, got base={dim}, official={official_dim}")
    if not 0 <= args.warmup_queries <= n_base:
        raise SystemExit(f"warmup-queries must be in [0, {n_base}]")
    if args.official_repeats < 1:
        raise SystemExit("official-repeats must be positive")

    official_ids, official_dists = load_gt_first(official_gt_path)
    if official_ids.size != n_official:
        raise SystemExit("official query/ground-truth count mismatch")

    prefix = args.output_prefix
    if prefix is None:
        prefix = data_dir / "workloads" / (
            f"fill_{args.warmup_queries // 1_000_000}m_official_{args.official_repeats}x"
        )
    prefix = prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    query_path = Path(f"{prefix}.u8bin")
    gt_path = Path(f"{prefix}_gt.bin")
    ids_path = Path(f"{prefix}_warmup_ids.bin")
    manifest_path = Path(f"{prefix}.json")

    rng = np.random.default_rng(args.seed)
    warmup_ids = rng.choice(n_base, size=args.warmup_queries, replace=False).astype(np.uint32, copy=False)
    warmup_ids.tofile(ids_path)

    eval_queries = n_official * args.official_repeats
    total_queries = args.warmup_queries + eval_queries
    with query_path.open("wb") as out:
        out.write(struct.pack("<II", total_queries, DIM))
        for begin in range(0, args.warmup_queries, args.chunk_size):
            end = min(begin + args.chunk_size, args.warmup_queries)
            np.asarray(base[warmup_ids[begin:end]], dtype=np.uint8).tofile(out)
            print(f"warmup vectors {end}/{args.warmup_queries}", flush=True)
        for _ in range(args.official_repeats):
            np.asarray(official, dtype=np.uint8).tofile(out)

    # DiskANN truthset layout is header, all IDs, then all float distances.
    with gt_path.open("wb") as out:
        out.write(struct.pack("<II", total_queries, 1))
        warmup_ids.tofile(out)
        for _ in range(args.official_repeats):
            official_ids.tofile(out)
        np.zeros(args.warmup_queries, dtype=np.float32).tofile(out)
        for _ in range(args.official_repeats):
            official_dists.tofile(out)

    manifest = {
        "dataset": "SIFT10M",
        "query_file": str(query_path),
        "gt_file": str(gt_path),
        "warmup_kind": "distinct base descriptors used as exact self-queries",
        "warmup_queries": args.warmup_queries,
        "official_kind": "BigANN query.public.10K",
        "official_queries": n_official,
        "official_repeats": args.official_repeats,
        "evaluation_begin": args.warmup_queries,
        "evaluation_end": total_queries,
        "seed": args.seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
