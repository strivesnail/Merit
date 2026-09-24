#!/usr/bin/env python3
"""Generate moving-region finite-Zipf self-query workloads for SIFT100M."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def map_u8bin(path: Path) -> tuple[np.memmap, int, int]:
    with path.open("rb") as source:
        count, dim = struct.unpack("<II", source.read(8))
    vectors = np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(count, dim))
    return vectors, count, dim


def finite_zipf_cdf(alpha: float, support_size: int) -> np.ndarray:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    ranks = np.arange(1, support_size + 1, dtype=np.float64)
    weights = np.power(ranks, -alpha)
    return np.cumsum(weights) / weights.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--region-query", type=Path)
    source.add_argument("--base", type=Path)
    parser.add_argument("--region-node-ids", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--region-size", type=int, default=100_000)
    parser.add_argument("--queries-per-round", type=int, default=200_000)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    needed_regions = args.rounds * args.region_size
    region_ids = np.fromfile(args.region_node_ids, dtype="<u4")
    if region_ids.size < needed_regions:
        raise SystemExit(
            f"region node-id file has {region_ids.size} IDs, but {needed_regions} are required"
        )
    regions = None
    base = None
    if args.region_query is not None:
        regions, region_count, dim = map_u8bin(args.region_query)
        if region_count < needed_regions:
            raise SystemExit(
                f"region query has {region_count} rows, but {needed_regions} are required"
            )
    else:
        base, base_count, dim = map_u8bin(args.base)
        if int(region_ids[:needed_regions].max()) >= base_count:
            raise SystemExit("region node ID exceeds base-vector count")

    total = args.rounds * args.queries_per_round
    query_ids = np.empty(total, dtype=np.uint32)
    cdf = finite_zipf_cdf(args.alpha, args.region_size)
    rng = np.random.default_rng(args.seed)
    segments: list[dict[str, int | float]] = []

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    query_path = args.output_prefix.with_suffix(".u8bin")
    with query_path.open("wb") as output:
        output.write(struct.pack("<II", total, dim))
        cursor = 0
        for round_index in range(args.rounds):
            ranks = np.searchsorted(
                cdf, rng.random(args.queries_per_round), side="left"
            )
            positions = round_index * args.region_size + ranks
            ids = np.asarray(region_ids[positions], dtype=np.uint32)
            if regions is not None:
                vectors = regions[positions]
            else:
                vectors = base[ids]
            np.asarray(vectors, dtype=np.uint8).tofile(output)
            query_ids[cursor : cursor + args.queries_per_round] = ids

            _, counts = np.unique(ids, return_counts=True)
            sorted_counts = np.sort(counts)
            top_10_share = float(
                sorted_counts[-10:].sum() / args.queries_per_round
            )
            top_1_percent_count = max(1, args.region_size // 100)
            top_1_percent_share = float(
                sorted_counts[-top_1_percent_count:].sum()
                / args.queries_per_round
            )
            segments.append(
                {
                    "segment": round_index + 1,
                    "begin": cursor,
                    "end": cursor + args.queries_per_round,
                    "unique_queries": int(counts.size),
                    "top_10_share": top_10_share,
                    "top_1_percent_share": top_1_percent_share,
                }
            )
            cursor += args.queries_per_round

    ids_path = args.output_prefix.parent / f"{args.output_prefix.name}_node_ids.bin"
    gt_path = args.output_prefix.parent / f"{args.output_prefix.name}_gt.bin"
    manifest_path = args.output_prefix.with_suffix(".json")
    query_ids.tofile(ids_path)
    with gt_path.open("wb") as output:
        output.write(struct.pack("<II", total, 1))
        query_ids.tofile(output)
        np.zeros(total, dtype=np.float32).tofile(output)

    manifest = {
        "dataset": "SIFT100M",
        "kind": "round-varying finite Zipfian",
        "rounds": args.rounds,
        "region_size": args.region_size,
        "queries_per_round": args.queries_per_round,
        "total_queries": total,
        "alpha": args.alpha,
        "seed": args.seed,
        "query_file": str(query_path),
        "gt_file": str(gt_path),
        "node_ids_file": str(ids_path),
        "segments": segments,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"generated {query_path} with {total:,} queries")


if __name__ == "__main__":
    main()
