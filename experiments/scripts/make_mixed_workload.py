#!/usr/bin/env python3
"""Interleave a skewed and a uniform perturbed workload into one stream.

Default: query i comes from the uniform workload with probability p and from the skewed
workload otherwise; both inputs must share the same sampled positions, so the exact
ground truth of each sampled query is taken from whichever workload supplied it.
With --union, all queries of both workloads are kept and randomly interleaved, and the
sampled queries of both workloads (with their ground truth) are carried over.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_u8bin(path: Path) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(int(n), int(d)))


def read_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    n, k = np.fromfile(path, dtype=np.int32, count=2)
    ids = np.fromfile(path, dtype=np.uint32, count=n * k, offset=8).reshape(n, k)
    dist = np.fromfile(path, dtype=np.float32, count=n * k, offset=8 + 4 * n * k).reshape(n, k)
    return ids, dist


def write_gt(path: Path, ids: np.ndarray, dist: np.ndarray) -> None:
    with path.open("wb") as f:
        np.array(ids.shape, dtype=np.int32).tofile(f)
        ids.astype(np.uint32).tofile(f)
        dist.astype(np.float32).tofile(f)


def union(args, xs, xu, ps, pu, gs, gu) -> None:
    ns, nu = xs.shape[0], xu.shape[0]
    rng = np.random.default_rng(args.seed)
    from_uniform = np.zeros(ns + nu, dtype=bool)
    from_uniform[rng.choice(ns + nu, size=nu, replace=False)] = True
    slot_s, slot_u = np.flatnonzero(~from_uniform), np.flatnonzero(from_uniform)
    out = np.empty((ns + nu, xs.shape[1]), dtype=np.uint8)
    out[slot_s], out[slot_u] = xs, xu

    pos = np.concatenate([slot_s[ps], slot_u[pu]])
    ids = np.concatenate([gs[0], gu[0]])
    dist = np.concatenate([gs[1], gu[1]])
    order = np.argsort(pos)
    pos, ids, dist = pos[order], ids[order], dist[order]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".u8bin").open("wb") as f:
        np.array(out.shape, dtype=np.int32).tofile(f)
        out.tofile(f)
    np.save(args.out.parent / (args.out.name + "_sample_pos.npy"), pos)
    np.save(args.out.parent / (args.out.name + "_from_uniform.npy"), from_uniform)
    write_gt(args.out.parent / (args.out.name + "_sample_gt10.bin"), ids, dist)
    meta = {"skewed": str(args.skewed), "uniform": str(args.uniform), "union": True,
            "seed": args.seed, "n": int(out.shape[0]), "n_skewed": int(ns), "n_uniform": int(nu),
            "sample": int(pos.size), "sampled_uniform_fraction": float(from_uniform[pos].mean())}
    (args.out.parent / (args.out.name + ".json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skewed", type=Path, required=True, help="prefix of the skewed workload")
    ap.add_argument("--uniform", type=Path, required=True, help="prefix of the uniform workload")
    ap.add_argument("--p", type=float, default=0.5, help="fraction of uniform queries")
    ap.add_argument("--union", action="store_true")
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    def files(prefix: Path) -> tuple[Path, Path, Path]:
        return (prefix.with_suffix(".u8bin"),
                prefix.parent / (prefix.name + "_sample_pos.npy"),
                prefix.parent / (prefix.name + "_sample_gt10.bin"))

    sq, sp, sg = files(args.skewed)
    uq, up, ug = files(args.uniform)
    xs, xu = read_u8bin(sq), read_u8bin(uq)
    if args.union:
        union(args, xs, xu, np.load(sp), np.load(up), read_gt(sg), read_gt(ug))
        return
    if xs.shape != xu.shape:
        raise SystemExit(f"shape mismatch: {xs.shape} vs {xu.shape}")
    pos = np.load(sp)
    if not np.array_equal(pos, np.load(up)):
        raise SystemExit("sampled positions differ between the two workloads")

    rng = np.random.default_rng(args.seed)
    from_uniform = rng.random(xs.shape[0]) < args.p
    out = np.where(from_uniform[:, None], xu, xs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".u8bin").open("wb") as f:
        np.array(out.shape, dtype=np.int32).tofile(f)
        out.tofile(f)
    np.save(args.out.parent / (args.out.name + "_sample_pos.npy"), pos)
    np.save(args.out.parent / (args.out.name + "_from_uniform.npy"), from_uniform)

    (si, sd), (ui, ud) = read_gt(sg), read_gt(ug)
    pick = from_uniform[pos][:, None]
    gi, gd = np.where(pick, ui, si), np.where(pick, ud, sd)
    with (args.out.parent / (args.out.name + "_sample_gt10.bin")).open("wb") as f:
        np.array(gi.shape, dtype=np.int32).tofile(f)
        gi.astype(np.uint32).tofile(f)
        gd.astype(np.float32).tofile(f)

    meta = {"skewed": str(args.skewed), "uniform": str(args.uniform), "p": args.p,
            "seed": args.seed, "n": int(out.shape[0]),
            "uniform_fraction": float(from_uniform.mean()),
            "sampled_uniform_fraction": float(from_uniform[pos].mean())}
    (args.out.parent / (args.out.name + ".json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
