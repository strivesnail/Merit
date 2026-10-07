#!/usr/bin/env python3
"""Generate uniform / clustered / hotspot / official query sequences (no GT).

clustered: 8 spatial clusters of equal query share.
hotspot: same clusters, but ~70% queries from cluster 0 (Zipf-like).
official: copy Big-ANN public 10K queries when present.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

DIM = 128
N_CLUSTERS = 8
HOTSPOT_SHARE = 0.70
SEED = 42


def save_u8bin(path: Path, arr: np.ndarray) -> None:
    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())


def load_u8bin_header(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
    return n, d


def gather_rows_u8(base_path: Path, ids: np.ndarray, dim: int = DIM) -> np.ndarray:
    ids = np.ascontiguousarray(ids, dtype=np.uint32)
    out = np.empty((len(ids), dim), dtype=np.uint8)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    with open(base_path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        assert d == dim
        last = None
        row = None
        for pos, nid in zip(order, sorted_ids):
            if last != nid:
                f.seek(8 + int(nid) * dim)
                row = np.frombuffer(f.read(dim), dtype=np.uint8).copy()
                last = nid
            out[pos] = row
    return out


def cluster_pools(base_path: Path, n_base: int, per_cluster: int, rng: np.random.Generator) -> list[np.ndarray]:
    centers = rng.choice(n_base, size=N_CLUSTERS, replace=False).astype(np.uint32)
    center_vecs = gather_rows_u8(base_path, centers).astype(np.float32)
    cnorm = np.sum(center_vecs * center_vecs, axis=1)

    best_d = np.full((N_CLUSTERS, per_cluster), np.inf, dtype=np.float32)
    best_i = np.zeros((N_CLUSTERS, per_cluster), dtype=np.uint32)
    chunk = 1_000_000
    with open(base_path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        assert n == n_base and d == DIM
        for j0 in range(0, n_base, chunk):
            j1 = min(j0 + chunk, n_base)
            raw = np.frombuffer(f.read((j1 - j0) * DIM), dtype=np.uint8).reshape(j1 - j0, DIM)
            b = raw.astype(np.float32)
            bn = np.sum(b * b, axis=1)
            # d2[c, i] = ||center_c - b_i||^2
            d2 = cnorm[:, None] + bn[None, :] - 2.0 * (center_vecs @ b.T)
            for c in range(N_CLUSTERS):
                cand_d = d2[c]
                cand_i = np.arange(j0, j1, dtype=np.uint32)
                merge_d = np.concatenate([best_d[c], cand_d])
                merge_i = np.concatenate([best_i[c], cand_i])
                keep = np.argpartition(merge_d, per_cluster - 1)[:per_cluster]
                best_d[c] = merge_d[keep]
                best_i[c] = merge_i[keep]
            print(f"  cluster scan {j1}/{n_base}", flush=True)
    return [best_i[c] for c in range(N_CLUSTERS)]


def sample_from_pools(pools: list[np.ndarray], shares: np.ndarray, nq: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.maximum(1, np.round(shares * nq).astype(np.int64))
    counts[-1] += nq - int(counts.sum())
    parts = []
    for pool, cnt in zip(pools, counts):
        replace = cnt > len(pool)
        parts.append(rng.choice(pool, size=int(cnt), replace=replace))
    ids = np.concatenate(parts).astype(np.uint32)
    rng.shuffle(ids)
    return ids[:nq]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--num-queries", type=int, default=10_000)
    ap.add_argument("--per-cluster", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = data_dir / "workloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    nq = args.num_queries
    rng = np.random.default_rng(args.seed)

    base = data_dir / "sift_base.u8bin"
    if not base.exists():
        base = data_dir / "base.1B.u8bin.crop_nb_100000000"
    n_base, dim = load_u8bin_header(base)
    assert dim == DIM
    tag = f"{nq // 1000}k" if nq % 1000 == 0 else str(nq)
    print(f"base {base} n={n_base}", flush=True)

    meta_path = out_dir / "workloads_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"workloads": {}}
    meta.update({"dataset": data_dir.name, "num_points": n_base, "num_queries": nq, "dim": DIM, "seed": args.seed})
    wls = meta.setdefault("workloads", {})

    # uniform: reuse if present, else sample
    uni = out_dir / f"uniform_{tag}.u8bin"
    if not uni.exists():
        ids = rng.integers(0, n_base, size=nq, dtype=np.uint32)
        save_u8bin(uni, gather_rows_u8(base, ids))
        (out_dir / f"uniform_{tag}_node_ids.bin").write_bytes(ids.tobytes())
        print(f"wrote {uni}", flush=True)
    wls["uniform"] = {"query_file": str(uni), "node_ids_file": str(out_dir / f"uniform_{tag}_node_ids.bin")}

    official_src = data_dir / "query.public.10K.u8bin"
    if official_src.exists() and nq == 10_000:
        dst = out_dir / "official_10k.u8bin"
        if not dst.exists():
            shutil.copy2(official_src, dst)
        wls["official"] = {"query_file": str(dst)}
        print(f"official -> {dst}", flush=True)

    print("building cluster pools (one sequential scan)...", flush=True)
    pools = cluster_pools(base, n_base, args.per_cluster, rng)
    pool_path = out_dir / f"cluster_pools_{N_CLUSTERS}x{args.per_cluster}.npz"
    np.savez(pool_path, **{f"c{i}": pools[i] for i in range(N_CLUSTERS)})

    equal = np.full(N_CLUSTERS, 1.0 / N_CLUSTERS)
    clustered_ids = sample_from_pools(pools, equal, nq, rng)
    cl_q = out_dir / f"clustered_{tag}.u8bin"
    save_u8bin(cl_q, gather_rows_u8(base, clustered_ids))
    (out_dir / f"clustered_{tag}_node_ids.bin").write_bytes(clustered_ids.tobytes())
    wls["clustered"] = {"query_file": str(cl_q), "node_ids_file": str(out_dir / f"clustered_{tag}_node_ids.bin")}
    print(f"wrote {cl_q}", flush=True)

    shares = np.full(N_CLUSTERS, (1.0 - HOTSPOT_SHARE) / (N_CLUSTERS - 1))
    shares[0] = HOTSPOT_SHARE
    hot_ids = sample_from_pools(pools, shares, nq, rng)
    ht_q = out_dir / f"hotspot_{tag}.u8bin"
    save_u8bin(ht_q, gather_rows_u8(base, hot_ids))
    (out_dir / f"hotspot_{tag}_node_ids.bin").write_bytes(hot_ids.tobytes())
    wls["hotspot"] = {"query_file": str(ht_q), "node_ids_file": str(out_dir / f"hotspot_{tag}_node_ids.bin")}
    print(f"wrote {ht_q}", flush=True)

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"meta -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
