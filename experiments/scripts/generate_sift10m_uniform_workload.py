#!/usr/bin/env python3
"""Generate uniform 10K-query workload for SIFT10M (uint8 u8bin base)."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

DIM = 128
SEED = 42


def load_u8bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        assert d == DIM, d
        arr = np.frombuffer(f.read(n * d), dtype=np.uint8).reshape(n, d)
    return arr


def save_u8bin(path: Path, arr: np.ndarray) -> None:
    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(arr.astype(np.uint8, copy=False).tobytes())


def save_gt(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    n, k = ids.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, k))
        f.write(ids.astype(np.uint32, copy=False).tobytes())
        f.write(dists.astype(np.float32, copy=False).tobytes())


def compute_gt_uint8(
    queries: np.ndarray,
    base: np.ndarray,
    k: int = 1,
    base_chunk: int = 250_000,
    qbatch: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    nb = base.shape[0]
    base_norm = np.sum(base.astype(np.uint32) ** 2, axis=1)
    nq = queries.shape[0]
    best_d = np.full(nq, np.inf, dtype=np.float32)
    best_i = np.zeros(nq, dtype=np.uint32)

    for j0 in range(0, nb, base_chunk):
        j1 = min(j0 + base_chunk, nb)
        b = base[j0:j1].astype(np.float32)
        bn = base_norm[j0:j1]
        for i0 in range(0, nq, qbatch):
            i1 = min(i0 + qbatch, nq)
            q = queries[i0:i1].astype(np.float32)
            qn = np.sum(q * q, axis=1, keepdims=True)
            d2 = qn + bn[None, :] - 2.0 * (q @ b.T)
            if k == 1:
                local_idx = np.argmin(d2, axis=1)
                local_d = d2[np.arange(i1 - i0), local_idx]
                global_idx = (j0 + local_idx).astype(np.uint32)
                mask = local_d < best_d[i0:i1]
                best_d[i0:i1] = np.where(mask, local_d, best_d[i0:i1])
                best_i[i0:i1] = np.where(mask, global_idx, best_i[i0:i1])
            else:
                raise NotImplementedError("k>1 not implemented")
        print(f"  gt chunk {j0}:{j1} / {nb}", flush=True)

    return best_i.reshape(nq, k), best_d.reshape(nq, k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--num-queries", type=int, default=10_000)
    ap.add_argument("--recall-at", type=int, default=1)
    ap.add_argument(
        "--gt-tool",
        type=Path,
        default=None,
        help="diskann compute_groundtruth binary (much faster for large nq)",
    )
    args = ap.parse_args()
    nq = args.num_queries

    data_dir = args.data_dir.resolve()
    out_dir = (args.out_dir or data_dir / "workloads").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = data_dir / "sift_base.u8bin"
    if not base_path.exists():
        base_path = data_dir / "base.1B.u8bin.crop_nb_10000000"

    print(f"loading base {base_path} ...", flush=True)
    base = load_u8bin(base_path)
    n_base = base.shape[0]
    print(f"base: {n_base} x {DIM}", flush=True)

    rng = np.random.default_rng(SEED)
    node_ids = rng.integers(0, n_base, size=nq, dtype=np.uint32)
    queries = base[node_ids]

    tag = f"uniform_{nq // 1000}k" if nq % 1000 == 0 else f"uniform_{nq}"
    qpath = out_dir / f"{tag}.u8bin"
    gtpath = out_dir / f"{tag}_gt.bin"
    idpath = out_dir / f"{tag}_node_ids.bin"

    save_u8bin(qpath, queries)
    idpath.write_bytes(node_ids.tobytes())
    print(f"queries -> {qpath}", flush=True)

    if args.gt_tool is not None:
        import subprocess

        print(f"computing GT via {args.gt_tool} ...", flush=True)
        subprocess.run(
            [
                str(args.gt_tool),
                "--data_type", "uint8",
                "--dist_fn", "l2",
                "--base_file", str(base_path),
                "--query_file", str(qpath),
                "--gt_file", str(gtpath),
                "--K", str(args.recall_at),
            ],
            check=True,
        )
    else:
        print("computing exact GT (chunked python)...", flush=True)
        gt_ids, gt_dists = compute_gt_uint8(queries, base, k=args.recall_at)
        save_gt(gtpath, gt_ids, gt_dists)
    print(f"gt -> {gtpath}", flush=True)

    meta = {
        "dataset": "sift10m",
        "num_points": int(n_base),
        "num_queries": nq,
        "dim": DIM,
        "dtype": "uint8",
        "seed": SEED,
        "workloads": {
            "uniform": {
                "query_file": str(qpath),
                "gt_file": str(gtpath),
                "node_ids_file": str(idpath),
            }
        },
    }
    meta_path = out_dir / "workloads_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"meta -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
