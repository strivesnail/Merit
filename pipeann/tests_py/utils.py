from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


SIFT_DATA_DIM = 128
SIFT_DATA_TYPE = "uint8"
SIFT_1M_PATH = "/mnt/nvme/data/bigann/bigann_1M.bbin"
SIFT_2M_PATH = "/mnt/nvme/data/bigann/bigann_2M.bbin"
SIFT_QUERY_PATH = "/mnt/nvme/data/bigann/bigann_query.bbin"
SIFT_1M_GT_PATH = "/mnt/nvme/indices_upd/bigann_gnd/idx_1M.ibin"
SIFT_2M_GT_PATH = "/mnt/nvme/indices_upd/bigann_gnd/2M_topk/gt_990000.bin"
SIFT_INDEX_PREFIX = "/mnt/nvme/indices/bigann/1M"


def bin_write(vectors: np.ndarray, filename: str | Path) -> None:
    vectors = np.ascontiguousarray(vectors)
    with open(filename, "wb") as writer:
        writer.write(struct.pack("<I", vectors.shape[0]))
        writer.write(struct.pack("<I", vectors.shape[1]))
        writer.write(vectors.tobytes())


def bin_read(filename: str | Path, dtype: str | np.dtype = "float32") -> np.ndarray:
    dtype = np.dtype(dtype)
    with open(filename, "rb") as reader:
        npts = struct.unpack("<I", reader.read(4))[0]
        dim = struct.unpack("<I", reader.read(4))[0]
        data = reader.read(npts * dim * dtype.itemsize)
    return np.frombuffer(data, dtype=dtype).reshape((npts, dim))


def write_spmat(rows: list[list[int]], values: list[list[float]], n_cols: int, filename: str | Path) -> None:
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for row_indices, row_values in zip(rows, values, strict=True):
        indices.extend(int(value) for value in row_indices)
        data.extend(float(value) for value in row_values)
        indptr.append(len(indices))

    with open(filename, "wb") as writer:
        writer.write(struct.pack("<qqq", len(rows), n_cols, len(indices)))
        writer.write(struct.pack(f"<{len(indptr)}q", *indptr))
        if indices:
            writer.write(struct.pack(f"<{len(indices)}i", *indices))
            writer.write(struct.pack(f"<{len(data)}f", *data))


def compute_recall(search_ids: np.ndarray, gt_ids: np.ndarray) -> float:
    recall = 0.0
    for ids, gt in zip(search_ids, gt_ids, strict=True):
        gt_set = set(int(value) for value in gt)
        recall += sum(int(value) in gt_set for value in ids) / len(gt)
    return recall / len(search_ids)


def random_clustered_vectors(
    rng: np.random.Generator,
    n_vectors: int,
    dim: int,
    n_centers: int,
    cluster_std: float = 0.18,
) -> np.ndarray:
    centers = rng.normal(0.0, 5.0, size=(n_centers, dim)).astype(np.float32)
    assignments = rng.integers(0, n_centers, size=n_vectors)
    noise = rng.normal(0.0, cluster_std, size=(n_vectors, dim)).astype(np.float32)
    return centers[assignments] + noise


def compute_gt(
    vectors: np.ndarray,
    queries: np.ndarray,
    topk: int,
    tags: np.ndarray | None = None,
    chunk_size: int = 8192,
) -> np.ndarray:
    tags = np.arange(len(vectors), dtype=np.int32) if tags is None else tags.astype(np.int32)
    gt = np.empty((len(queries), topk), dtype=np.int32)
    for qi, query in enumerate(queries):
        best_dists = np.empty(0, dtype=np.float32)
        best_ids = np.empty(0, dtype=np.int32)
        for start in range(0, len(vectors), chunk_size):
            end = min(start + chunk_size, len(vectors))
            dists = np.sum((vectors[start:end] - query) ** 2, axis=1)
            ids = tags[start:end]
            if len(best_ids):
                dists = np.concatenate((best_dists, dists))
                ids = np.concatenate((best_ids, ids))
            keep = np.argpartition(dists, min(topk - 1, len(dists) - 1))[:topk]
            best_dists = dists[keep]
            best_ids = ids[keep]
        order = np.argsort(best_dists)
        gt[qi] = best_ids[order[:topk]]
    return gt
