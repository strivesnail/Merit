from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeann import IndexPipeANN, Metric
from utils import bin_write, compute_gt, compute_recall, random_clustered_vectors


def count_hits_at_or_above(result_ids: np.ndarray, split: int) -> int:
    return sum(int(vid) >= split for row in result_ids for vid in row)


def count_hits_in(result_ids: np.ndarray, tags: np.ndarray) -> int:
    tag_set = {int(tag) for tag in tags}
    return sum(int(vid) in tag_set for row in result_ids for vid in row)


def main() -> None:
    rng = np.random.default_rng(42)
    dim = int(os.environ.get("PIPEANN_INSERT_DIM", "32"))
    n_base = int(os.environ.get("PIPEANN_INSERT_N_BASE", "10000"))
    n_insert = int(os.environ.get("PIPEANN_INSERT_N_INSERT", "10000"))
    n_queries = int(os.environ.get("PIPEANN_INSERT_N_QUERY", "10"))
    topk = int(os.environ.get("PIPEANN_INSERT_TOPK", "1"))
    search_L = int(os.environ.get("PIPEANN_INSERT_SEARCH_L", "1000"))
    n_centers = int(os.environ.get("PIPEANN_INSERT_N_CENTERS", "1"))
    n_delete = int(os.environ.get("PIPEANN_INSERT_N_DELETE", "10000"))
    recall_threshold = float(os.environ.get("PIPEANN_INSERT_RECALL", "0.9"))

    n_total = n_base + n_insert
    vectors = random_clustered_vectors(rng, n_total, dim, n_centers=n_centers)
    query_ids = rng.choice(n_insert, size=n_queries, replace=False)
    queries = vectors[n_base + query_ids].copy()
    queries += rng.normal(0.0, 0.02, size=queries.shape).astype(np.float32)

    base_vectors = np.ascontiguousarray(vectors[:n_base])
    insert_vectors = np.ascontiguousarray(vectors[n_base:])
    all_tags = np.arange(n_total, dtype=np.uint32)

    base_gt = compute_gt(base_vectors, queries, topk)
    inserted_gt = compute_gt(
        insert_vectors,
        queries,
        topk,
        tags=np.arange(n_base, n_total, dtype=np.int32),
    )
    all_gt = compute_gt(vectors, queries, topk, tags=all_tags.astype(np.int32))
    delete_tags = np.arange(n_delete, dtype=np.uint32)
    keep = np.ones(n_total, dtype=bool)
    keep[delete_tags] = False
    after_delete_gt = compute_gt(vectors[keep], queries, topk, tags=all_tags[keep].astype(np.int32))

    with tempfile.TemporaryDirectory(prefix="pipeann_insert_search_") as tmp_dir:
        tmp = Path(tmp_dir)
        data_path = tmp / "base.bin"
        index_prefix = tmp / "index"
        bin_write(base_vectors, data_path)

        idx = IndexPipeANN(dim, "float32", Metric.L2)
        idx.omp_set_num_threads(4)
        idx.build(
            str(data_path),
            str(index_prefix),
            build_mem_index=False,
            max_nbrs=32,
            build_L=64,
            PQ_bytes=16,
            memory_use_GB=1,
        )
        idx.load(str(index_prefix))

        before_ids, _ = idx.search(queries, topk=topk, L=search_L)
        before_recall = compute_recall(before_ids, base_gt)
        before_inserted_hits = count_hits_at_or_above(before_ids, n_base)

        for start in range(0, n_insert, 10000):
            end = min(start + 10000, n_insert)
            tags = np.arange(n_base + start, n_base + end, dtype=np.uint32)
            idx.add(insert_vectors[start:end], tags)
        idx.save(str(index_prefix))
        idx.load(str(index_prefix))

        after_insert_ids, _ = idx.search(queries, topk=topk, L=search_L)
        after_insert_recall = compute_recall(after_insert_ids, all_gt)
        after_insert_inserted_hits = count_hits_at_or_above(after_insert_ids, n_base)

        idx.remove(delete_tags)
        idx.save(str(index_prefix))
        idx.load(str(index_prefix))

        after_delete_ids, _ = idx.search(queries, topk=topk, L=search_L)
        after_delete_recall = compute_recall(after_delete_ids, after_delete_gt)
        after_delete_inserted_hits = count_hits_at_or_above(after_delete_ids, n_base)
        after_delete_deleted_hits = count_hits_in(after_delete_ids, delete_tags)

    print(
        f"Dataset: base={n_base}, insert={n_insert}, dim={dim}, "
        f"queries={n_queries}, centers={n_centers}, delete={n_delete}"
    )
    print(f"Recall@{topk} before insert: {before_recall:.4f}")
    print(f"Recall@{topk} after insert: {after_insert_recall:.4f}")
    print(f"Recall@{topk} after delete: {after_delete_recall:.4f}")
    print(f"Inserted hits before insert: {before_inserted_hits}/{before_ids.size}")
    print(f"Inserted hits after insert: {after_insert_inserted_hits}/{after_insert_ids.size}")
    print(f"Inserted hits after delete: {after_delete_inserted_hits}/{after_delete_ids.size}")
    print(f"Deleted hits after delete: {after_delete_deleted_hits}/{after_delete_ids.size}")
    print(f"Insert recall delta: {after_insert_recall - before_recall:+.4f}")
    print(f"Delete recall delta: {after_delete_recall - after_insert_recall:+.4f}")

    assert before_inserted_hits == 0, (
        f"search before insert returned inserted tags: {before_inserted_hits}/{before_ids.size}"
    )
    assert after_insert_inserted_hits > 0, (
        f"search after insert returned only original/base tags: {after_insert_ids.tolist()}"
    )
    assert after_delete_deleted_hits == 0, (
        f"search after delete returned deleted tags: {after_delete_ids.tolist()}"
    )
    assert after_delete_inserted_hits > 0, (
        f"search after deleting base returned no inserted tags: {after_delete_ids.tolist()}"
    )

    assert before_recall >= recall_threshold, (
        f"recall before insert is too low: {before_recall:.4f}"
    )
    assert after_insert_recall >= recall_threshold, (
        f"recall after insert is too low: {after_insert_recall:.4f}"
    )
    assert after_delete_recall >= recall_threshold, (
        f"recall after deleting base is too low: {after_delete_recall:.4f}"
    )
    print("PASS: disk build, insert, and delete recall stay above threshold.")


def test_insert_search_recall() -> None:
    main()


if __name__ == "__main__":
    main()
