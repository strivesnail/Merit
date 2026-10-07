#!/usr/bin/env python3
"""Motivation analysis on a DiskANN per-query read trace (no cache).

Inputs: hop_trace.csv from search_disk_index --dump_hop_trace_path, and the disk index
file (for adjacency lists). Outputs a JSON summary and prints a readable report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

SECTOR = 4096


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    txt = path.with_suffix(".reads.txt")
    if not txt.exists():
        subprocess.run(
            f"awk -F, '$3==\"frontier\" && $4==\"base\" {{print $1, $2, $5}}' {path} > {txt}",
            shell=True, check=True)
    arr = np.fromfile(txt, dtype=np.int64, sep=" ").reshape(-1, 3)
    q, it, n = arr[:, 0], arr[:, 1].astype(np.int32), arr[:, 2]
    order = np.lexsort((it, q))
    return q[order], it[order], n[order]


def read_meta(index_path: Path) -> tuple[int, int, int]:
    with index_path.open("rb") as f:
        hdr = np.frombuffer(f.read(8 + 8 * 9), dtype=np.uint64)
    # DiskANN disk index header: [nr, nc] then npts, ndims, medoid, max_node_len, nnodes_per_sector, ...
    npts, ndims, _medoid, max_node_len, nnps = (int(x) for x in hdr[1:6])
    return ndims, max_node_len, nnps


def read_neighbors(index_path: Path, nodes: np.ndarray, ndims: int, max_node_len: int, nnps: int,
                   elem_size: int = 1) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    fd = os.open(index_path, os.O_RDONLY)
    try:
        for u in nodes.tolist():
            sec = 1 + u // nnps
            off = (u % nnps) * max_node_len
            buf = os.pread(fd, SECTOR, sec * SECTOR)
            base = off + ndims * elem_size
            nn = int(np.frombuffer(buf, dtype=np.uint32, count=1, offset=base)[0])
            out[u] = np.frombuffer(buf, dtype=np.uint32, count=nn, offset=base + 4).astype(np.int64)
    finally:
        os.close(fd)
    return out


def load_vector_knn(trace_dir: Path, result: str = "hotknn_200_idx_uint32.bin") -> dict[int, np.ndarray]:
    """Nearest nodes in vector space of each hot node, from make_hot_queries.py + search_disk_index."""
    ids = np.load(trace_dir / "hot_ids.npy")
    raw = np.fromfile(trace_dir / result, dtype=np.uint32)
    k = int(raw[1])
    knn = raw[2:].reshape(-1, k).astype(np.int64)
    return {int(u): row[row != u] for u, row in zip(ids.tolist(), knn)}


def learn_coread(q, it, n, hot_set_mask_fn, window: int, page_slots: int) -> dict[int, np.ndarray]:
    us, vs = [], []
    for k in range(1, window + 1):
        same = q[:-k] == q[k:]
        u = n[:-k][same]
        v = n[k:][same]
        m = hot_set_mask_fn(u)
        us.append(u[m])
        vs.append(v[m])
    u = np.concatenate(us)
    v = np.concatenate(vs)
    keep = u != v
    u, v = u[keep], v[keep]
    key = (u << 32) | v
    uniq, cnt = np.unique(key, return_counts=True)
    uu = uniq >> 32
    vv = uniq & 0xFFFFFFFF
    order = np.lexsort((-cnt, uu))
    uu, vv = uu[order], vv[order]
    pages: dict[int, np.ndarray] = {}
    starts = np.flatnonzero(np.r_[True, uu[1:] != uu[:-1]])
    ends = np.r_[starts[1:], len(uu)]
    for s, e in zip(starts, ends):
        pages[int(uu[s])] = vv[s:min(e, s + page_slots)]
    return pages


def eval_pages(q, it, n, hot_mask, pages: dict[int, np.ndarray], page_slots: int) -> float:
    """Mean number of page members (excluding the hot node) read later in the same query."""
    key_all = (q << 30) | n
    order = np.argsort(key_all, kind="stable")
    key_sorted = key_all[order]
    it_sorted = it[order]

    occ = np.flatnonzero(hot_mask)
    occ_q = q[occ]
    occ_it = it[occ]
    occ_n = n[occ]
    members = np.full((len(occ), page_slots), -1, dtype=np.int64)
    for i, u in enumerate(occ_n.tolist()):
        p = pages.get(u)
        if p is not None and len(p):
            members[i, :len(p)] = p[:page_slots]
    hits = np.zeros(len(occ), dtype=np.int64)
    for s in range(page_slots):
        m = members[:, s]
        valid = m >= 0
        k = (occ_q << 30) | np.where(valid, m, 0)
        pos = np.searchsorted(key_sorted, k)
        pos_c = np.minimum(pos, len(key_sorted) - 1)
        found = valid & (key_sorted[pos_c] == k) & (it_sorted[pos_c] > occ_it)
        hits += found
    return float(hits.mean()) if len(hits) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hot", type=int, default=4693, help="number of hot nodes (= D-cache pages)")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--learn", type=str, default="0:300000")
    ap.add_argument("--eval", type=str, default="700000:1000000")
    args = ap.parse_args()

    q, it, n = load_trace(args.trace)
    nq = int(q.max()) + 1
    total_reads = len(n)
    res: dict[str, object] = {"queries": nq, "total_reads": total_reads,
                              "reads_per_query": total_reads / nq}

    uniq, cnt = np.unique(n, return_counts=True)
    cnt_sorted = np.sort(cnt)[::-1]
    csum = np.cumsum(cnt_sorted)
    res["unique_nodes_read"] = int(len(uniq))
    conc = {}
    for k in [100, 1000, 4693, 10000, 100000, 1000000]:
        if k <= len(cnt_sorted):
            conc[str(k)] = float(csum[k - 1] / total_reads)
    res["top_k_share"] = conc
    res["cdf_curve"] = [[int(k), float(csum[k - 1] / total_reads)]
                        for k in np.unique(np.logspace(0, np.log10(len(cnt_sorted)), 200).astype(int))]

    ndims, max_node_len, nnps = read_meta(args.index)
    res["nnodes_per_sector"] = nnps
    page_slots = nnps - 1

    l0, l1 = (int(x) for x in args.learn.split(":"))
    e0, e1 = (int(x) for x in args.eval.split(":"))
    lm = (q >= l0) & (q < l1)
    em = (q >= e0) & (q < e1)
    ql, itl, nl = q[lm], it[lm], n[lm]
    qe, ite, ne = q[em], it[em], n[em]

    u_l, c_l = np.unique(nl, return_counts=True)
    hot = u_l[np.argsort(-c_l)[:args.hot]]
    hot_sorted = np.sort(hot)

    def is_hot(x: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(hot_sorted, x)
        pos = np.minimum(pos, len(hot_sorted) - 1)
        return hot_sorted[pos] == x

    hot_mask_e = is_hot(ne)
    res["hot_nodes"] = int(len(hot))
    res["hot_share_of_eval_reads"] = float(hot_mask_e.mean())

    nbrs = read_neighbors(args.index, hot, ndims, max_node_len, nnps)
    res["mean_degree_hot"] = float(np.mean([len(v) for v in nbrs.values()]))

    id_pages = {}
    for u in hot.tolist():
        b = (u // nnps) * nnps
        id_pages[u] = np.array([x for x in range(b, b + nnps) if x != u], dtype=np.int64)
    nbr_pages = {u: v[:page_slots] for u, v in nbrs.items()}
    all_nbr_pages = {u: v for u, v in nbrs.items()}
    knn = load_vector_knn(args.trace.parent)
    missing = [u for u in hot.tolist() if u not in knn]
    if missing:
        raise SystemExit(f"{len(missing)} hot nodes lack vector kNN; rerun make_hot_queries.py with larger --top")
    vec_pages = {u: knn[u][:page_slots] for u in hot.tolist()}
    co_learn = learn_coread(ql, itl, nl, is_hot, args.window, page_slots)
    co_insample = learn_coread(qe, ite, ne, is_hot, args.window, page_slots)

    res["page_hits"] = {
        "id_layout": eval_pages(qe, ite, ne, hot_mask_e, id_pages, page_slots),
        "graph_neighbors": eval_pages(qe, ite, ne, hot_mask_e, nbr_pages, page_slots),
        "vector_nearest": eval_pages(qe, ite, ne, hot_mask_e, vec_pages, page_slots),
        "coread_learned_earlier": eval_pages(qe, ite, ne, hot_mask_e, co_learn, page_slots),
        "coread_in_sample": eval_pages(qe, ite, ne, hot_mask_e, co_insample, page_slots),
    }
    max_deg = max(len(v) for v in nbrs.values())
    res["all_neighbors_read_later"] = eval_pages(qe, ite, ne, hot_mask_e, all_nbr_pages, max_deg)

    overlap = []
    for u, p in co_learn.items():
        if len(p):
            nb = set(nbrs[u].tolist())
            overlap.append(np.mean([int(v) in nb for v in p.tolist()]))
    res["coread_partners_that_are_neighbors"] = float(np.mean(overlap)) if overlap else 0.0
    vec_overlap = [np.mean([int(v) in set(vec_pages[u].tolist()) for v in p.tolist()])
                   for u, p in co_learn.items() if len(p)]
    res["coread_partners_that_are_vector_nearest"] = float(np.mean(vec_overlap)) if vec_overlap else 0.0
    res["vector_nearest_that_are_neighbors"] = float(np.mean(
        [np.mean([int(v) in set(nbrs[u].tolist()) for v in vec_pages[u].tolist()]) for u in hot.tolist()]))

    args.out.write_text(json.dumps(res, indent=2))
    show = {k: v for k, v in res.items() if k != "cdf_curve"}
    print(json.dumps(show, indent=2))


if __name__ == "__main__":
    main()
