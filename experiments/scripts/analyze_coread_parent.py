#!/usr/bin/env python3
"""Motivation analysis with parent-based co-read (matches MERIT's query patching).

Within a query, the parent of a node c read from SSD is the first expanded node whose
adjacency list contains c (DiskANN inserts a node into the candidate list only when it is
first discovered). The co-read nodes of a hot node u are the nodes read from SSD that are
reachable from u along parent-child relations in the same query. Pages are formed from
the most frequent co-read nodes learned on an earlier window and evaluated on later queries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_motivation_trace import eval_pages  # noqa: E402
from simulate_dcache_trace import simulate  # noqa: E402

SECTOR = 4096


def read_adjacency(index: Path, nodes: np.ndarray, ndims=128, max_node_len=388, nnps=10, threads=32):
    fd = os.open(index, os.O_RDONLY)

    def one(u):
        buf = os.pread(fd, SECTOR, (1 + u // nnps) * SECTOR)
        base = (u % nnps) * max_node_len + ndims
        nn = int(np.frombuffer(buf, dtype=np.uint32, count=1, offset=base)[0])
        return np.frombuffer(buf, dtype=np.uint32, count=nn, offset=base + 4).astype(np.int64)

    try:
        with ThreadPoolExecutor(threads) as ex:
            return list(ex.map(one, nodes.tolist(), chunksize=4096))
    finally:
        os.close(fd)


@njit(cache=True)
def find_parents(qstart, it, nd, adj_ptr, adj_idx, nuniq):
    n = len(nd)
    par = np.full(n, -1, np.int64)
    stamp = np.full(nuniq, -1, np.int64)
    disc = np.zeros(nuniq, np.int64)
    for qi in range(len(qstart) - 1):
        s, e = qstart[qi], qstart[qi + 1]
        g = s
        while g < e:
            h = g
            while h < e and it[h] == it[g]:
                h += 1
            for j in range(g, h):
                v = nd[j]
                if stamp[v] == qi:
                    par[j] = disc[v]
            for j in range(g, h):
                u = nd[j]
                for k in range(adj_ptr[u], adj_ptr[u + 1]):
                    v = adj_idx[k]
                    if stamp[v] != qi:
                        stamp[v] = qi
                        disc[v] = j
            g = h
    return par


@njit(cache=True)
def count_pairs(lo, hi, nd, par, hot_flag):
    total = 0
    for j in range(lo, hi):
        a = par[j]
        while a >= lo:
            if hot_flag[nd[a]]:
                total += 1
            a = par[a]
    return total


@njit(cache=True)
def emit_pairs(lo, hi, nd, par, hot_flag, nuniq, keys, depth):
    c = 0
    for j in range(lo, hi):
        a = par[j]
        d = 1
        while a >= lo:
            if hot_flag[nd[a]]:
                keys[c] = nd[a] * nuniq + nd[j]
                depth[c] = min(d, 127)
                c += 1
            a = par[a]
            d += 1
    return c


def learn_pages(lo, hi, nd, par, hot_flag, uniq, keep):
    """Return {hot node id: [co-read node ids by descending count]} and depth stats."""
    nuniq = len(uniq)
    total = count_pairs(lo, hi, nd, par, hot_flag)
    keys = np.empty(total, np.int64)
    depth = np.empty(total, np.int8)
    emit_pairs(lo, hi, nd, par, hot_flag, nuniq, keys, depth)
    dhist = np.bincount(depth, minlength=128)
    k_all, c_all = np.unique(keys, return_counts=True)
    k1, c1 = np.unique(keys[depth == 1], return_counts=True)
    del keys, depth
    uu, vv = k_all // nuniq, k_all % nuniq
    order = np.lexsort((-c_all, uu))
    uu, vv, cc = uu[order], vv[order], c_all[order]
    starts = np.flatnonzero(np.r_[True, uu[1:] != uu[:-1]])
    ends = np.r_[starts[1:], len(uu)]
    pages, sel_keys, sel_cnt = {}, [], []
    for s, e in zip(starts, ends):
        t = min(e, s + keep)
        pages[int(uniq[uu[s]])] = uniq[vv[s:t]].tolist()
        sel_keys.append(uu[s:t] * nuniq + vv[s:t])
        sel_cnt.append(cc[s:t])
    return pages, dhist, (k1, c1), (sel_keys, sel_cnt)


def depth1_share(sel, k1c1, slots):
    k1, c1 = k1c1
    keys = np.concatenate([k[:slots] for k in sel[0]])
    cnt = np.concatenate([c[:slots] for c in sel[1]])
    pos = np.minimum(np.searchsorted(k1, keys), len(k1) - 1)
    d1 = np.where(k1[pos] == keys, c1[pos], 0)
    return float(d1.sum() / cnt.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--hot", type=int, default=4693)
    ap.add_argument("--mem", type=int, default=1173)
    ap.add_argument("--slots", type=int, default=9)
    ap.add_argument("--learn", default="0:300000")
    ap.add_argument("--eval", default="700000:1000000")
    ap.add_argument("--sim_eval", default="900000:1000000")
    args = ap.parse_args()

    q = np.load(args.dir / "reads_q.npy").astype(np.int64)
    it = np.load(args.dir / "reads_it.npy").astype(np.int32)
    n = np.load(args.dir / "reads_n.npy").astype(np.int64)
    uniq = np.unique(n)
    nd = np.searchsorted(uniq, n)
    print(f"reads={len(n)} unique={len(uniq)}", flush=True)

    adj = read_adjacency(args.index, uniq)
    deg = np.array([len(a) for a in adj])
    flat = np.concatenate(adj)
    pos = np.minimum(np.searchsorted(uniq, flat), len(uniq) - 1)
    keep = uniq[pos] == flat
    owner = np.repeat(np.arange(len(uniq)), deg)[keep]
    adj_idx = pos[keep]
    adj_ptr = np.zeros(len(uniq) + 1, np.int64)
    np.cumsum(np.bincount(owner, minlength=len(uniq)), out=adj_ptr[1:])
    print("adjacency loaded", flush=True)

    qstart = np.flatnonzero(np.r_[True, q[1:] != q[:-1], True]).astype(np.int64)
    par = find_parents(qstart, it, nd, adj_ptr, adj_idx, len(uniq))
    first = np.zeros(len(n), bool)
    first[qstart[:-1]] = True
    res = {"reads": int(len(n)), "unique_nodes": int(len(uniq)),
           "orphan_non_first_reads": float(((par < 0) & ~first).mean())}
    print(json.dumps(res), flush=True)

    def window(spec):
        a, b = (int(x) for x in spec.split(":"))
        return int(np.searchsorted(q, a)), int(np.searchsorted(q, b))

    llo, lhi = window(args.learn)
    elo, ehi = window(args.eval)
    slo, shi = window(args.sim_eval)

    u_l, c_l = np.unique(nd[llo:lhi], return_counts=True)
    rank = u_l[np.argsort(-c_l, kind="stable")]
    max_hot = max(args.hot, args.mem + args.hot)
    hot_flag = np.zeros(len(uniq), np.bool_)
    hot_flag[rank[:max_hot]] = True

    pages_l, dhist, k1c1, sel = learn_pages(llo, lhi, nd, par, hot_flag, uniq, args.slots * 4)
    pages_e, _, _, _ = learn_pages(elo, ehi, nd, par, hot_flag, uniq, args.slots * 4)
    print("co-read learned", flush=True)

    hot = uniq[rank[:args.hot]]
    hot_sorted = np.sort(hot)
    qe, ite, ne = q[elo:ehi], it[elo:ehi], n[elo:ehi]
    p_hot = np.minimum(np.searchsorted(hot_sorted, ne), len(hot_sorted) - 1)
    hot_mask_e = hot_sorted[p_hot] == ne

    def top(pages):
        return {u: np.array(pages.get(u, [])[:args.slots], np.int64) for u in hot.tolist()}

    res["page_hits"] = {
        "coread_parent_learned_earlier": eval_pages(qe, ite, ne, hot_mask_e, top(pages_l), args.slots),
        "coread_parent_in_sample": eval_pages(qe, ite, ne, hot_mask_e, top(pages_e), args.slots),
    }
    hot_idx = np.searchsorted(uniq, hot)
    nb_sets = {int(uniq[i]): set(adj[i].tolist()) for i in hot_idx}
    frac = [np.mean([v in nb_sets[u] for v in pages_l[u][:args.slots]])
            for u in hot.tolist() if pages_l.get(u)]
    res["page_members_that_are_graph_neighbors"] = float(np.mean(frac))
    res["page_member_occurrences_direct_child"] = depth1_share(sel, k1c1, args.slots)
    tot = dhist.sum()
    res["coread_occurrence_depth_share"] = {str(d): float(dhist[d] / tot) for d in range(1, 11)}
    res["coread_occurrence_depth_gt10"] = float(dhist[11:].sum() / tot)
    print(json.dumps(res, indent=2), flush=True)

    sb = np.flatnonzero(np.r_[True, q[slo + 1:shi] != q[slo:shi - 1], True]) + slo
    queries = [(it[s:e].tolist(), n[s:e].tolist()) for s, e in zip(sb[:-1], sb[1:])]
    rows = []
    for m in sorted({0, args.mem}):
        mem = set(uniq[rank[:m]].tolist())
        seeds = uniq[rank[m:m + args.hot]].tolist()

        def fill(src):
            return {s: [v for v in src.get(s, []) if v not in mem and v != s][:args.slots] for s in seeds}

        row = {"mem_nodes": m, "memory_only": simulate(queries, mem, {}),
               "coread_parent_pages": simulate(queries, mem, fill(pages_l))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    res["sim"] = {"eval_queries": len(queries), "pages": args.hot, "rows": rows}
    (args.dir / "motivation_parent.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
