#!/usr/bin/env python3
"""Trace-driven simulation of a memory cache plus SSD-resident packed pages (D-cache).

Hot nodes and co-read relations are learned on an earlier window of queries and evaluated
on a later window. Within a query, nodes brought in by a packed page serve only later
iterations, since nodes in the same beam iteration are read concurrently.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

SECTOR = 4096


def read_neighbors(index_path: Path, nodes, ndims=128, max_node_len=388, nnps=10):
    out = {}
    fd = os.open(index_path, os.O_RDONLY)
    try:
        for u in nodes:
            buf = os.pread(fd, SECTOR, (1 + u // nnps) * SECTOR)
            base = (u % nnps) * max_node_len + ndims
            nn = int(np.frombuffer(buf, dtype=np.uint32, count=1, offset=base)[0])
            out[u] = np.frombuffer(buf, dtype=np.uint32, count=nn, offset=base + 4).astype(np.int64).tolist()
    finally:
        os.close(fd)
    return out


def learn_coread(q, n, hot_sorted, window, keep):
    us, vs = [], []
    for k in range(1, window + 1):
        same = q[:-k] == q[k:]
        u, v = n[:-k][same], n[k:][same]
        pos = np.minimum(np.searchsorted(hot_sorted, u), len(hot_sorted) - 1)
        m = (hot_sorted[pos] == u) & (u != v)
        us.append(u[m])
        vs.append(v[m])
    u, v = np.concatenate(us), np.concatenate(vs)
    uniq, cnt = np.unique((u << 32) | v, return_counts=True)
    uu, vv = uniq >> 32, uniq & 0xFFFFFFFF
    order = np.lexsort((-cnt, uu))
    uu, vv = uu[order], vv[order]
    starts = np.flatnonzero(np.r_[True, uu[1:] != uu[:-1]])
    ends = np.r_[starts[1:], len(uu)]
    return {int(uu[s]): vv[s:min(e, s + keep)].tolist() for s, e in zip(starts, ends)}


def simulate(queries, mem: set, pages: dict) -> float:
    total = 0
    for its, ns in queries:
        avail = {}
        for it, u in zip(its, ns):
            if u in mem:
                continue
            a = avail.get(u)
            if a is not None and a < it:
                continue
            total += 1
            p = pages.get(u)
            if p:
                for v in p:
                    if v not in avail:
                        avail[v] = it
    return total / len(queries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--learn", default="0:300000")
    ap.add_argument("--eval", default="900000:1000000")
    ap.add_argument("--pages", type=int, default=4693)
    ap.add_argument("--slots", type=int, default=9)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--mem", default="0,100,300,1173,3000,10000,30000,100000")
    args = ap.parse_args()

    q = np.load(args.dir / "reads_q.npy")
    it = np.load(args.dir / "reads_it.npy")
    n = np.load(args.dir / "reads_n.npy")
    l0, l1 = (int(x) for x in args.learn.split(":"))
    e0, e1 = (int(x) for x in args.eval.split(":"))
    lm = (q >= l0) & (q < l1)
    ql, nl = q[lm].astype(np.int64), n[lm]
    u, c = np.unique(nl, return_counts=True)
    rank = u[np.argsort(-c, kind="stable")]

    mems = [int(x) for x in args.mem.split(",")]
    max_hot = max(mems) + args.pages
    hot_sorted = np.sort(rank[:max_hot])
    co = learn_coread(ql, nl, hot_sorted, args.window, args.slots * 4)
    nbrs = read_neighbors(args.index, rank[:max_hot].tolist())
    vec_ids = np.load(args.dir / "hot_ids.npy")
    raw = np.fromfile(args.dir / "hotknn_200_idx_uint32.bin", dtype=np.uint32)
    knn_rows = raw[2:].reshape(-1, int(raw[1])).astype(np.int64)
    vec = {int(u): [int(v) for v in row if v != u] for u, row in zip(vec_ids.tolist(), knn_rows)}

    em = (q >= e0) & (q < e1)
    qe, ite, ne = q[em], it[em], n[em]
    bounds = np.flatnonzero(np.r_[True, qe[1:] != qe[:-1], True])
    queries = [(ite[s:e].tolist(), ne[s:e].tolist()) for s, e in zip(bounds[:-1], bounds[1:])]

    res = {"eval_queries": len(queries), "pages": args.pages, "rows": []}
    for m in mems:
        mem = set(rank[:m].tolist())
        seeds = rank[m:m + args.pages].tolist()

        def fill(src):
            out = {}
            for s in seeds:
                cand = [v for v in src.get(s, []) if v not in mem and v != s]
                out[s] = cand[:args.slots]
            return out

        row = {"mem_nodes": m,
               "memory_only": simulate(queries, mem, {}),
               "neighbor_pages": simulate(queries, mem, fill(nbrs)),
               "coread_pages": simulate(queries, mem, fill(co))}
        if all(s in vec for s in seeds):
            row["vector_pages"] = simulate(queries, mem, fill(vec))
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    (args.dir / "dcache_sim.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
