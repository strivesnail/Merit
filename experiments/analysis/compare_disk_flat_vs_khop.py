#!/usr/bin/env python3
"""Compare flat heat-ordered disk cache vs k-hop-packed list (same profile, cap=100k)."""

from __future__ import annotations

import os
import struct
import sys
from collections import deque
from pathlib import Path

import numpy as np


def load_diskann_bin_u64(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        npts, ndim = struct.unpack("<II", f.read(8))
        assert ndim == 1
        data = np.frombuffer(f.read(npts * 8), dtype=np.uint64)
    return data


def load_diskann_bin_u32(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        npts, ndim = struct.unpack("<II", f.read(8))
        assert ndim == 1
        data = np.frombuffer(f.read(npts * 4), dtype=np.uint32)
    return data


def load_vamana_graph(mem_index: Path):
    with open(mem_index, "rb") as f:
        file_size = struct.unpack("<Q", f.read(8))[0]
        assert file_size == mem_index.stat().st_size
        width, medoid, frozen = struct.unpack("<IIQ", f.read(16))
        adj = []
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            nnbrs = struct.unpack("<I", hdr)[0]
            nbrs = []
            if nnbrs:
                nbrs = list(struct.unpack(f"<{nnbrs}I", f.read(4 * nnbrs)))
            if nnbrs > width:
                f.seek(4 * (nnbrs - width), 1)
            adj.append(nbrs[:width] if len(nbrs) > width else nbrs)
    return width, medoid, adj


def build_undirected_adj(adj):
    n = len(adj)
    und = [[] for _ in range(n)]
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if v < n:
                und[u].append(v)
                und[v].append(u)
    for u in range(n):
        und[u] = sorted(set(und[u]))
    return und


def flat_hot_list(node_expand: np.ndarray, max_nodes: int) -> list[int]:
    order = np.argsort(-node_expand, kind="mergesort")
    out = []
    for i in order:
        if node_expand[i] == 0:
            break
        out.append(int(i))
        if len(out) >= max_nodes:
            break
    return out


def k_hop_disk_list(
    node_expand: np.ndarray,
    und,
    nnodes_per_sector: int,
    k_hops: int,
    max_nodes: int,
    exclude: set[int],
) -> list[int]:
    n = len(und)
    placed = np.zeros(n, dtype=bool)
    for ex in exclude:
        if ex < n:
            placed[ex] = True

    order = np.argsort(-node_expand, kind="mergesort")
    node_list: list[int] = []

    while len(node_list) < max_nodes:
        page_cap = nnodes_per_sector
        page_nodes: list[int] = []
        seed = None
        for node in order:
            node = int(node)
            if not placed[node] and node_expand[node] > 0:
                seed = node
                break
        if seed is None:
            break
        page_nodes.append(seed)
        placed[seed] = True

        while len(page_nodes) < page_cap and len(node_list) + len(page_nodes) < max_nodes:
            dist = np.full(n, -1, dtype=np.int32)
            q = deque()
            for src in page_nodes:
                if dist[src] != -1:
                    continue
                dist[src] = 0
                q.append(src)
            while q:
                u = q.popleft()
                if dist[u] >= k_hops:
                    continue
                for v in und[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            best = None
            best_h = 0
            for i in range(n):
                if placed[i] or dist[i] == -1:
                    continue
                h = int(node_expand[i])
                if h == 0:
                    continue
                if h > best_h or (h == best_h and (best is None or i < best)):
                    best_h = h
                    best = i
            if best is None:
                break
            page_nodes.append(best)
            placed[best] = True

        for node in page_nodes:
            if node in exclude or node_expand[node] == 0:
                continue
            node_list.append(node)
            if len(node_list) >= max_nodes:
                break
    return node_list


def heat_mass(node_expand: np.ndarray, ids) -> float:
    total = float(node_expand.sum())
    if total == 0:
        return 0.0
    s = sum(int(node_expand[i]) for i in ids)
    return s / total


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(os.environ.get("DATA_DIR", repo_root / "data" / "sift1m"))
    profile = data_dir / "run2_profile_same_trace"
    mem_index = data_dir / "sift1m_index_mem.index"
    khop_nodes_file = data_dir / "five_way_k2_20260722_083546/disk01_merit_dc.nodes"
    max_nodes = 100_000
    nps = 5
    k_hops = 2

    node_expand = load_diskann_bin_u64(profile.with_name(profile.name + "_node_expand.bin"))
    assert len(node_expand) == 1_000_000

    flat = flat_hot_list(node_expand, max_nodes)
    flat_set = set(flat)

    if khop_nodes_file.is_file():
        khop_saved = load_diskann_bin_u32(khop_nodes_file).tolist()
    else:
        khop_saved = None

    print("Loading graph for k-hop recompute...")
    _, _, adj = load_vamana_graph(mem_index)
    und = build_undirected_adj(adj)
    khop = k_hop_disk_list(node_expand, und, nps, k_hops, max_nodes, set())
    khop_set = set(khop)

    if khop_saved is not None:
        same = khop_saved == khop
        print(f"Recomputed k-hop list matches saved sidecar: {same} (len {len(khop_saved)} vs {len(khop)})")

    total_expand = int(node_expand.sum())
    active = int((node_expand > 0).sum())
    print(f"\nProfile: total node_expand={total_expand:,}, nodes with expand>0={active:,}")

    def report(name: str, ids: list[int], idset: set[int]):
        mass = heat_mass(node_expand, ids)
        mean_h = np.mean([node_expand[i] for i in ids]) if ids else 0
        min_h = min(int(node_expand[i]) for i in ids) if ids else 0
        print(f"\n=== {name} (n={len(ids):,}) ===")
        print(f"  expand mass captured: {100 * mass:.2f}% of all expands")
        print(f"  mean / min expand per cached node: {mean_h:.1f} / {min_h}")

    report("Flat top-100k (heat order, expand>0)", flat, flat_set)
    report("K-hop k=2 packed (current disk cache)", khop, khop_set)

    only_flat = flat_set - khop_set
    only_khop = khop_set - flat_set
    both = flat_set & khop_set
    print(f"\n=== Set overlap ===")
    print(f"  |intersection|={len(both):,}  only_flat={len(only_flat):,}  only_khop={len(only_khop):,}")

    if only_flat:
        lost_mass = sum(int(node_expand[i]) for i in only_flat)
        print(f"  expand in flat-only (dropped by k-hop): {lost_mass:,} ({100 * lost_mass / total_expand:.2f}% of total)")
    if only_khop:
        extra_mass = sum(int(node_expand[i]) for i in only_khop)
        print(f"  expand in k-hop-only (not in flat top-100k): {extra_mass:,} ({100 * extra_mass / total_expand:.2f}% of total)")

    # Rank by heat: flat rank 0 = hottest
    rank = np.empty(len(node_expand), dtype=np.int64)
    order = np.argsort(-node_expand, kind="mergesort")
    for r, nid in enumerate(order):
        rank[nid] = r

    khop_ranks = [int(rank[i]) for i in khop]
    flat_ranks = [int(rank[i]) for i in flat]
    print(f"\n=== Heat rank (0=hottest) ===")
    print(f"  flat:   max_rank={max(flat_ranks):,}  p50={int(np.median(flat_ranks)):,}")
    print(f"  k-hop:  max_rank={max(khop_ranks):,}  p50={int(np.median(khop_ranks)):,}  p90={int(np.percentile(khop_ranks, 90)):,}")

    worst_khop = sorted(khop, key=lambda i: node_expand[i])[:5]
    print("  coldest nodes in k-hop cache (sample):", [(i, int(node_expand[i]), int(rank[i])) for i in worst_khop])

    # Proxy: if each expand event is proportional to node_expand counts, expected fraction of expand-events hitting cache
    print(f"\n=== Workload proxy (expand-weighted hit potential) ===")
    print(f"  P(expand node in cache) flat:  {100 * heat_mass(node_expand, flat):.2f}%")
    print(f"  P(expand node in cache) k-hop: {100 * heat_mass(node_expand, khop):.2f}%")

    # mem tier overlap: top 12843 like mem 0.01
    mem_n = 12_843
    mem_set = set(flat[:mem_n])
    khop_ex_mem = [i for i in khop if i not in mem_set]
    print(f"\n=== mem0.01 + disk (exclude memory tier) ===")
    print(f"  k-hop nodes after excluding top-{mem_n} memory: {len(khop_ex_mem):,}")
    print(f"  expand mass (k-hop ex mem): {100 * heat_mass(node_expand, khop_ex_mem):.2f}%")
    print(f"  expand mass (flat top-100k ex mem ranks): {100 * heat_mass(node_expand, flat[mem_n:mem_n+max_nodes]):.2f}%")


if __name__ == "__main__":
    main()
