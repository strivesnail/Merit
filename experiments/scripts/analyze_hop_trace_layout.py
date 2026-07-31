#!/usr/bin/env python3
"""Analyze hop trace vs MERIT disk cache page layout for one query."""

from __future__ import annotations

import argparse
import csv
import struct
from collections import defaultdict
from pathlib import Path


def load_nodes_bin(path: Path, nnodes_per_sector: int = 5) -> tuple[list[int], dict[int, tuple[int, int]], dict[int, list[int]]]:
    with path.open("rb") as f:
        npts, nd = struct.unpack("ii", f.read(8))
        assert nd == 1
        raw = f.read(4 * npts)
    nodes = list(struct.unpack(f"{npts}I", raw))
    loc: dict[int, tuple[int, int]] = {}
    page_nodes: dict[int, list[int]] = defaultdict(list)
    for i, nid in enumerate(nodes):
        sec = i // nnodes_per_sector
        slot = i % nnodes_per_sector
        loc[nid] = (sec, slot)
        page_nodes[sec].append(nid)
    return nodes, loc, page_nodes


def load_hop_trace(path: Path, query_id: int = 0) -> list[tuple[int, str, int]]:
    hops: list[tuple[int, str, int]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["query_id"]) != query_id:
                continue
            hops.append((int(row["hop"]), row["source"], int(row["node_id"])))
    return hops


def load_profile_edges(prefix: Path) -> dict[int, list[int]]:
    """Load parent->children from edge profile (u->v with counts)."""
    with (prefix.with_name(prefix.name + "_edge_u.bin")).open("rb") as f:
        nu, du = struct.unpack("ii", f.read(8))
        u = struct.unpack(f"{nu}I", f.read(4 * nu))
    with (prefix.with_name(prefix.name + "_edge_v.bin")).open("rb") as f:
        nv, dv = struct.unpack("ii", f.read(8))
        v = struct.unpack(f"{nv}I", f.read(4 * nv))
    parent_children: dict[int, list[int]] = defaultdict(list)
    for pu, cv in zip(u, v):
        parent_children[pu].append(cv)
    return parent_children


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop_trace", required=True)
    ap.add_argument("--nodes_bin", required=True)
    ap.add_argument("--profile_prefix", required=True)
    ap.add_argument("--query_id", type=int, default=0)
    ap.add_argument("--max_hops", type=int, default=8)
    args = ap.parse_args()

    nodes, loc, page_nodes = load_nodes_bin(Path(args.nodes_bin))
    hops = load_hop_trace(Path(args.hop_trace), args.query_id)
    edges = load_profile_edges(Path(args.profile_prefix))

    by_hop: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for hop, src, nid in hops:
        by_hop[hop].append((src, nid))

    print(f"# query {args.query_id}: {len(by_hop)} hops, {len(hops)} frontier reads")
    print()

    shown_pages: set[int] = set()
    for hop in sorted(by_hop.keys())[: args.max_hops]:
        items = by_hop[hop]
        merit = [nid for src, nid in items if src == "merit"]
        base = [nid for src, nid in items if src == "base"]
        print(f"## Hop {hop}  (beam frontier: merit={merit}, base={base})")

        hop_secs = set()
        for src, nid in items:
            if src != "merit" or nid not in loc:
                continue
            sec, slot = loc[nid]
            hop_secs.add(sec)
            mates = page_nodes[sec]
            same_hop_mates = [m for m in mates if m in merit]
            print(
                f"  - node {nid}: disk cache sector {sec} slot {slot}; "
                f"page mates {mates}; co-accessed this hop {same_hop_mates}"
            )
            if sec not in shown_pages and len(mates) >= 2:
                shown_pages.add(sec)
                # show profile path chain if consecutive nodes on page
                chain = " -> ".join(str(m) for m in mates)
                print(f"    [layout page] packed chain: {chain}")

        if len(hop_secs) > 1:
            print(f"  => this hop touches {len(hop_secs)} distinct disk cache sectors (ideally 1 if layout matched beam)")

        # Compare with profile parent expansion for first merit node
        if merit:
            p = merit[0]
            prof_nbrs = edges.get(p, [])[:8]
            prof_on_page = [n for n in prof_nbrs if n in loc and loc[n][0] == loc[p][0]]
            print(
                f"  profile parent {p} top neighbors (sample): {prof_nbrs[:6]}; "
                f"on same disk cache page: {prof_on_page}"
            )
        print()


if __name__ == "__main__":
    main()
