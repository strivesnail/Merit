#!/usr/bin/env python3
"""Offline seed/neighbor importance analysis from MERIT profile bins."""

from __future__ import annotations

import argparse
import struct
from collections import defaultdict
from pathlib import Path


def load_node_expand(prefix: Path) -> list[int]:
    path = Path(str(prefix) + "_node_expand.bin")
    with path.open("rb") as stream:
        n, d = struct.unpack("<II", stream.read(8))
        assert d == 1
        return list(struct.unpack(f"<{n}Q", stream.read(8 * n)))


def load_directed_edges(prefix: Path) -> dict[int, list[tuple[int, int]]]:
    base = Path(str(prefix))
    with (Path(str(base) + "_edge_u.bin")).open("rb") as stream:
        n, d = struct.unpack("<II", stream.read(8))
        assert d == 1
        us = struct.unpack(f"<{n}I", stream.read(4 * n))
    with (Path(str(base) + "_edge_v.bin")).open("rb") as stream:
        n2, d2 = struct.unpack("<II", stream.read(8))
        assert n2 == n and d2 == 1
        vs = struct.unpack(f"<{n}I", stream.read(4 * n))
    with (Path(str(base) + "_edge_count.bin")).open("rb") as stream:
        n3, d3 = struct.unpack("<II", stream.read(8))
        assert n3 == n and d3 == 1
        cs = struct.unpack(f"<{n}Q", stream.read(8 * n))

    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, c in zip(us, vs, cs):
        if c:
            out[u].append((v, c))
    for parent in out:
        out[parent].sort(key=lambda item: (-item[1], item[0]))
    return out


def pct_k(sorted_edges: list[tuple[int, int]], pct: float, page_cap: int) -> int:
    total = sum(c for _, c in sorted_edges)
    if total == 0:
        return 0
    cum = 0
    k = 0
    for _, c in sorted_edges:
        cum += c
        k += 1
        if cum / total >= pct:
            break
    if page_cap <= 0 or k == 0:
        return k
    first_cap = max(page_cap - 1, 0)
    if k <= first_cap:
        return min(first_cap, len(sorted_edges))
    rem = k - first_cap
    pages_after = (rem + page_cap - 1) // page_cap
    return min(first_cap + pages_after * page_cap, len(sorted_edges))


def hot_pages(nbr_count: int, page_cap: int) -> int:
    if nbr_count <= 0:
        return 0
    if nbr_count <= page_cap - 1:
        return 1
    rem = nbr_count - (page_cap - 1)
    return 1 + (rem + page_cap - 1) // page_cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-prefix", type=Path, required=True)
    parser.add_argument("--page-cap", type=int, default=8)
    parser.add_argument("--pct", type=float, default=1.0)
    parser.add_argument("--top-seeds", type=int, default=30)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    prefix = args.profile_prefix
    node_expand = load_node_expand(prefix)
    out_edges = load_directed_edges(prefix)

    seeds = []
    for parent, edges in out_edges.items():
        heat = sum(c for _, c in edges)
        k = pct_k(edges, args.pct, args.page_cap)
        pages = hot_pages(k, args.page_cap)
        benefit = heat / max(pages, 1)
        expand = node_expand[parent] if parent < len(node_expand) else 0
        seeds.append((benefit, heat, pages, parent, k, expand))

    seeds.sort(reverse=True)
    heat_rank = sorted(seeds, key=lambda item: (-item[1], item[3]))

    total_pages = sum(p for _, _, p, _, _, _ in seeds)
    total_heat = sum(h for _, h, _, _, _, _ in seeds)
    covered_heat_top50pct = sum(
        h for _, h, _, _, _, _ in sorted(seeds, key=lambda x: -x[1])[: max(1, len(seeds) // 2)]
    )

    print(f"profile={prefix} parents={len(seeds)} page_cap={args.page_cap} pct={args.pct}")
    print(f"total_out_heat={total_heat} total_hot_pages={total_pages} "
          f"top50%_seeds_cover_heat={covered_heat_top50pct/total_heat:.1%}")
    print("\nTop seeds by benefit density B(P)=H(P)/Pages(P):")
    print(f"{'rank':>4} {'parent':>10} {'heat':>10} {'expand':>8} {'pages':>6} {'nbrs':>6} {'benefit':>12}")
    for i, (benefit, heat, pages, parent, k, expand) in enumerate(seeds[: args.top_seeds], 1):
        print(f"{i:4d} {parent:10d} {heat:10d} {expand:8d} {pages:6d} {k:6d} {benefit:12.1f}")

    heat_order = {parent: idx for idx, (_, _, _, parent, _, _) in enumerate(heat_rank)}
    benefit_order = {parent: idx for idx, (_, _, _, parent, _, _) in enumerate(seeds)}
    deltas = [abs(heat_order[p] - benefit_order[p]) for p in heat_order]
    print(f"\nMean |rank_heat - rank_benefit| = {sum(deltas)/len(deltas):.1f}")
    print(f"Seeds with >=2 rank displacement: {sum(1 for d in deltas if d >= 2)}")

    # Expand-only: high expand, low out-heat (expand > 2x median, heat bottom quartile)
    expands = [e for *_, e in seeds]
    heats = [h for _, h, _, _, _, _ in seeds]
    med_expand = sorted(expands)[len(expands) // 2]
    q1_heat = sorted(heats)[len(heats) // 4]
    expand_only = [
        (parent, expand, heat)
        for _, heat, _, parent, _, expand in seeds
        if expand > 2 * med_expand and heat <= q1_heat
    ]
    print(f"\nExpand-heavy / low out-heat seeds (candidates to drop on seed axis): {len(expand_only)}")
    for parent, expand, heat in expand_only[:10]:
        print(f"  parent={parent} expand={expand} out_heat={heat}")

    if seeds:
        _, heat, pages, parent, k, expand = seeds[0]
        print(f"\nTop benefit seed {parent}: heat={heat} expand={expand} pages={pages} top-{k} neighbors:")
        for child, count in out_edges[parent][: min(10, k)]:
            print(f"  -> {child}: {count}")

    if args.json_out:
        import json

        payload = {
            "parents": len(seeds),
            "total_out_heat": total_heat,
            "total_hot_pages": total_pages,
            "top50pct_heat_coverage": covered_heat_top50pct / max(total_heat, 1),
            "mean_rank_displacement": sum(deltas) / max(len(deltas), 1),
            "expand_only_count": len(expand_only),
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
