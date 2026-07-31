#!/usr/bin/env python3
"""Quick profile stats for layout feasibility."""
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_bin(path: Path, fmt_char: str, item_size: int):
    with open(path, "rb") as f:
        npts, dim = struct.unpack("<II", f.read(8))
        assert dim == 1
        n = npts
        data = f.read(n * item_size)
        return list(struct.unpack(f"<{n}{fmt_char}", data))


def main():
    prefix = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/home/jianz/workplace/diskann/anndisk/data/sift1m/workloads/profiles/small_y_10k"
    )
    beam = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    page_cap = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    node_expand = read_bin(prefix.parent / f"{prefix.name}_node_expand.bin", "Q", 8)
    u = read_bin(prefix.parent / f"{prefix.name}_edge_u.bin", "I", 4)
    v = read_bin(prefix.parent / f"{prefix.name}_edge_v.bin", "I", 4)
    c = read_bin(prefix.parent / f"{prefix.name}_edge_count.bin", "Q", 8)

    hot_nodes = sum(1 for x in node_expand if x > 0)
    by_parent = defaultdict(list)
    for ui, vi, ci in zip(u, v, c):
        by_parent[ui].append((vi, ci))

    templates = Counter()
    template_sizes = []
    for children in by_parent.values():
        children.sort(key=lambda x: (-x[1], x[0]))
        take = min(beam, len(children))
        if take == 0:
            continue
        fr = tuple(sorted(ch[0] for ch in children[:take]))
        if len(fr) >= 2:
            templates[fr] += sum(ch[1] for ch in children[:take])
            template_sizes.append(len(fr))

    undirected = Counter()
    for ui, vi, ci in zip(u, v, c):
        a, b = (ui, vi) if ui < vi else (vi, ui)
        undirected[(a, b)] += ci

    fit = sum(1 for s in template_sizes if s <= page_cap)
    print(f"profile={prefix.name}")
    print(f"hot_nodes={hot_nodes}")
    print(f"directed_edges={len(u)} undirected_edges={len(undirected)}")
    print(f"frontier_templates(size>=2, beam={beam})={len(templates)}")
    if template_sizes:
        print(
            f"template_size min/max/avg={min(template_sizes)}/{max(template_sizes)}/"
            f"{sum(template_sizes)/len(template_sizes):.2f}"
        )
        print(f"fit_page_cap_{page_cap}={fit}/{len(template_sizes)} ({100*fit/len(template_sizes):.1f}%)")


if __name__ == "__main__":
    main()
