#!/usr/bin/env python3
"""Profile stats for page-cap / co-location feasibility (professor's skew hypothesis)."""
import struct
import sys
from collections import defaultdict
from pathlib import Path


def read_bin(path: Path, fmt_char: str, item_size: int):
    with open(path, "rb") as f:
        npts, dim = struct.unpack("<II", f.read(8))
        assert dim == 1
        data = f.read(npts * item_size)
        return list(struct.unpack(f"<{npts}{fmt_char}", data))


def main():
    prefix = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/home/jianz/workplace/diskann/anndisk/data/sift1m/workloads/profiles/small_y_10k"
    )
    beam = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    node_expand = read_bin(prefix.parent / f"{prefix.name}_node_expand.bin", "Q", 8)
    u = read_bin(prefix.parent / f"{prefix.name}_edge_u.bin", "I", 4)
    v = read_bin(prefix.parent / f"{prefix.name}_edge_v.bin", "I", 4)
    c = read_bin(prefix.parent / f"{prefix.name}_edge_count.bin", "Q", 8)

    by_parent: dict[int, list[int]] = defaultdict(list)
    for ui, vi, ci in zip(u, v, c):
        by_parent[ui].append(ci)

    parent_out_total = sum(sum(vs) for vs in by_parent.values())
    within_top1 = []
    for vs in by_parent.values():
        s = sorted(vs, reverse=True)
        t = sum(s)
        if t:
            within_top1.append(100 * s[0] / t)

    print(f"profile={prefix.name} parents={len(by_parent)} beam={beam}")
    print(
        f"within_parent_top1_child_pct: "
        f"median={sorted(within_top1)[len(within_top1)//2]:.1f}% "
        f"mean={sum(within_top1)/len(within_top1):.1f}%"
    )
    print(
        f"profile_children_per_parent: "
        f"avg={sum(len(v) for v in by_parent.values())/len(by_parent):.1f} "
        f"max={max(len(v) for v in by_parent.values())}"
    )

    for page_cap in (5, 10, 15, 20):
        cap_children = max(0, page_cap - 1)
        covered = 0
        overflow = 0
        for vs in by_parent.values():
            s = sorted(vs, reverse=True)
            take = min(cap_children, len(s))
            covered += sum(s[:take])
            if len(s) > cap_children:
                overflow += 1
        pct = 100 * covered / parent_out_total
        opct = 100 * overflow / len(by_parent)
        print(
            f"page_cap={page_cap} (1 parent + {cap_children} children): "
            f"outgoing_heat_captured={pct:.1f}% parents_overflow={overflow} ({opct:.1f}%)"
        )


if __name__ == "__main__":
    main()
