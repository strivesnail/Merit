#!/usr/bin/env python3
"""QPS and disk reads vs D-cache size (% of base index), one page vs two pages per seed."""
import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAGS = [("x1", "+0% hotspot", "tab:red"), ("x2", "+1% hotspot", "tab:orange"),
        ("x11", "+10% hotspot", "tab:green"), ("x51", "+50% hotspot", "tab:purple")]
BASE_SECTORS = 1e7
ROW = re.compile(r"\[same (\w+)\] unique=(\S+) d=\S+ QPS=(\S+) IOs=(\S+)")
PACKED = re.compile(r"packed file sectors=(\d+)")
BREAKDOWN = re.compile(r"breakdown: base_reads=(\S+) dcache_reads=(\S+)")


def last_match(path, pat):
    if not os.path.exists(path):
        return None
    m = pat.findall(open(path).read())
    return m[-1] if m else None


def load(roots):
    """Later roots override earlier ones (reruns)."""
    out = {}
    for root in roots.split(","):
        path = os.path.join(root, "status.txt")
        if not os.path.exists(path):
            continue
        for line in open(path):
            m = ROW.search(line)
            if not m:
                continue
            tag, f, qps = m.group(1), m.group(2), float(m.group(3))
            d = os.path.join(root, f"same_{tag}")
            sectors = last_match(os.path.join(d, f"u{f}_build", "run.out"), PACKED)
            bd = last_match(os.path.join(d, f"u{f}", "run.out"), BREAKDOWN)
            if sectors is None or bd is None:
                continue
            reads = float(bd[0]) + float(bd[1])
            out.setdefault(tag, {})[f] = (100.0 * int(sectors) / BASE_SECTORS, qps, reads)
    # Once every seed fits, larger budgets pack the same file; keep one point per size.
    for tag, pts in out.items():
        by_size = {}
        for f in sorted(pts, key=float):
            by_size[round(pts[f][0], 3)] = f
        out[tag] = {f: pts[f] for f in by_size.values()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one_page", required=True, help="run root(s), comma-separated; later ones override")
    ap.add_argument("--two_page", required=True, help="run root(s), comma-separated; later ones override")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title_suffix", default="")
    ap.add_argument("--diskann_qps", type=float, default=3604.0)
    ap.add_argument("--diskann_reads", type=float, default=145.2)
    args = ap.parse_args()

    rows = [("One page per seed", load(args.one_page)),
            ("Two pages per seed (children + grandchildren)", load(args.two_page))]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
    for r, (title, data) in enumerate(rows):
        for c, (ylabel, idx, ref, ymax) in enumerate([("QPS", 1, args.diskann_qps, 11000),
                                                       ("Disk reads / query", 2, args.diskann_reads, 150)]):
            ax = axes[r][c]
            for tag, label, color in TAGS:
                pts = sorted(data.get(tag, {}).values())
                if pts:
                    ax.plot([p[0] for p in pts], [p[idx] for p in pts], "-o", color=color, ms=5,
                            label=f"NeighborCache, {label}")
            ax.axhline(ref, color="tab:blue", lw=1.3, label="DiskANN")
            ax.set_title(title + args.title_suffix, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlim(0, 9)
            ax.set_ylim(0, ymax)
            ax.tick_params(labelsize=8)
            if r == 1:
                ax.set_xlabel("D-cache size (% of base index)", fontsize=9)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(args.out)


if __name__ == "__main__":
    main()
