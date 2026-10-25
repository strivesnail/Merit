#!/usr/bin/env python3
"""Schematic of NEIGHBORCACHE cache replacement and member update.

Redraws the replacement figure so it matches the runtime:
N-cache is CLOCK plus shadow admission, M-cache heat comes from sampled
storage co-access, page members are W(A) still resident in N-cache, and
NonSeed LRU only evicts metadata.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

FONT_DIR = os.path.expanduser("~/.fonts/libertinus")
if os.path.isdir(FONT_DIR):
    for name in os.listdir(FONT_DIR):
        if name.endswith(".otf"):
            font_manager.fontManager.addfont(os.path.join(FONT_DIR, name))
    plt.rcParams["font.family"] = "Libertinus Sans"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"

INK = "#1a1a1a"
MUTED = "#3f3f3f"
BLUE = "#1f5f8b"
SEED = "#2a2a2a"
SLOT = "#cfcfcf"
SLOT_ALT = "#e4e4e4"
PALE = "#f7f7f7"
EDGE = "#222222"

OUT_DIR = "/home/jianz/workload/code/NeighborCache/docs/figures"


def chip(ax, x, y, s, label, seed=False, face=None, dashed=False):
    fc = SEED if seed else (SLOT if face is None else face)
    ax.add_patch(
        Rectangle(
            (x, y),
            s,
            s,
            facecolor=fc,
            edgecolor=EDGE,
            linewidth=0.9,
            linestyle=(0, (2.2, 1.4)) if dashed else "solid",
            joinstyle="miter",
            zorder=3,
        )
    )
    ax.text(
        x + s / 2,
        y + s / 2,
        label,
        ha="center",
        va="center",
        color="white" if seed else INK,
        fontsize=10,
        zorder=4,
    )


def row(ax, x, y, labels, s=4.6, gap=0.55, seed0=True):
    for i, label in enumerate(labels):
        face = None if i % 2 == 0 else SLOT_ALT
        chip(ax, x + i * (s + gap), y, s, label, seed=(seed0 and i == 0), face=face)
    return x + len(labels) * (s + gap) - gap


def empty_row(ax, x, y, n, s=4.6, gap=0.55):
    for i in range(n):
        chip(ax, x + i * (s + gap), y, s, "", face="white")


def arrow(ax, start, end, text=None, text_xy=None, color=INK, dashed=False, rad=0.0, fs=8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.05,
            color=color,
            linestyle=(0, (3.2, 1.8)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )
    if text:
        tx, ty = text_xy if text_xy is not None else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(tx, ty, text, ha="center", va="center", color=color, fontsize=fs, zorder=5)


def heap(ax, root, children, labels):
    rx, ry, s = root
    ax.plot([rx + s / 2, children[0][0]], [ry, children[0][1] + 1.15], color=INK, lw=0.9, zorder=1)
    ax.plot([rx + s / 2, children[1][0]], [ry, children[1][1] + 1.15], color=INK, lw=0.9, zorder=1)
    chip(ax, rx, ry, s, labels[0], seed=True)
    for (cx, cy), lab in zip(children, labels[1:]):
        ax.add_patch(Circle((cx, cy), 1.35, facecolor=SLOT, edgecolor=EDGE, lw=0.8, zorder=3))
        ax.text(cx, cy, lab, ha="center", va="center", fontsize=8, color=INK, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(16.2, 8.05))
    ax.set_xlim(0, 162)
    ax.set_ylim(0, 80)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    # ----- N-cache -----
    ax.text(18, 77.2, "N-cache", ha="center", va="center", fontsize=13, color=INK)
    ny, ns = 67.4, 4.8
    xs = [2.4, 8.2, 14.0, 19.8, 28.6]
    for i, (x, lab) in enumerate(zip(xs, "Auvwp")):
        chip(ax, x, ny, ns, lab, seed=(lab == "A"), face=SLOT if i % 2 == 0 else SLOT_ALT, dashed=(lab == "p"))
    ax.annotate(
        "",
        xy=(31.0, 73.0),
        xytext=(10.0, 74.2),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.05, connectionstyle="arc3,rad=-0.28"),
    )
    ax.text(20.2, 75.15, "CLOCK", ha="center", va="center", fontsize=8, color=BLUE)
    ax.text(31.0, 66.15, "victim", ha="center", va="top", fontsize=7, color=MUTED)

    chip(ax, 2.4, 59.6, 4.4, "y", face=PALE)
    arrow(ax, (6.8, 64.1), (6.8, 67.3))
    ax.text(0.6, 62.4, "(1)", ha="center", va="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.text(8.2, 62.4, "adaptive admit", ha="left", va="center", fontsize=8, color=INK)
    ax.text(16.5, 57.6, "shadow: MRU  /  Hop-12  /  Hop-8", ha="center", va="center", fontsize=7.5, color=MUTED)

    # A,u,v,w stay resident and can enter the max-heap. p is only the CLOCK victim.
    ax.plot([22.2, 22.2, 60.5], [67.2, 49.6, 49.6], color=BLUE, lw=0.95, linestyle=(0, (3, 1.6)))
    arrow(ax, (60.5, 49.6), (64.6, 45.2), color=BLUE, dashed=True)
    ax.text(40.5, 50.35, "N-cache-resident NonSeed", ha="center", va="bottom", fontsize=7.5, color=BLUE)

    # ----- M-cache -----
    ax.text(78, 77.2, "M-cache", ha="center", va="center", fontsize=13, color=INK)
    ax.add_patch(
        FancyBboxPatch(
            (43.0, 68.6),
            16.4,
            6.6,
            boxstyle="round,pad=0.2,rounding_size=0.5",
            facecolor="white",
            edgecolor=EDGE,
            lw=0.8,
            zorder=3,
        )
    )
    ax.text(51.2, 71.9, "sampled\nstorage access", ha="center", va="center", fontsize=7.3, color=INK, zorder=4)
    arrow(ax, (59.6, 71.9), (67.2, 71.9), color=BLUE, dashed=True)
    ax.text(63.4, 73.15, "(2)  H, W", ha="center", va="center", fontsize=8, color=BLUE)

    tx, ty, tw, th = 67.0, 63.4, 36.0, 11.6
    ax.add_patch(Rectangle((tx, ty), tw, th, facecolor="white", edgecolor=EDGE, lw=0.9, zorder=3))
    cols = [tx + 4.4, tx + 12.0, tx + 24.0]
    for cx, header in zip(cols, ("ID", "H", "W")):
        ax.text(cx, ty + th - 1.9, header, ha="center", va="center", fontsize=8.5, color=INK, zorder=4)
    ax.plot([tx, tx + tw], [ty + th - 3.45, ty + th - 3.45], color=EDGE, lw=0.6, zorder=4)
    for i, rec in enumerate((("A", "9", "u:6,  v:4,  w:3"), ("B", "2", "r:3,  s:2"), ("…", "…", "…"))):
        yy = ty + th - 5.5 - i * 2.35
        for cx, val in zip(cols, rec):
            ax.text(cx, yy, val, ha="center", va="center", fontsize=8, color=INK, zorder=4)

    lx, ly, lw, lh = 70.5, 56.6, 29.0, 5.4
    ax.add_patch(
        FancyBboxPatch(
            (lx, ly),
            lw,
            lh,
            boxstyle="round,pad=0.12,rounding_size=0.35",
            facecolor="#f3f6f8",
            edgecolor=BLUE,
            lw=0.7,
            zorder=3,
        )
    )
    ax.text(lx + lw / 2, ly + 3.45, "NonSeed LRU:   head → ··· → tail", ha="center", va="center", fontsize=7.5, color=INK, zorder=4)
    ax.text(lx + lw / 2, ly + 1.55, "metadata eviction only", ha="center", va="center", fontsize=7, color=MUTED, zorder=4)

    # H leaves the table on both sides of the LRU box, so the two structures stay separate.
    arrow(ax, (68.4, 63.2), (68.4, 46.2), color=BLUE, dashed=True)
    arrow(ax, (101.6, 63.2), (96.2, 46.2), color=BLUE, dashed=True)
    ax.text(66.2, 54.4, "H", ha="center", va="center", fontsize=8, color=BLUE)

    # ----- heaps -----
    ax.text(62.6, 43.6, "Max", ha="right", va="center", fontsize=8.5, color=INK)
    ax.text(100.6, 43.6, "Min", ha="left", va="center", fontsize=8.5, color=INK)
    heap(ax, (66.2, 40.2, 4.8), [(63.6, 33.4), (74.2, 33.4)], ["A", "D", "E"])
    heap(ax, (90.4, 40.2, 4.8), [(87.6, 33.4), (98.4, 33.4)], ["B", "C", "F"])
    ax.text(80.8, 44.6, "(3)", ha="center", va="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.text(80.8, 42.35, "H(A) > H(B)", ha="center", va="center", fontsize=8, color=BLUE)
    ax.text(80.8, 40.55, "if D-cache is full", ha="center", va="center", fontsize=7, color=MUTED)
    ax.text(68.6, 30.2, "NonSeed → ReadyToInsertion", ha="center", va="center", fontsize=7.2, color=INK)
    ax.text(92.8, 30.2, "Seed → ReadyToDeletion", ha="center", va="center", fontsize=7.2, color=INK)
    ax.text(80.8, 28.15, "circles: heap order, not page members", ha="center", va="center", fontsize=7, color=MUTED)

    # ----- D-cache -----
    ax.text(132, 77.2, "D-cache", ha="center", va="center", fontsize=13, color=INK)
    ax.text(116.2, 74.15, "Before", ha="center", va="center", fontsize=8, color=MUTED)
    ax.text(147.6, 74.15, "After", ha="center", va="center", fontsize=8, color=MUTED)
    row(ax, 108.2, 68.2, list("Brst"), s=4.4, gap=0.4)
    row(ax, 139.2, 68.2, list("Auvw"), s=4.4, gap=0.4)
    arrow(ax, (127.2, 70.4), (138.6, 70.4))
    row(ax, 108.2, 61.6, list("Cjkl"), s=4.4, gap=0.4)
    row(ax, 139.2, 61.6, list("Cjkl"), s=4.4, gap=0.4)
    arrow(ax, (127.2, 63.8), (138.6, 63.8), color=MUTED, dashed=True)

    ax.text(132.0, 59.15, "on commit:   A → Seed,    B → NonSeed", ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(112.6, 55.7, "Pending", ha="center", va="center", fontsize=8, color=MUTED)
    row(ax, 122.8, 53.2, list("Auvw"), s=4.4, gap=0.4)
    ax.text(133.2, 51.55, "top W(A) still in N-cache", ha="center", va="center", fontsize=7.2, color=MUTED)
    arrow(ax, (133.2, 53.0), (133.2, 49.2))
    empty_row(ax, 122.8, 44.0, 4, s=4.4, gap=0.4)
    ax.text(133.2, 42.4, "Commit queue", ha="center", va="center", fontsize=8, color=MUTED)

    ax.plot([145.6, 158.6, 158.6], [46.2, 46.2, 70.4], color=INK, lw=1.05)
    arrow(ax, (158.6, 70.4), (157.2, 70.4))
    ax.text(159.4, 57.5, "Async\nwrite", ha="left", va="center", fontsize=7.5, color=INK)
    ax.text(152.8, 72.55, "(5)", ha="center", va="center", fontsize=9, color=BLUE, fontweight="bold")

    arrow(ax, (101.2, 38.4), (122.2, 46.2), color=INK, rad=-0.12)
    ax.text(112.6, 44.7, "(4)", ha="center", va="center", fontsize=9, color=BLUE, fontweight="bold")

    # ----- member update -----
    ax.plot([2, 160], [24.8, 24.8], color="#d0d0d0", lw=0.8)
    ax.text(8.6, 16.6, "Member\nupdate", ha="center", va="center", fontsize=11, color=INK)
    row(ax, 20.0, 14.2, list("Auvw"), s=4.6, gap=0.45)
    ax.text(42.4, 16.5, "+", ha="center", va="center", fontsize=12, color=INK)
    chip(ax, 45.0, 14.2, 4.6, "x", face=PALE)
    ax.text(47.3, 12.35, "fetched\nthis query", ha="center", va="top", fontsize=6.6, color=MUTED)
    arrow(ax, (50.2, 16.5), (60.6, 16.5))
    ax.text(55.4, 18.15, "(6)", ha="center", va="center", fontsize=9, color=BLUE, fontweight="bold")
    empty_row(ax, 61.4, 14.2, 4, s=4.6, gap=0.45)
    ax.text(71.6, 12.55, "Patch queue", ha="center", va="center", fontsize=8, color=MUTED)
    arrow(ax, (82.4, 16.5), (97.2, 16.5), text="Async write", text_xy=(89.8, 18.15), fs=8)
    row(ax, 98.0, 14.2, list("Axuv"), s=4.6, gap=0.45)
    ax.text(124.5, 12.55, "insert x after the seed, drop the tail", ha="center", va="center", fontsize=7.5, color=MUTED)

    ax.text(81, 4.2, "NEIGHBORCACHE cache replacement and update", ha="center", va="center", fontsize=12, color=INK)

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, "neighborcache_cache_replacement.png")
    pdf = os.path.join(OUT_DIR, "neighborcache_cache_replacement.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
