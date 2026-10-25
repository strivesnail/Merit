#!/usr/bin/env python3
"""Vector redraw of NeighborCache overview (matches reference screenshot layout)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, FancyBboxPatch as FBP

OUT = Path(__file__).resolve().parent
PDF = OUT / "neighborcache_overview_swap_label.pdf"
PNG = OUT / "neighborcache_overview_swap_label.png"

BLUE = "#2F6FED"
RED = "#E03131"
LG = "#E8E8E8"
MG = "#C4C4C4"
QBLUE = "#A8C8E8"
EDGE = "#222222"
FS, FS_S, FS_L = 10.5, 8.5, 13.0


def sq(ax, x, y, s, text=None, fc="white", tc="black", fs=FS, lw=0.9, z=3, weight="normal"):
    ax.add_patch(Rectangle((x, y), s, s, facecolor=fc, edgecolor=EDGE, lw=lw, zorder=z))
    if text is not None:
        ax.text(x + s / 2, y + s / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight=weight, zorder=z + 1)


def rbox(ax, x, y, w, h, lw=1.0, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="white", edgecolor=EDGE, lw=lw, zorder=z,
    ))


def arr(ax, a, b, c=BLUE, lw=1.4, ms=12, rad=0.0, z=6):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>,head_length=2.6,head_width=1.6",
        color=c, lw=lw, mutation_scale=ms, zorder=z,
        connectionstyle=f"arc3,rad={rad}",
    ))


def dash(ax, a, b, rad=0.2, c="#777", lw=1.05):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
        color=c, lw=lw, linestyle=(0, (2.5, 1.7)), zorder=2,
    ))


def node(ax, x, y, r, t, fc=LG, tc="black"):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=EDGE, lw=0.95, zorder=4))
    ax.text(x, y, t, ha="center", va="center", fontsize=FS_S, color=tc,
            fontweight="bold", zorder=5)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    # Match reference aspect ~1024x566
    fig, ax = plt.subplots(figsize=(11.0, 6.08), dpi=240)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 55.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- N-cache ----
    ax.text(9.5, 53.3, "N-cache", ha="center", va="bottom", fontsize=FS_L, fontweight="bold")
    ax.text(9.5, 51.7, "LRU", ha="center", va="bottom", fontsize=FS_S, color="#444")

    s = 3.05
    nx, top = 8.0, 48.0
    labs = ["z", "v", "u", "A", "w"]
    pos = {}
    for i, t in enumerate(labs):
        y = top - i * s
        sq(ax, nx, y, s, t, fc=LG)
        pos[t] = (nx + s / 2, y + s / 2)
    ax.text(nx + s / 2, top - 5 * s + 0.45, "⋮", ha="center", va="center", fontsize=14, color="#555")
    py = top - 6.15 * s
    sq(ax, nx, py, s, "p", fc=LG)
    m = 0.38
    ax.plot([nx + m, nx + s - m], [py + m, py + s - m], color=BLUE, lw=2.0, zorder=7)
    ax.plot([nx + m, nx + s - m], [py + s - m, py + m], color=BLUE, lw=2.0, zorder=7)

    arr(ax, (3.6, pos["z"][1]), (nx - 0.1, pos["z"][1]))
    ax.text(2.5, pos["z"][1] + 0.25, "(1)", color=BLUE, fontsize=FS, fontweight="bold", ha="center")

    # ---- M-cache ----
    ax.text(74.5, 53.3, "M-cache", ha="center", va="bottom", fontsize=FS_L, fontweight="bold")
    ax.text(74.5, 51.7, "LRU", ha="center", va="bottom", fontsize=FS_S, color="#444")

    tx, ty = 55.0, 38.6
    cw = [4.8, 10.2, 21.5]
    rh = 2.7
    heads = ["ID", "H", "W"]
    tw, th = sum(cw), rh * 5

    x = tx
    for h, w in zip(heads, cw):
        ax.add_patch(Rectangle((x, ty + th - rh), w, rh, fc="#EEEEEE", ec=EDGE, lw=0.75, zorder=3))
        ax.text(x + w / 2, ty + th - rh / 2, h, ha="center", va="center", fontsize=FS_S, fontweight="bold")
        x += w
    for ri in range(4):
        y = ty + th - (ri + 2) * rh
        x = tx
        for w in cw:
            ax.add_patch(Rectangle((x, y), w, rh, fc="white", ec=EDGE, lw=0.75, zorder=3))
            x += w
    ax.add_patch(Rectangle((tx, ty), tw, th, fill=False, ec=EDGE, lw=1.05, zorder=5))

    def cy(ri):
        return ty + th - (ri + 1.5) * rh

    ax.text(tx + cw[0] / 2, cy(0), "A", ha="center", va="center", fontsize=FS_S)
    # 13 -> 14
    ax.text(tx + cw[0] + 1.3, cy(0), "13", color=RED, ha="center", va="center", fontsize=FS_S)
    ax.annotate("", xy=(tx + cw[0] + 5.3, cy(0)), xytext=(tx + cw[0] + 2.5, cy(0)),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9, mutation_scale=8))
    ax.text(tx + cw[0] + 6.9, cy(0), "14", color=RED, ha="center", va="center", fontsize=FS_S)
    # z:6 -> 7
    wx = tx + cw[0] + cw[1] + 0.45
    ax.text(wx, cy(0), "z:", ha="left", va="center", fontsize=FS_S - 0.4)
    ax.text(wx + 1.5, cy(0), "6", color=RED, ha="left", va="center", fontsize=FS_S - 0.4)
    ax.annotate("", xy=(wx + 3.55, cy(0)), xytext=(wx + 2.5, cy(0)),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.85, mutation_scale=7))
    ax.text(wx + 3.7, cy(0), "7,", color=RED, ha="left", va="center", fontsize=FS_S - 0.4)
    ax.text(wx + 5.4, cy(0), "v:4, w:3", ha="left", va="center", fontsize=FS_S - 0.4)

    for ri, row in enumerate([("B", "5", "r:3, s:2"), ("...", "...", "..."), ("q", "1", "y:1")], 1):
        ax.text(tx + cw[0] / 2, cy(ri), row[0], ha="center", va="center", fontsize=FS_S)
        ax.text(tx + cw[0] + cw[1] / 2, cy(ri), row[1], ha="center", va="center", fontsize=FS_S)
        ax.text(tx + cw[0] + cw[1] + 0.55, cy(ri), row[2], ha="left", va="center", fontsize=FS_S)

    zb = 2.45
    zx, zy = 48.0, cy(0) - zb / 2
    sq(ax, zx, zy, zb, "z", fc="white", fs=FS_S)
    arr(ax, (zx + zb + 0.05, zy + zb / 2), (tx - 0.1, cy(0)))
    ax.text(zx + zb / 2, zy + zb + 0.5, "(2)", color=BLUE, fontsize=FS, fontweight="bold", ha="center")

    # ---- Heaps ----
    ax.text(62.5, 36.0, "Max Heap", ha="center", va="bottom", fontsize=FS_S, fontweight="bold")
    ax.text(86.5, 36.0, "Min Heap", ha="center", va="bottom", fontsize=FS_S, fontweight="bold")
    hr = 1.35
    mx, my = 62.5, 32.6
    node(ax, mx, my, hr, "A", fc=LG)
    node(ax, mx - 2.6, my - 3.0, hr * 0.92, "D", fc=LG)
    node(ax, mx + 2.6, my - 3.0, hr * 0.92, "E", fc=LG)
    ax.plot([mx, mx - 2.6], [my - hr, my - 3.0 + hr * 0.92], color=EDGE, lw=0.9, zorder=3)
    ax.plot([mx, mx + 2.6], [my - hr, my - 3.0 + hr * 0.92], color=EDGE, lw=0.9, zorder=3)

    sx, sy = 86.5, 32.6
    node(ax, sx, sy, hr, "B", fc="black", tc="white")
    node(ax, sx - 2.6, sy - 3.0, hr * 0.92, "C", fc="black", tc="white")
    node(ax, sx + 2.6, sy - 3.0, hr * 0.92, "F", fc="black", tc="white")
    ax.plot([sx, sx - 2.6], [sy - hr, sy - 3.0 + hr * 0.92], color=EDGE, lw=0.9, zorder=3)
    ax.plot([sx, sx + 2.6], [sy - hr, sy - 3.0 + hr * 0.92], color=EDGE, lw=0.9, zorder=3)

    ax.text(74.5, 31.8, r"$H(A)>H(B)$", color=BLUE, fontsize=FS, fontweight="bold", ha="center")
    ax.text(74.5, 29.9, "(3)", color=BLUE, fontsize=FS, fontweight="bold", ha="center")

    # ---- Assembly ----
    cs = 2.9
    g = 0.15
    items = ["A", "z", "v", "w"]
    ax0, ay0 = 27.0, 27.8
    aw = 4 * cs + 3 * g + 0.7
    ah = cs + 0.7
    rbox(ax, ax0, ay0, aw, ah)
    ac = {}
    for i, t in enumerate(items):
        x = ax0 + 0.35 + i * (cs + g)
        y = ay0 + 0.35
        sq(ax, x, y, cs, t, fc=MG if t == "A" else LG)
        ac[t] = (x + cs / 2, y + cs / 2)

    arr(ax, (mx - hr - 0.05, my), (ax0 + aw + 0.05, ay0 + ah * 0.55), rad=-0.02)
    ax.text(50.5, 30.8, "(4)", color=BLUE, fontsize=FS, fontweight="bold", ha="center")

    for t, rad in zip(items, [0.30, 0.22, 0.12, 0.04]):
        dash(ax, (pos[t][0] + s / 2 + 0.05, pos[t][1]),
             (ac[t][0] - cs / 2 - 0.05, ac[t][1] + 0.1), rad=rad)
    ax.text(19.5, 37.5, "(5)", color=BLUE, fontsize=FS, fontweight="bold", ha="center")

    # ---- Commit queue ----
    qs = 2.9
    qn = 6
    qx, qy = 26.8, 21.4
    qw = qn * qs + 0.5
    rbox(ax, qx - 0.25, qy - 0.25, qw, qs + 0.5)
    last = None
    for i in range(qn):
        x = qx + i * qs
        sq(ax, x, qy, qs, None, fc=QBLUE if i == qn - 1 else LG)
        if i == qn - 1:
            last = (x + qs / 2, qy + qs / 2)
    ax.plot([ax0 + 0.4, qx + 0.2], [ay0, qy + qs], color="#AAA", lw=0.7, ls=(0, (1.5, 1.2)), zorder=1)
    ax.plot([ax0 + aw - 0.4, qx + qw - 0.8], [ay0, qy + qs], color="#AAA", lw=0.7, ls=(0, (1.5, 1.2)), zorder=1)
    ax.plot([ax0 + aw / 2, last[0]], [ay0, qy + qs], color="#888", lw=0.9, ls=(0, (2.2, 1.6)), zorder=2)

    ax.annotate("", xy=(qx + 0.8, qy - 1.35), xytext=(qx + 14.5, qy - 1.35),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.0, mutation_scale=9))
    ax.text(qx + 15.0, qy - 1.35, "Commit queue", ha="left", va="center", fontsize=FS_S, color="#333")

    # ---- separator + D-cache ----
    sep = 18.0
    ax.plot([1.5, 98.5], [sep, sep], color=BLUE, lw=1.5, ls=(0, (1.0, 1.3)))

    ax.text(50, 16.5, "D-cache", ha="center", va="top", fontsize=FS_L + 1, fontweight="bold")

    def page(lab, y, vals):
        ax.text(22.5, y + 1.45, lab, ha="right", va="center", fontsize=FS)
        x0, c = 24.0, 2.95
        for i, t in enumerate(vals):
            if i == 0:
                sq(ax, x0 + i * c, y, c, t, fc="black", tc="white", weight="bold")
            else:
                sq(ax, x0 + i * c, y, c, t, fc=LG)
        return x0 + c / 2, y + c

    p1x, p1top = page("Page1", 10.8, ["A", "z", "v", "w"])
    page("Page2", 6.5, ["C", "j", "k", "l"])
    ax.text(22.5, 3.5, "Pagen", ha="right", va="center", fontsize=FS)
    ax.text(25.5, 3.5, "...", ha="left", va="center", fontsize=FS + 1)

    arr(ax, (last[0], qy - 0.05), (p1x, p1top + 0.1), rad=0.0)
    ax.text(last[0] + 2.3, 14.6, "(6)", color=BLUE, fontsize=FS, fontweight="bold", ha="center")

    fig.savefig(PDF, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(PNG, bbox_inches="tight", pad_inches=0.05, dpi=240)
    plt.close(fig)
    print("Wrote", PDF)
    print("Wrote", PNG)


if __name__ == "__main__":
    main()
