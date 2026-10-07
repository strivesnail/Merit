#!/usr/bin/env python3
"""Write the vectors of the most frequently read nodes as a DiskANN query file.

The hot nodes are ranked on the learn window of a recorded read trace. Searching the index
with these vectors yields each hot node's nearest nodes in vector space, which is the
grouping criterion of distance-based layouts (PageANN, VeloANN).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

SECTOR = 4096


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--learn", default="0:300000")
    ap.add_argument("--top", type=int, default=6500)
    ap.add_argument("--ndims", type=int, default=128)
    ap.add_argument("--max_node_len", type=int, default=388)
    ap.add_argument("--nnps", type=int, default=10)
    args = ap.parse_args()

    q = np.load(args.dir / "reads_q.npy")
    n = np.load(args.dir / "reads_n.npy")
    l0, l1 = (int(x) for x in args.learn.split(":"))
    lm = (q >= l0) & (q < l1)
    u, c = np.unique(n[lm], return_counts=True)
    hot = u[np.argsort(-c, kind="stable")[:args.top]].astype(np.int64)

    vecs = np.empty((len(hot), args.ndims), dtype=np.uint8)
    fd = os.open(args.index, os.O_RDONLY)
    try:
        for i, v in enumerate(hot.tolist()):
            buf = os.pread(fd, SECTOR, (1 + v // args.nnps) * SECTOR)
            off = (v % args.nnps) * args.max_node_len
            vecs[i] = np.frombuffer(buf, dtype=np.uint8, count=args.ndims, offset=off)
    finally:
        os.close(fd)

    np.save(args.dir / "hot_ids.npy", hot)
    with (args.dir / "hot_queries.u8bin").open("wb") as f:
        np.array([len(hot), args.ndims], dtype=np.int32).tofile(f)
        vecs.tofile(f)
    print(f"wrote {len(hot)} hot nodes")


if __name__ == "__main__":
    main()
