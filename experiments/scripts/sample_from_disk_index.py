#!/usr/bin/env python3
"""Sample whole pages of a DiskANN disk index and write the vectors and ids as PipeANN slice files.

Node ids within a page are consecutive, but ids carry no spatial order in SIFT1B, so whole-page sampling
is an unbiased node sample that needs one read per nnodes_per_sector nodes.
Outputs <prefix>_data.bin (uint8 bin) and <prefix>_ids.bin (uint32 bin), as gen_random_slice does.
"""

from __future__ import annotations

import argparse
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SECTOR = 4096


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="path to *_disk.index")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--pages", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    with open(args.index, "rb") as f:
        hdr = f.read(SECTOR)
    npts, dim, _, node_len, nnps = struct.unpack_from("<5Q", hdr, 8)
    npages = (npts + nnps - 1) // nnps
    rng = np.random.default_rng(args.seed)
    pages = np.sort(rng.choice(npages - 1, size=args.pages, replace=False))  # skip a possibly partial last page

    fd = os.open(args.index, os.O_RDONLY)
    vecs = np.empty((args.pages * nnps, dim), dtype=np.uint8)

    def read(i: int) -> None:
        buf = np.frombuffer(os.pread(fd, SECTOR, (1 + int(pages[i])) * SECTOR), dtype=np.uint8)
        for j in range(nnps):
            vecs[i * nnps + j] = buf[j * node_len:j * node_len + dim]

    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(read, range(args.pages), chunksize=4096))
    os.close(fd)

    ids = (pages[:, None] * nnps + np.arange(nnps)[None, :]).ravel().astype(np.uint32)
    with open(args.prefix + "_data.bin", "wb") as f:
        f.write(struct.pack("<ii", len(ids), dim))
        vecs.tofile(f)
    with open(args.prefix + "_ids.bin", "wb") as f:
        f.write(struct.pack("<ii", len(ids), 1))
        ids.tofile(f)
    print(f"sampled {len(ids)} nodes from {args.pages} pages (nnps={nnps}, node_len={node_len})")


if __name__ == "__main__":
    main()
