#!/usr/bin/env python3
"""Convert a DiskANN 4-section PQ pivot file to the older 5-section format (used by Starling,
MARGO and GPS) by inserting an identity dimension rearrangement. PQ codes are unchanged."""
import sys

import numpy as np


def main(src, dst):
    b = open(src, 'rb').read()
    nsec = int(np.frombuffer(b[:4], dtype=np.int32)[0])
    if nsec == 5:
        open(dst, 'wb').write(b)
        return
    assert nsec == 4, nsec
    off = np.frombuffer(b[8:8 + 32], dtype=np.uint64).astype(int)
    tab, cen, chk = b[off[0]:off[1]], b[off[1]:off[2]], b[off[2]:off[3]]
    dim = int(np.frombuffer(tab[4:8], dtype=np.int32)[0])
    rearr = np.array([dim, 1], dtype=np.int32).tobytes() + np.arange(dim, dtype=np.uint32).tobytes()
    secs = [tab, cen, rearr, chk]
    o = [4096]
    for s in secs:
        o.append(o[-1] + len(s))
    hdr = np.array([5, 1], dtype=np.int32).tobytes() + np.array(o, dtype=np.uint64).tobytes()
    open(dst, 'wb').write(hdr + b'\0' * (4096 - len(hdr)) + b''.join(secs))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
