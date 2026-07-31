#!/usr/bin/env python3
"""Measure random vs sequential read latency on a real disk index file."""
import argparse
import ctypes
import os
import random
import statistics
import time

O_DIRECT = getattr(os, "O_DIRECT", 0)
O_RDONLY = os.O_RDONLY


def open_direct(path: str):
    flags = O_RDONLY
    if O_DIRECT:
        try:
            return os.open(path, flags | O_DIRECT)
        except OSError:
            pass
    return os.open(path, flags)


def align_up(n: int, align: int) -> int:
    return (n + align - 1) // align * align


def bench_random(fd: int, file_size: int, block: int, n_ops: int, align: int = 4096):
    max_off = file_size - block
    if max_off <= 0:
        raise ValueError("file too small")
    buf_size = align_up(block, align)
    buf = ctypes.create_string_buffer(buf_size)
    mv = memoryview(buf)[:block]

    # warmup
    for _ in range(min(200, n_ops // 10)):
        off = random.randrange(0, max_off // align) * align
        os.pread(fd, block, off)

    lat_us = []
    t0 = time.perf_counter()
    for _ in range(n_ops):
        off = random.randrange(0, max_off // align) * align
        t1 = time.perf_counter()
        os.pread(fd, block, off)
        lat_us.append((time.perf_counter() - t1) * 1e6)
    total_s = time.perf_counter() - t0
    return lat_us, total_s


def bench_sequential(fd: int, file_size: int, block: int, n_ops: int, align: int = 4096):
    buf_size = align_up(block, align)
    buf = ctypes.create_string_buffer(buf_size)
    stride = align  # consecutive aligned sectors
    max_start = file_size - block - stride * (n_ops - 1)
    if max_start <= 0:
        raise ValueError("file too small for sequential run")
    start = random.randrange(0, max(1, max_start // align)) * align

    lat_us = []
    t0 = time.perf_counter()
    off = start
    for _ in range(n_ops):
        t1 = time.perf_counter()
        os.pread(fd, block, off)
        lat_us.append((time.perf_counter() - t1) * 1e6)
        off += stride
    total_s = time.perf_counter() - t0
    return lat_us, total_s


def summarize(name: str, block: int, lat_us: list, total_s: float, n_ops: int):
    lat_us.sort()
    p50 = lat_us[len(lat_us) // 2]
    p99 = lat_us[int(len(lat_us) * 0.99)]
    mb_s = (n_ops * block / (1024 * 1024)) / total_s if total_s > 0 else 0
    print(
        f"{name:12s} block={block//1024:3d}KB  n={n_ops:5d}  "
        f"total={total_s:6.3f}s  throughput={mb_s:7.1f} MiB/s  "
        f"lat_us p50={p50:7.1f} p99={p99:7.1f} mean={statistics.mean(lat_us):7.1f}"
    )
    return {
        "name": name,
        "block": block,
        "n_ops": n_ops,
        "total_s": total_s,
        "mb_s": mb_s,
        "p50_us": p50,
        "p99_us": p99,
        "mean_us": statistics.mean(lat_us),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="path to disk index or test file")
    ap.add_argument("--ops", type=int, default=5000, help="reads per scenario")
    ap.add_argument("--blocks", default="4096,8192,16384")
    args = ap.parse_args()

    path = args.file
    blocks = [int(x) for x in args.blocks.split(",")]
    n_ops = args.ops
    file_size = os.path.getsize(path)

    print(f"file={path}")
    print(f"size={file_size / (1024**2):.1f} MiB  ops={n_ops}  O_DIRECT={'yes' if O_DIRECT else 'no'}")
    print()

    fd = open_direct(path)
    try:
        rows = []
        for block in blocks:
            print(f"--- block {block} bytes ---")
            lat_r, tot_r = bench_random(fd, file_size, block, n_ops)
            rows.append(summarize("random", block, lat_r, tot_r, n_ops))
            lat_s, tot_s = bench_sequential(fd, file_size, block, n_ops)
            rows.append(summarize("sequential", block, lat_s, tot_s, n_ops))
            ratio = rows[-2]["mean_us"] / rows[-1]["mean_us"] if rows[-1]["mean_us"] > 0 else 0
            print(f"             random/sequential mean latency ratio: {ratio:.2f}x")
            print()

        print("=== Cross-size random read (same op count) ===")
        for r in [x for x in rows if x["name"] == "random"]:
            equiv_4k_ops = r["n_ops"] * (r["block"] / 4096)
            bytes_total = r["n_ops"] * r["block"]
            print(
                f"  {r['block']//1024:3d}KB x {r['n_ops']} ops = {bytes_total/1024:.0f} KiB total, "
                f"4KB-equiv ops={equiv_4k_ops:.0f}, time={r['total_s']:.3f}s"
            )
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
