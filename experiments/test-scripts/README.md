# Benchmark and test scripts

This directory contains dataset preparation, index builds, baseline searches and I/O diagnostics. For example, `build_sift100m_index_and_profile.sh` builds and profiles a SIFT100M index, and `run_100m_x1_diskann_bfs_sweep.sh` runs the DiskANN BFS baseline.

Use the [path cache guide](../../diskann/README-path-seed.md) for the current NeighborCache. Before a disk measurement, consult the [experiment ledger](../neighborcache_alignment/EXPERIMENTS.md) and record a successful TRIM for every participating SSD.

The [older benchmark scripts](https://github.com/strivesnail/Merit/tree/4efb25d/experiments/test-scripts) accompany measurements made with earlier cache implementations.
