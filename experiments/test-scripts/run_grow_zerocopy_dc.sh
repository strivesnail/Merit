#!/bin/bash
# Growing-hotspot sweep, D-cache lines only (none / lrucold / bfs at 400MB), zero-copy shared page cache build.
cd /home/jianz/workload/code/NeighborCache/experiments/test-scripts
for r in ${LEVELS:-1 4 8 12 16 20 24 28 32 36 40}; do
  TAG=k3000h5kg$r WNAME=knn3000_h5000g${r}_block_s4_pct MBS=400 REPS=${REPS:-1} DC_ONLY=1 RDATE=${RDATE:-zc20261014} \
    bash run_100m_seed50_ncache_4lines.sh
done
echo GROW_ZC_DONE
