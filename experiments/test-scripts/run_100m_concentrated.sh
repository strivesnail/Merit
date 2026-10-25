#!/bin/bash
# Small-sigma concentrated hotspots on SIFT100M (k-means cell or kNN-of-pool region), L=128, single NVMe.
# GT is computed in batches of 3 files to stay within GPU memory.
export PATH=/home/jianz/miniconda3/bin:$PATH
set -uo pipefail
WL=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
TESTS=/home/jianz/workload/code/NeighborCache/experiments/test-scripts
OUT=/mnt/nvme_single/sift100m/runs
PRES=${PRES:-"cluster10000 knn200"}
SIGMAS=${SIGMAS:-"4 8"}
FRACS=${FRACS:-"0.1 0.2"}
WTAGS=${WTAGS:-"x2 x10 x50"}
SUF=${SUF:-}
cd "$WL"
for pre in $PRES; do
  for s in $SIGMAS; do
    files=()
    for t in $WTAGS; do
      [ -f "${pre}_${t}_block_s${s}${SUF}_gt10.bin" ] || files+=("${pre}_${t}_block_s${s}${SUF}.u8bin")
    done
    [ ${#files[@]} -gt 0 ] && python /home/jianz/workload/code/NeighborCache/experiments/scripts/gt_bruteforce_disk_index.py \
      --index /mnt/nvme_single/sift100m/sift100m_index_disk.index --npts 100000000 --queries "${files[@]}" >> gt_s.log 2>&1
  done
done
sync
for pre in $PRES; do
  for s in $SIGMAS; do
    WPRE=$pre WSUF=_block_s$s$SUF TAGS="$WTAGS" L=128 RECORD_PROFILE=1 DEL_PROFILE=1 MODES=same UNIQUE_FRACS="$FRACS" \
      INDEX=/mnt/nvme_single/sift100m/sift100m_index ROOT=$OUT/dcache_${pre}_s${s}${SUF}_L128${RTAG:-}_20261010 bash "$TESTS/run_100m_dcache_only.sh"
  done
done
echo CONCENTRATED DONE
