#!/bin/bash
# knn2000 0% hotspot: plain DiskANN (no D-cache), BFS node cache size sweep (MB / 388B per node).
set -u
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/nvme_single/sift100m/sift100m_index
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
R=/mnt/nvme_single/sift100m/runs/dcache_knn2000_x1_seed50_nc_vs_bfs_20261012
REPS=${REPS:-3}
mkdir -p $R
log() { echo "$(date -Is) $*" | tee -a $R/status.txt; }
q=$WLD/knn2000_x1_block_s4_pct.u8bin; gt=$WLD/knn2000_x1_block_s4_pct_gt10.bin
for mb in ${MBS:-0 10 50 100 200 400 500 600 800 1600}; do
  nc=$(( mb * 1048576 / 388 ))
  for rep in $(seq 1 $REPS); do
    out=$R/diskann${mb}_r$rep; mkdir -p $out
    env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 \
      $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
      --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result --num_nodes_to_cache $nc > $out/run.out 2>&1
    r=$(grep -E '^\s+128\s+4\s' $out/run.out | tail -1)
    log "[x1] kind=diskann ncMB=$mb nodes=$nc rep=$rep QPS=$(echo $r | awk '{print $3}') recall=$(echo $r | awk '{print $NF}') base_reads=$(echo $r | awk '{print $6}') dcache_reads=0"
  done
done
log X1_DISKANN_BFS_SWEEP_DONE
