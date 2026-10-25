#!/bin/bash
# knn2000 0% hotspot: cold-start LRU N-cache (empty at start), size sweep.
# kind=lrucold: D-cache (seed 50%, all children, 240MB shared cache) + LRU N-cache.
# kind=lruonlycold: LRU N-cache only, no D-cache.
set -u
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/nvme_single/sift100m/sift100m_index
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
R=/mnt/nvme_single/sift100m/runs/dcache_knn2000_x1_seed50_nc_vs_bfs_20261012
P=/mnt/hdd/merit_park/tmp_profiles/knn2000_seedfrac_allch240_20261012
REPS=${REPS:-3}
mkdir -p $R
log() { echo "$(date -Is) $*" | tee -a $R/status.txt; }
run() { local out=$1 nc=$2; shift 2; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=0.5 NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=60000 \
    NEIGHBORCACHE_NCACHE_CAP_NODES=$nc NEIGHBORCACHE_NCACHE_COLD=1 \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result --num_nodes_to_cache 0 "$@" > $out/run.out 2>&1; }
row() { local f=$1/run.out r=$(grep -E '^\s+128\s+4\s' $1/run.out | tail -1)
  echo "QPS=$(echo $r | awk '{print $3}') recall=$(echo $r | awk '{print $NF}') $(grep -o 'base_reads=.*' $f | tail -1)"; }
q=$WLD/knn2000_x1_block_s4_pct.u8bin; gt=$WLD/knn2000_x1_block_s4_pct_gt10.bin
DC="--neighborcache_profile_prefix $P/x1 --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
NCA="--neighborcache_memory_gb 4 --neighborcache_memory_runtime_admit true"
b=$R/s0.5_cold_build
run $b 0 $DC
sync; sleep 3
for mb in ${MBS:-10 50 100 200 400 500 600 800 1600}; do
  nc=$(( mb * 1048576 / 388 ))
  for rep in $(seq 1 $REPS); do
    run $R/lrucold${mb}_r$rep $nc $DC --neighborcache_disk_cache_reuse_prefix $b/result $NCA --neighborcache_disk_cache_exclude_memory false
    log "[x1] seedfrac=0.5 kind=lrucold ncMB=$mb nodes=$nc rep=$rep $(row $R/lrucold${mb}_r$rep)"
    run $R/lruonlycold${mb}_r$rep $nc --neighborcache_profile_prefix $P/x1 $NCA
    log "[x1] kind=lruonlycold ncMB=$mb nodes=$nc rep=$rep $(row $R/lruonlycold${mb}_r$rep)"
  done
done
rm -f $b/*_neighborcache_dc.data $b/*_neighborcache_dc.nodes $b/*_neighborcache_dc.seed_*
log X1_SEED50_LRU_COLD_DONE
