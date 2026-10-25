#!/bin/bash
# Pure D-cache (seed 50%, all children, 240MB shared cache, no N-cache): per-query disk reads per workload.
set -u
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/nvme_single/sift100m/sift100m_index
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
R=/mnt/nvme_single/sift100m/runs/dcache_knn2000_dcache_split_reads_20261013
P=/mnt/hdd/merit_park/tmp_profiles/knn2000_seedfrac_allch240_20261012
mkdir -p $R
run() { local out=$1; shift; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=0.5 NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=60000 \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result --num_nodes_to_cache 0 \
    --dump_query_stats $out/qstats.csv "$@" > $out/run.out 2>&1; }
for T in ${TAGS:-x1 x71 x71q0 x71q0.5 x71q0.8 x71q1 x71q1.2 x71q1.5}; do
  q=$WLD/knn2000_${T}_block_s4_pct.u8bin; gt=$WLD/knn2000_${T}_block_s4_pct_gt10.bin
  DC="--neighborcache_profile_prefix $P/$T --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
  b=$R/$T/build
  run $b $DC
  sync; sleep 3
  run $R/$T/dcache $DC --neighborcache_disk_cache_reuse_prefix $b/result
  rm -f $b/*_neighborcache_dc.data $b/*_neighborcache_dc.nodes $b/*_neighborcache_dc.seed_* $R/$T/*/result_*.bin
  echo "$T done" >> $R/status.txt
done
echo DONE >> $R/status.txt
