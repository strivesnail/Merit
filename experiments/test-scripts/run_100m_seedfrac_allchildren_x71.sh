#!/bin/bash
# x71 all-children seed-frac sweep, shared cache 60000 pages only; reuses the existing profile.
set -u
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/nvme_single/sift100m/sift100m_index
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
R=/mnt/nvme_single/sift100m/runs/dcache_knn2000_seedfrac_allch_x71_20261012
P=/mnt/hdd/merit_park/tmp_profiles/knn2000_seedfrac_allch_x71_20261012
log() { echo "$(date -Is) $*" | tee -a $R/status.txt; }
run() { local out=$1 q=$2 gt=$3; shift 3; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=60000 \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result --num_nodes_to_cache 0 "$@" > $out/run.out 2>&1; }
row() { local f=$1/run.out r=$(grep -E '^\s+128\s+4\s' $1/run.out | tail -1)
  echo "QPS=$(echo $r | awk '{print $3}') recall=$(echo $r | awk '{print $NF}') $(grep -o 'base_reads=.*' $f | tail -1)"; }
t=x71
q=$WLD/knn2000_${t}_block_s4_pct.u8bin; gt=$WLD/knn2000_${t}_block_s4_pct_gt10.bin
for fr in ${FRACS:-0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}; do
  DC="--neighborcache_profile_prefix $P/$t --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
  b=$R/$t/s${fr}_build
  NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=$fr run $b $q $gt $DC
  sync; sleep 3
  NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=$fr run $R/$t/s${fr}_c60000 $q $gt $DC --neighborcache_disk_cache_reuse_prefix $b/result
  log "[$t] seedfrac=$fr cap=60000 $(grep -o 'packed file sectors=[0-9]*' $b/run.out) $(row $R/$t/s${fr}_c60000)"
  rm -f $b/*_neighborcache_dc.data $b/*_neighborcache_dc.nodes $b/*_neighborcache_dc.seed_*
done
log SEEDFRAC_ALLCH_X71_240_DONE
