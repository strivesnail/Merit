#!/bin/bash
# knn2000 hotspot 0/30/50/70%: all children per seed (unlimited pages), shared cache 60000 pages (240MB),
# plus a profile Top-K N-cache of NC nodes (40MB / 388B). Baseline: DiskANN with a 240MB BFS node cache.
set -u
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/nvme_single/sift100m/sift100m_index
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
R=/mnt/nvme_single/sift100m/runs/dcache_knn2000_seedfrac_allch240_nc40m_20261012
P=/mnt/hdd/merit_park/tmp_profiles/knn2000_seedfrac_allch240_20261012
NC=${NC:-108100}
BFS=${BFS:-633402}
mkdir -p $R $P
log() { echo "$(date -Is) $*" | tee -a $R/status.txt; }
base() { local out=$1 q=$2 gt=$3; shift 3; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result "$@" > $out/run.out 2>&1; }
run() { local out=$1 q=$2 gt=$3; shift 3; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=60000 NEIGHBORCACHE_NCACHE_CAP_NODES=$NC NEIGHBORCACHE_MISS_DIAG=1 \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result --num_nodes_to_cache 0 "$@" > $out/run.out 2>&1; }
row() { local f=$1/run.out r=$(grep -E '^\s+128\s+4\s' $1/run.out | tail -1)
  echo "QPS=$(echo $r | awk '{print $3}') IOs=$(echo $r | awk '{print $6}') recall=$(echo $r | awk '{print $NF}') $(grep -o 'base_reads=.*' $f | tail -1)"; }
for t in ${TAGS:-x71 x51 x31 x1}; do
  q=$WLD/knn2000_${t}_block_s4_pct.u8bin; gt=$WLD/knn2000_${t}_block_s4_pct_gt10.bin
  [ -f $P/${t}_node_visit.bin ] || run $R/$t/profile_run $q $gt --enable_access_profile --access_profile_prefix $P/$t
  for fr in ${FRACS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}; do
    DC="--neighborcache_profile_prefix $P/$t --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
    b=$R/$t/s${fr}_build
    NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=$fr run $b $q $gt $DC
    sync; sleep 3
    NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=$fr run $R/$t/s$fr $q $gt $DC --neighborcache_disk_cache_reuse_prefix $b/result \
      --neighborcache_memory_gb 1 --neighborcache_disk_cache_exclude_memory false
    log "[$t] seedfrac=$fr cap=60000 nc=$NC $(grep -o 'packed file sectors=[0-9]*' $b/run.out) $(row $R/$t/s$fr) $(grep -o 'not_in_dcache=.*' $R/$t/s$fr/run.out | tail -1)"
    rm -f $b/*_neighborcache_dc.data $b/*_neighborcache_dc.nodes $b/*_neighborcache_dc.seed_*
  done
done
log SEEDFRAC_ALLCH240_NC40M_DONE
