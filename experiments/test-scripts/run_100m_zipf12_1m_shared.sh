#!/bin/bash
# SIFT100M stable Zipf a=1.2 1M (old graid N2.5% setup: L=100 W=4 t20, N=1162 nodes, BFS=4134 nodes),
# rerun on nvme_single with the current seed-replica D-cache + shared page cache.
set -u
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/nvme_single/sift100m/sift100m_index
W=/home/jianz/workload/data/sift100m/workloads/zipf_a1p2_1m_stable
q=$W.u8bin; gt=${W}_gt.bin
R=/mnt/nvme_single/sift100m/runs/zipf_a1p2_1m_shared_20261014
P=/mnt/hdd/merit_park/tmp_profiles/zipf_a1p2_1m_stable_20261014
T=a1p2
SFS=${SFS:-0.5 1.0}
NC=${NC:-1162}
BFS=${BFS:-4134}
mkdir -p $R $P
log() { echo "$(date -Is) $*" | tee -a $R/status.txt; }
run() { local out=$1 nc=$2 sf=$3; shift 3; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=$sf NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=60000 \
    NEIGHBORCACHE_NCACHE_CAP_NODES=$nc NEIGHBORCACHE_NCACHE_COLD=1 \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 100 --beamwidth 4 --num_threads 20 --result_path $out/result "$@" > $out/run.out 2>&1; }
row() { local f=$1/run.out r=$(grep -E '^\s+100\s+4\s' $1/run.out | tail -1)
  local br=$(grep -o 'base_reads=.*' $f | tail -1)
  [ -z "$br" ] && br="base_reads=$(echo $r | awk '{print $6}') dcache_reads=0"
  echo "QPS=$(echo $r | awk '{print $3}') recall=$(echo $r | awk '{print $NF}') $br"; }
NCA="--neighborcache_memory_gb 4 --neighborcache_memory_runtime_admit true"

[ -f $P/${T}_node_visit.bin ] || { run $R/profile_run 0 0.5 --num_nodes_to_cache 0 --enable_access_profile --access_profile_prefix $P/$T; log "profile done"; }
if [ "${SKIP_BASE:-0}" != 1 ]; then
  run $R/diskann0 0 0.5 --num_nodes_to_cache 0
  log "kind=diskann nodes=0 $(row $R/diskann0)"
  run $R/bfs$BFS 0 0.5 --num_nodes_to_cache $BFS
  log "kind=diskann_bfs nodes=$BFS $(row $R/bfs$BFS)"
  run $R/lruonly$NC $NC 0.5 --neighborcache_profile_prefix $P/$T --num_nodes_to_cache 0 $NCA
  log "kind=lru_only nodes=$NC $(row $R/lruonly$NC)"
fi
for sf in $SFS; do
  DC="--neighborcache_profile_prefix $P/$T --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
  b=$R/build_sf$sf
  run $b 0 $sf $DC --num_nodes_to_cache 0
  log "build sf=$sf $(grep -o 'packed file sectors=[0-9]* bytes=[0-9]*' $b/run.out)"
  sync; sleep 3
  run $R/dc_sf$sf 0 $sf $DC --neighborcache_disk_cache_reuse_prefix $b/result --num_nodes_to_cache 0
  log "kind=dcache sf=$sf nodes=0 $(row $R/dc_sf$sf)"
  run $R/dc_lru_sf$sf $NC $sf $DC --neighborcache_disk_cache_reuse_prefix $b/result --num_nodes_to_cache 0 $NCA \
    --neighborcache_disk_cache_exclude_memory false
  log "kind=dcache_lru sf=$sf nodes=$NC $(row $R/dc_lru_sf$sf)"
  run $R/dc_bfs_sf$sf 0 $sf $DC --neighborcache_disk_cache_reuse_prefix $b/result --num_nodes_to_cache $BFS
  log "kind=dcache_bfs sf=$sf nodes=$BFS $(row $R/dc_bfs_sf$sf)"
  rm -f $b/*_neighborcache_dc.data $b/*_neighborcache_dc.nodes $b/*_neighborcache_dc.seed_* $R/*/result_100_*.bin
done
log ZIPF12_DONE
