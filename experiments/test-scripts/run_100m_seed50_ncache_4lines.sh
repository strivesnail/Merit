#!/bin/bash
# knn2000 hotspot TAG (default x31), seed SEEDFRAC (default 50%), all children, 240MB shared cache. N-cache size sweep, 4 lines:
#   lrucold: D-cache + cold-start LRU N-cache    bfs: D-cache + DiskANN BFS cache
#   lruonlycold: cold-start LRU N-cache only     diskann: DiskANN + BFS cache (no D-cache)
set -u
SEARCH=${SEARCH_BIN:-/tmp/search_disk_index_zerocopy}
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=${INDEX:-/mnt/nvme_single/sift100m/sift100m_index}
RUNS=${RUNS:-/mnt/nvme_single/sift100m/runs}
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
T=${TAG:-x31}
SF=${SEEDFRAC:-0.5}
DC_ONLY=${DC_ONLY:-0}
if [ "$SF" = "0.5" ]; then
  R=$RUNS/dcache_knn2000_${T}_seed50_nc4_${RDATE:-20261012}
else
  R=$RUNS/dcache_knn2000_${T}_seed${SF}_nc4_20261013
fi
P=${PROFILE_DIR:-/mnt/hdd/merit_park/tmp_profiles/knn2000_seedfrac_allch240_20261012}
REPS=${REPS:-3}
mkdir -p $R
log() { echo "$(date -Is) $*" | tee -a $R/status.txt; }
run() { local out=$1 nc=$2; shift 2; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=$SF NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=60000 \
    NEIGHBORCACHE_NCACHE_CAP_NODES=$nc NEIGHBORCACHE_NCACHE_COLD=1 \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result "$@" > $out/run.out 2>&1; }
row() { local f=$1/run.out r=$(grep -E '^\s+128\s+4\s' $1/run.out | tail -1)
  local br=$(grep -o 'base_reads=.*' $f | tail -1)
  [ -z "$br" ] && br="base_reads=$(echo $r | awk '{print $6}') dcache_reads=0"
  echo "QPS=$(echo $r | awk '{print $3}') recall=$(echo $r | awk '{print $NF}') $br"; }
W=${WNAME:-knn2000_${T}_block_s4_pct}
q=${QFILE:-$WLD/$W.u8bin}; gt=${GTFILE:-$WLD/${W}_gt10.bin}
DC="--neighborcache_profile_prefix $P/$T --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
NCA="--neighborcache_memory_gb 4 --neighborcache_memory_runtime_admit true"
[ -f $P/${T}_node_visit.bin ] || run $R/profile_run 0 --num_nodes_to_cache 0 --enable_access_profile --access_profile_prefix $P/$T
b=$R/s0.5_build
run $b 0 $DC --num_nodes_to_cache 0
sync; sleep 3
for rep in $(seq 1 $REPS); do
  run $R/none_r$rep 0 $DC --neighborcache_disk_cache_reuse_prefix $b/result --num_nodes_to_cache 0
  log "[$T] kind=none ncMB=0 nodes=0 rep=$rep $(row $R/none_r$rep)"
  [ "$DC_ONLY" = 1 ] && continue
  run $R/diskann0_r$rep 0 --num_nodes_to_cache 0
  log "[$T] kind=diskann ncMB=0 nodes=0 rep=$rep $(row $R/diskann0_r$rep)"
done
for mb in ${MBS:-10 50 100 200 400 500 600 800 1600}; do
  nc=$(( mb * 1048576 / 388 ))
  for rep in $(seq 1 $REPS); do
    run $R/lrucold${mb}_r$rep $nc $DC --neighborcache_disk_cache_reuse_prefix $b/result --num_nodes_to_cache 0 $NCA \
      --neighborcache_disk_cache_exclude_memory false
    log "[$T] kind=lrucold ncMB=$mb nodes=$nc rep=$rep $(row $R/lrucold${mb}_r$rep)"
    run $R/bfs${mb}_r$rep 0 $DC --neighborcache_disk_cache_reuse_prefix $b/result --num_nodes_to_cache $nc
    log "[$T] kind=bfs ncMB=$mb nodes=$nc rep=$rep $(row $R/bfs${mb}_r$rep)"
    [ "$DC_ONLY" = 1 ] && continue
    run $R/lruonlycold${mb}_r$rep $nc --neighborcache_profile_prefix $P/$T --num_nodes_to_cache 0 $NCA
    log "[$T] kind=lruonlycold ncMB=$mb nodes=$nc rep=$rep $(row $R/lruonlycold${mb}_r$rep)"
    run $R/diskann${mb}_r$rep 0 --num_nodes_to_cache $nc
    log "[$T] kind=diskann ncMB=$mb nodes=$nc rep=$rep $(row $R/diskann${mb}_r$rep)"
  done
done
rm -f $b/*_neighborcache_dc.data $b/*_neighborcache_dc.nodes $b/*_neighborcache_dc.seed_*
log SEED50_NC4_DONE
