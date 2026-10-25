#!/bin/bash
# A/B: shared page cache mutex (old binary) vs optimistic lock-free reads (new binary), same D-cache.
set -u
NEW=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
OLD=${OLD:-/tmp/search_disk_index_mutex}
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/nvme_single/sift100m/sift100m_index
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
T=${TAG:-k3000h5kg28}
W=${WNAME:-knn3000_h5000g28_block_s4_pct}
REPS=${REPS:-3}
R=/mnt/nvme_single/sift100m/runs/ab_shared_lock_${T}_${RUN_SUFFIX:-20261014}
P=/mnt/hdd/merit_park/tmp_profiles/knn2000_seedfrac_allch240_20261012
mkdir -p $R
log() { echo "$(date -Is) $*" | tee -a $R/status.txt; }
run() { local bin=$1 out=$2 nc=$3; shift 3; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=0.5 NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=60000 \
    NEIGHBORCACHE_NCACHE_CAP_NODES=$nc NEIGHBORCACHE_NCACHE_COLD=1 \
    $bin --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result "$@" > $out/run.out 2>&1; }
row() { local f=$1/run.out r=$(grep -E '^\s+128\s+4\s' $1/run.out | tail -1)
  echo "QPS=$(echo $r | awk '{print $3}') recall=$(echo $r | awk '{print $NF}') $(grep -o 'base_reads=.*' $f | tail -1)"; }
q=$WLD/$W.u8bin; gt=$WLD/${W}_gt10.bin
DC="--neighborcache_profile_prefix $P/$T --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
NCA="--neighborcache_memory_gb 4 --neighborcache_memory_runtime_admit true"
nc=$(( 400 * 1048576 / 388 ))
b=${BUILD_DIR:-$R/s0.5_build}
if ! ls $b/*_neighborcache_dc.data > /dev/null 2>&1; then
  run $NEW $b 0 $DC --num_nodes_to_cache 0
  sync; sleep 3
fi
for rep in $(seq 1 $REPS); do
  for v in old new; do
    bin=$NEW; [ $v = old ] && bin=$OLD
    run $bin $R/${v}_none_r$rep 0 $DC --neighborcache_disk_cache_reuse_prefix $b/result --num_nodes_to_cache 0
    log "[$T] lock=$v kind=none rep=$rep $(row $R/${v}_none_r$rep)"
    run $bin $R/${v}_bfs_r$rep 0 $DC --neighborcache_disk_cache_reuse_prefix $b/result --num_nodes_to_cache $nc
    log "[$T] lock=$v kind=bfs rep=$rep $(row $R/${v}_bfs_r$rep)"
    run $bin $R/${v}_lrucold_r$rep $nc $DC --neighborcache_disk_cache_reuse_prefix $b/result --num_nodes_to_cache 0 $NCA \
      --neighborcache_disk_cache_exclude_memory false
    log "[$T] lock=$v kind=lrucold rep=$rep $(row $R/${v}_lrucold_r$rep)"
  done
  if [ "${LRUONLY:-0}" = 1 ]; then
    run $NEW $R/lruonlycold_r$rep $nc --neighborcache_profile_prefix $P/$T --num_nodes_to_cache 0 $NCA
    log "[$T] lock=- kind=lruonlycold rep=$rep $(row $R/lruonlycold_r$rep)"
  fi
done
rm -f $b/*_neighborcache_dc.data $b/*_neighborcache_dc.nodes $b/*_neighborcache_dc.seed_*
log AB_DONE
