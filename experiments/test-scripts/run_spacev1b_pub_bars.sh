#!/usr/bin/env bash
# SPACEV1B: same 4-bar QPS setup as SIFT100M pub10k (DiskANN / BFS400 / LRU only / LRU+LRP).
# Requires DiskANN index on NVMe and workloads under DATA/workloads (see prep_spacev1b_workloads.py).
# Usage:
#   WNAME=log_test INDEX=/mnt/nvme_single/spacev1b/spacev1b_index \
#   bash run_spacev1b_pub_bars.sh
set -uo pipefail
SEARCH=${SEARCH_BIN:-/tmp/search_disk_index_zerocopy}
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=${INDEX:-/mnt/nvme_single/spacev1b/spacev1b_index}
DATA=${DATA:-/home/jianz/workload/data/spacev1b}
WLD=${WLD:-$DATA/workloads}
W=${WNAME:-log_test}
q=${QFILE:-$WLD/${W}.i8bin}
gt=${GTFILE:-$WLD/${W}_gt10.bin}
RDATE=${RDATE:-$(date +%Y%m%d)}
RUNS=${RUNS:-/mnt/nvme_single/spacev1b/runs}
R=$RUNS/dcache_${W}_seed50_nc4_${RDATE}
P=${PROFILE_DIR:-/mnt/hdd/merit_park/tmp_profiles/spacev1b_${W}_${RDATE}}
REPS=${REPS:-3}
MBS=${MBS:-400}
SF=${SEEDFRAC:-0.5}
mkdir -p "$R" "$P"
log() { echo "$(date -Is) $*" | tee -a "$R/status.txt"; }

run() {
  local out=$1 nc=$2; shift 2; mkdir -p "$out"
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=$SF NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=60000 \
    NEIGHBORCACHE_NCACHE_CAP_NODES=$nc NEIGHBORCACHE_NCACHE_COLD=1 \
    $SEARCH --data_type int8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt \
    --recall_at 10 --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result "$@" \
    > $out/run.out 2>&1
}
row() {
  local f=$1/run.out r=$(grep -E '^\s+128\s+4\s' $1/run.out | tail -1)
  local br=$(grep -o 'base_reads=.*' $f | tail -1)
  [ -z "$br" ] && br="base_reads=$(echo $r | awk '{print $6}') dcache_reads=0"
  echo "QPS=$(echo $r | awk '{print $3}') recall=$(echo $r | awk '{print $NF}') $br"
}

[ -f "$q" ] && [ -f "$gt" ] || { log "missing $q or $gt"; exit 1; }
[ -f "${INDEX}_disk.index" ] || { log "missing index ${INDEX}_disk.index"; exit 1; }

DC="--neighborcache_profile_prefix $P/prof --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
NCA="--neighborcache_memory_gb 4 --neighborcache_memory_runtime_admit true"

[ -f $P/prof_node_visit.bin ] || run $R/profile_run 0 --num_nodes_to_cache 0 --enable_access_profile --access_profile_prefix $P/prof
b=$R/s0.5_build
run $b 0 $DC --num_nodes_to_cache 0
sync; sleep 3
for rep in $(seq 1 $REPS); do
  run $R/diskann0_r$rep 0 --num_nodes_to_cache 0
  log "[$W] kind=diskann ncMB=0 nodes=0 rep=$rep $(row $R/diskann0_r$rep)"
done
for mb in $MBS; do
  # SPACEV node ~ (100+4+64*4)=360B packed roughly; use 388 like SIFT for comparable DRAM MB
  nc=$(( mb * 1048576 / 388 ))
  for rep in $(seq 1 $REPS); do
    run $R/diskann${mb}_r$rep 0 --num_nodes_to_cache $nc
    log "[$W] kind=diskann ncMB=$mb nodes=$nc rep=$rep $(row $R/diskann${mb}_r$rep)"
    run $R/lruonlycold${mb}_r$rep $nc --neighborcache_profile_prefix $P/prof --num_nodes_to_cache 0 $NCA
    log "[$W] kind=lruonlycold ncMB=$mb nodes=$nc rep=$rep $(row $R/lruonlycold${mb}_r$rep)"
    run $R/lrucold${mb}_r$rep $nc $DC --neighborcache_disk_cache_reuse_prefix $b/result --num_nodes_to_cache 0 $NCA \
      --neighborcache_disk_cache_exclude_memory false
    log "[$W] kind=lrucold ncMB=$mb nodes=$nc rep=$rep $(row $R/lrucold${mb}_r$rep)"
  done
done
rm -f $b/*_neighborcache_dc.data $b/*_neighborcache_dc.nodes $b/*_neighborcache_dc.seed_*
log SPACEV_PUB_BARS_DONE
