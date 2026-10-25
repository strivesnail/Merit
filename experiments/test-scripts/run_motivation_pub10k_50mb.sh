#!/bin/bash
# Motivation fig bars at 50MB DRAM (paper-honest):
#   DiskANN (0), BFS (50MB), N-cache only (50MB cold), N+D (40MB N + 10MB shared prefetch).
# Full-neighborhood D-cache ceiling (seedfrac=0.5, all children), same as prior pub10k bars.
# Usage:
#   TAG=pub10k QFILE=... GTFILE=... bash run_motivation_pub10k_50mb.sh
set -uo pipefail
SEARCH=${SEARCH_BIN:-/tmp/search_disk_index_zerocopy}
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=${INDEX:-/mnt/nvme_single/sift100m/sift100m_index}
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
T=${TAG:?set TAG=pub10k|pub10kz08|pub10kx5}
SF=${SEEDFRAC:-0.5}
REPS=${REPS:-3}
NODE_B=${NODE_B:-388}
PAGE_B=4096
DRAM_MB=50
N_MB=40
SHARED_MB=10
BFS_NC=$(( DRAM_MB * 1048576 / NODE_B ))
NONLY_NC=$(( DRAM_MB * 1048576 / NODE_B ))
ND_NC=$(( N_MB * 1048576 / NODE_B ))
SHARED_PAGES=$(( SHARED_MB * 1048576 / PAGE_B ))
MEM_GB_NONLY=$(python3 -c "print($DRAM_MB/1024.0)")
MEM_GB_ND=$(python3 -c "print($N_MB/1024.0)")

R=${OUTROOT:-/mnt/raid5/fig_motivation_50mb}/dcache_${T}_seed50_50mb_$(date +%Y%m%d)
P=${PROFILE_DIR:-/mnt/hdd/merit_park/tmp_profiles/knn2000_seedfrac_allch240_20261012}
case "$T" in
  pub10k)
    q=${QFILE:-/home/jianz/workload/data/sift100m/query.public.10K.u8bin}
    gt=${GTFILE:-/home/jianz/workload/data/sift100m/bigann-100M}
    ;;
  pub10kz08)
    q=${QFILE:-$WLD/pub10k_zipf0p8_10k_shuffled.u8bin}
    gt=${GTFILE:-$WLD/pub10k_zipf0p8_10k_shuffled_gt10.bin}
    ;;
  pub10kx5)
    q=${QFILE:-$WLD/pub10kx5_shuffled.u8bin}
    gt=${GTFILE:-$WLD/pub10kx5_shuffled_gt10.bin}
    ;;
  *) echo "unknown TAG=$T"; exit 1 ;;
esac

mkdir -p "$R"
log() { echo "$(date -Is) $*" | tee -a "$R/status.txt"; }
log "START TAG=$T DRAM=${DRAM_MB}MB N+D=${N_MB}+${SHARED_MB} BFS_NC=$BFS_NC NONLY_NC=$NONLY_NC ND_NC=$ND_NC SHARED_PAGES=$SHARED_PAGES q=$q"

run() {
  local out=$1 nc=$2 shared=$3; shift 3
  mkdir -p "$out"
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=$SF NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 \
    NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=$shared \
    NEIGHBORCACHE_NCACHE_CAP_NODES=$nc NEIGHBORCACHE_NCACHE_COLD=1 \
    "$SEARCH" --data_type uint8 --dist_fn l2 --index_path_prefix "$INDEX" \
    --query_file "$q" --gt_file "$gt" --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path "$out/result" \
    "$@" > "$out/run.out" 2>&1
}

row() {
  local f=$1/run.out r
  r=$(grep -E '^\s+128\s+4\s' "$f" | tail -1)
  local br
  br=$(grep -o 'base_reads=.*' "$f" | tail -1)
  [ -z "$br" ] && br="base_reads=$(echo "$r" | awk '{print $6}') dcache_reads=0"
  echo "QPS=$(echo "$r" | awk '{print $3}') recall=$(echo "$r" | awk '{print $NF}') $br"
}

[[ -f "$q" && -f "$gt" ]] || { log "missing q/gt"; exit 1; }
[[ -f "${INDEX}_disk.index" ]] || { log "missing index"; exit 1; }
[[ -x "$SEARCH" || -f "$SEARCH" ]] || { log "missing SEARCH=$SEARCH"; exit 1; }
[[ -f "$P/${T}_node_visit.bin" ]] || { log "missing profile $P/${T}_node_visit.bin"; exit 1; }

DC="--neighborcache_profile_prefix $P/$T --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
b=$R/s0.5_build
log "build full-neighborhood D-cache"
run "$b" 0 0 $DC --num_nodes_to_cache 0
sync; sleep 2

for rep in $(seq 1 "$REPS"); do
  run "$R/diskann0_r$rep" 0 0 --num_nodes_to_cache 0
  log "[$T] kind=diskann ncMB=0 nodes=0 rep=$rep $(row "$R/diskann0_r$rep")"
done

for rep in $(seq 1 "$REPS"); do
  # BFS: all 50MB in DiskANN BFS node cache
  run "$R/diskann${DRAM_MB}_r$rep" 0 0 --num_nodes_to_cache $BFS_NC
  log "[$T] kind=diskann ncMB=$DRAM_MB nodes=$BFS_NC rep=$rep $(row "$R/diskann${DRAM_MB}_r$rep")"

  # N-cache only: all 50MB, cold
  run "$R/lruonlycold${DRAM_MB}_r$rep" $NONLY_NC 0 \
    --neighborcache_profile_prefix "$P/$T" --num_nodes_to_cache 0 \
    --neighborcache_memory_gb "$MEM_GB_NONLY" --neighborcache_memory_runtime_admit true
  log "[$T] kind=lruonlycold ncMB=$DRAM_MB nodes=$NONLY_NC rep=$rep $(row "$R/lruonlycold${DRAM_MB}_r$rep")"

  # N+D: 40MB N-cache + 10MB shared prefetch; D-cache on SSD (full-neighborhood ceiling)
  run "$R/lrucold${DRAM_MB}_r$rep" $ND_NC $SHARED_PAGES \
    $DC --neighborcache_disk_cache_reuse_prefix "$b/result" --num_nodes_to_cache 0 \
    --neighborcache_memory_gb "$MEM_GB_ND" --neighborcache_memory_runtime_admit true \
    --neighborcache_disk_cache_exclude_memory false
  log "[$T] kind=lrucold ncMB=$DRAM_MB nodes=$ND_NC shared_pages=$SHARED_PAGES rep=$rep $(row "$R/lrucold${DRAM_MB}_r$rep")"
done

rm -f "$b"/*_neighborcache_dc.data "$b"/*_neighborcache_dc.nodes "$b"/*_neighborcache_dc.seed_* 2>/dev/null || true
log "DONE $T"
echo ALLDONE | tee -a "$R/status.txt"
