#!/bin/bash
# SIFT100M official 10K queries, same-trace profiling: Run1 records an access profile without caches,
# Run2 builds static N-cache (Top-N by node_expand) and D-cache (profile-edge packing) from it and
# replays the same queries. Baseline: DiskANN BFS cache with the same node count.
export PATH=/home/jianz/miniconda3/bin:$PATH
set -uo pipefail
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/raid5/sift100m/sift100m_index
QUERY=${QUERY:-/home/jianz/workload/data/sift100m/query.public.10K.u8bin}
GT=${GT:-/home/jianz/workload/data/sift100m/bigann-100M}
RUNS=${RUNS:-/mnt/raid5/sift100m/runs/official_profiled_$(date +%Y%m%d)}
PROFILE=$RUNS/profile
N=${N:-100000}
DRATIO=${DRATIO:-0.02}
LAYOUT=${LAYOUT:-directed_beam}
L=${L:-100}
W=${W:-4}
THREADS=${THREADS:-20}
STATUS=$RUNS/status.txt
mkdir -p "$RUNS"
log() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

run() { # $1=tag, rest = extra args
  local tag=$1; shift
  local out=$RUNS/$tag
  mkdir -p "$out"
  env LD_LIBRARY_PATH="$LIB" OMP_NUM_THREADS=$THREADS NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_NCACHE_CAP_NODES=$N \
    NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    "$SEARCH" --data_type uint8 --dist_fn l2 --index_path_prefix "$INDEX" --query_file "${RUNQ:-$QUERY}" \
    --gt_file "${RUNGT:-$GT}" --recall_at 10 --search_list "$L" --beamwidth "$W" --num_threads "$THREADS" \
    --result_path "$out/result" "$@" > "$out/run.out" 2>&1
  local row
  row=$(grep -E "^\s+$L\s+$W\s" "$out/run.out" | tail -1)
  log "$tag QPS=$(echo "$row" | awk '{print $3}') IOs=$(echo "$row" | awk '{print $6}') recall=$(echo "$row" | awk '{print $NF}')"
}

if [ ! -f "${PROFILE}_node_expand.bin" ]; then
  log "Run1 profile"
  # PROFILE_QUERY: record the profile on a different query file than the measured one.
  RUNQ=${PROFILE_QUERY:-$QUERY} RUNGT=${PROFILE_GT:-$GT} \
    run profile_run --num_nodes_to_cache 0 --enable_access_profile --access_profile_prefix "$PROFILE"
fi
run bfs_N --num_nodes_to_cache "$N"
run static_n --num_nodes_to_cache 0 --neighborcache_memory_gb 1 --neighborcache_profile_prefix "$PROFILE"
run static_nd --num_nodes_to_cache 0 --neighborcache_memory_gb 1 --neighborcache_profile_prefix "$PROFILE" \
  --neighborcache_disk_cache_ratio "$DRATIO" --neighborcache_disk_cache_layout "$LAYOUT" --neighborcache_disk_cache_exclude_memory true
log "PROFILED100M DONE"
