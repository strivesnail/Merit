#!/bin/bash
# SIFT1B perturbed Zipf workload: equal-memory DiskANN BFS, MERIT-N, and MERIT at one or more L.
# Usage: QUERY=... OUTROOT=... LS="50 100 200" SYSTEMS="bfs_eqmem ncache merit" run_1b_perturbed.sh
export PATH=/home/jianz/miniconda3/bin:$PATH
set -uo pipefail
SEARCH=${SEARCH:-/home/jianz/workload/code/Merit/diskann/build/apps/search_disk_index}
LIB=/tmp/merit-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=${INDEX:-/mnt/graid_single/sift1b/sift1b_index}
: "${QUERY:?}" "${OUTROOT:?}"
LS=${LS:-100}
SYSTEMS=${SYSTEMS:-"bfs_eqmem ncache merit"}
UNIQUE=${UNIQUE:-46929}
NCAP=${NCAP:-1173}
DPAGES=${DPAGES:-4693}
BFS_EQMEM=${BFS_EQMEM:-345505}
THREADS=${THREADS:-20}
K=${K:-10}
W=${W:-4}
mkdir -p "$OUTROOT"
PIN=(OMP_PROC_BIND=close "OMP_PLACES={0}:20" MERIT_BG_CPUS=20-23)
NC_ENV=(MERIT_NCACHE_SHARDS=128 MERIT_HEAP_SHARDS=4 MERIT_NCACHE_SPIN=0
  MERIT_NCACHE_ADMISSION=adaptive_hop_reject MERIT_NCACHE_HOP_THRESHOLD=12
  MERIT_NCACHE_ADAPT_HALF_LIFE_QUERIES=25000 MERIT_NCACHE_ADAPT_MIN_HOP=8
  MERIT_NCACHE_GHOST_PERCENT=10 MERIT_NCACHE_FAST_MISS=1 MERIT_NCACHE_CLOCK=0)

common() { # $1=L
  echo --data_type uint8 --dist_fn l2 --index_path_prefix "$INDEX" --query_file "$QUERY" \
    --gt_file null --recall_at "$K" --search_list "$1" --beamwidth "$W" --num_threads "$THREADS"
}

run_bfs() { # $1=out $2=L $3=nodes
  env LD_LIBRARY_PATH="$LIB" OMP_NUM_THREADS=$THREADS "${PIN[@]}" MERIT_USE_RAMFS=0 \
    "$SEARCH" $(common $2) --num_nodes_to_cache $3 \
    --result_path "$1/result" --dump_query_stats "$1/qstats.csv" >> "$1/run.out" 2>&1
}

run_ncache() { # $1=out $2=L
  env LD_LIBRARY_PATH="$LIB" OMP_NUM_THREADS=$THREADS "${PIN[@]}" \
    MERIT_USE_RAMFS=0 MERIT_DYNAMIC_3CACHE=1 MERIT_NCACHE_ONLY=1 \
    MERIT_CACHE_FROM_WORKLOAD=1 MERIT_WORKLOAD_UNIQUE_NODES=$UNIQUE \
    MERIT_NCACHE_WORKLOAD_FRAC=0.025 MERIT_NCACHE_CAP_NODES=$NCAP MERIT_DCACHE_WORKLOAD_FRAC=0 \
    MERIT_DCACHE_CAP=0 MERIT_MCACHE_CAP=0 \
    MERIT_DYNAMIC_PREFILL_STATIC=0 MERIT_REQUIRE_FULL_DCACHE_WARMUP=0 \
    MERIT_MEASURE_SKIP_QUERIES=${SKIP:-10000} "${NC_ENV[@]}" MERIT_DCACHE_WRITER_THREADS=0 \
    "$SEARCH" $(common $2) --num_nodes_to_cache ${MERIT_BFS:-0} \
    --merit_memory_gb 2 --merit_memory_runtime_admit true \
    --merit_disk_cache_ratio 0 --merit_disk_cache_layout bfs --merit_disk_cache_exclude_memory false \
    --result_path "$1/result" --dump_query_stats "$1/qstats.csv" >> "$1/run.out" 2>&1
}

run_merit() { # $1=out $2=L
  env LD_LIBRARY_PATH="$LIB" OMP_NUM_THREADS=$THREADS MERIT_SECTION_PROFILE=0 "${PIN[@]}" \
    MERIT_USE_RAMFS=0 MERIT_DYNAMIC_3CACHE=1 MERIT_NCACHE_ONLY=0 \
    MERIT_CACHE_FROM_WORKLOAD=1 MERIT_WORKLOAD_UNIQUE_NODES=$UNIQUE \
    MERIT_NCACHE_WORKLOAD_FRAC=0.025 MERIT_DCACHE_WORKLOAD_FRAC=1.0 \
    MERIT_NCACHE_CAP_NODES=$NCAP MERIT_DCACHE_CAP=$DPAGES MERIT_DCACHE_NODE_CAP=$((DPAGES * 10)) \
    MERIT_DYNAMIC_PREFILL_STATIC=0 MERIT_REQUIRE_FULL_DCACHE_WARMUP=0 \
    MERIT_MEASURE_SKIP_QUERIES=${SKIP:-10000} MERIT_RESOLVE_PARENT=0 \
    MERIT_REAL_IO_COACCESS=1 MERIT_REAL_IO_WINDOW=4 MERIT_REAL_IO_SAMPLE_QUERIES=64 \
    MERIT_SEED_T=2 MERIT_DCACHE_EVICT_NCACHE_ON_COMMIT=0 \
    MERIT_DEFERRED_NCACHE_PAGE_WRITE=${DEFERRED_WRITE:-1} MERIT_PENDING_REQUIRE_FULL_PAGE=0 \
    MERIT_PENDING_BUFFER_BYTES=4194304 "${NC_ENV[@]}" \
    MERIT_MCACHE_ADAPTIVE_UPDATE=0 MERIT_MCACHE_EXPAND_SAMPLE=1 \
    MERIT_MCACHE_MAX_EDGES=${MCACHE_MAX_EDGES:-64} \
    MERIT_DCACHE_NET_GATE=${NET_GATE:-0} MERIT_DCACHE_NET_WRITE_WEIGHT=${NET_WRITE_WEIGHT:-0} \
    MERIT_DCACHE_BASE_READ_US=${BASE_READ_US:-40} MERIT_DCACHE_READ_US=${DCACHE_READ_US:-50} \
    MERIT_DCACHE_WRITE_US=${DCACHE_WRITE_US:-56} MERIT_DCACHE_CPU_US=${DCACHE_CPU_US:-20} \
    MERIT_DCACHE_WRITER_THREADS=${WRITER_THREADS:-1} \
    MERIT_DCACHE_COMMIT_THREADS=${COMMIT_THREADS:-1} MERIT_DCACHE_COMMIT_QUEUE=${COMMIT_QUEUE:-256} \
    MERIT_DCACHE_QUERY_PATCH=1 MERIT_DCACHE_QUERY_PATCH_ASYNC=1 MERIT_DCACHE_QUERY_PATCH_MAX=4 \
    MERIT_DCACHE_QUERY_PATCH_Q=64 MERIT_DCACHE_QUERY_PATCH_BATCH_US=100 MERIT_DCACHE_QUERY_PATCH_NO_DISK=0 \
    MERIT_DCACHE_SECOND_PAGE=0 MERIT_DYN_MEMBIT=1 MERIT_DYN_LOC_SHARDS=${LOC_SHARDS:-64} \
    "$SEARCH" $(common $2) --num_nodes_to_cache ${MERIT_BFS:-0} \
    --merit_memory_gb 2 --merit_memory_runtime_admit true \
    --merit_disk_cache_ratio 0.01 --merit_disk_cache_layout bfs --merit_disk_cache_exclude_memory false \
    --enable_query_sector_cache true \
    --result_path "$1/result" --dump_query_stats "$1/qstats.csv" >> "$1/run.out" 2>&1
}

for L in $LS; do
  for s in $SYSTEMS; do
    out=$OUTROOT/${s}_L$L
    rm -rf "$out"; mkdir -p "$out"
    echo "==== START $s L=$L $(date -Is) ====" > "$out/run.out"
    case $s in
      bfs_eqmem) run_bfs "$out" $L $BFS_EQMEM ;;
      bfs0) run_bfs "$out" $L 0 ;;
      ncache) run_ncache "$out" $L ;;
      merit) run_merit "$out" $L ;;
    esac
    echo "==== END $s L=$L $(date -Is) ====" >> "$out/run.out"
    rm -f "$out"/*.dyn.data
    echo "$s L=$L QPS=$(grep -E "^\s+$L\s+$W\s" "$out/run.out" | awk '{print $3}')" | tee -a "$OUTROOT/summary.txt"
  done
done
echo ALLDONE | tee -a "$OUTROOT/summary.txt"
