#!/usr/bin/env bash
# Five-way MERIT compare (configurable dataset / memory / disk layout).
#   baseline | BFS static cache | MERIT mem | MERIT disk | MERIT mem+disk (disjoint)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
INDEX_PREFIX="${INDEX_PREFIX:-${DATA_DIR}/sift1m_index}"
QUERY_FILE="${QUERY_FILE:-${DATA_DIR}/sift_query.fbin}"
GT_FILE="${GT_FILE:-${DATA_DIR}/sift_groundtruth.bin}"
PROFILE="${PROFILE:-${DATA_DIR}/run2_profile_same_trace}"
# BFS static cache node count (align with MERIT mem tier when possible)
N_NODES="${N_NODES:-12843}"
MEM_GB="${MEM_GB:-0.01}"
DISK_RATIO="${DISK_RATIO:-0.1}"
DISK_K_HOPS="${DISK_K_HOPS:-2}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/five_way_runs}"
LOG="${LOG:-${OUT_DIR}/run_merit_five_way.log}"

SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
THREADS="${THREADS:-16}"
L="${L:-100}"
K="${K:-10}"
W="${W:-2}"
DATA_TYPE="${DATA_TYPE:-float}"
DIST_FN="${DIST_FN:-l2}"

if [[ ! -f "${INDEX_PREFIX}_disk.index" ]]; then
  echo "ERROR: missing ${INDEX_PREFIX}_disk.index" >&2
  exit 1
fi
if [[ ! -f "${SEARCH}" ]]; then
  echo "ERROR: missing ${SEARCH} (build search_disk_index first)" >&2
  exit 1
fi

extract_row() { grep -E "^\s+${L}\s+${W}\s+" "$1" | tail -1 || true; }

COMMON=(
  --data_type "${DATA_TYPE}" --dist_fn "${DIST_FN}"
  --index_path_prefix "${INDEX_PREFIX}"
  --query_file "${QUERY_FILE}"
  --gt_file "${GT_FILE}"
  --recall_at "${K}" --search_list "${L}" --beamwidth "${W}"
  --num_threads "${THREADS}"
  --enable_query_sector_cache
)

mkdir -p "${OUT_DIR}"

{
  echo "Five-way MERIT compare $(date -Is)"
  echo "index=${INDEX_PREFIX} query=${QUERY_FILE} profile=${PROFILE}"
  echo "BFS N_NODES=${N_NODES} MEM_GB=${MEM_GB} DISK_RATIO=${DISK_RATIO} DISK_K_HOPS=${DISK_K_HOPS} L=${L} W=${W} threads=${THREADS}"
  echo "OUT_DIR=${OUT_DIR}"
  echo ""

  run_case() {
    local tag="$1"
    shift
    local out="${OUT_DIR}/${tag}.out"
    echo "======== ${tag} ========"
    "$SEARCH" "$@" >"${out}" 2>&1
    grep -E 'Caching|MERIT memory pool|rank_skip|MERIT disk-cache ready|MERIT disk-cache k_hops|flat Top|k-hop packing' \
      "${out}" 2>/dev/null | head -6 || true
    extract_row "${out}" | sed 's/^/  /'
    echo ""
  }

  run_case baseline \
    "${COMMON[@]}" --result_path "${OUT_DIR}/baseline" --num_nodes_to_cache 0

  run_case bfs \
    "${COMMON[@]}" --result_path "${OUT_DIR}/bfs" --num_nodes_to_cache "${N_NODES}"

  run_case mem \
    "${COMMON[@]}" --result_path "${OUT_DIR}/mem" --num_nodes_to_cache 0 \
    --merit_memory_gb "${MEM_GB}" --merit_profile_prefix "${PROFILE}"

  run_case disk \
    "${COMMON[@]}" --result_path "${OUT_DIR}/disk" --num_nodes_to_cache 0 \
    --merit_disk_cache_ratio "${DISK_RATIO}" --merit_profile_prefix "${PROFILE}" \
    --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case mem_disk \
    "${COMMON[@]}" --result_path "${OUT_DIR}/mem_disk" --num_nodes_to_cache 0 \
    --merit_memory_gb "${MEM_GB}" --merit_disk_cache_ratio "${DISK_RATIO}" \
    --merit_profile_prefix "${PROFILE}" --merit_disk_cache_exclude_memory true \
    --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  echo "===== TABLE (L=${L} W=${W}) ====="
  printf "%-16s %10s %12s %10s %12s %8s %8s\n" "Case" "QPS" "Latency_us" "MeanIOs" "MeanIO_us" "Recall" "MeritDC"
  declare -A LABEL=(
    [baseline]=baseline
    [bfs]=BFS
    [mem]=mem
    [disk]=disk
    [mem_disk]=mem+disk
  )
  for tag in baseline bfs mem disk mem_disk; do
    row=$(extract_row "${OUT_DIR}/${tag}.out")
    if [[ -z "$row" ]]; then
      printf "%-16s %s\n" "${LABEL[$tag]}" "(no row)"
      continue
    fi
    qps=$(echo "$row" | awk '{print $3}')
    lat=$(echo "$row" | awk '{print $4}')
    ios=$(echo "$row" | awk '{print $6}')
    recall=$(echo "$row" | awk '{print $NF}')
    nf=$(echo "$row" | awk '{print NF}')
    if [[ "$nf" -ge 14 ]]; then
      ious=$(echo "$row" | awk '{print $(NF-3)}')
      mdc=$(echo "$row" | awk '{print $(NF-4)}')
    else
      ious=$(echo "$row" | awk '{print $(NF-2)}')
      mdc="-"
    fi
    printf "%-16s %10s %12s %10s %12s %8s %8s\n" "${LABEL[$tag]}" "$qps" "$lat" "$ios" "$ious" "$recall" "$mdc"
  done
  echo ""
  echo "Outputs: ${OUT_DIR}/*.out"
  echo "Done $(date -Is)"
} 2>&1 | tee "${LOG}"
