#!/usr/bin/env bash
# Regenerate Run2 profile on QUERY_FILE, then run all MERIT compare cases with the same trace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
# shellcheck source=merit_ramfs_env.sh disable=SC1091
source "${SCRIPT_DIR}/merit_ramfs_env.sh"
PERSIST_DATA_DIR="${DATA_DIR}"
merit_ramfs_activate "${PERSIST_DATA_DIR}"
DATA_DIR="${MERIT_RAMFS_ACTIVE_DATA_DIR}"
QUERY_FILE="${QUERY_FILE:-${DATA_DIR}/sift_query.fbin}"
GT_FILE="${GT_FILE:-${DATA_DIR}/sift_groundtruth.bin}"
INDEX="${INDEX:-${DATA_DIR}/sift1m_index}"
# Profile written by Run2 on the same query/trace as benchmarks below
PROFILE="${PROFILE:-${DATA_DIR}/run2_profile_same_trace}"
SKIP_RUN2="${SKIP_RUN2:-0}"
N_NODES="${N_NODES:-12843}"
DISK_K_HOPS="${DISK_K_HOPS:-2}"

SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
THREADS="${THREADS:-16}"
L="${L:-100}"
K="${K:-10}"
W="${W:-2}"
OUT_DIR="${OUT_DIR:-${PERSIST_DATA_DIR}/same_trace_runs}"
LOG="${LOG:-${PERSIST_DATA_DIR}/run_merit_same_trace.log}"

extract_row() { grep -E "^\s+${L}\s+${W}\s+" "$1" | tail -1 || true; }

COMMON=(
  --data_type float --dist_fn l2
  --index_path_prefix "${INDEX}"
  --query_file "${QUERY_FILE}"
  --gt_file "${GT_FILE}"
  --recall_at "${K}" --search_list "${L}" --beamwidth "${W}"
  --num_threads "${THREADS}"
)

mkdir -p "${OUT_DIR}"
: > "${LOG}"

{
  echo "=== Same-trace MERIT suite $(date -Is) ==="
  echo "query_file=${QUERY_FILE}"
  echo "profile_prefix=${PROFILE}"
  echo "L=${L} W=${W} threads=${THREADS} N_NODES=${N_NODES} DISK_K_HOPS=${DISK_K_HOPS}"
  echo ""

  if [[ "${SKIP_RUN2}" != "1" ]]; then
    echo "======== Run2: profiling on same query file (no cache) ========"
    "${SEARCH}" \
      "${COMMON[@]}" \
      --num_nodes_to_cache 0 \
      --result_path "${OUT_DIR}/run2_search" \
      --enable_access_profile \
      --access_profile_prefix "${PROFILE}" \
      >"${OUT_DIR}/run2_profile.out" 2>&1
    grep -E 'Access profiling|node_expand|CDF|^\s+'"${L}"'\s+'"${W}"'' "${OUT_DIR}/run2_profile.out" | tail -8
    if [[ ! -f "${PROFILE}_node_expand.bin" ]]; then
      echo "ERROR: missing ${PROFILE}_node_expand.bin after Run2"
      exit 1
    fi
  else
    echo "SKIP_RUN2=1: using existing ${PROFILE}_node_expand.bin"
  fi

  run_case() {
    local tag="$1"
    shift
    local out="${OUT_DIR}/${tag}.out"
    echo ""
    echo "======== ${tag} ========"
    "${SEARCH}" "$@" >"${out}" 2>&1
    grep -E 'Caching|MERIT memory pool|rank_skip|MERIT disk-cache ready' "${out}" 2>/dev/null | head -4 || true
    extract_row "${out}" | sed 's/^/  /'
  }

  run_case baseline \
    "${COMMON[@]}" --result_path "${OUT_DIR}/baseline" --num_nodes_to_cache 0

  run_case diskann_bfs \
    "${COMMON[@]}" --result_path "${OUT_DIR}/diskann_bfs" --num_nodes_to_cache "${N_NODES}"

  run_case merit_mem001 \
    "${COMMON[@]}" --result_path "${OUT_DIR}/merit_mem001" --num_nodes_to_cache 0 \
    --merit_memory_gb 0.01 --merit_profile_prefix "${PROFILE}"

  run_case disk_only_01 \
    "${COMMON[@]}" --result_path "${OUT_DIR}/disk_only_01" --num_nodes_to_cache 0 \
    --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case mem001_disk01_exclude \
    "${COMMON[@]}" --result_path "${OUT_DIR}/mem001_disk01_ex" --num_nodes_to_cache 0 \
    --merit_memory_gb 0.01 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" \
    --merit_disk_cache_exclude_memory true --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case mem001_disk01_overlap \
    "${COMMON[@]}" --result_path "${OUT_DIR}/mem001_disk01_ov" --num_nodes_to_cache 0 \
    --merit_memory_gb 0.01 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" \
    --merit_disk_cache_exclude_memory false --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case mem01_disk01 \
    "${COMMON[@]}" --result_path "${OUT_DIR}/mem01_disk01" --num_nodes_to_cache 0 \
    --merit_memory_gb 0.1 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" \
    --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case diskann_bfs_disk01 \
    "${COMMON[@]}" --result_path "${OUT_DIR}/bfs_disk01" --num_nodes_to_cache "${N_NODES}" \
    --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case batch_evict \
    "${COMMON[@]}" --result_path "${OUT_DIR}/batch_evict" --num_nodes_to_cache 0 \
    --merit_memory_gb 0.1 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" \
    --merit_evict_memory_gb 0.01 --merit_evict_disk_ratio 0.01 --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case runtime_admit_t1 \
    --data_type float --dist_fn l2 --index_path_prefix "${INDEX}" \
    --query_file "${QUERY_FILE}" --gt_file "${GT_FILE}" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads 1 --num_nodes_to_cache 0 \
    --result_path "${OUT_DIR}/runtime_admit" \
    --merit_memory_gb 0.01 --merit_profile_prefix "${PROFILE}" \
    --merit_memory_runtime_admit true

  echo ""
  echo "===== PERFORMANCE TABLE (trace=${QUERY_FILE}, profile=${PROFILE}) ====="
  printf "%-28s %10s %12s %10s %12s %8s %8s\n" "Case" "QPS" "Latency_us" "MeanIOs" "MeanIO_us" "Recall" "MeritDC"
  declare -A LABEL=(
    [baseline]=baseline
    [diskann_bfs]=diskann_bfs_N
    [merit_mem001]=merit_mem_0.01
    [disk_only_01]=disk_only_0.1
    [mem001_disk01_exclude]=mem0.01+disk0.1_ex
    [mem001_disk01_overlap]=mem0.01+disk0.1_ov
    [mem01_disk01]=mem0.1+disk0.1
    [diskann_bfs_disk01]=bfs_N+disk0.1
    [batch_evict]=batch_evict
    [runtime_admit_t1]=runtime_admit_t1
  )
  for tag in baseline diskann_bfs merit_mem001 disk_only_01 mem001_disk01_exclude mem001_disk01_overlap \
             mem01_disk01 diskann_bfs_disk01 batch_evict runtime_admit_t1; do
    row=$(extract_row "${OUT_DIR}/${tag}.out")
    if [[ -z "$row" ]]; then
      printf "%-28s %s\n" "${LABEL[$tag]}" "(no row)"
      continue
    fi
    qps=$(echo "$row" | awk '{print $3}')
    lat=$(echo "$row" | awk '{print $4}')
    ios=$(echo "$row" | awk '{print $6}')
    nf=$(echo "$row" | awk '{print NF}')
    recall=$(echo "$row" | awk '{print $NF}')
    if [[ "$nf" -ge 14 ]]; then
      ious=$(echo "$row" | awk '{print $(NF-3)}')
      meritdc=$(echo "$row" | awk '{print $(NF-4)}')
    else
      ious=$(echo "$row" | awk '{print $(NF-2)}')
      meritdc="-"
    fi
    printf "%-28s %10s %12s %10s %12s %8s %8s\n" "${LABEL[$tag]}" "$qps" "$lat" "$ios" "$ious" "$recall" "$meritdc"
  done
  echo ""
  echo "Done $(date -Is)"
} 2>&1 | tee "${LOG}"
