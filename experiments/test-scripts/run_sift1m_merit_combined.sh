#!/usr/bin/env bash
# SIFT1M: MERIT memory / disk-cache / combined vs baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
PROFILE="${PROFILE:-${DATA_DIR}/run2_profile}"
DISK_K_HOPS="${DISK_K_HOPS:-2}"
LOG="${LOG:-${DATA_DIR}/run_merit_combined.log}"

SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
THREADS="${THREADS:-16}"
L="${L:-100}"
K="${K:-10}"
W="${W:-2}"

run_case() {
  local name="$1"
  shift
  echo ""
  echo "========== ${name} =========="
  echo "CMD: ${SEARCH} $*"
  "${SEARCH}" "$@"
}

: > "${LOG}"
{
  echo "MERIT combined test $(date -Is)"
  echo "build=${DISKANN_BUILD} data=${DATA_DIR}"

  run_case "1 baseline" \
    --data_type float --dist_fn l2 --index_path_prefix "${DATA_DIR}/sift1m_index" \
    --result_path "${DATA_DIR}/merit_test_base" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads "${THREADS}" --num_nodes_to_cache 0

  run_case "2 disk only ratio=0.1" \
    --data_type float --dist_fn l2 --index_path_prefix "${DATA_DIR}/sift1m_index" \
    --result_path "${DATA_DIR}/merit_test_dc01" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads "${THREADS}" --num_nodes_to_cache 0 \
    --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case "3 memory only 0.01GB" \
    --data_type float --dist_fn l2 --index_path_prefix "${DATA_DIR}/sift1m_index" \
    --result_path "${DATA_DIR}/merit_test_mem001" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads "${THREADS}" --num_nodes_to_cache 0 \
    --merit_memory_gb 0.01 --merit_profile_prefix "${PROFILE}"

  run_case "4 memory 0.01GB + disk 0.1" \
    --data_type float --dist_fn l2 --index_path_prefix "${DATA_DIR}/sift1m_index" \
    --result_path "${DATA_DIR}/merit_test_both" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads "${THREADS}" --num_nodes_to_cache 0 \
    --merit_memory_gb 0.01 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case "5 memory 0.1GB + disk 0.1" \
    --data_type float --dist_fn l2 --index_path_prefix "${DATA_DIR}/sift1m_index" \
    --result_path "${DATA_DIR}/merit_test_both2" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads "${THREADS}" --num_nodes_to_cache 0 \
    --merit_memory_gb 0.1 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  echo ""
  echo "===== SUMMARY (grep L=${L} result row) ====="
} 2>&1 | tee "${LOG}"

grep -E '^\s+'"${L}"'\s+' "${LOG}" || true
