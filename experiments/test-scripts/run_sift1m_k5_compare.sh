#!/usr/bin/env bash
# SIFT1M: regenerate relayout orders with k_hops=5 (edge + hotnode), apply, search.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-${REPO_ROOT}/diskann/build}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
INDEX_PREFIX="${INDEX_PREFIX:-${DATA_DIR}/sift1m_index}"
PROFILE_PREFIX="${PROFILE_PREFIX:-${DATA_DIR}/run2_profile}"
K_HOPS="${K_HOPS:-5}"
LOG="${LOG:-${DATA_DIR}/run_1m_k${K_HOPS}_compare.log}"

RELAYOUT="${DISKANN_BUILD}/apps/relayout_disk_index"
APPLY="${DISKANN_BUILD}/apps/apply_disk_permutation"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
BASE_FILE="${DATA_DIR}/sift_base.fbin"

L=100 K=10 W=2 THREADS=16

run_search() {
  local label="$1" index="$2" order="${3:-}"
  echo ""
  echo "=== $(date -Is) ${label} ==="
  local extra=()
  if [[ -n "${order}" ]]; then
    extra=(--relayout_order_prefix "${order}")
  fi
  "${SEARCH}" \
    --data_type float --dist_fn l2 \
    --index_path_prefix "${index}" \
    --result_path "${DATA_DIR}/k${K_HOPS}_${label}_results" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads "${THREADS}" --num_nodes_to_cache 0 \
    "${extra[@]}"
}

offline_layout() {
  local layout="$1" order_out="$2" index_out="$3"
  echo "=== $(date -Is) relayout layout=${layout} k_hops=${K_HOPS} ==="
  "${RELAYOUT}" \
    --mem_index "${INDEX_PREFIX}_mem.index" \
    --disk_index "${INDEX_PREFIX}_disk.index" \
    --profile_prefix "${PROFILE_PREFIX}" \
    --output_order "${order_out}" \
    --k_hops "${K_HOPS}" \
    --layout "${layout}"
  "${APPLY}" \
    --data_type float \
    --base_file "${BASE_FILE}" \
    --mem_index "${INDEX_PREFIX}_mem.index" \
    --index_prefix "${INDEX_PREFIX}" \
    --order_file "${order_out}" \
    --output_prefix "${index_out}"
}

: > "${LOG}"
{
  echo "SIFT1M k_hops=${K_HOPS} edge vs hotnode (search only after offline apply)"
  offline_layout edge "${DATA_DIR}/relayout_order_k${K_HOPS}" "${DATA_DIR}/sift1m_relayout_index_k${K_HOPS}"
  offline_layout hotnode "${DATA_DIR}/relayout_order_hotnode_k${K_HOPS}" "${DATA_DIR}/sift1m_hotnode_relayout_index_k${K_HOPS}"
  run_search baseline "${INDEX_PREFIX}"
  run_search edge "${DATA_DIR}/sift1m_relayout_index_k${K_HOPS}" "${DATA_DIR}/relayout_order_k${K_HOPS}"
  run_search hotnode "${DATA_DIR}/sift1m_hotnode_relayout_index_k${K_HOPS}" "${DATA_DIR}/relayout_order_hotnode_k${K_HOPS}"
  echo "=== $(date -Is) done ==="
} 2>&1 | tee -a "${LOG}"
