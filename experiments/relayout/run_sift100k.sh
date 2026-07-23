#!/usr/bin/env bash
# SIFT100K re-layout experiment: Run1 (baseline) -> Run2 (profile) -> offline relayout -> Run3 (relayout)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-${REPO_ROOT}/diskann/build}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift100k}"
INDEX_PREFIX="${INDEX_PREFIX:-${DATA_DIR}/sift100k_index}"
RELAYOUT_PREFIX="${RELAYOUT_PREFIX:-${DATA_DIR}/sift100k_relayout_index}"
PROFILE_PREFIX="${PROFILE_PREFIX:-${DATA_DIR}/run2_profile}"
QUERY_FILE="${QUERY_FILE:-${DATA_DIR}/sift_query.bin}"
RELAYOUT_ORDER="${RELAYOUT_ORDER:-${DATA_DIR}/relayout_order}"
GT_FILE="${GT_FILE:-${DATA_DIR}/sift_groundtruth.bin}"
BASE_FILE="${BASE_FILE:-${DATA_DIR}/sift_base.bin}"

L="${L:-100}"
K="${K:-10}"
W="${W:-2}"
THREADS="${THREADS:-16}"

SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
RELAYOUT="${DISKANN_BUILD}/apps/relayout_disk_index"
APPLY="${DISKANN_BUILD}/apps/apply_disk_permutation"

common_search_args=(
  --data_type float
  --dist_fn l2
  --query_file "${QUERY_FILE}"
  --gt_file "${GT_FILE}"
  --recall_at "${K}"
  --search_list "${L}"
  --beamwidth "${W}"
  --num_threads "${THREADS}"
  --num_nodes_to_cache 0
)

echo "=== Run1: original index, profiling OFF (baseline) ==="
"${SEARCH}" \
  "${common_search_args[@]}" \
  --index_path_prefix "${INDEX_PREFIX}" \
  --result_path "${DATA_DIR}/run1_results"

echo "=== Run2: original index, profiling ON (CDF + relayout input) ==="
"${SEARCH}" \
  "${common_search_args[@]}" \
  --index_path_prefix "${INDEX_PREFIX}" \
  --result_path "${DATA_DIR}/run2_results" \
  --enable_access_profile \
  --access_profile_prefix "${PROFILE_PREFIX}"

echo "=== Offline: hot-node k-hop relayout + rewrite index (k_hops=2) ==="
"${RELAYOUT}" \
  --mem_index "${INDEX_PREFIX}_mem.index" \
  --profile_prefix "${PROFILE_PREFIX}" \
  --output_order "${DATA_DIR}/relayout_order" \
  --k_hops 2

"${APPLY}" \
  --data_type float \
  --base_file "${BASE_FILE}" \
  --mem_index "${INDEX_PREFIX}_mem.index" \
  --index_prefix "${INDEX_PREFIX}" \
  --order_file "${DATA_DIR}/relayout_order" \
  --output_prefix "${RELAYOUT_PREFIX}"

echo "=== Run3: relayout index, profiling OFF (compare IO vs Run1) ==="
"${SEARCH}" \
  "${common_search_args[@]}" \
  --index_path_prefix "${RELAYOUT_PREFIX}" \
  --result_path "${DATA_DIR}/run3_results" \
  --relayout_order_prefix "${RELAYOUT_ORDER}"

echo "Done. Compare Run1 vs Run3 Mean IOs in search output above."
