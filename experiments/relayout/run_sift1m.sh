#!/usr/bin/env bash
# SIFT1M re-layout experiment: download → build → Run1/2/3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-${REPO_ROOT}/diskann/build}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
INDEX_PREFIX="${INDEX_PREFIX:-${DATA_DIR}/sift1m_index}"
RELAYOUT_PREFIX="${RELAYOUT_PREFIX:-${DATA_DIR}/sift1m_relayout_index}"
RELAYOUT_ORDER="${RELAYOUT_ORDER:-${DATA_DIR}/relayout_order_hotnode}"
SKIP_RUN2="${SKIP_RUN2:-0}"

BASE_FILE="${BASE_FILE:-${DATA_DIR}/sift_base.fbin}"
QUERY_FILE="${QUERY_FILE:-${DATA_DIR}/sift_query.fbin}"
GT_FILE="${GT_FILE:-${DATA_DIR}/sift_groundtruth.bin}"

L="${L:-100}"
K="${K:-10}"
W="${W:-2}"
THREADS="${THREADS:-16}"
K_HOPS="${K_HOPS:-2}"
BUILD_R="${BUILD_R:-64}"
BUILD_L="${BUILD_L:-100}"
BUILD_B="${BUILD_B:-64}"
BUILD_M="${BUILD_M:-64}"

SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
BUILD="${DISKANN_BUILD}/apps/build_disk_index"
RELAYOUT="${DISKANN_BUILD}/apps/relayout_disk_index"
APPLY="${DISKANN_BUILD}/apps/apply_disk_permutation"
FVECS_BIN="${DISKANN_BUILD}/apps/utils/fvecs_to_bin"
GT_BIN="${DISKANN_BUILD}/apps/utils/compute_groundtruth"

mkdir -p "${DATA_DIR}"

prepare_data() {
  if [[ -f "${BASE_FILE}" && -f "${QUERY_FILE}" && -f "${GT_FILE}" ]]; then
    echo "Data already present under ${DATA_DIR}"
    return 0
  fi

  local raw_dir="${DATA_DIR}/texmex"
  mkdir -p "${raw_dir}"
  if [[ ! -f "${raw_dir}/sift/sift_base.fvecs" ]]; then
    echo "=== Download TexMex SIFT1M (sift.tar.gz) ==="
    if [[ ! -f "${raw_dir}/sift.tar.gz" ]]; then
      wget -O "${raw_dir}/sift.tar.gz" ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    fi
    tar -xf "${raw_dir}/sift.tar.gz" -C "${raw_dir}"
  fi

  echo "=== Convert fvecs → fbin ==="
  "${FVECS_BIN}" uint8 "${raw_dir}/sift/sift_base.fvecs" "${BASE_FILE}"
  "${FVECS_BIN}" uint8 "${raw_dir}/sift/sift_query.fvecs" "${QUERY_FILE}"

  if [[ ! -f "${GT_FILE}" ]]; then
    echo "=== Compute ground truth (K=${K}) — may take several minutes ==="
    "${GT_BIN}" \
      --data_type uint8 \
      --dist_fn l2 \
      --base_file "${BASE_FILE}" \
      --query_file "${QUERY_FILE}" \
      --gt_file "${GT_FILE}" \
      -K "${K}"
  fi
}

build_index() {
  if [[ -f "${INDEX_PREFIX}_disk.index" && -f "${INDEX_PREFIX}_mem.index" ]]; then
    echo "Index already built: ${INDEX_PREFIX}"
    return 0
  fi
  echo "=== Build disk index (1M points, R=${BUILD_R}, L=${BUILD_L}) ==="
  "${BUILD}" \
    --data_type uint8 \
    --dist_fn l2 \
    --data_path "${BASE_FILE}" \
    --index_path_prefix "${INDEX_PREFIX}" \
    -R "${BUILD_R}" \
    -L "${BUILD_L}" \
    -B "${BUILD_B}" \
    -M "${BUILD_M}" \
    -T "$(nproc)"
}

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

prepare_data
build_index

echo "=== Run1: original index, profiling OFF ==="
"${SEARCH}" \
  "${common_search_args[@]}" \
  --index_path_prefix "${INDEX_PREFIX}" \
  --result_path "${DATA_DIR}/run1_results"

echo "=== Run2: original index, profiling ON ==="
if [[ "${SKIP_RUN2}" == "1" ]]; then
  echo "SKIP_RUN2=1: using existing profile at ${PROFILE_PREFIX}"
else
"${SEARCH}" \
  "${common_search_args[@]}" \
  --index_path_prefix "${INDEX_PREFIX}" \
  --result_path "${DATA_DIR}/run2_results" \
  --enable_access_profile \
  --access_profile_prefix "${PROFILE_PREFIX}"
fi

echo "=== Offline relayout (hot-node, k_hops=${K_HOPS}) ==="
"${RELAYOUT}" \
  --mem_index "${INDEX_PREFIX}_mem.index" \
  --disk_index "${INDEX_PREFIX}_disk.index" \
  --profile_prefix "${PROFILE_PREFIX}" \
  --output_order "${RELAYOUT_ORDER}" \
  --k_hops "${K_HOPS}"

"${APPLY}" \
  --data_type float \
  --base_file "${BASE_FILE}" \
  --mem_index "${INDEX_PREFIX}_mem.index" \
  --index_prefix "${INDEX_PREFIX}" \
  --order_file "${RELAYOUT_ORDER}" \
  --output_prefix "${RELAYOUT_PREFIX}"

echo "=== Run3: relayout index ==="
"${SEARCH}" \
  "${common_search_args[@]}" \
  --index_path_prefix "${RELAYOUT_PREFIX}" \
  --result_path "${DATA_DIR}/run3_results" \
  --relayout_order_prefix "${RELAYOUT_ORDER}"

echo "Done. Compare Run1 vs Run3 Mean Pages / Mean IOs above."
