#!/usr/bin/env bash
# Build SIFT10M disk index + same-trace access profile (Big-ANN uint8).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift10m}"
INDEX_PREFIX="${INDEX_PREFIX:-${DATA_DIR}/sift10m_index}"
BASE_FILE="${BASE_FILE:-${DATA_DIR}/sift_base.u8bin}"
QUERY_FILE="${QUERY_FILE:-${DATA_DIR}/sift_query.u8bin}"
GT_FILE="${GT_FILE:-${DATA_DIR}/sift_groundtruth.bin}"
PROFILE="${PROFILE:-${DATA_DIR}/run_profile_same_trace}"

BUILD_R="${BUILD_R:-64}"
BUILD_L="${BUILD_L:-100}"
BUILD_B="${BUILD_B:-1}"
BUILD_M="${BUILD_M:-16}"
BUILD_THREADS="${BUILD_THREADS:-$(nproc)}"
SEARCH_THREADS="${SEARCH_THREADS:-8}"
L="${L:-50}"
K="${K:-10}"
W="${W:-4}"

BUILD="${DISKANN_BUILD}/apps/build_disk_index"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
LOG="${DATA_DIR}/build_and_profile.log"

mkdir -p "${DATA_DIR}"

{
  echo "=== SIFT10M build+profile $(date -Is) ==="
  for f in "${BASE_FILE}" "${QUERY_FILE}" "${GT_FILE}"; do
    [[ -e "$f" ]] || { echo "ERROR missing $f" >&2; exit 1; }
  done

  if [[ -f "${INDEX_PREFIX}_disk.index" ]]; then
    echo "Index exists: ${INDEX_PREFIX}_disk.index"
  else
    echo "=== build_disk_index ==="
    "${BUILD}" \
      --data_type uint8 --dist_fn l2 \
      --data_path "${BASE_FILE}" \
      --index_path_prefix "${INDEX_PREFIX}" \
      -R "${BUILD_R}" -L "${BUILD_L}" \
      -B "${BUILD_B}" -M "${BUILD_M}" \
      -T "${BUILD_THREADS}"
  fi

  if [[ -f "${PROFILE}_node_expand.bin" ]]; then
    echo "Profile exists: ${PROFILE}_node_expand.bin"
  else
    echo "=== access profile (L=${L} W=${W}) ==="
    "${SEARCH}" \
      --data_type uint8 --dist_fn l2 \
      --index_path_prefix "${INDEX_PREFIX}" \
      --query_file "${QUERY_FILE}" \
      --gt_file "${GT_FILE}" \
      --result_path "${DATA_DIR}/profile_results" \
      --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
      --num_threads "${SEARCH_THREADS}" --num_nodes_to_cache 0 \
      --enable_access_profile --access_profile_prefix "${PROFILE}"
  fi

  echo "Done $(date -Is)"
} 2>&1 | tee -a "${LOG}"
