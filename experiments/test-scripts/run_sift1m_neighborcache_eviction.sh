#!/usr/bin/env bash
# SIFT1M: NEIGHBORCACHE tier exclude vs overlap; batch memory/disk eviction after warmup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
# shellcheck source=neighborcache_ramfs_env.sh disable=SC1091
source "${SCRIPT_DIR}/neighborcache_ramfs_env.sh"
PERSIST_DATA_DIR="${DATA_DIR}"
neighborcache_ramfs_activate "${PERSIST_DATA_DIR}"
DATA_DIR="${NEIGHBORCACHE_RAMFS_ACTIVE_DATA_DIR}"
PROFILE="${PROFILE:-${DATA_DIR}/run2_profile}"
DISK_K_HOPS="${DISK_K_HOPS:-2}"
LOG="${LOG:-${PERSIST_DATA_DIR}/run_neighborcache_eviction.log}"

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
  "${SEARCH}" "$@"
}

: > "${LOG}"
{
  echo "NEIGHBORCACHE eviction test $(date -Is)"

  run_case "A exclude_memory ON" \
    --data_type float --dist_fn l2 --index_path_prefix "${DATA_DIR}/sift1m_index" \
    --result_path "${DATA_DIR}/neighborcache_ev_a" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads "${THREADS}" --num_nodes_to_cache 0 \
    --neighborcache_memory_gb 0.01 --neighborcache_disk_cache_ratio 0.1 --neighborcache_profile_prefix "${PROFILE}" \
    --neighborcache_disk_cache_exclude_memory true --neighborcache_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case "B exclude_memory OFF" \
    --data_type float --dist_fn l2 --index_path_prefix "${DATA_DIR}/sift1m_index" \
    --result_path "${DATA_DIR}/neighborcache_ev_b" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads "${THREADS}" --num_nodes_to_cache 0 \
    --neighborcache_memory_gb 0.01 --neighborcache_disk_cache_ratio 0.1 --neighborcache_profile_prefix "${PROFILE}" \
    --neighborcache_disk_cache_exclude_memory false --neighborcache_disk_cache_k_hops "${DISK_K_HOPS}"

  run_case "C batch evict after warmup" \
    --data_type float --dist_fn l2 --index_path_prefix "${DATA_DIR}/sift1m_index" \
    --result_path "${DATA_DIR}/neighborcache_ev_c" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads "${THREADS}" --num_nodes_to_cache 0 \
    --neighborcache_memory_gb 0.1 --neighborcache_disk_cache_ratio 0.1 --neighborcache_profile_prefix "${PROFILE}" \
    --neighborcache_evict_memory_gb 0.01 --neighborcache_evict_disk_ratio 0.01 --neighborcache_disk_cache_k_hops "${DISK_K_HOPS}"

  echo ""
  echo "===== KEY LINES (evict / rank_skip / L=${L}) ====="
} 2>&1 | tee "${LOG}"

grep -E 'evict|rank_skip|^\s+'"${L}"'\s+' "${LOG}" || true
