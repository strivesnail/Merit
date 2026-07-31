#!/usr/bin/env bash
# ~0.01GB tier: Mem 0.01 + Disk 0.1 vs baseline / BFS / mem-only / disk-only
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
PROFILE="${PROFILE:-${DATA_DIR}/run2_profile}"
DISK_K_HOPS="${DISK_K_HOPS:-2}"
N_NODES="${N_NODES:-12843}"
LOG="${LOG:-${PERSIST_DATA_DIR}/run_merit_mem001_disk01.log}"

SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
THREADS="${THREADS:-16}"
L="${L:-100}"
K="${K:-10}"
W="${W:-2}"

extract_row() { grep -E "^\s+${L}\s+${W}\s+" "$1" | tail -1; }

: > "${LOG}"
{
  echo "Mem 0.01GB + Disk 0.1 tier compare $(date -Is)"
  COMMON=(--data_type float --dist_fn l2 --index_path_prefix "${DATA_DIR}/sift1m_index"
    --query_file "${DATA_DIR}/sift_query.fbin" --gt_file "${DATA_DIR}/sift_groundtruth.bin"
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" --num_threads "${THREADS}")

  run() { local t="$1"; shift; local o="${PERSIST_DATA_DIR}/md01_${t}.out"; echo "== $t =="; "$SEARCH" "$@" >"$o"; extract_row "$o"; }

  run baseline "${COMMON[@]}" --result_path "${PERSIST_DATA_DIR}/md01_base" --num_nodes_to_cache 0
  run diskann_bfs "${COMMON[@]}" --result_path "${PERSIST_DATA_DIR}/md01_bfs" --num_nodes_to_cache "${N_NODES}"
  run merit_mem_only "${COMMON[@]}" --result_path "${PERSIST_DATA_DIR}/md01_mem" --num_nodes_to_cache 0 \
    --merit_memory_gb 0.01 --merit_profile_prefix "${PROFILE}"
  run disk_only "${COMMON[@]}" --result_path "${PERSIST_DATA_DIR}/md01_disk" --num_nodes_to_cache 0 \
    --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" --merit_disk_cache_k_hops "${DISK_K_HOPS}"
  run mem001_disk01 "${COMMON[@]}" --result_path "${PERSIST_DATA_DIR}/md01_both" --num_nodes_to_cache 0 \
    --merit_memory_gb 0.01 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" \
    --merit_disk_cache_exclude_memory true --merit_disk_cache_k_hops "${DISK_K_HOPS}"
} 2>&1 | tee "${LOG}"
