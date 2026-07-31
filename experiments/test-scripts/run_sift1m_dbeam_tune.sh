#!/usr/bin/env bash
# 1M tune: directed_beam w=1 — disk ratio sweep + unified vs disk cache.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
PERSIST_DATA_DIR="${DATA_DIR:-/home/jianz/workplace/diskann/anndisk/data/sift1m}"
# shellcheck source=merit_ramfs_env.sh disable=SC1091
source "${SCRIPT_DIR}/merit_ramfs_env.sh"
merit_ramfs_activate "${PERSIST_DATA_DIR}"
DATA_DIR="${MERIT_RAMFS_ACTIVE_DATA_DIR}"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/1m_dbeam_tune_${STAMP}"
PROFILE="${DATA_DIR}/run2_profile_same_trace"
LOG="${OUT_ROOT}/run.log"
mkdir -p "${OUT_ROOT}"

L=100
W=2
K=10
THREADS=16

BASE=(
  --data_type float --dist_fn l2
  --index_path_prefix "${DATA_DIR}/sift1m_index"
  --query_file "${DATA_DIR}/sift_query.fbin"
  --gt_file "${DATA_DIR}/sift_groundtruth.bin"
  --recall_at "${K}" --search_list "${L}" --beamwidth "${W}"
  --num_threads "${THREADS}"
  --num_nodes_to_cache 0
  --enable_query_sector_cache
  --merit_memory_gb 0.01
  --merit_profile_prefix "${PROFILE}"
  --merit_disk_cache_exclude_memory true
  --merit_disk_cache_layout directed_beam
  --merit_disk_cache_k_hops 1
)

run_case() {
  local tag="$1"
  shift
  local out="${OUT_ROOT}/${tag}"
  mkdir -p "${out}"
  echo "======== ${tag} ========"
  /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
    "${BASE[@]}" "$@" \
    --result_path "${out}/run" \
    >"${OUT_ROOT}/${tag}.out" 2>&1
  grep -E 'selected|packing|unified|ready|^\s+'"${L}"'\s+'"${W}"'\s+' "${OUT_ROOT}/${tag}.out" | tail -4
  echo ""
}

parse_row() {
  local f="$1"
  grep -E "^\s+${L}\s+${W}\s+" "$f" | tail -1
}

{
  echo "1M directed_beam tune $(date -Is)"
  echo "profile=${PROFILE} OUT=${OUT_ROOT}"

  echo "--- ratio sweep ---"
  run_case ratio_010 --merit_disk_cache_ratio 0.10 --merit_unified_disk_cache false
  run_case ratio_015 --merit_disk_cache_ratio 0.15 --merit_unified_disk_cache false
  run_case ratio_020 --merit_disk_cache_ratio 0.20 --merit_unified_disk_cache false

  echo "--- unified vs disk cache (ratio=0.1) ---"
  run_case disk_cache_010 --merit_disk_cache_ratio 0.10 --merit_unified_disk_cache false
  run_case unified_010 --merit_disk_cache_ratio 0.10 --merit_unified_disk_cache true

  echo "===== SUMMARY L=${L} W=${W} ====="
  printf "%-14s %12s %12s %12s %10s %8s\n" "case" "DiskReads" "DiskCachePg" "MeritDcHit" "Selected" "Recall"
  for tag in ratio_010 ratio_015 ratio_020 disk_cache_010 unified_010; do
    row=$(parse_row "${OUT_ROOT}/${tag}.out")
    dr=$(echo "$row" | awk '{print $10}')
    sp=$(echo "$row" | awk '{print $8}')
    md=$(echo "$row" | awk '{print $15}')
    rc=$(echo "$row" | awk '{print $NF}')
    sel=$(grep -oE 'selected [0-9]+ nodes' "${OUT_ROOT}/${tag}.out" | tail -1 | awk '{print $2}')
    printf "%-14s %12s %12s %12s %10s %8s\n" "$tag" "$dr" "$sp" "$md" "$sel" "$rc"
  done
  echo "Done $(date -Is)"
} 2>&1 | tee "${LOG}"
