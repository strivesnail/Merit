#!/usr/bin/env bash
# Beyond ratio=0.20: higher ratio + memory tier sweep (directed_beam w=1).
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
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/1m_dbeam_ratio20_plus_${STAMP}"
PROFILE="${DATA_DIR}/run2_profile_same_trace"
LOG="${OUT_ROOT}/run.log"
mkdir -p "${OUT_ROOT}"

L=100 W=2 K=10 THREADS=16

BASE=(
  --data_type float --dist_fn l2
  --index_path_prefix "${DATA_DIR}/sift1m_index"
  --query_file "${DATA_DIR}/sift_query.fbin"
  --gt_file "${DATA_DIR}/sift_groundtruth.bin"
  --recall_at "${K}" --search_list "${L}" --beamwidth "${W}"
  --num_threads "${THREADS}"
  --num_nodes_to_cache 0
  --enable_query_sector_cache
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
  grep -E 'selected|packing|MERIT memory|^\s+'"${L}"'\s+'"${W}"'\s+' "${OUT_ROOT}/${tag}.out" | tail -4
  echo ""
}

{
  echo "1M dbeam ratio20+ sweep $(date -Is)"
  run_case r020_mem001 --merit_memory_gb 0.01 --merit_disk_cache_ratio 0.20
  run_case r025_mem001 --merit_memory_gb 0.01 --merit_disk_cache_ratio 0.25
  run_case r030_mem001 --merit_memory_gb 0.01 --merit_disk_cache_ratio 0.30
  run_case r020_mem010 --merit_memory_gb 0.10 --merit_disk_cache_ratio 0.20
  run_case r020_mem010_r015 --merit_memory_gb 0.10 --merit_disk_cache_ratio 0.15
  run_case r020_mem010_r025 --merit_memory_gb 0.10 --merit_disk_cache_ratio 0.25

  echo "===== SUMMARY L=${L} W=${W} ====="
  printf "%-22s %12s %12s %12s %12s %10s\n" "case" "DiskReads" "BasePages" "DiskCachePg" "MeritDcHit" "Recall"
  for tag in r020_mem001 r025_mem001 r030_mem001 r020_mem010 r020_mem010_r015 r020_mem010_r025; do
    row=$(grep -E "^\s+${L}\s+${W}\s+" "${OUT_ROOT}/${tag}.out" | tail -1)
    dr=$(echo "$row" | awk '{print $10}')
    bp=$(echo "$row" | awk '{print $7}')
    sp=$(echo "$row" | awk '{print $8}')
    md=$(echo "$row" | awk '{print $15}')
    rc=$(echo "$row" | awk '{print $NF}')
    printf "%-22s %12s %12s %12s %12s %10s\n" "$tag" "$dr" "$bp" "$sp" "$md" "$rc"
  done
  echo "Results: ${OUT_ROOT}"
} 2>&1 | tee "${LOG}"
