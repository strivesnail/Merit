#!/usr/bin/env bash
# Disk-cache-only sweep (no memory, no relayout): directed_beam w=1 ratio 0.20–0.50.
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
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/1m_disk_cache_only_${STAMP}"
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
  --merit_memory_gb 0
  --merit_profile_prefix "${PROFILE}"
  --merit_disk_cache_layout directed_beam
  --merit_disk_cache_k_hops 1
)

run_case() {
  local tag="$1" ratio="$2"
  local out="${OUT_ROOT}/${tag}"
  mkdir -p "${out}"
  echo "======== ${tag} ratio=${ratio} ========"
  /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
    "${BASE[@]}" \
    --merit_disk_cache_ratio "${ratio}" \
    --result_path "${out}/run" \
    >"${OUT_ROOT}/${tag}.out" 2>&1
  grep -E 'selected|packing|^\s+'"${L}"'\s+'"${W}"'\s+' "${OUT_ROOT}/${tag}.out" | tail -3
  echo ""
}

{
  echo "1M disk-cache-only ratio sweep $(date -Is)"
  run_case r020 0.20
  run_case r025 0.25
  run_case r030 0.30
  run_case r035 0.35
  run_case r040 0.40
  run_case r050 0.50

  echo "===== SUMMARY L=${L} W=${W} (no memory, no relayout) ====="
  printf "%-8s %10s %12s %12s %12s %10s %10s\n" "case" "ratio" "DiskReads" "BasePages" "DiskCachePg" "MeritDcHit" "Recall"
  for tag in r020 r025 r030 r035 r040 r050; do
    ratio="${tag#r}"
    ratio="0.${ratio}"
    row=$(grep -E "^\s+${L}\s+${W}\s+" "${OUT_ROOT}/${tag}.out" | tail -1)
    dr=$(echo "$row" | awk '{print $10}')
    bp=$(echo "$row" | awk '{print $7}')
    sp=$(echo "$row" | awk '{print $8}')
    md=$(echo "$row" | awk '{print $15}')
    rc=$(echo "$row" | awk '{print $NF}')
    printf "%-8s %10s %12s %12s %12s %12s %10s\n" "$tag" "$ratio" "$dr" "$bp" "$sp" "$md" "$rc"
  done
  echo "Results: ${OUT_ROOT}"
} 2>&1 | tee "${LOG}"
