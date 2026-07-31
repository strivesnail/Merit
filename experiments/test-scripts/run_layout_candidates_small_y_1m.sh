#!/usr/bin/env bash
# Compare directed_beam vs hybrid / edge_star_dup / dir_edge_star (small_y + 1M).
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
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/layout_candidates_${STAMP}"
LOG="${OUT_ROOT}/run.log"
mkdir -p "${OUT_ROOT}"

KH=1
CASES=(
  directed_beam:directed_beam
  dbeam_hybrid:directed_beam_hybrid
  edge_star_dup:edge_star_dup
  dir_edge_star:dir_edge_star
)

run_case() {
  local suite="$1" tag="$2" layout="$3"
  local out="${OUT_ROOT}/${suite}/${tag}"
  shift 3
  mkdir -p "${out}"
  echo "======== ${suite}/${tag} layout=${layout} k_hops=${KH} ========"
  /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
    "$@" \
    --merit_disk_cache_layout "${layout}" \
    --merit_disk_cache_k_hops "${KH}" \
    --result_path "${out}/run" \
    >"${OUT_ROOT}/${suite}/${tag}.out" 2>&1
  grep -E 'packing|selected|slots=|list stats|^\s+' "${OUT_ROOT}/${suite}/${tag}.out" | tail -4
  echo ""
}

print_row() {
  local suite="$1" tag="$2" f="$3"
  local l w
  if [[ "${suite}" == "small_y" ]]; then l=50; w=4; else l=100; w=2; fi
  local row bp sp dr rec
  row=$(grep -E "^\s+${l}\s+${w}\s+" "$f" | tail -1)
  bp=$(echo "$row" | awk '{print $7}')
  sp=$(echo "$row" | awk '{print $8}')
  dr=$(echo "$row" | awk '{print $10}')
  rec=$(echo "$row" | awk '{print $NF}')
  printf "%-18s %12s %12s %12s %10s\n" "$tag" "$dr" "$bp" "$sp" "$rec"
}

{
  echo "Layout candidates vs directed_beam $(date -Is)"
  echo "OUT=${OUT_ROOT}"
  echo "No memory; disk ratio=0.1; k_hops=${KH}"

  SMALL=(
    --data_type float --dist_fn l2
    --index_path_prefix "${DATA_DIR}/sift1m_index"
    --query_file "${DATA_DIR}/workloads/small_y_10k.fbin"
    --gt_file "${DATA_DIR}/workloads/small_y_10k_gt.bin"
    --recall_at 1 --search_list 50 --beamwidth 4
    --num_threads 8 --num_nodes_to_cache 0
    --enable_query_sector_cache
    --merit_memory_gb 0
    --merit_profile_prefix "${DATA_DIR}/workloads/profiles/small_y_10k"
    --merit_disk_cache_ratio 0.1
  )

  M1M=(
    --data_type float --dist_fn l2
    --index_path_prefix "${DATA_DIR}/sift1m_index"
    --query_file "${DATA_DIR}/sift_query.fbin"
    --gt_file "${DATA_DIR}/sift_groundtruth.bin"
    --recall_at 10 --search_list 100 --beamwidth 2
    --num_threads 16 --num_nodes_to_cache 0
    --enable_query_sector_cache
    --merit_memory_gb 0
    --merit_profile_prefix "${DATA_DIR}/run2_profile_same_trace"
    --merit_disk_cache_ratio 0.1
  )

  for spec in "${CASES[@]}"; do
    tag="${spec%%:*}"
    layout="${spec##*:}"
    run_case small_y "${tag}" "${layout}" "${SMALL[@]}"
  done

  for spec in "${CASES[@]}"; do
    tag="${spec%%:*}"
    layout="${spec##*:}"
    run_case 1m "${tag}" "${layout}" "${M1M[@]}"
  done

  echo "===== small_y L=50 W=4 ====="
  printf "%-18s %12s %12s %12s %10s\n" "layout" "DiskReads" "BasePages" "DiskCachePg" "Recall"
  for spec in "${CASES[@]}"; do
    tag="${spec%%:*}"
    print_row small_y "${tag}" "${OUT_ROOT}/small_y/${tag}.out"
  done

  echo ""
  echo "===== 1M L=100 W=2 ====="
  printf "%-18s %12s %12s %12s %10s\n" "layout" "DiskReads" "BasePages" "DiskCachePg" "Recall"
  for spec in "${CASES[@]}"; do
    tag="${spec%%:*}"
    print_row 1m "${tag}" "${OUT_ROOT}/1m/${tag}.out"
  done

  echo ""
  echo "Results: ${OUT_ROOT}"
} 2>&1 | tee "${LOG}"
