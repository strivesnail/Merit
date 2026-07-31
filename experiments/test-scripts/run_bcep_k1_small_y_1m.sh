#!/usr/bin/env bash
# Compare Layout B/C/E/P at k_hops=1 on small_y and 1M (Disk Reads).
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
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/bcep_k1_${STAMP}"
LOG="${OUT_ROOT}/run.log"
mkdir -p "${OUT_ROOT}"

KH=1
LAYOUTS=(B:node C:edge E:directed_beam P:parent)

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
  grep -E 'Layout E|packing|selected|^\s+' "${OUT_ROOT}/${suite}/${tag}.out" | tail -3
  echo ""
}

run_baseline() {
  local suite="$1"
  local out="${OUT_ROOT}/${suite}/baseline"
  shift
  mkdir -p "${out}"
  echo "======== ${suite}/baseline (no MERIT) ========"
  /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
    "$@" \
    --result_path "${out}/run" \
    >"${OUT_ROOT}/${suite}/baseline.out" 2>&1
  parse_row "${OUT_ROOT}/${suite}/baseline.out" "${suite}" | tail -1 || true
  echo ""
}

parse_row() {
  local f="$1" suite="$2"
  local l w
  if [[ "${suite}" == "small_y" ]]; then l=50; w=4; else l=100; w=2; fi
  grep -E "^\s+${l}\s+${w}\s+" "$f" | tail -1
}

print_summary_row() {
  local label="$1" f="$2" suite="$3"
  local row l w dr bp sp md rc
  if [[ "${suite}" == "small_y" ]]; then l=50; w=4; else l=100; w=2; fi
  row=$(grep -E "^\s+${l}\s+${w}\s+" "$f" | tail -1)
  rc=$(echo "$row" | awk '{print $NF}')
  if [[ "${label}" == "baseline" ]]; then
    # No DiskCachePg column: BasePages=$7, Disk Reads=$9
    bp=$(echo "$row" | awk '{print $7}')
    dr=$(echo "$row" | awk '{print $9}')
    printf "%-10s %12s %12s %12s %12s %10s\n" "$label" "$dr" "$bp" "-" "-" "$rc"
  else
    # With disk cache: BasePages=$7, DiskCachePg=$8, Disk Reads=$10 (= $7+$8)
    bp=$(echo "$row" | awk '{print $7}')
    sp=$(echo "$row" | awk '{print $8}')
    dr=$(echo "$row" | awk '{print $10}')
    md=$(echo "$row" | awk '{print $15}')
    printf "%-10s %12s %12s %12s %12s %10s\n" "$label" "$dr" "$bp" "$sp" "$md" "$rc"
  fi
}

{
  echo "Layout baseline+B/C/E/P k_hops=${KH} compare $(date -Is)"
  echo "OUT=${OUT_ROOT}"
  echo "No memory tier; disk cache ratio=0.1 for B/C/E/P only."

  SMALL_COMMON=(
    --data_type float --dist_fn l2
    --index_path_prefix "${DATA_DIR}/sift1m_index"
    --query_file "${DATA_DIR}/workloads/small_y_10k.fbin"
    --gt_file "${DATA_DIR}/workloads/small_y_10k_gt.bin"
    --recall_at 1 --search_list 50 --beamwidth 4
    --num_threads 8 --num_nodes_to_cache 0
    --enable_query_sector_cache
    --merit_profile_prefix "${DATA_DIR}/workloads/profiles/small_y_10k"
    --merit_disk_cache_ratio 0.1
  )

  SMALL_BASE=(
    --data_type float --dist_fn l2
    --index_path_prefix "${DATA_DIR}/sift1m_index"
    --query_file "${DATA_DIR}/workloads/small_y_10k.fbin"
    --gt_file "${DATA_DIR}/workloads/small_y_10k_gt.bin"
    --recall_at 1 --search_list 50 --beamwidth 4
    --num_threads 8 --num_nodes_to_cache 0
    --enable_query_sector_cache
  )

  M1M_COMMON=(
    --data_type float --dist_fn l2
    --index_path_prefix "${DATA_DIR}/sift1m_index"
    --query_file "${DATA_DIR}/sift_query.fbin"
    --gt_file "${DATA_DIR}/sift_groundtruth.bin"
    --recall_at 10 --search_list 100 --beamwidth 2
    --num_threads 16 --num_nodes_to_cache 0
    --enable_query_sector_cache
    --merit_memory_gb 0
    --merit_disk_cache_ratio 0.1
    --merit_profile_prefix "${DATA_DIR}/run2_profile_same_trace"
  )

  M1M_BASE=(
    --data_type float --dist_fn l2
    --index_path_prefix "${DATA_DIR}/sift1m_index"
    --query_file "${DATA_DIR}/sift_query.fbin"
    --gt_file "${DATA_DIR}/sift_groundtruth.bin"
    --recall_at 10 --search_list 100 --beamwidth 2
    --num_threads 16 --num_nodes_to_cache 0
    --enable_query_sector_cache
  )

  run_baseline small_y "${SMALL_BASE[@]}"
  run_baseline 1m "${M1M_BASE[@]}"

  for spec in "${LAYOUTS[@]}"; do
    letter="${spec%%:*}"
    layout="${spec##*:}"
    run_case small_y "${letter}" "${layout}" "${SMALL_COMMON[@]}"
  done

  for spec in "${LAYOUTS[@]}"; do
    letter="${spec%%:*}"
    layout="${spec##*:}"
    run_case 1m "${letter}" "${layout}" "${M1M_COMMON[@]}"
  done

  echo "===== small_y L=50 W=4 (no memory) ====="
  printf "%-10s %12s %12s %12s %12s %10s\n" "Layout" "DiskReads" "BasePages" "DiskCachePg" "MeritDcHit" "Recall"
  print_summary_row baseline "${OUT_ROOT}/small_y/baseline.out" small_y
  for spec in "${LAYOUTS[@]}"; do
    letter="${spec%%:*}"
    print_summary_row "$letter" "${OUT_ROOT}/small_y/${letter}.out" small_y
  done

  echo ""
  echo "===== 1M L=100 W=2 (no memory, disk0.1 for B/C/E/P) ====="
  printf "%-10s %12s %12s %12s %12s %10s\n" "Layout" "DiskReads" "BasePages" "DiskCachePg" "MeritDcHit" "Recall"
  print_summary_row baseline "${OUT_ROOT}/1m/baseline.out" 1m
  for spec in "${LAYOUTS[@]}"; do
    letter="${spec%%:*}"
    print_summary_row "$letter" "${OUT_ROOT}/1m/${letter}.out" 1m
  done

  echo ""
  echo "Results: ${OUT_ROOT}"
} 2>&1 | tee "${LOG}"
