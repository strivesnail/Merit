#!/usr/bin/env bash
# 1M: Layout A (k_hops=0) vs B (k_hops=2), disk cache ratio 50%, mem+disk (same as prior A/B compare)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
INDEX_PREFIX="${DATA_DIR}/sift1m_index"
QUERY_FILE="${DATA_DIR}/sift_query.fbin"
GT_FILE="${DATA_DIR}/sift_groundtruth.bin"
PROFILE="${DATA_DIR}/run2_profile_same_trace"
MEM_GB=0.01
DISK_RATIO=0.5
THREADS=16
L=100 K=10 W=2
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${DATA_DIR}/layout_ab_disk50_${STAMP}"
LOG="${OUT_DIR}/run.log"
mkdir -p "${OUT_DIR}"

extract_row() { grep -E "^\s+${L}\s+${W}\s+" "$1" | tail -1 || true; }

run_case() {
  local tag="$1" kh="$2"
  shift 2
  local out="${OUT_DIR}/${tag}_kh${kh}.out"
  echo "======== ${tag} k_hops=${kh} ========"
  "$SEARCH" \
    --data_type float --dist_fn l2 \
    --index_path_prefix "${INDEX_PREFIX}" \
    --query_file "${QUERY_FILE}" --gt_file "${GT_FILE}" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads "${THREADS}" --enable_query_sector_cache \
    --result_path "${OUT_DIR}/${tag}_kh${kh}" --num_nodes_to_cache 0 \
    "$@" >"${out}" 2>&1
  grep -E 'MERIT disk-cache|flat Top|k-hop packing|selected_nodes' "${out}" | head -5 || true
  extract_row "${out}" | sed 's/^/  /'
  echo ""
}

{
  echo "Layout A/B disk ratio=${DISK_RATIO} 1M $(date -Is) OUT=${OUT_DIR}"
  for kh in 0 2; do
    run_case disk "${kh}" \
      --merit_disk_cache_ratio "${DISK_RATIO}" \
      --merit_profile_prefix "${PROFILE}" \
      --merit_disk_cache_k_hops "${kh}"
    run_case mem_disk "${kh}" \
      --merit_memory_gb "${MEM_GB}" \
      --merit_disk_cache_ratio "${DISK_RATIO}" \
      --merit_profile_prefix "${PROFILE}" \
      --merit_disk_cache_exclude_memory true \
      --merit_disk_cache_k_hops "${kh}"
  done
  echo "===== summary L=${L} W=${W} ====="
  printf "%-12s %-6s %10s %10s %10s %10s %8s\n" "case" "kh" "MeanIOs" "BasePages" "DiskReads" "SidecarPg" "Recall"
  for tag in disk mem_disk; do
    for kh in 0 2; do
      row=$(extract_row "${OUT_DIR}/${tag}_kh${kh}.out")
      [[ -z "$row" ]] && continue
      ios=$(echo "$row" | awk '{print $6}')
      bp=$(echo "$row" | awk '{print $7}')
      dr=$(echo "$row" | awk '{print $10}')
      sp=$(echo "$row" | awk '{print $8}')
      rc=$(echo "$row" | awk '{print $NF}')
      printf "%-12s %-6s %10s %10s %10s %10s %8s\n" "$tag" "$kh" "$ios" "$bp" "$dr" "$sp" "$rc"
    done
  done
  echo "Done $(date -Is)"
} 2>&1 | tee "${LOG}"
