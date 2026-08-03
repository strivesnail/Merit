#!/usr/bin/env bash
# small_y: Layout E — disk cache 0.1 vs memory 0.01 + disk cache 0.1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
PERSIST_DATA_DIR="${DATA_DIR:-/home/jianz/workplace/diskann/anndisk/data/sift1m}"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
PROFILE="${PROFILE:-${PERSIST_DATA_DIR}/workloads/profiles/small_y_10k}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_ROOT:-${PERSIST_DATA_DIR}/workloads_disk01_runs/e_mem001_vs_disk01_${STAMP}}"
mkdir -p "${OUT}/disk_only" "${OUT}/mem001_disk01"

COMMON=(
  --data_type float --dist_fn l2
  --index_path_prefix "${PERSIST_DATA_DIR}/sift1m_index"
  --query_file "${PERSIST_DATA_DIR}/workloads/small_y_10k.fbin"
  --gt_file "${PERSIST_DATA_DIR}/workloads/small_y_10k_gt.bin"
  --recall_at 1 --search_list 50 --beamwidth 4
  --num_threads 8 --num_nodes_to_cache 0
  --enable_query_sector_cache
  --merit_profile_prefix "${PROFILE}"
  --merit_disk_cache_ratio 0.1
  --merit_disk_cache_layout directed_beam
  --merit_disk_cache_k_hops 1
)

run_one() {
  local tag="$1"
  shift
  mkdir -p "${OUT}/${tag}"
  echo "== ${tag} =="
  "${SEARCH}" "$@" --result_path "${OUT}/${tag}/run" >"${OUT}/${tag}.out" 2>&1
  grep -E 'MERIT memory|selected_nodes|entries=|^\s+50\s+4\s+' "${OUT}/${tag}.out" | tail -5
}

{
  echo "E mem0.01+disk0.1 vs disk0.1 $(date -Is)"
  echo "OUT=${OUT}"
  run_one disk_only "${COMMON[@]}"
  run_one mem001_disk01 "${COMMON[@]}" \
    --merit_memory_gb 0.01 --merit_disk_cache_exclude_memory true
  echo ""
  echo "Config             Disk Reads  DiskCachePg   MeritDcHit   Recall@1"
  for tag in disk_only mem001_disk01; do
    row=$(grep -E '^\s+50\s+4\s+' "${OUT}/${tag}.out" | tail -1)
    printf "%-16s %12s %12s %12s %10s\n" "$tag" \
      "$(echo "$row" | awk '{print $10}')" \
      "$(echo "$row" | awk '{print $8}')" \
      "$(echo "$row" | awk '{print $15}')" \
      "$(echo "$row" | awk '{print $NF}')"
  done
} 2>&1 | tee "${OUT}/run.log"
