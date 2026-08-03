#!/usr/bin/env bash
# mem 0.01 + disk 0.1 — E vs pct80/90/100, 8 threads, concise summary only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
PERSIST_DATA_DIR="${DATA_DIR:-/home/jianz/workplace/diskann/anndisk/data/sift1m}"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
PROFILE="${PROFILE:-${PERSIST_DATA_DIR}/workloads/profiles/small_y_10k}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_ROOT:-${PERSIST_DATA_DIR}/workloads_disk01_runs/e_pct_compare_8t_${STAMP}}"
mkdir -p "${OUT}"

COMMON=(
  --data_type float --dist_fn l2
  --index_path_prefix "${PERSIST_DATA_DIR}/sift1m_index"
  --query_file "${PERSIST_DATA_DIR}/workloads/small_y_10k.fbin"
  --gt_file "${PERSIST_DATA_DIR}/workloads/small_y_10k_gt.bin"
  --recall_at 1 --search_list 50 --beamwidth 4
  --num_threads 8 --num_nodes_to_cache 0
  --enable_query_sector_cache
  --merit_profile_prefix "${PROFILE}"
  --merit_memory_gb 0.01
  --merit_disk_cache_ratio 0.1
  --merit_disk_cache_exclude_memory true
  --merit_disk_cache_k_hops 1
)

run_one() {
  local tag="$1" layout="$2"
  mkdir -p "${OUT}/${tag}/run"
  echo "== ${tag} =="
  "${SEARCH}" "${COMMON[@]}" \
    --merit_disk_cache_layout "${layout}" \
    --result_path "${OUT}/${tag}/run" >"${OUT}/${tag}.out" 2>&1
}

{
  echo "mem0.01+disk0.1 E vs pct80/90/100 (8t) $(date -Is)"
  echo "OUT=${OUT}"
  run_one E directed_beam
  run_one pct80 directed_beam_pct80
  run_one pct90 directed_beam_pct90
  run_one pct100 directed_beam_pct100
  echo ""
  printf "%-8s %12s %12s %10s\n" "Layout" "Disk Reads" "DiskCachePg" "Recall@1"
  for tag in E pct80 pct90 pct100; do
    row=$(grep -E '^\s+50\s+4\s+' "${OUT}/${tag}.out" | tail -1)
    printf "%-8s %12s %12s %10s\n" "$tag" \
      "$(echo "$row" | awk '{print $10}')" \
      "$(echo "$row" | awk '{print $8}')" \
      "$(echo "$row" | awk '{print $NF}')"
  done
} 2>&1 | tee "${OUT}/run.log"
