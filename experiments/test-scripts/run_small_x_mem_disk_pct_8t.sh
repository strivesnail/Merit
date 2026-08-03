#!/usr/bin/env bash
# small_x 8t: baseline | mem0.01 | mem+disk directed_beam/pct80/90/100
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
PERSIST_DATA_DIR="${DATA_DIR:-/home/jianz/workplace/diskann/anndisk/data/sift1m}"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
PROFILE="${PROFILE:-${PERSIST_DATA_DIR}/workloads/profiles/small_x_10k}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_ROOT:-${PERSIST_DATA_DIR}/workloads_disk01_runs/small_x_mem_disk_pct_8t_${STAMP}}"
mkdir -p "${OUT}"

BASE=(
  --data_type float --dist_fn l2
  --index_path_prefix "${PERSIST_DATA_DIR}/sift1m_index"
  --query_file "${PERSIST_DATA_DIR}/workloads/small_x_10k.fbin"
  --gt_file "${PERSIST_DATA_DIR}/workloads/small_x_10k_gt.bin"
  --recall_at 1 --search_list 50 --beamwidth 4
  --num_threads 8 --num_nodes_to_cache 0
  --enable_query_sector_cache
)

run_one() {
  local tag="$1"
  shift
  mkdir -p "${OUT}/${tag}/run"
  echo "== ${tag} =="
  "${SEARCH}" "${BASE[@]}" "$@" \
    --result_path "${OUT}/${tag}/run" >"${OUT}/${tag}.out" 2>&1
}

print_row() {
  local label="$1" file="$2"
  local row hdr
  row=$(grep -E '^\s+50\s+4\s+' "${file}" | tail -1)
  hdr=$(grep -E 'L   Beamwidth' "${file}" | tail -1)
  if [[ -z "${row}" ]]; then
    printf "%-28s %s\n" "${label}" "(no row)"
    return
  fi
  if [[ "${hdr}" == *DiskCachePg* ]]; then
    printf "%-28s %12s %10s\n" "${label}" \
      "$(echo "$row" | awk '{print $10}')" \
      "$(echo "$row" | awk '{print $NF}')"
  else
    printf "%-28s %12s %10s\n" "${label}" \
      "$(echo "$row" | awk '{print $9}')" \
      "$(echo "$row" | awk '{print $NF}')"
  fi
}

{
  echo "small_x mem/disk/pct compare 8t $(date -Is)"
  echo "OUT=${OUT}"

  run_one baseline
  run_one mem_only \
    --merit_profile_prefix "${PROFILE}" \
    --merit_memory_gb 0.01
  run_one mem_disk_beam \
    --merit_profile_prefix "${PROFILE}" \
    --merit_memory_gb 0.01 \
    --merit_disk_cache_ratio 0.1 \
    --merit_disk_cache_exclude_memory true \
    --merit_disk_cache_k_hops 1 \
    --merit_disk_cache_layout directed_beam
  run_one mem_disk_pct80 \
    --merit_profile_prefix "${PROFILE}" \
    --merit_memory_gb 0.01 \
    --merit_disk_cache_ratio 0.1 \
    --merit_disk_cache_exclude_memory true \
    --merit_disk_cache_k_hops 1 \
    --merit_disk_cache_layout directed_beam_pct80
  run_one mem_disk_pct90 \
    --merit_profile_prefix "${PROFILE}" \
    --merit_memory_gb 0.01 \
    --merit_disk_cache_ratio 0.1 \
    --merit_disk_cache_exclude_memory true \
    --merit_disk_cache_k_hops 1 \
    --merit_disk_cache_layout directed_beam_pct90
  run_one mem_disk_pct100 \
    --merit_profile_prefix "${PROFILE}" \
    --merit_memory_gb 0.01 \
    --merit_disk_cache_ratio 0.1 \
    --merit_disk_cache_exclude_memory true \
    --merit_disk_cache_k_hops 1 \
    --merit_disk_cache_layout directed_beam_pct100

  echo ""
  printf "%-28s %12s %10s\n" "Config" "Disk Reads" "Recall@1"
  print_row baseline "${OUT}/baseline.out"
  print_row "mem only" "${OUT}/mem_only.out"
  print_row "mem+disk (directed_beam)" "${OUT}/mem_disk_beam.out"
  print_row "mem+disk (pct80)" "${OUT}/mem_disk_pct80.out"
  print_row "mem+disk (pct90)" "${OUT}/mem_disk_pct90.out"
  print_row "mem+disk (pct100)" "${OUT}/mem_disk_pct100.out"
} 2>&1 | tee "${OUT}/run.log"
