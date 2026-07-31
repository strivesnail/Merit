#!/usr/bin/env bash
# Sequentially try layout candidates vs edge_star baseline (small_y, ratio=0.1).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
PERSIST_DATA_DIR="${DATA_DIR:-/home/jianz/workplace/diskann/anndisk/data/sift1m}"
# shellcheck source=merit_ramfs_env.sh disable=SC1091
source "${SCRIPT_DIR}/merit_ramfs_env.sh"
merit_ramfs_activate "${PERSIST_DATA_DIR}"
DATA_DIR="${MERIT_RAMFS_ACTIVE_DATA_DIR}"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/small_y/dig_try_all"
PROFILE_BASE="${DATA_DIR}/workloads/profiles/small_y_10k"
PROFILE_HOP="${DATA_DIR}/workloads/profiles/small_y_10k_hop"
QF="${DATA_DIR}/workloads/small_y_10k.fbin"
GT="${DATA_DIR}/workloads/small_y_10k_gt.bin"
mkdir -p "${OUT_ROOT}"

COMMON=(
  --data_type float --dist_fn l2
  --index_path_prefix "${DATA_DIR}/sift1m_index"
  --query_file "${QF}" --gt_file "${GT}"
  --recall_at 1 --search_list 50 --beamwidth 4
  --num_threads 8 --num_nodes_to_cache 0
  --enable_query_sector_cache
)

profile_hop() {
  echo "======== STEP 0: profile hop-frontier templates ========"
  if [[ -f "${PROFILE_HOP}_frontier_templates.bin" ]]; then
    echo "reuse ${PROFILE_HOP}_frontier_templates.bin"
    return 0
  fi
  mkdir -p "${OUT_ROOT}/profile_hop"
  "${SEARCH}" \
    "${COMMON[@]}" \
    --merit_disk_cache_ratio 0 \
    --enable_access_profile \
    --access_profile_prefix "${PROFILE_HOP}" \
    --result_path "${OUT_ROOT}/profile_hop/run" \
    >"${OUT_ROOT}/profile_hop.out" 2>&1
  grep -E "Saved frontier|Saved access" "${OUT_ROOT}/profile_hop.out" | tail -3
}

run_case() {
  local tag="$1" layout="$2" kh="$3" profile="$4"
  local out="${OUT_ROOT}/${tag}"
  mkdir -p "${out}"
  echo "======== ${tag} layout=${layout} k=${kh} profile=$(basename "${profile}") ========"
  /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
    "${COMMON[@]}" \
    --merit_profile_prefix "${profile}" \
    --merit_disk_cache_ratio 0.1 \
    --merit_disk_cache_layout "${layout}" \
    --merit_disk_cache_k_hops "${kh}" \
    --result_path "${out}/run" \
    >"${OUT_ROOT}/${tag}.out" 2>&1
  grep -E "layout=|packing|templates|selected|slots=|^\s+50\s+4\s+" "${OUT_ROOT}/${tag}.out" | tail -4
  echo ""
}

{
  echo "dig try-all $(date -Is)"
  profile_hop
  python3 "${SCRIPT_DIR}/analyze_profile_layout.py" "${PROFILE_HOP}" 4 5 || true
  echo ""
  run_case star_k1 edge_star 1 "${PROFILE_BASE}"
  run_case dbeam_w4 directed_beam 4 "${PROFILE_BASE}"
  run_case parent_p parent 1 "${PROFILE_BASE}"
  run_case fpage_syn frontier_page 4 "${PROFILE_BASE}"
  run_case fpage_hop frontier_page 4 "${PROFILE_HOP}"
  run_case frontier_d frontier 4 "${PROFILE_HOP}"
  echo "===== SUMMARY (Disk Reads) ====="
  printf "%-14s %12s %12s %12s\n" "tag" "DiskReads" "DiskCachePg" "MeritDcHit"
  for tag in star_k1 dbeam_w4 parent_p fpage_syn fpage_hop frontier_d; do
    row=$(grep -E '^\s+50\s+4\s+' "${OUT_ROOT}/${tag}.out" | tail -1)
    dr=$(echo "$row" | awk '{print $10}')
    sp=$(echo "$row" | awk '{print $8}')
    md=$(echo "$row" | awk '{print $15}')
    printf "%-14s %12s %12s %12s\n" "$tag" "$dr" "$sp" "$md"
  done
} 2>&1 | tee "${OUT_ROOT}/run.log"
