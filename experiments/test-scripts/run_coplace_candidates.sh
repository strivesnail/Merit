#!/usr/bin/env bash
# Page co-place candidates vs directed_beam on small_y (hop-frontier profile).
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
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/coplace_${STAMP}"
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"

KH=4
PROFILE="${DATA_DIR}/workloads/profiles/small_y_10k_hop"
if [[ ! -f "${PROFILE}_node_expand.bin" ]]; then
  PROFILE="${DATA_DIR}/workloads/profiles/small_y_10k"
  echo "WARN: hop profile missing; cooccur/frontier_topk use synthesized templates" >&2
fi

COMMON=(
  --data_type float --dist_fn l2
  --index_path_prefix "${DATA_DIR}/sift1m_index"
  --query_file "${DATA_DIR}/workloads/small_y_10k.fbin"
  --gt_file "${DATA_DIR}/workloads/small_y_10k_gt.bin"
  --recall_at 1 --search_list 50 --beamwidth 4
  --num_threads 8 --num_nodes_to_cache 0
  --enable_query_sector_cache
  --merit_profile_prefix "${PROFILE}"
  --merit_disk_cache_ratio 0.1
  --merit_disk_cache_k_hops "${KH}"
)

run_one() {
  local tag="$1" layout="$2"
  local out="${OUT_ROOT}/${tag}"
  mkdir -p "${out}"
  echo "======== ${tag} layout=${layout} k=${KH} profile=$(basename "${PROFILE}") ========"
  /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
    "${COMMON[@]}" \
    --merit_disk_cache_layout "${layout}" \
    --result_path "${out}/run" \
    >"${out}.out" 2>&1
  grep -E 'packing|cooccur|top_k|selected|^\s+50\s+4\s+' "${out}.out" | tail -5
  echo ""
}

{
  echo "co-place candidates $(date -Is)"
  echo "OUT=${OUT_ROOT} PROFILE=${PROFILE}"
  run_one ref_dbeam directed_beam
  run_one dir_edge_star dir_edge_star
  run_one dbeam_tight directed_beam_tight
  run_one cooccur cooccur_star
  run_one dbeam_coc dbeam_cooccur
  run_one ftopk frontier_topk

  echo "===== SUMMARY small_y L=50 W=4 (Disk Reads = col 10) ====="
  printf "%-14s %12s %12s %10s\n" "layout" "DiskReads" "DiskCachePg" "Recall@1"
  for tag in ref_dbeam dir_edge_star dbeam_tight cooccur dbeam_coc ftopk; do
    row=$(grep -E '^\s+50\s+4\s+' "${OUT_ROOT}/${tag}.out" | tail -1)
    dr=$(echo "$row" | awk '{print $10}')
    sp=$(echo "$row" | awk '{print $8}')
    rc=$(echo "$row" | awk '{print $NF}')
    printf "%-14s %12s %12s %10s\n" "$tag" "$dr" "$sp" "$rc"
  done
} | tee "${LOG}"

echo "Log: ${LOG}"
