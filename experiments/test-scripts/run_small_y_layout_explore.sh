#!/usr/bin/env bash
# Explore disk-cache layouts on small_y; compare Disk Reads vs node k=1 baseline (50.52).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
PERSIST_DATA_DIR="${DATA_DIR:-/home/jianz/workplace/diskann/anndisk/data/sift1m}"
# shellcheck source=merit_ramfs_env.sh disable=SC1091
source "${SCRIPT_DIR}/merit_ramfs_env.sh"
merit_ramfs_activate "${PERSIST_DATA_DIR}"
DATA_DIR="${MERIT_RAMFS_ACTIVE_DATA_DIR}"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/small_y/layout_explore"
PROFILE="${DATA_DIR}/workloads/profiles/small_y_10k"
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
  --merit_profile_prefix "${PROFILE}"
  --merit_disk_cache_ratio 0.1
)

run_case() {
  local tag="$1" layout="$2" kh="$3"
  local out="${OUT_ROOT}/${tag}"
  mkdir -p "${out}"
  echo "======== ${tag} layout=${layout} k_hops=${kh} ========"
  /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
    "${COMMON[@]}" \
    --result_path "${out}/run" \
    --merit_disk_cache_layout "${layout}" \
    --merit_disk_cache_k_hops "${kh}" \
    >"${OUT_ROOT}/${tag}.out" 2>&1
  grep -E 'layout=|packing|selected|^\s+50\s+4\s+' "${OUT_ROOT}/${tag}.out" | tail -4
  echo ""
}

{
  echo "small_y layout explore $(date -Is)"
  run_case node_k1 node 1
  run_case edge_k1 edge 1
  run_case edge_k2 edge 2
  run_case frontier_w4 frontier 4
  run_case parent_p parent 1
  run_case edge_dir_k1 edge_dir 1
  run_case edge_star_k1 edge_star 1
  run_case edge_u_k1 edge_u 1
  run_case edge_pair_k1 edge_pair 1
  echo "===== SUMMARY (Disk Reads) ====="
  printf "%-14s %12s %12s %12s\n" "layout" "DiskReads" "DiskCachePg" "MeritDcHit"
  for tag in node_k1 edge_k1 edge_k2 frontier_w4 parent_p edge_dir_k1 edge_star_k1 edge_u_k1 edge_pair_k1; do
    row=$(grep -E '^\s+50\s+4\s+' "${OUT_ROOT}/${tag}.out" | tail -1)
    dr=$(echo "$row" | awk '{print $10}')
    sp=$(echo "$row" | awk '{print $8}')
    md=$(echo "$row" | awk '{print $15}')
    printf "%-14s %12s %12s %12s\n" "$tag" "$dr" "$sp" "$md"
  done
} 2>&1 | tee "${OUT_ROOT}/run.log"
