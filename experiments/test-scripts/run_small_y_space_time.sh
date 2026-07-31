#!/usr/bin/env bash
# Space-for-time layout exploration on small_y (assume larger disk budget).
# Baseline: edge_star @ ratio=0.1 (DiskReads 46.06).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
PERSIST_DATA_DIR="${DATA_DIR:-/home/jianz/workplace/diskann/anndisk/data/sift1m}"
# shellcheck source=merit_ramfs_env.sh disable=SC1091
source "${SCRIPT_DIR}/merit_ramfs_env.sh"
merit_ramfs_activate "${PERSIST_DATA_DIR}"
DATA_DIR="${MERIT_RAMFS_ACTIVE_DATA_DIR}"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/small_y/space_time"
PROFILE="${DATA_DIR}/workloads/profiles/small_y_10k"
QF="${DATA_DIR}/workloads/small_y_10k.fbin"
GT="${DATA_DIR}/workloads/small_y_10k_gt.bin"
mkdir -p "${OUT_ROOT}"

run_case() {
  local tag="$1" layout="$2" kh="$3" ratio="$4"
  local out="${OUT_ROOT}/${tag}"
  mkdir -p "${out}"
  echo "======== ${tag} layout=${layout} k_hops=${kh} ratio=${ratio} ========"
  /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
    --data_type float --dist_fn l2 \
    --index_path_prefix "${DATA_DIR}/sift1m_index" \
    --query_file "${QF}" --gt_file "${GT}" \
    --recall_at 1 --search_list 50 --beamwidth 4 \
    --num_threads 8 --num_nodes_to_cache 0 \
    --enable_query_sector_cache \
    --merit_profile_prefix "${PROFILE}" \
    --merit_disk_cache_ratio "${ratio}" \
    --merit_disk_cache_layout "${layout}" \
    --merit_disk_cache_k_hops "${kh}" \
    --result_path "${out}/run" \
    >"${OUT_ROOT}/${tag}.out" 2>&1
  grep -E 'layout=|packing|selected|slots=|duplicates=|^\s+50\s+4\s+' "${OUT_ROOT}/${tag}.out" | tail -5
  echo ""
}

{
  echo "small_y space-for-time explore $(date -Is)"
  # Unique packing, larger budget
  run_case star_r01 edge_star 1 0.1
  run_case star_r03 edge_star 1 0.3
  run_case star_r10 edge_star 1 1.0
  run_case clique_r01 edge_clique 1 0.1
  run_case clique_r03 edge_clique 1 0.3
  # Replica packing (nodes may repeat across pages)
  run_case replica_r01 edge_replica 1 0.1
  run_case replica_r03 edge_replica 1 0.3
  run_case replica_r10 edge_replica 1 1.0
  run_case replica_r20 edge_replica 1 2.0
  echo "===== SUMMARY (Disk Reads) ====="
  printf "%-14s %12s %12s %12s\n" "layout" "DiskReads" "DiskCachePg" "MeritDcHit"
  for tag in star_r01 star_r03 star_r10 clique_r01 clique_r03 replica_r01 replica_r03 replica_r10 replica_r20; do
    row=$(grep -E '^\s+50\s+4\s+' "${OUT_ROOT}/${tag}.out" | tail -1)
    dr=$(echo "$row" | awk '{print $10}')
    sp=$(echo "$row" | awk '{print $8}')
    md=$(echo "$row" | awk '{print $15}')
    printf "%-14s %12s %12s %12s\n" "$tag" "$dr" "$sp" "$md"
  done
} 2>&1 | tee "${OUT_ROOT}/run.log"
