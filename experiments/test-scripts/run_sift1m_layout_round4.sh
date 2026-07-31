#!/usr/bin/env bash
# 1M round4: directed_beam w=1/2/3, parent_star, dbeam_estar_split vs baseline.
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
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/1m_round4_${STAMP}"
PROFILE="${DATA_DIR}/run2_profile_same_trace"
LOG="${OUT_ROOT}/run.log"
mkdir -p "${OUT_ROOT}"

L=100 W=2 K=10 THREADS=16

BASE=(
  --data_type float --dist_fn l2
  --index_path_prefix "${DATA_DIR}/sift1m_index"
  --query_file "${DATA_DIR}/sift_query.fbin"
  --gt_file "${DATA_DIR}/sift_groundtruth.bin"
  --recall_at "${K}" --search_list "${L}" --beamwidth "${W}"
  --num_threads "${THREADS}"
  --num_nodes_to_cache 0
  --enable_query_sector_cache
  --merit_memory_gb 0.01
  --merit_disk_cache_ratio 0.1
  --merit_profile_prefix "${PROFILE}"
  --merit_disk_cache_exclude_memory true
)

run_case() {
  local tag="$1" layout="$2" kh="$3"
  local out="${OUT_ROOT}/${tag}"
  mkdir -p "${out}"
  echo "======== ${tag} layout=${layout} k_hops=${kh} ========"
  /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
    "${BASE[@]}" \
    --merit_disk_cache_layout "${layout}" \
    --merit_disk_cache_k_hops "${kh}" \
    --result_path "${out}/run" \
    >"${OUT_ROOT}/${tag}.out" 2>&1
  grep -E 'packing|split|selected|^\s+'"${L}"'\s+'"${W}"'\s+' "${OUT_ROOT}/${tag}.out" | tail -3
  echo ""
}

{
  echo "1M layout round4 $(date -Is)"
  run_case dbeam_w1 directed_beam 1
  run_case dbeam_w2 directed_beam 2
  run_case dbeam_w3 directed_beam 3
  run_case parent_star parent_star 1
  run_case desplit dbeam_estar_split 1
  echo "===== SUMMARY L=${L} W=${W} ====="
  printf "%-16s %12s %12s %12s %10s\n" "case" "DiskReads" "DiskCachePg" "MeritDcHit" "Recall"
  for tag in dbeam_w1 dbeam_w2 dbeam_w3 parent_star desplit; do
    row=$(grep -E "^\s+${L}\s+${W}\s+" "${OUT_ROOT}/${tag}.out" | tail -1)
    dr=$(echo "$row" | awk '{print $10}')
    sp=$(echo "$row" | awk '{print $8}')
    md=$(echo "$row" | awk '{print $15}')
    rc=$(echo "$row" | awk '{print $NF}')
    printf "%-16s %12s %12s %12s %10s\n" "$tag" "$dr" "$sp" "$md" "$rc"
  done
  echo "Results: ${OUT_ROOT}"
} 2>&1 | tee "${LOG}"
