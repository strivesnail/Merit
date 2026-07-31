#!/usr/bin/env bash
# 1M: base index relayout (directed_beam / edge_star / hotnode) vs MERIT disk cache baseline.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
PERSIST_DATA_DIR="${DATA_DIR:-/home/jianz/workplace/diskann/anndisk/data/sift1m}"
# shellcheck source=merit_ramfs_env.sh disable=SC1091
source "${SCRIPT_DIR}/merit_ramfs_env.sh"
merit_ramfs_activate "${PERSIST_DATA_DIR}"
DATA_DIR="${MERIT_RAMFS_ACTIVE_DATA_DIR}"
INDEX_PREFIX="${DATA_DIR}/sift1m_index"
PROFILE="${DATA_DIR}/run2_profile_same_trace"
BASE_FILE="${PERSIST_DATA_DIR}/sift_base.fbin"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/1m_relayout_${STAMP}"
LOG="${OUT_ROOT}/run.log"
mkdir -p "${OUT_ROOT}"

RELAYOUT="${DISKANN_BUILD}/apps/relayout_disk_index"
APPLY="${DISKANN_BUILD}/apps/apply_disk_permutation"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"

L=100
W=2
K=10
THREADS=16
DISK_RATIO=0.1

COMMON=(
  --data_type float --dist_fn l2
  --query_file "${DATA_DIR}/sift_query.fbin"
  --gt_file "${DATA_DIR}/sift_groundtruth.bin"
  --recall_at "${K}" --search_list "${L}" --beamwidth "${W}"
  --num_threads "${THREADS}"
  --num_nodes_to_cache 0
  --enable_query_sector_cache
)

MERIT=(
  --merit_memory_gb 0.01
  --merit_disk_cache_ratio "${DISK_RATIO}"
  --merit_profile_prefix "${PROFILE}"
  --merit_disk_cache_exclude_memory true
  --merit_disk_cache_layout directed_beam
  --merit_disk_cache_k_hops 1
)

parse_row() {
  grep -E "^\s+${L}\s+${W}\s+" "$1" | tail -1
}

offline_relayout() {
  local layout="$1" kh="$2" order="$3" out_prefix="$4"
  echo "=== offline relayout layout=${layout} k_hops=${kh} ==="
  "${RELAYOUT}" \
    --mem_index "${INDEX_PREFIX}_mem.index" \
    --disk_index "${INDEX_PREFIX}_disk.index" \
    --profile_prefix "${PROFILE}" \
    --output_order "${order}" \
    --layout "${layout}" \
    --k_hops "${kh}"
  "${APPLY}" \
    --data_type float \
    --base_file "${BASE_FILE}" \
    --mem_index "${INDEX_PREFIX}_mem.index" \
    --index_prefix "${INDEX_PREFIX}" \
    --order_file "${order}" \
    --output_prefix "${out_prefix}"
  for suffix in _mem.index _mem.index.data _sample_data.bin _sample_ids.bin; do
    if [[ ! -e "${out_prefix}${suffix}" && -e "${INDEX_PREFIX}${suffix}" ]]; then
      cp -a "${INDEX_PREFIX}${suffix}" "${out_prefix}${suffix}"
    fi
  done
}

run_search() {
  local tag="$1" index="$2" order="${3:-}" merit="${4:-0}"
  local out="${OUT_ROOT}/${tag}"
  mkdir -p "${out}"
  echo "======== search ${tag} index=${index} merit=${merit} ========"
  local extra=()
  if [[ -n "${order}" ]]; then
    extra+=(--relayout_order_prefix "${order}")
  fi
  if [[ "${merit}" == "1" ]]; then
    /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
      "${COMMON[@]}" "${MERIT[@]}" \
      --index_path_prefix "${index}" \
      --result_path "${out}/run" \
      "${extra[@]}" \
      >"${OUT_ROOT}/${tag}.out" 2>&1
  else
    /usr/bin/time -f "elapsed=%es" "${SEARCH}" \
      "${COMMON[@]}" \
      --index_path_prefix "${index}" \
      --result_path "${out}/run" \
      "${extra[@]}" \
      >"${OUT_ROOT}/${tag}.out" 2>&1
  fi
  parse_row "${OUT_ROOT}/${tag}.out" | sed 's/^/  /'
  echo ""
}

{
  echo "1M relayout vs disk cache $(date -Is)"
  echo "OUT=${OUT_ROOT}"

  ORDER_DBEAM="${OUT_ROOT}/order_dbeam_w1"
  INDEX_DBEAM="${OUT_ROOT}/index_dbeam"
  ORDER_NODE="${OUT_ROOT}/order_node_k1"
  INDEX_NODE="${OUT_ROOT}/index_node"

  offline_relayout directed_beam 1 "${ORDER_DBEAM}" "${INDEX_DBEAM}"
  offline_relayout node 1 "${ORDER_NODE}" "${INDEX_NODE}"

  run_search orig_plain "${INDEX_PREFIX}"
  run_search orig_merit "${INDEX_PREFIX}" "" 1
  run_search dbeam_plain "${INDEX_DBEAM}" "${ORDER_DBEAM}"
  run_search dbeam_merit "${INDEX_DBEAM}" "${ORDER_DBEAM}" 1
  run_search node_plain "${INDEX_NODE}" "${ORDER_NODE}"

  echo "===== SUMMARY L=${L} W=${W} ====="
  printf "%-16s %10s %10s %10s %10s %8s\n" "case" "MeanIOs" "BasePages" "DiskReads" "DiskCachePg" "Recall"
  for tag in orig_plain orig_merit dbeam_plain dbeam_merit node_plain; do
    row=$(parse_row "${OUT_ROOT}/${tag}.out")
    ios=$(echo "$row" | awk '{print $6}')
    bp=$(echo "$row" | awk '{print $7}')
    dr=$(echo "$row" | awk '{print $10}')
    sp=$(echo "$row" | awk '{print $8}')
    rc=$(echo "$row" | awk '{print $NF}')
    printf "%-16s %10s %10s %10s %10s %8s\n" "$tag" "$ios" "$bp" "$dr" "$sp" "$rc"
  done
  echo "Done $(date -Is)"
} 2>&1 | tee "${LOG}"
