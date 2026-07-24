#!/usr/bin/env bash
# Five-way compare (same query trace + profile):
#   1) baseline
#   2) baseline + DiskANN BFS static cache (~N nodes, default ≈ mem 0.01GB)
#   3) MERIT memory 0.01GB
#   4) MERIT disk cache ratio 0.1 only
#   5) MERIT memory 0.01GB + disk 0.1 (exclude memory from disk tier)
#
# Disk-cache page packing: DISK_K_HOPS (CLI --merit_disk_cache_k_hops), default 2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
QUERY_FILE="${QUERY_FILE:-${DATA_DIR}/sift_query.fbin}"
PROFILE="${PROFILE:-${DATA_DIR}/run2_profile_same_trace}"
N_NODES="${N_NODES:-12843}"
MEM_GB="${MEM_GB:-0.01}"
DISK_RATIO="${DISK_RATIO:-0.1}"
DISK_K_HOPS="${DISK_K_HOPS:-2}"
DISK_LAYOUT="${DISK_LAYOUT:-node}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/five_way_runs}"
LOG="${LOG:-${DATA_DIR}/run_merit_five_way.log}"

SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
THREADS="${THREADS:-16}"
L="${L:-100}"
K="${K:-10}"
W="${W:-2}"

extract_row() { grep -E "^\s+${L}\s+${W}\s+" "$1" | tail -1 || true; }

COMMON=(
  --data_type float --dist_fn l2
  --index_path_prefix "${DATA_DIR}/sift1m_index"
  --query_file "${QUERY_FILE}"
  --gt_file "${DATA_DIR}/sift_groundtruth.bin"
  --recall_at "${K}" --search_list "${L}" --beamwidth "${W}"
  --num_threads "${THREADS}"
  --enable_query_sector_cache
)

mkdir -p "${OUT_DIR}"
: > "${LOG}"

{
  echo "Five-way MERIT compare $(date -Is)"
  echo "query=${QUERY_FILE} profile=${PROFILE}"
  echo "BFS N_NODES=${N_NODES} MEM_GB=${MEM_GB} DISK_RATIO=${DISK_RATIO} DISK_K_HOPS=${DISK_K_HOPS} DISK_LAYOUT=${DISK_LAYOUT} L=${L} W=${W} threads=${THREADS}"
  echo ""

  run_case() {
    local tag="$1"
    shift
    local out="${OUT_DIR}/${tag}.out"
    echo "======== ${tag} ========"
    "$SEARCH" "$@" >"${out}" 2>&1
    grep -E 'Caching|MERIT memory pool|rank_skip|MERIT disk-cache ready|unified' "${out}" 2>/dev/null | head -4 || true
    extract_row "${out}" | sed 's/^/  /'
    echo ""
  }

  run_case baseline \
    "${COMMON[@]}" --result_path "${OUT_DIR}/baseline" --num_nodes_to_cache 0

  run_case bfs \
    "${COMMON[@]}" --result_path "${OUT_DIR}/bfs" --num_nodes_to_cache "${N_NODES}"

  run_case mem001 \
    "${COMMON[@]}" --result_path "${OUT_DIR}/mem001" --num_nodes_to_cache 0 \
    --merit_memory_gb "${MEM_GB}" --merit_profile_prefix "${PROFILE}"

  run_case disk01 \
    "${COMMON[@]}" --result_path "${OUT_DIR}/disk01" --num_nodes_to_cache 0 \
    --merit_disk_cache_ratio "${DISK_RATIO}" --merit_profile_prefix "${PROFILE}" \
    --merit_disk_cache_k_hops "${DISK_K_HOPS}" --merit_disk_cache_layout "${DISK_LAYOUT}"

  run_case mem001_disk01 \
    "${COMMON[@]}" --result_path "${OUT_DIR}/mem001_disk01" --num_nodes_to_cache 0 \
    --merit_memory_gb "${MEM_GB}" --merit_disk_cache_ratio "${DISK_RATIO}" \
    --merit_profile_prefix "${PROFILE}" --merit_disk_cache_exclude_memory true \
    --merit_disk_cache_k_hops "${DISK_K_HOPS}" --merit_disk_cache_layout "${DISK_LAYOUT}"

  echo "===== TABLE (L=${L} W=${W}) ====="
  printf "%-22s %10s %12s %10s %12s %8s %8s\n" "Case" "QPS" "Latency_us" "MeanIOs" "MeanIO_us" "Recall" "MeritDC"
  declare -A LABEL=(
    [baseline]=baseline
    [bfs]=baseline+BFS
    [mem001]=memory_0.01
    [disk01]=disk_0.1
    [mem001_disk01]=mem0.01+disk0.1
  )
  for tag in baseline bfs mem001 disk01 mem001_disk01; do
    row=$(extract_row "${OUT_DIR}/${tag}.out")
    if [[ -z "$row" ]]; then
      printf "%-22s %s\n" "${LABEL[$tag]}" "(no row)"
      continue
    fi
    qps=$(echo "$row" | awk '{print $3}')
    lat=$(echo "$row" | awk '{print $4}')
    ios=$(echo "$row" | awk '{print $6}')
    nf=$(echo "$row" | awk '{print NF}')
    recall=$(echo "$row" | awk '{print $NF}')
    if [[ "$nf" -ge 14 ]]; then
      ious=$(echo "$row" | awk '{print $(NF-3)}')
      mdc=$(echo "$row" | awk '{print $(NF-4)}')
    else
      ious=$(echo "$row" | awk '{print $(NF-2)}')
      mdc="-"
    fi
    printf "%-22s %10s %12s %10s %12s %8s %8s\n" "${LABEL[$tag]}" "$qps" "$lat" "$ios" "$ious" "$recall" "$mdc"
  done
  echo ""
  echo "Outputs: ${OUT_DIR}/*.out"
  echo "Done $(date -Is)"
} 2>&1 | tee "${LOG}"
