#!/usr/bin/env bash
# Compare MERIT disk cache: separate disk cache file (_merit_dc.data) vs unified single-file + merged IO batch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
# shellcheck source=merit_ramfs_env.sh disable=SC1091
source "${SCRIPT_DIR}/merit_ramfs_env.sh"
PERSIST_DATA_DIR="${DATA_DIR}"
merit_ramfs_activate "${PERSIST_DATA_DIR}"
DATA_DIR="${MERIT_RAMFS_ACTIVE_DATA_DIR}"
QUERY_FILE="${QUERY_FILE:-${DATA_DIR}/sift_query.fbin}"
PROFILE="${PROFILE:-${DATA_DIR}/run2_profile_same_trace}"
DISK_K_HOPS="${DISK_K_HOPS:-2}"
OUT_DIR="${OUT_DIR:-${PERSIST_DATA_DIR}/unified_disk_bench}"
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
  --num_threads "${THREADS}" --num_nodes_to_cache 0
  --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}"
  --merit_disk_cache_k_hops "${DISK_K_HOPS}"
)

mkdir -p "${OUT_DIR}"

run() {
  local tag="$1"; shift
  local out="${OUT_DIR}/${tag}.out"
  echo "== ${tag} =="
  "$SEARCH" "$@" >"${out}" 2>&1
  grep -E 'unified|MERIT disk-cache ready|rank_skip' "${out}" | head -3 || true
  extract_row "${out}"
}

echo "Unified vs disk cache disk cache bench $(date -Is)"
echo "profile=${PROFILE} query=${QUERY_FILE}"

run disk_cache_file_only \
  "${COMMON[@]}" --result_path "${OUT_DIR}/disk_cache_file" \
  --merit_unified_disk_cache false

run unified_disk_only \
  "${COMMON[@]}" --result_path "${OUT_DIR}/unified_disk" \
  --merit_unified_disk_cache true

run disk_cache_mem001_disk \
  "${COMMON[@]}" --result_path "${OUT_DIR}/disk_cache_both" \
  --merit_memory_gb 0.01 --merit_unified_disk_cache false

run unified_mem001_disk \
  "${COMMON[@]}" --result_path "${OUT_DIR}/unified_both" \
  --merit_memory_gb 0.01 --merit_unified_disk_cache true

echo ""
echo "===== TABLE ====="
printf "%-28s %10s %12s %10s %12s %8s %8s\n" "Case" "QPS" "Latency_us" "MeanIOs" "MeanIO_us" "Recall" "MeritDC"
for tag in disk_cache_file_only unified_disk_only disk_cache_mem001_disk unified_mem001_disk; do
  row=$(extract_row "${OUT_DIR}/${tag}.out")
  qps=$(echo "$row" | awk '{print $3}')
  lat=$(echo "$row" | awk '{print $4}')
  ios=$(echo "$row" | awk '{print $6}')
  recall=$(echo "$row" | awk '{print $NF}')
  nf=$(echo "$row" | awk '{print NF}')
  if [[ "$nf" -ge 14 ]]; then
    ious=$(echo "$row" | awk '{print $(NF-3)}')
    mdc=$(echo "$row" | awk '{print $(NF-4)}')
  else
    ious=$(echo "$row" | awk '{print $(NF-2)}')
    mdc="-"
  fi
  printf "%-28s %10s %12s %10s %12s %8s %8s\n" "$tag" "$qps" "$lat" "$ios" "$ious" "$recall" "$mdc"
done
