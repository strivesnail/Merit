#!/usr/bin/env bash
# SIFT1M: verify MERIT features + performance comparison table.
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
PROFILE="${PROFILE:-${DATA_DIR}/run2_profile}"
DISK_K_HOPS="${DISK_K_HOPS:-2}"
LOG="${LOG:-${PERSIST_DATA_DIR}/run_merit_full_verify.log}"

SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
THREADS="${THREADS:-16}"
L="${L:-100}"
K="${K:-10}"
W="${W:-2}"

extract_row() {
  grep -E "^\s+${L}\s+${W}\s+" "$1" | tail -1 || true
}

run_expect_ok() {
  local tag="$1"
  local out="$2"
  shift 2
  echo ""
  echo "======== ${tag} (expect OK) ========"
  if "$SEARCH" "$@" >"$out" 2>&1; then
    echo "  status: OK"
  else
    echo "  status: FAIL exit=$?"
    tail -5 "$out"
    return 1
  fi
  extract_row "$out" | sed "s/^/  row: /"
}

run_expect_fail() {
  local tag="$1"
  local out="$2"
  shift 2
  echo ""
  echo "======== ${tag} (expect FAIL) ========"
  set +e
  "$SEARCH" "$@" >"$out" 2>&1
  local ec=$?
  set -e
  if [[ "$ec" -ne 0 ]]; then
    echo "  status: OK (exit=$ec)"
    grep -E 'ERROR|Failed|requires' "$out" | tail -3 | sed 's/^/  /'
  else
    echo "  status: UNEXPECTED SUCCESS"
    return 1
  fi
}

COMMON=(
  --data_type float --dist_fn l2
  --index_path_prefix "${DATA_DIR}/sift1m_index"
  --query_file "${DATA_DIR}/sift_query.fbin"
  --gt_file "${DATA_DIR}/sift_groundtruth.bin"
  --recall_at "${K}" --search_list "${L}" --beamwidth "${W}"
  --num_threads "${THREADS}" --num_nodes_to_cache 0
)

TMP="${DATA_DIR}/merit_verify_tmp"
mkdir -p "$TMP"

: > "${LOG}"
{
  echo "MERIT full verify $(date -Is)"
  echo "build=${DISKANN_BUILD} threads=${THREADS} L=${L} W=${W}"

  run_expect_ok "F1 baseline" "${TMP}/f1.out" \
    "${COMMON[@]}" --result_path "${TMP}/f1"

  run_expect_ok "F2 memory pool 0.01GB" "${TMP}/f2.out" \
    "${COMMON[@]}" --result_path "${TMP}/f2" \
    --merit_memory_gb 0.01 --merit_profile_prefix "${PROFILE}"

  run_expect_ok "F3 disk cache ratio 0.1" "${TMP}/f3.out" \
    "${COMMON[@]}" --result_path "${TMP}/f3" \
    --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_expect_ok "F4 memory+disk exclude" "${TMP}/f4.out" \
    "${COMMON[@]}" --result_path "${TMP}/f4" \
    --merit_memory_gb 0.01 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" \
    --merit_disk_cache_exclude_memory true --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_expect_ok "F5 memory+disk overlap" "${TMP}/f5.out" \
    "${COMMON[@]}" --result_path "${TMP}/f5" \
    --merit_memory_gb 0.01 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" \
    --merit_disk_cache_exclude_memory false --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_expect_ok "F6 memory 0.1GB + disk 0.1" "${TMP}/f6.out" \
    "${COMMON[@]}" --result_path "${TMP}/f6" \
    --merit_memory_gb 0.1 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" \
    --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_expect_ok "F7 batch evict after warmup" "${TMP}/f7.out" \
    "${COMMON[@]}" --result_path "${TMP}/f7" \
    --merit_memory_gb 0.1 --merit_disk_cache_ratio 0.1 --merit_profile_prefix "${PROFILE}" \
    --merit_evict_memory_gb 0.01 --merit_evict_disk_ratio 0.01 --merit_disk_cache_k_hops "${DISK_K_HOPS}"

  run_expect_ok "F8 runtime admit (1 thread)" "${TMP}/f8.out" \
    --data_type float --dist_fn l2 \
    --index_path_prefix "${DATA_DIR}/sift1m_index" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" \
    --num_threads 1 --num_nodes_to_cache 0 \
    --result_path "${TMP}/f8" \
    --merit_memory_gb 0.01 --merit_profile_prefix "${PROFILE}" \
    --merit_memory_runtime_admit true

  run_expect_fail "F9 oversize memory budget" "${TMP}/f9.out" \
    "${COMMON[@]}" --result_path "${TMP}/f9" \
    --merit_memory_gb 200 --merit_profile_prefix "${PROFILE}" \
    --merit_host_memory_gb 188 --merit_memory_reserve_gb 8

  run_expect_fail "F10 memory without profile" "${TMP}/f10.out" \
    "${COMMON[@]}" --result_path "${TMP}/f10" \
    --merit_memory_gb 0.01

  echo ""
  echo "===== FUNCTION CHECKS (grep) ====="
  grep -E 'MERIT memory pool loaded|rank_skip|evicted|runtime admit' "${TMP}/f2.out" "${TMP}/f4.out" "${TMP}/f7.out" "${TMP}/f8.out" 2>/dev/null | head -20

  echo ""
  echo "===== PERFORMANCE TABLE (L=${L} W=${W}) ====="
  printf "%-30s %10s %12s %10s %12s %8s %8s\n" "Case" "QPS" "Latency" "MeanIOs" "MeanIO_us" "Recall" "MeritDC"
  declare -A LABEL=(
    [f1]="baseline"
    [f2]="mem 0.01GB pool"
    [f3]="disk ratio 0.1"
    [f4]="mem+disk exclude"
    [f5]="mem+disk overlap"
    [f6]="mem 0.1 + disk 0.1"
    [f7]="batch evict"
    [f8]="runtime admit t=1"
  )
  for f in f1 f2 f3 f4 f5 f6 f7 f8; do
    row=$(extract_row "${TMP}/${f}.out")
    if [[ -z "$row" ]]; then
      printf "%-30s %s\n" "${LABEL[$f]:-$f}" "(no row)"
      continue
    fi
    read -r qps lat lat999 ios pages dup _rest <<< "$row"
    # Parse from full row with awk (handles optional MeritDcHit column)
    qps=$(echo "$row" | awk '{print $3}')
    lat=$(echo "$row" | awk '{print $4}')
    ios=$(echo "$row" | awk '{print $6}')
    nf=$(echo "$row" | awk '{print NF}')
    recall=$(echo "$row" | awk '{print $NF}')
    if [[ "$nf" -ge 14 ]]; then
      ious=$(echo "$row" | awk '{print $(NF-3)}')
      meritdc=$(echo "$row" | awk '{print $(NF-4)}')
    else
      ious=$(echo "$row" | awk '{print $(NF-2)}')
      meritdc="-"
    fi
    printf "%-30s %10s %12s %10s %12s %8s %8s\n" "${LABEL[$f]}" "$qps" "$lat" "$ios" "$ious" "$recall" "$meritdc"
  done

  echo ""
  echo "Done $(date -Is)"
} 2>&1 | tee "${LOG}"
