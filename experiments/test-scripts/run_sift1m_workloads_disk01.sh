#!/usr/bin/env bash
# For each clustered workload (10K queries): Run2 profile on same trace, then baseline vs disk 0.1 (Layout B).
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
WORKLOAD_DIR="${WORKLOAD_DIR:-${DATA_DIR}/workloads}"
INDEX="${INDEX:-${DATA_DIR}/sift1m_index}"
OUT_ROOT="${OUT_ROOT:-${PERSIST_DATA_DIR}/workloads_disk01_runs}"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"

THREADS="${THREADS:-8}"
L="${L:-50}"
W="${W:-4}"
K="${K:-1}"
DISK_RATIO="${DISK_RATIO:-0.1}"
DISK_K_HOPS="${DISK_K_HOPS:-2}"
DISK_LAYOUT="${DISK_LAYOUT:-node}"

WORKLOADS=(
  uniform
  small_x
  small_y
  large_x
  large_y
  x_only
  x_then_y
  y_only
)

extract_row() { grep -E "^\s+${L}\s+${W}\s+" "$1" | tail -1 || true; }

COMMON=(
  --data_type float --dist_fn l2
  --index_path_prefix "${INDEX}"
  --recall_at "${K}" --search_list "${L}" --beamwidth "${W}"
  --num_threads "${THREADS}"
  --num_nodes_to_cache 0
  --enable_query_sector_cache
)

mkdir -p "${OUT_ROOT}"
SUMMARY="${OUT_ROOT}/summary.tsv"
: > "${SUMMARY}"
echo -e "workload\tphase\tqps\tmean_lat\tp99\tmean_io\tdisk_reads\tbase_pages\tdisk_cache_pg\tmerit_hits\trecall" >> "${SUMMARY}"

append_row() {
  local wl="$1" phase="$2" out="$3"
  local row
  row="$(extract_row "${out}")"
  if [[ -z "${row}" ]]; then
    echo -e "${wl}\t${phase}\tMISSING\t-\t-\t-\t-\t-\t-\t-\t-" >> "${SUMMARY}"
    return
  fi
  # baseline: no DiskCachePg/MeritDcHit; disk01: has both
  echo "${row}" | awk -v wl="${wl}" -v phase="${phase}" '{
    rec=$NF
    if (NF >= 17) {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", wl, phase, $3, $4, $5, $6, $11, $7, $8, $14, rec
    } else {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t-\t-\t%s\n", wl, phase, $3, $4, $5, $6, $9, $7, rec
    }
  }' >> "${SUMMARY}"
}

{
  echo "=== Workload disk0.1 suite $(date -Is) ==="
  echo "persistent DATA=${PERSIST_DATA_DIR} active DATA=${DATA_DIR} (tmpfs)"
  echo "workloads=${WORKLOAD_DIR} OUT=${OUT_ROOT}"
  echo "L=${L} W=${W} K=${K} threads=${THREADS} disk=${DISK_RATIO} layout=${DISK_LAYOUT} k_hops=${DISK_K_HOPS}"
  echo ""

  for wl in "${WORKLOADS[@]}"; do
    QF="${WORKLOAD_DIR}/${wl}_10k.fbin"
    GT="${WORKLOAD_DIR}/${wl}_10k_gt.bin"
    PROFILE="${WORKLOAD_DIR}/profiles/${wl}_10k"
    OUT="${OUT_ROOT}/${wl}"
    mkdir -p "${OUT}" "${WORKLOAD_DIR}/profiles"

    if [[ ! -f "${QF}" || ! -f "${GT}" ]]; then
      echo "ERROR: missing ${QF} or ${GT}; run generate_sift1m_cluster_workloads.py first"
      exit 1
    fi

    echo "======== ${wl}: Run2 profile ========"
    if [[ ! -f "${PROFILE}_node_expand.bin" ]]; then
      "${SEARCH}" \
        "${COMMON[@]}" \
        --query_file "${QF}" \
        --gt_file "${GT}" \
        --result_path "${OUT}/run2" \
        --enable_access_profile \
        --access_profile_prefix "${PROFILE}" \
        >"${OUT}/run2_profile.out" 2>&1
      grep -E 'Access profiling|node_expand|^\s+'"${L}"'\s+'"${W}"'' "${OUT}/run2_profile.out" | tail -5 || true
    else
      echo "  reuse profile ${PROFILE}"
    fi

    echo "======== ${wl}: baseline ========"
    "${SEARCH}" \
      "${COMMON[@]}" \
      --query_file "${QF}" \
      --gt_file "${GT}" \
      --result_path "${OUT}/baseline" \
      >"${OUT}/baseline.out" 2>&1
    grep -E 'MERIT disk-cache ready|^\s+'"${L}"'\s+'"${W}"'' "${OUT}/baseline.out" | tail -3 || true
    append_row "${wl}" baseline "${OUT}/baseline.out"

    echo "======== ${wl}: disk ${DISK_RATIO} ========"
    "${SEARCH}" \
      "${COMMON[@]}" \
      --query_file "${QF}" \
      --gt_file "${GT}" \
      --result_path "${OUT}/disk01" \
      --merit_profile_prefix "${PROFILE}" \
      --merit_disk_cache_ratio "${DISK_RATIO}" \
      --merit_disk_cache_layout "${DISK_LAYOUT}" \
      --merit_disk_cache_k_hops "${DISK_K_HOPS}" \
      >"${OUT}/disk01.out" 2>&1
    grep -E 'hot-node|appended|MERIT disk-cache ready|^\s+'"${L}"'\s+'"${W}"'' "${OUT}/disk01.out" | tail -5 || true
    append_row "${wl}" disk01 "${OUT}/disk01.out"
    echo ""
  done

  echo "===== SUMMARY ====="
  column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
} | tee "${OUT_ROOT}/run.log"
