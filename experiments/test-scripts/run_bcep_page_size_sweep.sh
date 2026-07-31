#!/usr/bin/env bash
# Page-size sweep for Layout B/C/E on small_y (professor's co-location hypothesis).
# Rebuilds _disk.index at 4/8/16 KiB sectors (via create_disk_layout) and compares Disk Reads.
#
# Hypothesis: within-parent hot neighbors are not very skewed; ~5 nodes/page limits co-location.
# If true, doubling/quadruling page size should yield modest Disk Reads gains on B/C/E.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PERSIST_DATA_DIR="${DATA_DIR:-/home/jianz/workplace/diskann/anndisk/data/sift1m}"
# shellcheck source=merit_ramfs_env.sh disable=SC1091
source "${SCRIPT_DIR}/merit_ramfs_env.sh"
if [[ "${MERIT_USE_RAMFS:-1}" == "0" ]]; then
  DATA_DIR="${PERSIST_DATA_DIR}"
  echo "MERIT_USE_RAMFS=0: using persistent DATA_DIR=${DATA_DIR}" >&2
else
  merit_ramfs_activate "${PERSIST_DATA_DIR}"
  DATA_DIR="${MERIT_RAMFS_ACTIVE_DATA_DIR}"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${PERSIST_DATA_DIR}/workloads_disk01_runs/page_size_sweep_${STAMP}"
LOG="${OUT_ROOT}/run.log"
mkdir -p "${OUT_ROOT}"

# 4KB / 8KB / 16KB (SIFT1M max_node_len=772B → ~5 / ~10 / ~21 nodes per sector)
SECTORS=(4096 8192 16384)
LAYOUTS=(B:node C:edge E:directed_beam)
KH="${KH:-1}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DEFAULT_BUILD="${REPO_ROOT}/diskann/build"
OMP_PATH="${OMP_PATH:-/opt/intel/oneapi/compiler/2025.3/lib}"

build_dir_for_sector() {
  local sect="$1"
  if [[ "${sect}" == "4096" ]]; then
    echo "${DEFAULT_BUILD}"
  else
    echo "${REPO_ROOT}/diskann/build_sect${sect}"
  fi
}

SRC_PREFIX="${DATA_DIR}/sift1m_index"
BASE_FILE="${BASE_FILE:-${PERSIST_DATA_DIR}/sift_base.fbin}"
MEM_INDEX="${SRC_PREFIX}_mem.index"

ensure_sector_build() {
  local sect="$1"
  local bdir
  bdir="$(build_dir_for_sector "${sect}")"
  if [[ -x "${bdir}/apps/search_disk_index" && -x "${bdir}/apps/utils/create_disk_layout" ]]; then
    echo "reuse build: ${bdir} (SECTOR_LEN=${sect})"
    return 0
  fi
  if [[ "${SKIP_BUILD}" == "1" ]]; then
    echo "SKIP_BUILD=1 but missing ${bdir}/apps/search_disk_index" >&2
    exit 1
  fi
  echo "=== cmake/build SECTOR_LEN=${sect} -> ${bdir} ==="
  cmake -S "${REPO_ROOT}/diskann" -B "${bdir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDISKANN_SECTOR_LEN="${sect}" \
    -DOMP_PATH="${OMP_PATH}"
  cmake --build "${bdir}" -j"$(nproc)" --target search_disk_index create_disk_layout
}

index_prefix_for_sector() {
  local sect="$1"
  if [[ "${sect}" == "4096" ]]; then
    echo "${DATA_DIR}/sift1m_index"
  else
    local ram="${DATA_DIR}/sift1m_index_s${sect}"
    local disk="${PERSIST_DATA_DIR}/sift1m_index_s${sect}"
    if [[ -f "${disk}_disk.index" ]]; then
      echo "${disk}"
    else
      echo "${ram}"
    fi
  fi
}

ensure_disk_index() {
  local sect="$1"
  if [[ "${sect}" == "4096" ]]; then
    echo "use existing 4KB index: ${SRC_PREFIX}_disk.index"
    return 0
  fi
  local bdir
  bdir="$(build_dir_for_sector "${sect}")"
  local prefix
  prefix="$(index_prefix_for_sector "${sect}")"
  local disk_out="${prefix}_disk.index"

  if [[ -f "${disk_out}" ]]; then
    echo "disk index exists: ${disk_out}"
    return 0
  fi

  # Also accept index already on persistent disk when running without ramfs.
  local persist_out="${PERSIST_DATA_DIR}/sift1m_index_s${sect}_disk.index"
  if [[ "${disk_out}" != "${persist_out}" && -f "${persist_out}" ]]; then
    echo "disk index exists on persistent storage: ${persist_out}"
    return 0
  fi

  echo "=== create_disk_layout SECTOR_LEN=${sect} -> ${disk_out} ==="
  for suffix in pq_compressed.bin pq_pivots.bin mem.index; do
    local dst="${prefix}_${suffix}"
    local src="${SRC_PREFIX}_${suffix}"
    if [[ ! -e "${dst}" ]]; then
      ln -sf "${src}" "${dst}"
    fi
  done

  "${bdir}/apps/utils/create_disk_layout" \
    float "${BASE_FILE}" "${MEM_INDEX}" "${disk_out}"
}

run_case() {
  local sect="$1" letter="$2" layout="$3"
  local bdir
  bdir="$(build_dir_for_sector "${sect}")"
  local prefix
  prefix="$(index_prefix_for_sector "${sect}")"
  local out="${OUT_ROOT}/s${sect}/${letter}"
  mkdir -p "${out}"

  if [[ "${DROP_CACHES:-0}" == "1" ]]; then
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || sync
  fi

  echo "-------- sector=${sect} bytes layout=${layout} (${letter}) index=${prefix}_disk.index --------"
  "${bdir}/apps/search_disk_index" \
    --data_type float --dist_fn l2 \
    --index_path_prefix "${prefix}" \
    --query_file "${DATA_DIR}/workloads/small_y_10k.fbin" \
    --gt_file "${DATA_DIR}/workloads/small_y_10k_gt.bin" \
    --recall_at 1 --search_list 50 --beamwidth 4 \
    --num_threads 8 --num_nodes_to_cache 0 \
    --enable_query_sector_cache \
    --merit_profile_prefix "${DATA_DIR}/workloads/profiles/small_y_10k" \
    --merit_disk_cache_ratio 0.1 \
    --merit_disk_cache_layout "${layout}" \
    --merit_disk_cache_k_hops "${KH}" \
    --result_path "${out}/run" \
    >"${out}.out" 2>&1

  grep -E 'nodes per sector|packing|selected|^\s+50\s+4\s+' "${out}.out" | tail -4
}

parse_dr() {
  local f="$1"
  grep -E '^\s+50\s+4\s+' "$f" | tail -1 | awk '{print $10, $NF}'
}

{
  echo "Page-size sweep B/C/E k_hops=${KH} small_y $(date -Is)"
  echo "OUT=${OUT_ROOT}"
  echo ""

  echo "=== Offline: parent neighbor skew + page_cap coverage ==="
  python3 "${SCRIPT_DIR}/analyze_page_cap_sweep.py" \
    "${DATA_DIR}/workloads/profiles/small_y_10k" 4 \
    | tee "${OUT_ROOT}/profile_page_cap.txt"
  echo ""

  for sect in "${SECTORS[@]}"; do
    ensure_sector_build "${sect}"
    ensure_disk_index "${sect}"
  done

  for sect in "${SECTORS[@]}"; do
    echo ""
    echo "======== sector ${sect} bytes ========"
    for spec in "${LAYOUTS[@]}"; do
      letter="${spec%%:*}"
      layout="${spec##*:}"
      run_case "${sect}" "${letter}" "${layout}"
    done
  done

  echo ""
  echo "===== Summary: Mean Latency / Disk Reads / Recall@1 (small_y L=50 W=4) ====="
  printf "%-8s %-6s %12s %12s %12s %10s\n" "Sector" "Layout" "MeanLat_us" "MeanIO_us" "DiskReads" "Recall@1"
  for sect in "${SECTORS[@]}"; do
    for spec in "${LAYOUTS[@]}"; do
      letter="${spec%%:*}"
      f="${OUT_ROOT}/s${sect}/${letter}.out"
      row=$(grep -E "^\s+50\s+4\s+" "$f" | tail -1)
      ml=$(echo "$row" | awk '{print $4}')
      mio=$(echo "$row" | awk '{print $(NF-2)}')
      dr=$(echo "$row" | awk '{print $10}')
      rc=$(echo "$row" | awk '{print $NF}')
      printf "%-8s %-6s %12s %12s %12s %10s\n" "${sect}B" "${letter}" "${ml}" "${mio}" "${dr}" "${rc}"
    done
  done
  echo ""
  echo "===== Disk Reads only ====="
  printf "%-8s %-6s %12s %12s %10s\n" "Sector" "Layout" "DiskReads" "DiskCachePg" "Recall@1"
  for sect in "${SECTORS[@]}"; do
    for spec in "${LAYOUTS[@]}"; do
      letter="${spec%%:*}"
      f="${OUT_ROOT}/s${sect}/${letter}.out"
      row=$(grep -E '^\s+50\s+4\s+' "$f" | tail -1)
      dr=$(echo "$row" | awk '{print $10}')
      sp=$(echo "$row" | awk '{print $8}')
      rc=$(echo "$row" | awk '{print $NF}')
      printf "%-8s %-6s %12s %12s %10s\n" "${sect}B" "${letter}" "${dr}" "${sp}" "${rc}"
    done
  done
  echo ""
  echo "Results: ${OUT_ROOT}"
} 2>&1 | tee "${LOG}"
