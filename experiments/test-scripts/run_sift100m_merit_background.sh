#!/usr/bin/env bash
# Background MERIT five-way for 100M-scale index (MEM 2GB, disk ratio 0.1).
# Prereqs:
#   - ${DATA_DIR}/${INDEX_PREFIX}_disk.index  (~76 GiB for 100M SIFT)
#   - query / gt / profile from same trace (baseline profile run first)
#
# Usage:
#   ./run_sift100m_merit_background.sh          # start nohup (flat + k-hop=2)
#   ./run_sift100m_merit_background.sh --fg     # foreground
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift100m}"
INDEX_PREFIX="${INDEX_PREFIX:-${DATA_DIR}/sift100m_index}"
QUERY_FILE="${QUERY_FILE:-${DATA_DIR}/sift_query.u8bin}"
GT_FILE="${GT_FILE:-${DATA_DIR}/sift_groundtruth.bin}"
PROFILE="${PROFILE:-${DATA_DIR}/run_profile_same_trace}"
MEM_GB="${MEM_GB:-2}"
DISK_RATIO="${DISK_RATIO:-0.1}"
THREADS="${THREADS:-16}"
# ~2GB MERIT mem pool @ ~836 B/node ≈ 2.4M nodes; override after `plan` if needed
N_NODES="${N_NODES:-2400000}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${DATA_DIR}"
MASTER_LOG="${DATA_DIR}/merit100m_bg_${STAMP}.log"
RUNNER="${SCRIPT_DIR}/run_merit_five_way.sh"

if [[ ! -x "${RUNNER}" ]]; then
  chmod +x "${RUNNER}" "${SCRIPT_DIR}/run_sift100m_merit_background.sh" 2>/dev/null || true
fi

run_suite() {
  local kh="$1"
  local label="$2"
  local out="${DATA_DIR}/five_way_mem${MEM_GB}gb_disk${DISK_RATIO}_kh${kh}_${STAMP}"
  echo "=== DISK_K_HOPS=${kh} (${label}) OUT_DIR=${out} ==="
  env DATA_DIR="${DATA_DIR}" INDEX_PREFIX="${INDEX_PREFIX}" QUERY_FILE="${QUERY_FILE}" \
    GT_FILE="${GT_FILE}" PROFILE="${PROFILE}" MEM_GB="${MEM_GB}" DISK_RATIO="${DISK_RATIO}" \
    DISK_K_HOPS="${kh}" N_NODES="${N_NODES}" THREADS="${THREADS}" \
    DATA_TYPE=uint8 DIST_FN=l2 OUT_DIR="${out}" \
    LOG="${out}/run.log" bash "${RUNNER}"
}

main() {
  if [[ ! -f "${INDEX_PREFIX}_disk.index" ]]; then
    echo "ERROR: 100M index not found: ${INDEX_PREFIX}_disk.index" >&2
    echo "Build index + same-trace profile first, then re-run." >&2
    exit 1
  fi
  echo "MERIT 100M background suite $(date -Is)" | tee -a "${MASTER_LOG}"
  echo "index=${INDEX_PREFIX} MEM_GB=${MEM_GB} DISK_RATIO=${DISK_RATIO} N_NODES(BFS)=${N_NODES}" | tee -a "${MASTER_LOG}"
  run_suite 0 "flat" 2>&1 | tee -a "${MASTER_LOG}"
  run_suite 2 "k-hop" 2>&1 | tee -a "${MASTER_LOG}"
  echo "All done $(date -Is)" | tee -a "${MASTER_LOG}"
}

if [[ "${1:-}" == "--fg" ]]; then
  main
else
  if [[ ! -f "${INDEX_PREFIX}_disk.index" ]]; then
    echo "ERROR: 100M index not found: ${INDEX_PREFIX}_disk.index" >&2
    echo "Build index + profile, then re-run." >&2
    exit 1
  fi
  nohup bash "${SCRIPT_DIR}/run_sift100m_merit_background.sh" --fg >>"${MASTER_LOG}" 2>&1 &
  echo "Started PID $! master log: ${MASTER_LOG}"
  echo "Tail: tail -f ${MASTER_LOG}"
fi
