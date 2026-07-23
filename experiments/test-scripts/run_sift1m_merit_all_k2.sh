#!/usr/bin/env bash
# Re-run MERIT benchmark suite: disk cache k-hop=2, same trace profile.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
export DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
export DISK_K_HOPS="${DISK_K_HOPS:-2}"
export PROFILE="${PROFILE:-${DATA_DIR}/run2_profile_same_trace}"
export SKIP_RUN2="${SKIP_RUN2:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
export OUT_DIR="${OUT_DIR:-${DATA_DIR}/rerun_k2_${STAMP}}"
LOG="${DATA_DIR}/run_merit_all_k2_${STAMP}.log"

echo "=== MERIT full rerun k_hops=${DISK_K_HOPS} ===" | tee "${LOG}"
echo "profile=${PROFILE} OUT_DIR=${OUT_DIR}" | tee -a "${LOG}"

cmake --build "${DISKANN_BUILD}" -j"$(nproc)" --target search_disk_index 2>&1 | tee -a "${LOG}"

run_sub() {
  local name="$1"
  shift
  echo "" | tee -a "${LOG}"
  echo ">>> ${name}" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

export OUT_DIR="${OUT_DIR}/five_way"
run_sub "five_way" bash "${SCRIPT_DIR}/run_sift1m_merit_five_way.sh"

export OUT_DIR="${DATA_DIR}/rerun_k2_${STAMP}/vs_diskann"
run_sub "vs_diskann_cache" bash "${SCRIPT_DIR}/run_sift1m_merit_vs_diskann_cache.sh"

export OUT_DIR="${DATA_DIR}/rerun_k2_${STAMP}/same_trace"
export LOG="${DATA_DIR}/run_merit_same_trace_k2_${STAMP}.log"
run_sub "same_trace" bash "${SCRIPT_DIR}/run_sift1m_merit_same_trace.sh"

export LOG="${DATA_DIR}/run_merit_combined_k2_${STAMP}.log"
run_sub "combined" bash "${SCRIPT_DIR}/run_sift1m_merit_combined.sh"

export LOG="${DATA_DIR}/run_merit_full_verify_k2_${STAMP}.log"
run_sub "full_verify" bash "${SCRIPT_DIR}/run_sift1m_merit_full_verify.sh"

export LOG="${DATA_DIR}/run_merit_eviction_k2_${STAMP}.log"
run_sub "eviction" bash "${SCRIPT_DIR}/run_sift1m_merit_eviction.sh"

export LOG="${DATA_DIR}/run_merit_mem001_disk01_k2_${STAMP}.log"
run_sub "mem001_disk01" bash "${SCRIPT_DIR}/run_sift1m_merit_mem001_disk01.sh"

export OUT_DIR="${DATA_DIR}/rerun_k2_${STAMP}/five_way_disk02"
export DISK_RATIO=0.2
export LOG="${DATA_DIR}/run_merit_five_way_disk02_k2_${STAMP}.log"
run_sub "five_way_disk_ratio_0.2" bash "${SCRIPT_DIR}/run_sift1m_merit_five_way.sh"

echo "" | tee -a "${LOG}"
echo "All done. Master log: ${LOG}" | tee -a "${LOG}"
