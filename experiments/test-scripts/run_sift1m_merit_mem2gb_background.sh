#!/usr/bin/env bash
# 1M smoke / pipeline: MEM_GB=2, disk ratio 0.1, flat + k-hop (while 100M index is not ready).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${DATA_DIR}/merit1m_mem2gb_bg_${STAMP}.log"
# 2GB pool caps at 1M points on this index; BFS uses full point count for ~fair RAM tier
N_NODES="${N_NODES:-1000000}"

run_one() {
  local kh="$1"
  local out="${DATA_DIR}/five_way_mem2gb_kh${kh}_${STAMP}"
  env DATA_DIR="${DATA_DIR}" INDEX_PREFIX="${DATA_DIR}/sift1m_index" \
    PROFILE="${DATA_DIR}/run2_profile_same_trace" \
    QUERY_FILE="${DATA_DIR}/sift_query.fbin" \
    GT_FILE="${DATA_DIR}/sift_groundtruth.bin" \
    MEM_GB=2 DISK_RATIO=0.1 DISK_K_HOPS="${kh}" N_NODES="${N_NODES}" \
    OUT_DIR="${out}" LOG="${out}/run.log" \
    bash "${SCRIPT_DIR}/run_merit_five_way.sh"
}

if [[ "${1:-}" == "--fg" ]]; then
  {
    echo "1M MEM 2GB suite $(date -Is)"
    run_one 0
    run_one 2
    echo "Done $(date -Is)"
  } 2>&1 | tee "${MASTER_LOG}"
else
  chmod +x "${SCRIPT_DIR}/run_merit_five_way.sh" "${SCRIPT_DIR}/run_sift1m_merit_mem2gb_background.sh" 2>/dev/null || true
  nohup bash "${SCRIPT_DIR}/run_sift1m_merit_mem2gb_background.sh" --fg >>"${MASTER_LOG}" 2>&1 &
  echo "Started PID $! log: ${MASTER_LOG}"
fi
