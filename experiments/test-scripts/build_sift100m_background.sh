#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift100m}"
mkdir -p "${DATA_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${DATA_DIR}/build_bg_${STAMP}.log"
chmod +x "${SCRIPT_DIR}/build_sift100m_index_and_profile.sh" 2>/dev/null || true
if [[ "${1:-}" == "--fg" ]]; then
  bash "${SCRIPT_DIR}/build_sift100m_index_and_profile.sh" 2>&1 | tee "${LOG}"
else
  nohup bash "${SCRIPT_DIR}/build_sift100m_background.sh" --fg >>"${LOG}" 2>&1 &
  echo "Started PID $! log: ${LOG}"
fi
