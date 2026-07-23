#!/usr/bin/env bash
# Resume after index build: profile + MERIT (skip build if disk.index exists)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift100m}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${DATA_DIR}/resume100m_${STAMP}.log"

{
  echo "=== resume 100M $(date -Is) ==="
  echo "aio-max-nr=$(cat /proc/sys/fs/aio-max-nr 2>/dev/null) (profile uses SEARCH_THREADS=16)"
  bash "${SCRIPT_DIR}/build_sift100m_index_and_profile.sh"
  bash "${SCRIPT_DIR}/run_sift100m_merit_background.sh" --fg
  echo "=== resume done $(date -Is) ==="
} 2>&1 | tee "${LOG}"
