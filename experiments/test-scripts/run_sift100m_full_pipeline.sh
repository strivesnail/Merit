#!/usr/bin/env bash
# SIFT100M: build index → same-trace profile → MERIT five-way (MEM 2GB, disk 0.1, kh0+kh2)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift100m}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER="${DATA_DIR}/pipeline100m_${STAMP}.log"

{
  echo "=== 100M pipeline start $(date -Is) ==="
  bash "${SCRIPT_DIR}/build_sift100m_index_and_profile.sh"
  echo "=== build+profile done, starting MERIT ==="
  bash "${SCRIPT_DIR}/run_sift100m_merit_background.sh" --fg
  echo "=== 100M pipeline all done $(date -Is) ==="
} 2>&1 | tee "${MASTER}"
