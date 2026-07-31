#!/usr/bin/env bash
# SIFT1M: cold-ish run + per-query sector cache; baseline / edge / node relayout.
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
DROP_SCRIPT="${DROP_SCRIPT:-${REPO_ROOT}/scripts/drop_system_caches.sh}"
LOG="${LOG:-${PERSIST_DATA_DIR}/run_1m_sector_cache_compare.log}"

SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
THREADS="${THREADS:-16}"
L="${L:-100}"
K="${K:-10}"
W="${W:-2}"

drop_caches() {
  echo "=== $(date -Is) drop page cache ==="
  if sudo -n bash "${DROP_SCRIPT}" 2>/dev/null; then
    return 0
  fi
  echo "sudo drop_caches unavailable; evicting page cache via memory pressure (no root)."
  python3 <<'PY'
import gc, os
def avail_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0
target = max(0, int(avail_kb() * 1024 * 0.85))
block = 64 * 1024 * 1024
chunks, off = [], 0
while off < target:
    n = min(block, target - off)
    b = bytearray(n); b[0] = 1; b[-1] = 2
    chunks.append(b); off += n
del chunks; gc.collect(); os.sync()
print(f"memory pressure done (~{target // (1024*1024)} MiB touched)")
PY
}

run_search() {
  local label="$1"
  local index_prefix="$2"
  local result_prefix="$3"
  local order_prefix="${4:-}"
  echo ""
  echo "=== $(date -Is) ${label} ==="
  drop_caches
  local extra=(--enable_query_sector_cache)
  if [[ -n "${order_prefix}" ]]; then
    extra+=(--relayout_order_prefix "${order_prefix}")
  fi
  "${SEARCH}" \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix "${index_prefix}" \
    --result_path "${result_prefix}" \
    --query_file "${DATA_DIR}/sift_query.fbin" \
    --gt_file "${DATA_DIR}/sift_groundtruth.bin" \
    --recall_at "${K}" \
    --search_list "${L}" \
    --beamwidth "${W}" \
    --num_threads "${THREADS}" \
    --num_nodes_to_cache 0 \
    "${extra[@]}"
}

: > "${LOG}"
{
  echo "SIFT1M sector-cache + cold comparison, L=${L} W=${W} K=${K} threads=${THREADS}"
  run_search "Run1 baseline (original layout)" \
    "${DATA_DIR}/sift1m_index" \
    "${PERSIST_DATA_DIR}/scache_baseline_results"
  run_search "Run3 edge-count relayout (k_hops=2)" \
    "${DATA_DIR}/sift1m_relayout_index" \
    "${PERSIST_DATA_DIR}/scache_edge_results" \
    "${DATA_DIR}/relayout_order_k2"
  run_search "Run3 node-count relayout (hot-node, k_hops=2)" \
    "${DATA_DIR}/sift1m_hotnode_relayout_index" \
    "${PERSIST_DATA_DIR}/scache_node_results" \
    "${DATA_DIR}/relayout_order_hotnode"
  echo "=== $(date -Is) all searches done ==="
} 2>&1 | tee "${LOG}"
