#!/usr/bin/env bash
# Fair-ish compare: same ~0.01GB DRAM hot nodes
#   A) baseline (no cache)
#   B) DiskANN static: BFS medoid + load_cache_list (num_nodes_to_cache=N)
#   C) MERIT: profile Top-N + memory pool (merit_memory_gb=0.01), no runtime admit
#
# N defaults to 12843 (= 0.01GB / ~836B per node on SIFT1M float). Override with N_NODES=.
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
N_NODES="${N_NODES:-12843}"
LOG="${LOG:-${PERSIST_DATA_DIR}/run_merit_vs_diskann_cache.log}"

SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
THREADS="${THREADS:-16}"
L="${L:-100}"
K="${K:-10}"
W="${W:-2}"
MERIT_GB="${MERIT_GB:-0.01}"

extract_row() { grep -E "^\s+${L}\s+${W}\s+" "$1" | tail -1; }

: > "${LOG}"
{
  echo "MERIT vs DiskANN static cache (~${MERIT_GB}GB / N=${N_NODES}) $(date -Is)"

  run_one() {
    local tag="$1"; shift
    local out="${DATA_DIR}/cmp_${tag}.out"
    echo ""
    echo "======== ${tag} ========"
    "$SEARCH" "$@" >"$out" 2>&1
    grep -E 'Caching|MERIT memory pool|loading Top' "$out" | head -3 || true
    extract_row "$out" | sed 's/^/  /'
  }

  COMMON=(--data_type float --dist_fn l2 --index_path_prefix "${DATA_DIR}/sift1m_index"
    --query_file "${DATA_DIR}/sift_query.fbin" --gt_file "${DATA_DIR}/sift_groundtruth.bin"
    --recall_at "${K}" --search_list "${L}" --beamwidth "${W}" --num_threads "${THREADS}")

  run_one baseline "${COMMON[@]}" --result_path "${DATA_DIR}/cmp_baseline" --num_nodes_to_cache 0

  run_one diskann_bfs "${COMMON[@]}" --result_path "${DATA_DIR}/cmp_diskann_bfs" --num_nodes_to_cache "${N_NODES}"

  run_one merit_pool "${COMMON[@]}" --result_path "${DATA_DIR}/cmp_merit_pool" --num_nodes_to_cache 0 \
    --merit_memory_gb "${MERIT_GB}" --merit_profile_prefix "${PROFILE}"

  echo ""
  echo "===== TABLE ====="
  printf "%-18s %10s %12s %10s %12s %8s\n" "Config" "QPS" "Latency_us" "MeanIOs" "MeanIO_us" "Recall"
  for tag in baseline diskann_bfs merit_pool; do
    row=$(extract_row "${DATA_DIR}/cmp_${tag}.out")
    qps=$(echo "$row" | awk '{print $3}')
    lat=$(echo "$row" | awk '{print $4}')
    ios=$(echo "$row" | awk '{print $6}')
    ious=$(echo "$row" | awk '{print $(NF-2)}')
    rec=$(echo "$row" | awk '{print $NF}')
    printf "%-18s %10s %12s %10s %12s %8s\n" "$tag" "$qps" "$lat" "$ios" "$ious" "$rec"
  done
  echo ""
  echo "Note: DiskANN=BFS medoid selection; MERIT=Run2 node_expand Top-N (same node count, different nodes)."
} 2>&1 | tee "${LOG}"
