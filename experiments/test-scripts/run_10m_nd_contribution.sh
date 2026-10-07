#!/usr/bin/env bash
# Four-way ablation for Shapley attribution of n-cache and d-cache QPS gains.
# Common to all points: 1000-node static BFS cache, SIFT10M Zipf 1.2, L=100,
# W=4, 20 pinned threads. "Off" n-cache uses one pinned medoid slot.
set -uo pipefail
cd "$(dirname "$0")"

WL=/home/jianz/workload/data/sift10m/workloads/core/zipf_a1p2_1m_s24
ROOT=/mnt/graid_single/sift10m/runs/nd_contribution_a1p2
mkdir -p "$ROOT"

run_point() { # tag, N, D, system
  local tag=$1 n=$2 d=$3 system=$4 out=$ROOT/$1
  if [ -f "$out/recall_qps.json" ]; then
    echo "skip $tag"
    return
  fi
  env INDEX=/mnt/graid_single/sift10m/sift10m_index \
      SEARCH=/tmp/search_disk_index.wgate \
      NET_GATE=1 NET_WRITE_WEIGHT=1 MERIT_BFS=1000 \
      NCAP="$n" DPAGES="$d" UNIQUE=46929 \
      QUERY="$WL.u8bin" OUTROOT="$out" LS=100 SYSTEMS="$system" \
      ./run_1b_perturbed.sh
  (
    cd ../scripts
    python collect_recall_qps.py --root "$out" \
      --pos "${WL}_sample_pos.npy" --gt "${WL}_sample_gt10.bin"
  )
}

# Q00: neither dynamic cache (one n-cache slot is required to pin the medoid).
run_point q00_none 1 0 ncache
# Q10: all extra 18 MiB assigned to n-cache.
run_point q10_n 48359 0 ncache
# Q01: baseline d-cache, n-cache effectively off.
run_point q01_d 1 4693 merit
# Q11: selected final configuration.
run_point q11_both 48359 4693 merit
echo ALLDONE
