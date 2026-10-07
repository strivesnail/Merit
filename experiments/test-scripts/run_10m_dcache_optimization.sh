#!/usr/bin/env bash
# Focused regression/ablation for the 12%-to-N point that exposed D-cache overhead.
set -uo pipefail
cd "$(dirname "$0")"

WL=/home/jianz/workload/data/sift10m/workloads/core/zipf_a1p2_1m_s24
ROOT=${ROOT:-/mnt/graid_single/sift10m/runs/dcache_optimization_a1p2}
NEW_SEARCH=${NEW_SEARCH:-/home/jianz/workload/code/Merit/diskann/build/apps/search_disk_index}
OLD_SEARCH=${OLD_SEARCH:-/tmp/search_disk_index.wgate}
NCAP=6835
DPAGES=6721
mkdir -p "$ROOT"

run_one() { # tag, binary, system, writer threads, commit threads, loc shards
  local tag=$1 search=$2 system=$3 writers=$4 commits=$5 shards=$6
  local out="$ROOT/$tag"
  if [ -f "$out/recall_qps.json" ]; then
    echo "skip $tag"
    return
  fi
  echo "$tag writers=$writers commits=$commits loc_shards=$shards $(date -Is)"
  env INDEX=/mnt/graid_single/sift10m/sift10m_index \
      SEARCH="$search" NET_GATE=1 NET_WRITE_WEIGHT=1 MERIT_BFS=1000 \
      NCAP="$NCAP" DPAGES="$DPAGES" UNIQUE=46929 \
      WRITER_THREADS="$writers" COMMIT_THREADS="$commits" LOC_SHARDS="$shards" \
      QUERY="$WL.u8bin" OUTROOT="$out" LS=100 SYSTEMS="$system" \
      ./run_1b_perturbed.sh
  (
    cd ../scripts
    python collect_recall_qps.py --root "$out" \
      --pos "${WL}_sample_pos.npy" --gt "${WL}_sample_gt10.bin"
  )
}

run_one n_only "$NEW_SEARCH" ncache 1 1 64
if [ -x "$OLD_SEARCH" ]; then
  run_one legacy "$OLD_SEARCH" merit 1 1 64
fi
run_one async_w1 "$NEW_SEARCH" merit 1 1 64
run_one async_w2 "$NEW_SEARCH" merit 2 2 64
run_one async_w4 "$NEW_SEARCH" merit 4 4 64

echo ALLDONE
