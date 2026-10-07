#!/usr/bin/env bash
# Per-split N-only and D-only runs for pointwise Shapley attribution.
# Combined N+D runs are reused from run_10m_extra18_split.sh.
set -uo pipefail
cd "$(dirname "$0")"

WL=/home/jianz/workload/data/sift10m/workloads/core/zipf_a1p2_1m_s24
ROOT=${ROOT:-/mnt/graid_single/sift10m/runs/nd_contribution_curve_a1p2}
SEARCH_BIN=${SEARCH_BIN:-/tmp/search_disk_index.wgate}
BASE_N=1173
BASE_D=4693
EXTRA_N=47186
EXTRA_D=2304
POINTS=${POINTS:-"0 12 25 37 50 62 75 87 100"}
RUN_N=${RUN_N:-1}
RUN_D=${RUN_D:-1}
mkdir -p "$ROOT"

run_point() { # output directory, N, D, system
  local out=$1 n=$2 d=$3 system=$4
  if [ -f "$out/recall_qps.json" ]; then
    return
  fi
  env INDEX=/mnt/graid_single/sift10m/sift10m_index \
      SEARCH="$SEARCH_BIN" \
      NET_GATE=1 NET_WRITE_WEIGHT=1 MERIT_BFS=1000 \
      NCAP="$n" DPAGES="$d" UNIQUE=46929 \
      QUERY="$WL.u8bin" OUTROOT="$out" LS=100 SYSTEMS="$system" \
      ./run_1b_perturbed.sh
  (
    cd ../scripts
    /home/jianz/miniconda3/bin/python collect_recall_qps.py --root "$out" \
      --pos "${WL}_sample_pos.npy" --gt "${WL}_sample_gt10.bin"
  )
}

for pct in $POINTS; do
  n=$((BASE_N + (EXTRA_N * pct + 50) / 100))
  d=$((BASE_D + (EXTRA_D * (100 - pct) + 50) / 100))
  tag=$(printf "n%03d" "$pct")
  if [ "$RUN_N" != 0 ]; then
    echo "$tag N-only N=$n $(date -Is)"
    run_point "$ROOT/$tag/n_only" "$n" 0 ncache
  fi
  if [ "$RUN_D" != 0 ]; then
    echo "$tag D-only D=$d $(date -Is)"
    run_point "$ROOT/$tag/d_only" 1 "$d" merit
  fi
done
echo ALLDONE
