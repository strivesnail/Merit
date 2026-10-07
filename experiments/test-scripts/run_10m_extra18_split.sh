#!/usr/bin/env bash
# Split MERIT's extra 18 MiB between n-cache and the coupled m/d-cache side.
# Workload: SIFT10M Zipf alpha=1.2, sigma=24, L=100, W=4, 20 pinned threads.
#
# Calibration from prior capacity sweeps:
#   18 MiB ~= 47,186 n-cache nodes ~= 2,304 d-cache pages.
# m-cache is constrained by the implementation to at least 2 entries per
# d-cache page, so increasing d-cache also increases m-cache.
set -uo pipefail
cd "$(dirname "$0")"

WL=/home/jianz/workload/data/sift10m/workloads/core/zipf_a1p2_1m_s24
ROOT=${ROOT:-/mnt/graid_single/sift10m/runs/extra18_split_a1p2}
SEARCH_BIN=${SEARCH_BIN:-/tmp/search_disk_index.wgate}
BASE_N=1173
BASE_D=4693
EXTRA_N=47186
EXTRA_D=2304

# Percent of the 18 MiB assigned to n-cache.
POINTS=${POINTS:-"0 12 25 37 50 62 75 87 100"}
mkdir -p "$ROOT"

for pct in $POINTS; do
  n=$((BASE_N + (EXTRA_N * pct + 50) / 100))
  d=$((BASE_D + (EXTRA_D * (100 - pct) + 50) / 100))
  tag=$(printf "n%03d" "$pct")
  out=$ROOT/$tag
  if [ -f "$out/recall_qps.json" ]; then
    echo "skip $tag"
    continue
  fi
  echo "$tag pct_n=$pct NCAP=$n DPAGES=$d MCAP=$((2*d)) $(date -Is)"
  env INDEX=/mnt/graid_single/sift10m/sift10m_index \
      SEARCH="$SEARCH_BIN" \
      NET_GATE=1 NET_WRITE_WEIGHT=1 MERIT_BFS=1000 \
      NCAP="$n" DPAGES="$d" UNIQUE=46929 \
      QUERY="$WL.u8bin" OUTROOT="$out" LS=100 SYSTEMS=merit \
      ./run_1b_perturbed.sh
  (
    cd ../scripts
    /home/jianz/miniconda3/bin/python collect_recall_qps.py --root "$out" \
      --pos "${WL}_sample_pos.npy" --gt "${WL}_sample_gt10.bin"
  )
done
echo ALLDONE
