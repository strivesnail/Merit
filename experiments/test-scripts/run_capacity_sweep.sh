#!/usr/bin/env bash
# N and D capacity sweep on the 2M union workload (1M Zipf a=1.2 + 1M uniform, sigma=24).
# Ratios are relative to the 1,046,384 distinct keys of the workload.
# DiskANN BFS caches N + MERIT_BFS + 2*D nodes, counting every M-cache entry as a full node.
set -u
cd "$(dirname "$0")/../.."
export PATH=/home/jianz/miniconda3/bin:$PATH
KEYS=1046384
M=/home/jianz/workload/data/sift1b/workloads/mixed
WL=union_a1p2_uniform_2m_s24
ROOT=/mnt/graid_single/sift1b/runs/capsweep
STATIC=1000
DEFAULT_D=4693
DEFAULT_N=26160

point() { # $1=tag $2=N $3=D $4=systems
  local out=$ROOT/$1
  if [ -f "$out/recall_qps.json" ]; then echo "skip $1"; return; fi
  UNIQUE=$KEYS NCAP=$2 DPAGES=$3 BFS_EQMEM=$(( $2 + STATIC + 2 * $3 )) MERIT_BFS=$STATIC \
    SEARCH=/tmp/search_disk_index.wgate NET_GATE=1 NET_WRITE_WEIGHT=1 SYSTEMS="$4" \
    QUERY=$M/$WL.u8bin OUTROOT=$out LS=100 experiments/test-scripts/run_1b_perturbed.sh > "/tmp/capsweep_$1.log" 2>&1
  (cd experiments/scripts && python collect_recall_qps.py --root "$out" --pos "$M/${WL}_sample_pos.npy" \
    --gt "$M/${WL}_sample_gt10.bin") | sed "s/^/$1 /"
}

for r in 0.001 0.01 0.025 0.1 0.25; do
  n=$(python -c "print(round($KEYS*$r))")
  point "n${r}" "$n" $DEFAULT_D "bfs_eqmem ncache merit"
done
for r in 0.01 0.025 0.05 0.1 0.25 1.0; do
  d=$(python -c "print(round($KEYS*$r/10))")
  point "d${r}" $DEFAULT_N "$d" "bfs_eqmem merit"
done
for r in 0.01 0.025 0.1 0.25; do
  n=$(python -c "print(round($KEYS*$r))")
  point "n${r}_d2x" "$n" $(( DEFAULT_D * 2 )) "merit"
done
echo ALLDONE
