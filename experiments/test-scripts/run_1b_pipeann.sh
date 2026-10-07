#!/usr/bin/env bash
# PipeANN (libaio build, read-only) on the SIFT1B core workloads, same pinning and thread count as run_1b_perturbed.sh.
# pipe = PipeANN pipelined search (mode 2), coro = coroutine inter-query search (mode 3).
# Usage: run_1b_pipeann.sh [workload ...]   (default: all core workloads)
set -u
cd "$(dirname "$0")/../.."
export PATH=/home/jianz/miniconda3/bin:$PATH
BIN=${BIN:-/home/jianz/workload/code/baselines/PipeANN/build_aio/tests/search_disk_index}
INDEX=${INDEX:-/mnt/graid_single/sift1b/pipeann/sift1b}
ROOT=${ROOT:-/mnt/graid_single/sift1b/runs/pipeann}
THREADS=${THREADS:-20}
MEM_L=${MEM_L:-0}
TAG=${TAG:-}
P=/home/jianz/workload/data/sift1b/workloads/perturbed
X=/home/jianz/workload/data/sift1b/workloads/mixed
declare -A WL=(
  [zipf_a1p2]=$P/zipf_a1p2_1m_s24 [zipf_a1p0]=$P/zipf_a1p0_1m_s24
  [zipf_a0p8]=$P/zipf_a0p8_1m_s24 [zipf_a0p6]=$P/zipf_a0p6_1m_s24
  [uniform]=$P/uniform_1m_s24 [union]=$X/union_a1p2_uniform_2m_s24
)
ORDER=${*:-"zipf_a1p2 zipf_a1p0 zipf_a0p8 zipf_a0p6 uniform union"}
MODES=${MODES:-"pipe:2:32:90 100 coro:3:4:100"}

run_mode() { # $1=out $2=mode $3=width $4..=Ls
  local out=$1 mode=$2 width=$3; shift 3
  rm -rf "$out"; mkdir -p "$out"
  env OMP_PROC_BIND=close "OMP_PLACES={0}:$THREADS" LD_LIBRARY_PATH=/home/jianz/miniconda3/lib \
    PIPEANN_RESULT_PREFIX="$out/result" /usr/bin/time -f "RSS_KB=%M usr=%U sys=%S wall=%e" \
    "$BIN" uint8 "$INDEX" "$THREADS" "$width" "$QUERY" null 10 l2 pq "$mode" "$MEM_L" "$@" > "$out/run.out" 2>&1
}

for w in $ORDER; do
  base=${WL[$w]}
  QUERY=$base.u8bin
  root=$ROOT/${w}${TAG}
  if [ -f "$root/recall_qps.json" ]; then echo "skip $w"; continue; fi
  echo "==== $w $(date -Is)"
  # MODES entries look like "name:mode:width:L1 L2" separated by the next "name:" token.
  for spec in $(echo "$MODES" | grep -oE '[a-z]+:[0-9]+:[0-9]+:[0-9 ]+' | tr ' ' '_'); do
    IFS=: read -r name mode width ls <<< "$spec"
    run_mode "$root/$name" "$mode" "$width" ${ls//_/ }
  done
  (cd experiments/scripts && python collect_pipeann.py --root "$root" --pos "${base}_sample_pos.npy" \
    --gt "${base}_sample_gt10.bin") | sed "s/^/$w /"
done
echo ALLDONE
