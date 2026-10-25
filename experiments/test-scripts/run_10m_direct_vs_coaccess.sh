#!/usr/bin/env bash
# SIFT10M: NEIGHBORCACHE co-access restricted to direct graph neighbors (REAL_IO_DIRECT=1) vs. window co-access (0),
# with the run_10m_compare.sh NEIGHBORCACHE settings, on the core workloads.
set -uo pipefail
cd "$(dirname "$0")"
SRC=/mnt/raid5/graid_offload/sift10m
IDX_DIR=/mnt/nvme_single/sift10m
WL=/home/jianz/workload/data/sift10m/workloads/core
OUT=${OUT:-/mnt/raid5/graid_offload/sift10m/runs/direct_vs_coaccess}
WORKLOADS=${WORKLOADS:-"zipf_a1p2_1m_s24 zipf_a0p8_1m_s24 uniform_1m_s24"}
export LS=${LS:-"20 100"}
mkdir -p "$IDX_DIR" "$OUT"
for f in disk.index pq_compressed.bin pq_pivots.bin sample_data.bin sample_ids.bin; do
  [ -f "$IDX_DIR/sift10m_index_$f" ] || cp "$SRC/sift10m_index_$f" "$IDX_DIR/"
done
for w in $WORKLOADS; do
  for mode in direct coaccess; do
    d=1; [ $mode = coaccess ] && d=0
    env REAL_IO_DIRECT=$d INDEX=$IDX_DIR/sift10m_index NET_GATE=1 NET_WRITE_WEIGHT=1 NEIGHBORCACHE_BFS=1000 \
      QUERY=$WL/$w.u8bin OUTROOT=$OUT/$w/$mode SYSTEMS=neighborcache ./run_1b_perturbed.sh > /dev/null
    for L in $LS; do
      f=$OUT/$w/$mode/neighborcache_L$L/run.out
      echo "$w $mode L=$L $(grep -E "^\s+$L\s+4\s" "$f" | awk '{print "QPS="$3, "lat_us="$4, "disk_reads="$10, "dcache_pages="$8, "dc_hit="$15}') \
$(grep -o 'direct_only=[a-z]* window_pairs=[0-9]* adjacent_pairs=[0-9]*' "$f")" | tee -a "$OUT/compare.txt"
    done
  done
done
echo ALLDONE | tee -a "$OUT/compare.txt"
