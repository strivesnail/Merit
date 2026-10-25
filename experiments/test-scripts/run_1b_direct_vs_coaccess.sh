#!/usr/bin/env bash
# SIFT1B Zipf 1.2 (perturbed s24): NEIGHBORCACHE with window co-access (default) vs. co-access restricted to direct
# graph neighbors (NEIGHBORCACHE_REAL_IO_DIRECT=1). Same binary, same sampling and window; only the pair filter differs.
set -uo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-/mnt/raid5/graid_offload/sift1b/runs/direct_vs_coaccess_s24}
export INDEX=/mnt/nvme_single/sift1b/sift1b_index
export QUERY=/home/jianz/workload/data/sift1b/workloads/perturbed/zipf_a1p2_1m_s24.u8bin
export LS=${LS:-"100 50"}
mkdir -p "$OUT"
for mode in coaccess direct; do
  d=0; [ $mode = direct ] && d=1
  env REAL_IO_DIRECT=$d OUTROOT=$OUT/$mode SYSTEMS=neighborcache ./run_1b_perturbed.sh
done
for mode in coaccess direct; do
  for L in $LS; do
    f=$OUT/$mode/neighborcache_L$L/run.out
    echo "$mode L=$L $(grep -E "^\s+$L\s+4\s" "$f" | awk '{print "QPS="$3, "mean_lat_us="$4, "mean_ios="$6, "base_pages="$7, "dcache_pages="$8, "disk_reads="$10}') \
$(grep -o 'window_pairs=[0-9]* adjacent_pairs=[0-9]*' "$f")" | tee -a "$OUT/compare.txt"
  done
done
echo ALLDONE | tee -a "$OUT/compare.txt"
