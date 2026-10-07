#!/usr/bin/env bash
# Extra L points for recall-matched comparison on SIFT10M (page search reaches higher recall at the same L).
# Waits for run_10m_compare.sh to finish, then runs MERIT/BFS at L=30,50 and Starling/MARGO at L=15,50.
set -uo pipefail
cd "$(dirname "$0")"
while pgrep -f run_10m_compare.sh > /dev/null; do sleep 30; done
WL=/home/jianz/workload/data/sift10m/workloads/core
R=/mnt/graid_single/sift10m/runs
WORKLOADS=${WORKLOADS:-"zipf_a1p2_1m_s24 zipf_a1p0_1m_s24 zipf_a0p8_1m_s24 zipf_a0p6_1m_s24 uniform_1m_s24 union_a1p2_uniform_2m_s24"}
for w in $WORKLOADS; do
  env INDEX=/mnt/graid_single/sift10m/sift10m_index SEARCH=/tmp/search_disk_index.wgate NET_GATE=1 NET_WRITE_WEIGHT=1 \
    MERIT_BFS=1000 QUERY=$WL/$w.u8bin OUTROOT=$R/diskann/$w LS="30 50" SYSTEMS=merit ./run_1b_perturbed.sh
  env INDEX=/mnt/graid_single/sift10m/sift10m_index BFS_EQMEM=139600 QUERY=$WL/$w.u8bin OUTROOT=$R/diskann/$w \
    LS="30 50" SYSTEMS=bfs_eqmem ./run_1b_perturbed.sh
  env SYS=starling CACHE=30000 MEM_L=10 LS="15 50" TAG=c30000_m10_x WORKLOADS=$w ./run_10m_pagesearch.sh
  env SYS=margo CACHE=29000 MEM_L=10 LS="15 50" TAG=c29000_m10_x WORKLOADS=$w ./run_10m_pagesearch.sh
done
echo ALLDONE_EXTRA
