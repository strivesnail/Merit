#!/usr/bin/env bash
# SIFT10M equal-memory comparison: MERIT, DiskANN BFS cache, Starling, MARGO on the core workloads.
# Memory budget beyond PQ and runtime buffers = MERIT peak RSS - cache-less DiskANN peak RSS (88.7 MB on Zipf 1.2);
# BFS 651 B/node -> 139600 nodes; Starling/MARGO spend 38 MB on the navigation graph and ~1.6 KB per cached node.
set -uo pipefail
cd "$(dirname "$0")"
WL=/home/jianz/workload/data/sift10m/workloads/core
R=/mnt/graid_single/sift10m/runs
WORKLOADS=${WORKLOADS:-"zipf_a1p2_1m_s24 zipf_a1p0_1m_s24 zipf_a0p8_1m_s24 zipf_a0p6_1m_s24 uniform_1m_s24 union_a1p2_uniform_2m_s24"}
LS=${LS:-"20 100"}
SYSTEMS=${SYSTEMS:-"merit bfs starling margo"}
for w in $WORKLOADS; do
  for s in $SYSTEMS; do
    case $s in
      merit) env INDEX=/mnt/graid_single/sift10m/sift10m_index SEARCH=/tmp/search_disk_index.wgate NET_GATE=1 \
               NET_WRITE_WEIGHT=1 MERIT_BFS=1000 QUERY=$WL/$w.u8bin OUTROOT=$R/diskann/$w LS="$LS" SYSTEMS=merit \
               ./run_1b_perturbed.sh ;;
      bfs) env INDEX=/mnt/graid_single/sift10m/sift10m_index BFS_EQMEM=139600 QUERY=$WL/$w.u8bin \
             OUTROOT=$R/diskann/$w LS="$LS" SYSTEMS=bfs_eqmem ./run_1b_perturbed.sh ;;
      starling) env SYS=starling CACHE=30000 MEM_L=10 LS="$LS" WORKLOADS=$w ./run_10m_pagesearch.sh ;;
      margo) env SYS=margo CACHE=29000 MEM_L=10 LS="$LS" WORKLOADS=$w ./run_10m_pagesearch.sh ;;
    esac
  done
done
echo ALLDONE
