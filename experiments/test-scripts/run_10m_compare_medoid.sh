#!/usr/bin/env bash
# Starling / MARGO on SIFT10M starting from the single medoid (mem_L=0), like DiskANN and MERIT.
# The 38 MB navigation graph budget goes to the node cache: ~86.7 MB / ~1.6 KB per node.
set -uo pipefail
cd "$(dirname "$0")"
R=/mnt/graid_single/sift10m/runs/medoid
WORKLOADS=${WORKLOADS:-"zipf_a1p2_1m_s24 zipf_a1p0_1m_s24 zipf_a0p8_1m_s24 zipf_a0p6_1m_s24 uniform_1m_s24 union_a1p2_uniform_2m_s24"}
for w in $WORKLOADS; do
  env SYS=starling CACHE=54000 MEM_L=0 LS="15 20 30 50 100" OUTROOT=$R/starling WORKLOADS=$w ./run_10m_pagesearch.sh
  env SYS=margo CACHE=52000 MEM_L=0 LS="15 20 30 50 100" OUTROOT=$R/margo WORKLOADS=$w ./run_10m_pagesearch.sh
done
echo ALLDONE
