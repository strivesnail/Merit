#!/usr/bin/env bash
# Starling / MARGO page search on SIFT10M core workloads (synchronous, 20 pinned threads, W=4).
# Usage: SYS=starling|margo CACHE=<nodes> MEM_L=<nav L> LS="..." WORKLOADS="..." run_10m_pagesearch.sh
set -uo pipefail
: "${SYS:?}"
case $SYS in
  starling) BIN=/home/jianz/workload/code/baselines/starling/build/tests/search_disk_index ;;
  margo) BIN=/home/jianz/workload/code/baselines/MARGO/build/tests/search_disk_index ;;
esac
P=/mnt/graid_single/sift10m/$SYS/sift10m
NAV=/mnt/graid_single/sift10m/starling/mem/_index
WL=/home/jianz/workload/data/sift10m/workloads/core
OUTROOT=${OUTROOT:-/mnt/graid_single/sift10m/runs/$SYS}
WORKLOADS=${WORKLOADS:-"zipf_a1p2_1m_s24 zipf_a1p0_1m_s24 zipf_a0p8_1m_s24 zipf_a0p6_1m_s24 uniform_1m_s24 union_a1p2_uniform_2m_s24"}
LS=${LS:-100}
CACHE=${CACHE:-0}
MEM_L=${MEM_L:-10}
THREADS=${THREADS:-20}
W=${W:-4}
TAG=${TAG:-c${CACHE}_m${MEM_L}}
for w in $WORKLOADS; do
  out=$OUTROOT/$w/$TAG
  mkdir -p "$out"
  env LD_LIBRARY_PATH=/home/jianz/miniconda3/lib OMP_NUM_THREADS=$THREADS OMP_PROC_BIND=close "OMP_PLACES={0}:20" \
    /usr/bin/time -f "RSS_KB=%M ELAPSED=%e USER=%U SYS=%S" \
    "$BIN" --data_type uint8 --dist_fn l2 --index_path_prefix "$P" --query_file "$WL/$w.u8bin" --gt_file null \
    -K 10 -L $LS -W "$W" -T "$THREADS" --num_nodes_to_cache "$CACHE" --mem_L "$MEM_L" --mem_index_path "$NAV" \
    --use_page_search 1 --use_ratio 1.0 --disk_file_path "${P}_disk.index" --result_path "$out/result" \
    > "$out/run.out" 2>&1
  rm -f "$out"/result_*_dists_float.bin
  echo "$SYS $w $TAG done $(date -Is)"
done
