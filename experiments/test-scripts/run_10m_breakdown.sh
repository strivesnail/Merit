#!/usr/bin/env bash
# SIFT10M Zipf-1.2 read breakdown inside the Starling/MARGO code base (same 30000-node cache, 20 threads, W=4):
# beam search on the original layout with/without the navigation graph, page search on the relaid-out index
# with/without it, and MARGO. Per-query CSVs (STARLING_QSTATS) carry reads, cache hits, same-page skips,
# exact evaluations, hops, and the read count at which the final top-1 / top-10 were first evaluated.
set -uo pipefail
W=/home/jianz/workload/data/sift10m/workloads/core/zipf_a1p2_1m_s24
NAV=/mnt/graid_single/sift10m/starling/mem/_index
OUT=/mnt/graid_single/sift10m/runs/breakdown/zipf_a1p2_1m_s24
LS=${LS:-"15 20 30 50 100"}
CONFIGS=${CONFIGS:-"beam_nonav beam_nav page_nonav page_nav margo_page_nav"}
for c in $CONFIGS; do
  case $c in
    beam_nonav) sys=starling ps=0 ml=0 f=_disk_beam_search.index ;;
    beam_nav) sys=starling ps=0 ml=10 f=_disk_beam_search.index ;;
    page_nonav) sys=starling ps=1 ml=0 f=_disk.index ;;
    page_nav) sys=starling ps=1 ml=10 f=_disk.index ;;
    margo_page_nav) sys=margo ps=1 ml=10 f=_disk.index ;;
  esac
  B=/home/jianz/workload/code/baselines/$([ $sys = margo ] && echo MARGO || echo starling)/build/tests/search_disk_index
  P=/mnt/graid_single/sift10m/$sys/sift10m
  d=$OUT/$c
  mkdir -p "$d"
  env LD_LIBRARY_PATH=/home/jianz/miniconda3/lib OMP_NUM_THREADS=20 OMP_PROC_BIND=close "OMP_PLACES={0}:20" \
    STARLING_QSTATS="$d/qstats" "$B" --data_type uint8 --dist_fn l2 --index_path_prefix "$P" --query_file "$W.u8bin" \
    --gt_file null -K 10 -L $LS -W 4 -T 20 --num_nodes_to_cache 30000 --mem_L $ml --mem_index_path "$NAV" \
    --use_page_search $ps --disk_file_path "$P$f" --result_path "$d/result" > "$d/run.out" 2>&1
  rm -f "$d"/result_*_dists_float.bin
  echo "$c done $(date -Is)"
done
