#!/bin/bash
# SIFT1B official 10K queries with a profiling pass: every NEIGHBORCACHE system first runs the whole query set
# once, then the same set is measured. DiskANN BFS has a static cache and needs no profiling pass.
export PATH=/home/jianz/miniconda3/bin:$PATH
set -uo pipefail
SCRIPTS=/home/jianz/workload/code/NeighborCache/experiments/scripts
TESTS=/home/jianz/workload/code/NeighborCache/experiments/test-scripts
INDEX=/mnt/nvme_single/sift1b/sift1b_index
Q=/home/jianz/workload/data/sift1b/workloads/official/query.public.10K
RUNS=${RUNS:-/mnt/nvme_single/sift1b/runs/official_profiled_$(date +%Y%m%d)}
BIGD=${BIGD:-50000}
STATUS=$RUNS/status.txt
mkdir -p "$RUNS"
log() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

collect() { # $1=root
  python "$SCRIPTS/collect_recall_qps.py" --root "$1" --pos "${Q}_sample_pos.npy" --gt "${Q}_gt10.bin" \
    | tee "$1/summary_recall.txt" | tee -a "$STATUS"
}

log "default caches"
INDEX=$INDEX QUERY=$Q.u8bin OUTROOT=$RUNS/default LS=100 SKIP=0 WARMUP_Q=10000 \
  bash "$TESTS/run_1b_perturbed.sh" >> "$RUNS/run.log" 2>&1
collect "$RUNS/default"
log "neighborcache DPAGES=$BIGD"
INDEX=$INDEX QUERY=$Q.u8bin OUTROOT=$RUNS/d$BIGD LS=100 SKIP=0 WARMUP_Q=10000 SYSTEMS=neighborcache DPAGES=$BIGD \
  bash "$TESTS/run_1b_perturbed.sh" >> "$RUNS/run.log" 2>&1
collect "$RUNS/d$BIGD"
log "PROFILED DONE"
