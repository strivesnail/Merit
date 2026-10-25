#!/bin/bash
# Follow-up on the R=1K region-uniform workload: a larger D-cache at sigma=24, and sigma=8.
export PATH=/home/jianz/miniconda3/bin:$PATH
set -uo pipefail
SCRIPTS=/home/jianz/workload/code/NeighborCache/experiments/scripts
TESTS=/home/jianz/workload/code/NeighborCache/experiments/test-scripts
INDEX=/mnt/nvme_single/sift1b/sift1b_index
W=/home/jianz/workload/data/sift1b/workloads
OUT=$W/region_uniform
RUNS=${RUNS:-/mnt/nvme_single/sift1b/runs/region_uniform_20261010}
BIGD=${BIGD:-50000}
STATUS=$RUNS/status.txt
log() { echo "$(date -Is) $*" | tee -a "$STATUS"; }
cd "$SCRIPTS"

collect() { # $1=root $2=workload prefix
  python collect_recall_qps.py --root "$1" --pos "${2}_sample_pos.npy" --gt "${2}_sample_gt10.bin" \
    | tee "$1/summary_recall.txt" | tee -a "$STATUS"
}

p24=$OUT/region1000_1m_s24
log "run R=1000 sigma=24 neighborcache DPAGES=$BIGD"
INDEX=$INDEX QUERY=$p24.u8bin OUTROOT=$RUNS/region1000_d$BIGD LS=100 SYSTEMS=neighborcache DPAGES=$BIGD \
  bash "$TESTS/run_1b_perturbed.sh" >> "$RUNS/run.log" 2>&1
collect "$RUNS/region1000_d$BIGD" "$p24"

p8=$OUT/region1000_1m_s8
[ -f "$p8.u8bin" ] || python make_region_uniform_workload.py --index "${INDEX}_disk.index" \
  --region-node-ids "$W/small_regions_5x100k_node_ids.bin" --region-size 1000 --sigma 8 --out "$p8" | tee -a "$STATUS"
[ -f "${p8}_sample_gt10.bin" ] || python gt_bruteforce_disk_index.py --index "${INDEX}_disk.index" \
  --queries "${p8}_sample.u8bin" || { log "GT failed"; exit 1; }
log "run R=1000 sigma=8"
INDEX=$INDEX QUERY=$p8.u8bin OUTROOT=$RUNS/region1000_s8 LS=100 \
  bash "$TESTS/run_1b_perturbed.sh" >> "$RUNS/run.log" 2>&1
collect "$RUNS/region1000_s8" "$p8"
log "run R=1000 sigma=8 neighborcache DPAGES=$BIGD"
INDEX=$INDEX QUERY=$p8.u8bin OUTROOT=$RUNS/region1000_s8_d$BIGD LS=100 SYSTEMS=neighborcache DPAGES=$BIGD \
  bash "$TESTS/run_1b_perturbed.sh" >> "$RUNS/run.log" 2>&1
collect "$RUNS/region1000_s8_d$BIGD" "$p8"
log "FOLLOWUP DONE"
