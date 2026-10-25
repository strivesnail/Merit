#!/bin/bash
# SIFT1B: queries spread uniformly over a graph-local region of R nodes (no hot item),
# compared with equal-memory DiskANN BFS, NEIGHBORCACHE-N, and NEIGHBORCACHE. Waits for the 100M build queue.
# Usage: SIZES="1000 10000 100000" run_1b_region_uniform.sh
export PATH=/home/jianz/miniconda3/bin:$PATH
set -uo pipefail
SCRIPTS=/home/jianz/workload/code/NeighborCache/experiments/scripts
TESTS=/home/jianz/workload/code/NeighborCache/experiments/test-scripts
INDEX=/mnt/nvme_single/sift1b/sift1b_index
W=/home/jianz/workload/data/sift1b/workloads
OUT=$W/region_uniform
# The D-cache replica file is written under RUNS, so RUNS must be on NVMe, not the SATA root disk.
RUNS=${RUNS:-/mnt/nvme_single/sift1b/runs/region_uniform_$(date +%Y%m%d)}
SIZES=${SIZES:-"1000 10000 100000"}
STATUS=$RUNS/status.txt
mkdir -p "$OUT" "$RUNS"
log() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

if [ "${WAIT_BUILD:-1}" = 1 ]; then
  log "waiting for build_baselines_100m.sh"
  while pgrep -f build_baselines_100m.sh > /dev/null; do sleep 300; done
  log "build queue finished"
fi

cd "$SCRIPTS"
samples=()
for R in $SIZES; do
  p=$OUT/region${R}_1m_s24
  [ -f "$p.u8bin" ] || python make_region_uniform_workload.py --index "${INDEX}_disk.index" \
    --region-node-ids "$W/small_regions_5x100k_node_ids.bin" --region-size "$R" --out "$p" | tee -a "$STATUS"
  [ -f "${p}_sample_gt10.bin" ] || samples+=("${p}_sample.u8bin")
done
if [ ${#samples[@]} -gt 0 ]; then
  log "GT for ${samples[*]}"
  python gt_bruteforce_disk_index.py --index "${INDEX}_disk.index" --queries "${samples[@]}" || { log "GT failed"; exit 1; }
fi

for R in $SIZES; do
  p=$OUT/region${R}_1m_s24
  log "run R=$R"
  INDEX=$INDEX QUERY=$p.u8bin OUTROOT=$RUNS/region$R LS=100 bash "$TESTS/run_1b_perturbed.sh" >> "$RUNS/run.log" 2>&1
  python collect_recall_qps.py --root "$RUNS/region$R" --pos "${p}_sample_pos.npy" --gt "${p}_sample_gt10.bin" \
    | tee "$RUNS/region$R/summary_recall.txt" | tee -a "$STATUS"
done
log ALLDONE
