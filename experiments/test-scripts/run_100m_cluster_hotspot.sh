#!/bin/bash
# SIFT100M official queries plus a k-means-cell hotspot of distinct queries; same-trace profiling per workload.
export PATH=/home/jianz/miniconda3/bin:$PATH
set -uo pipefail
SCRIPTS=/home/jianz/workload/code/NeighborCache/experiments/scripts
TESTS=/home/jianz/workload/code/NeighborCache/experiments/test-scripts
WL=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
ROOT=${ROOT:-/mnt/raid5/sift100m/runs/cluster_hotspot_$(date +%Y%m%d)}
TAGS=${TAGS:-"x2 x10 x50 half"}
mkdir -p "$ROOT"
log() { echo "$(date -Is) $*" | tee -a "$ROOT/status.txt"; }

need=()
for t in $TAGS; do
  [ -f "$WL/cluster100_${t}_gt10.bin" ] || need+=("$WL/cluster100_${t}.u8bin")
done
if [ ${#need[@]} -gt 0 ]; then
  log "GT for ${need[*]}"
  python "$SCRIPTS/gt_bruteforce_disk_index.py" --index /mnt/raid5/sift100m/sift100m_index_disk.index \
    --npts 100000000 --queries "${need[@]}" || { log "GT failed"; exit 1; }
fi
for t in $TAGS; do
  log "run $t"
  QUERY=$WL/cluster100_$t.u8bin GT=$WL/cluster100_${t}_gt10.bin RUNS=$ROOT/$t \
    bash "$TESTS/run_100m_official_profiled.sh" > /dev/null 2>&1
  cat "$ROOT/$t/status.txt" | grep -v 'Run1\|DONE' | sed "s/^/[$t] /" | tee -a "$ROOT/status.txt"
done
log "CLUSTER HOTSPOT DONE"
