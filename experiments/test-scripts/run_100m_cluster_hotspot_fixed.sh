#!/bin/bash
# Fixed-total (10K) cluster-hotspot workloads on SIFT100M. Two evaluations per workload:
#   same : profile and measure on all 10K queries;
#   split: profile on the first 5K queries, measure on the last 5K.
export PATH=/home/jianz/miniconda3/bin:$PATH
set -uo pipefail
SCRIPTS=/home/jianz/workload/code/NeighborCache/experiments/scripts
TESTS=/home/jianz/workload/code/NeighborCache/experiments/test-scripts
WL=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
ROOT=${ROOT:-/mnt/raid5/sift100m/runs/cluster_hotspot_fixed_$(date +%Y%m%d)}
TAGS=${TAGS:-"x1 x2 x10 x50"}
mkdir -p "$ROOT"
log() { echo "$(date -Is) $*" | tee -a "$ROOT/status.txt"; }

need=()
for t in $TAGS; do
  [ -f "$WL/cluster100_${t}_fixed_gt10.bin" ] || need+=("$WL/cluster100_${t}_fixed.u8bin")
done
if [ ${#need[@]} -gt 0 ]; then
  log "GT for ${need[*]}"
  python "$SCRIPTS/gt_bruteforce_disk_index.py" --index /mnt/raid5/sift100m/sift100m_index_disk.index \
    --npts 100000000 --queries "${need[@]}" || { log "GT failed"; exit 1; }
fi

for t in $TAGS; do
  p=$WL/cluster100_${t}_fixed
  python - "$p" <<'EOF'
import sys, numpy as np
p = sys.argv[1]
def rd(path, dt):
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return int(n), int(d), path
n, d, _ = rd(p + ".u8bin", np.uint8)
q = np.fromfile(p + ".u8bin", dtype=np.uint8, offset=8).reshape(n, d)
gn, gk = np.fromfile(p + "_gt10.bin", dtype=np.int32, count=2)
raw = np.fromfile(p + "_gt10.bin", dtype=np.uint8, offset=8)
ids = raw[:gn * gk * 4].view(np.uint32).reshape(gn, gk)
dists = raw[gn * gk * 4:].view(np.float32).reshape(gn, gk)
h = n // 2
for tag, sl in (("A", slice(0, h)), ("B", slice(h, n))):
    with open(f"{p}_{tag}.u8bin", "wb") as f:
        np.array(q[sl].shape, dtype=np.int32).tofile(f); q[sl].tofile(f)
    with open(f"{p}_{tag}_gt10.bin", "wb") as f:
        np.array(ids[sl].shape, dtype=np.int32).tofile(f); ids[sl].tofile(f); dists[sl].tofile(f)
EOF
  log "same $t"
  QUERY=$p.u8bin GT=${p}_gt10.bin RUNS=$ROOT/same_$t bash "$TESTS/run_100m_official_profiled.sh" > /dev/null 2>&1
  grep -v 'Run1\|DONE' "$ROOT/same_$t/status.txt" | sed "s/^/[same $t] /" | tee -a "$ROOT/status.txt"
  log "split $t"
  QUERY=${p}_B.u8bin GT=${p}_B_gt10.bin PROFILE_QUERY=${p}_A.u8bin PROFILE_GT=${p}_A_gt10.bin RUNS=$ROOT/split_$t \
    bash "$TESTS/run_100m_official_profiled.sh" > /dev/null 2>&1
  grep -v 'Run1\|DONE' "$ROOT/split_$t/status.txt" | sed "s/^/[split $t] /" | tee -a "$ROOT/status.txt"
done
log "FIXED HOTSPOT DONE"
