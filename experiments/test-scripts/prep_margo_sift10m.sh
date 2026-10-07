#!/usr/bin/env bash
# MARGO on SIFT10M following my_gp/mcrun.sh: one-shot weighted Vamana build (R=64 L=100, same as our index),
# weighted min-cut page layout (new_mincut, 256 clusters) -> index_relayout.
# The PQ files are replaced by the shared SIFT10M PQ so that every system uses identical compressed vectors.
set -euo pipefail
export PATH=/home/jianz/miniconda3/bin:$PATH
G=/home/jianz/workload/code/baselines/MARGO/build
SRC=/mnt/graid_single/sift10m/sift10m_index
BASE=/home/jianz/workload/data/sift10m/base.1B.u8bin.crop_nb_10000000
FBASE=/mnt/graid_single/sift10m/base_10m.fbin
OUT=/mnt/graid_single/sift10m/margo
P=$OUT/sift10m
T=${T:-24}
mkdir -p "$OUT"

[ -f "$FBASE" ] || python - "$BASE" "$FBASE" <<'EOF'
import sys, numpy as np
n, d = np.fromfile(sys.argv[1], dtype=np.int32, count=2)
x = np.memmap(sys.argv[1], dtype=np.uint8, mode="r", offset=8, shape=(int(n), int(d)))
with open(sys.argv[2], "wb") as f:
    np.array([n, d], dtype=np.int32).tofile(f)
    for b in range(0, int(n), 1_000_000):
        x[b:b + 1_000_000].astype(np.float32).tofile(f)
EOF

if [ ! -f "${P}_mem.index.weights" ]; then
  /usr/bin/time -v "$G/tests/my_build_disk_index" --data_type uint8 --dist_fn l2 --data_path "$BASE" \
    --index_path_prefix "$P" -R 64 -L 100 -B 1.0 -M 50 -T "$T" > "$OUT/build.log" 2>&1
fi

if [ ! -f "${P}_partition.bin" ]; then
  (cd "$G" && /usr/bin/time -v ./my_gp/new_mincut "$P" "$FBASE") > "$OUT/mincut.log" 2>&1
fi
if [ ! -f "${P}_disk_beam_search.index" ]; then
  /usr/bin/time -v "$G/tests/utils/index_relayout" "${P}_disk.index" "${P}_partition.bin" > "$OUT/relayout.log" 2>&1
  mv "${P}_disk.index" "${P}_disk_beam_search.index"
  mv "${P}_partition_tmp.index" "${P}_disk.index"
fi

for f in pq_compressed.bin sample_data.bin sample_ids.bin; do
  [ -L "${P}_$f" ] || { [ -f "${P}_$f" ] && mv "${P}_$f" "${P}_$f.margo"; ln -sf "${SRC}_$f" "${P}_$f"; }
done
# MARGO's PQ loader expects the older 5-section pivot file (with an identity dimension rearrangement).
[ -L "${P}_pq_pivots.bin" ] || mv "${P}_pq_pivots.bin" "${P}_pq_pivots.bin.margo"
ln -sf "${SRC}_pq_pivots_v5.bin" "${P}_pq_pivots.bin"
ls -la "$OUT"
