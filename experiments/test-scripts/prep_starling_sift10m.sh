#!/usr/bin/env bash
# Starling on SIFT10M from the existing DiskANN index (R=64, same PQ), following scripts/run_benchmark.sh:
# LDG graph partition (16 rounds) -> index_relayout -> 1% random-sample in-memory navigation graph (R48 L128 a1.2).
set -euo pipefail
export PATH=/home/jianz/miniconda3/bin:$PATH
S=/home/jianz/workload/code/baselines/starling
SRC=/mnt/graid_single/sift10m/sift10m_index
BASE=/home/jianz/workload/data/sift10m/base.1B.u8bin.crop_nb_10000000
OUT=/mnt/graid_single/sift10m/starling
P=$OUT/sift10m
T=${T:-20}
mkdir -p "$OUT/gp" "$OUT/mem_sample" "$OUT/mem"

for f in pq_compressed.bin sample_data.bin sample_ids.bin; do ln -sf "${SRC}_$f" "${P}_$f"; done
# Starling's PQ loader expects the older 5-section pivot file (with an identity dimension rearrangement).
ln -sf "${SRC}_pq_pivots_v5.bin" "${P}_pq_pivots.bin"
[ -f "${P}_disk_beam_search.index" ] || cp "${SRC}_disk.index" "${P}_disk_beam_search.index"

if [ ! -f "$OUT/gp/_part.bin" ]; then
  /usr/bin/time -v "$S/graph_partition/build/partitioner" --index_file "${P}_disk_beam_search.index" \
    --data_type uint8 --gp_file "$OUT/gp/_part.bin" -T "$T" --ldg_times 16 > "$OUT/gp/_part.bin.log" 2>&1
fi
if [ ! -f "${P}_partition.bin" ]; then
  /usr/bin/time -v "$S/build/tests/utils/index_relayout" "${P}_disk_beam_search.index" "$OUT/gp/_part.bin" \
    > "$OUT/gp/relayout.log" 2>&1
  mv "$OUT/gp/_part_tmp.index" "${P}_disk.index"
  cp "$OUT/gp/_part.bin" "${P}_partition.bin"
fi

if [ ! -f "$OUT/mem/_index" ]; then
  "$S/build/tests/utils/gen_random_slice" uint8 "$BASE" "$OUT/mem_sample/" 0.01 > "$OUT/mem_sample/sample.log" 2>&1
  /usr/bin/time -v "$S/build/tests/build_memory_index" --data_type uint8 --dist_fn l2 \
    --data_path "$OUT/mem_sample/" --index_path_prefix "$OUT/mem/_index" -R 48 -L 128 --alpha 1.2 \
    > "$OUT/mem/build.log" 2>&1
fi
ls -la "$OUT" "$OUT/gp" "$OUT/mem"
