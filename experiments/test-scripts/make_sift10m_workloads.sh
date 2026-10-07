#!/usr/bin/env bash
# SIFT10M core workloads, generated the same way as the SIFT1B ones:
# one 100K-node BFS region, finite Zipf (1M queries) at alpha 1.2/1.0/0.8/0.6, 1M uniform queries,
# Gaussian noise sigma=24, exact top-10 GT on 10K sampled positions, and the 2M Zipf-1.2 + uniform union.
set -euo pipefail
cd "$(dirname "$0")/../scripts"
export PATH=/home/jianz/miniconda3/bin:$PATH
D=/home/jianz/workload/data/sift10m
BASE=$D/base.1B.u8bin.crop_nb_10000000
INDEX=/mnt/graid_single/sift10m/sift10m_index_disk.index
OUT=$D/workloads/core
mkdir -p "$OUT"

[ -f "$OUT/small_regions_1x100k_node_ids.bin" ] || python generate_multiregion_rounds.py --base "$BASE" \
  --disk-index "$INDEX" --out-dir "$OUT" --dataset SIFT10M --rounds 1 --region-size 100000 --seed 42 \
  --output-tag ""

for a in 1p2 1p0 0p8 0p6; do
  [ -f "$OUT/zipf_a${a}_1m.u8bin" ] || python generate_sift100m_finite_zipf_rounds.py --base "$BASE" \
    --region-node-ids "$OUT/small_regions_1x100k_node_ids.bin" --output-prefix "$OUT/zipf_a${a}_1m" \
    --rounds 1 --region-size 100000 --queries-per-round 1000000 --alpha "${a/p/.}"
done

[ -f "$OUT/uniform_1m.u8bin" ] || python - "$BASE" "$OUT/uniform_1m" <<'EOF'
import sys, numpy as np
base, out = sys.argv[1], sys.argv[2]
n, d = np.fromfile(base, dtype=np.int32, count=2)
x = np.memmap(base, dtype=np.uint8, mode="r", offset=8, shape=(int(n), int(d)))
ids = np.random.default_rng(20261005).choice(int(n), size=1_000_000, replace=True)
np.save(out + "_node_ids.npy", ids)
with open(out + ".u8bin", "wb") as f:
    np.array([len(ids), d], dtype=np.int32).tofile(f)
    np.ascontiguousarray(x[np.sort(ids)][np.argsort(np.argsort(ids))]).tofile(f)
print("uniform unique ids", len(np.unique(ids)))
EOF

for w in zipf_a1p2_1m zipf_a1p0_1m zipf_a0p8_1m zipf_a0p6_1m uniform_1m; do
  [ -f "$OUT/${w}_s24.u8bin" ] || python make_perturbed_workload.py --src "$OUT/$w.u8bin" --out "$OUT/${w}_s24" \
    --sigma 24
done

samples=()
for w in zipf_a1p2_1m zipf_a1p0_1m zipf_a0p8_1m zipf_a0p6_1m uniform_1m; do
  [ -f "$OUT/${w}_s24_sample_gt10.bin" ] || samples+=("$OUT/${w}_s24_sample.u8bin")
done
[ ${#samples[@]} -eq 0 ] || python gt_bruteforce_disk_index.py --index "$INDEX" --queries "${samples[@]}" \
  --npts 10000000 --node_len 388 --nnps 10

[ -f "$OUT/union_a1p2_uniform_2m_s24.u8bin" ] || python make_mixed_workload.py --skewed "$OUT/zipf_a1p2_1m_s24" \
  --uniform "$OUT/uniform_1m_s24" --union --out "$OUT/union_a1p2_uniform_2m_s24"
ls -la "$OUT"
