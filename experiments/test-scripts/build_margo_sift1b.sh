#!/usr/bin/env bash
# MARGO on SIFT1B end to end, unattended and resumable (every stage leaves a marker):
#   1. validate the out-of-core path on SIFT10M: sharded weighted build (already in $V) laid out by the original
#      new_mincut and by ooc_mincut; the two page layouts must give the same recall (|diff| <= 1%)
#   2. SIFT1B sharded weighted Vamana (R=64 L=100) + weighted merge + disk layout
#   3. ooc_mincut (256 clusters) -> ooc_relayout
#   4. shared PQ / samples / Starling navigation graph symlinks + smoke search
# Progress: $OUT/status.txt ; logs next to it.
set -euo pipefail
export PATH=/home/jianz/miniconda3/bin:$PATH
MB=/home/jianz/workload/code/baselines/MARGO/build
RELAYOUT=/home/jianz/workload/code/baselines/starling/graph_partition/build/ooc_relayout
T=${T:-24}

V=/mnt/raid5/tmp_margo10m
V_BASE=/home/jianz/workload/data/sift10m/base.1B.u8bin.crop_nb_10000000
V_FBASE=/mnt/raid5/graid_offload/sift10m/base_10m.fbin
V_SHARED=/mnt/raid5/graid_offload/sift10m/sift10m_index
V_NAV=/mnt/raid5/graid_offload/sift10m/starling/mem/_index
V_Q=/home/jianz/workload/data/sift10m/query.public.10K.u8bin
V_GT=/home/jianz/workload/data/sift10m/bigann-10M

OUT=/mnt/raid5/sift1b/margo
P=$OUT/sift1b
BASE=/mnt/raid5/sift1b/base.1B.u8bin
Q=/mnt/raid5/sift1b/query.public.10K.u8bin
GT=/mnt/raid5/sift1b/GT.public.1B.ibin
SHARED=/mnt/nvme_single/sift1b
NAV=/mnt/nvme_single/sift1b/starling/mem/_index
RAM_GB=${RAM_GB:-24}

mkdir -p "$OUT"
status() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }

link_shared() {  # <prefix> <shared index prefix> <pivots>
  ln -sfn "$2_pq_compressed.bin" "$1_pq_compressed.bin"
  ln -sfn "$3" "$1_pq_pivots.bin"
  ln -sfn "$2_sample_data.bin" "$1_sample_data.bin"
  ln -sfn "$2_sample_ids.bin" "$1_sample_ids.bin"
}

search() {  # <prefix> <nav> <query> <gt> <log>
  env LD_LIBRARY_PATH=/home/jianz/miniconda3/lib OMP_NUM_THREADS=20 \
    "$MB/tests/search_disk_index" --data_type uint8 --dist_fn l2 --index_path_prefix "$1" --query_file "$3" \
    --gt_file "$4" -K 10 -L 50 100 -W 4 -T 20 --num_nodes_to_cache 0 --mem_L 10 --mem_index_path "$2" \
    --use_page_search 1 --disk_file_path "$1_disk.index" --result_path "$(dirname "$1")/res" > "$5" 2>&1
  rm -f "$(dirname "$1")"/res_*
}

recall_at() {  # <log> <L> -> recall column of that L row
  awk -v L="$2" '$1==L && NF>=8 {print $NF; exit}' "$1"
}

# ---------------- 1. 10M validation ----------------
if [ ! -f "$V/validate.done" ]; then
  status "10M validation start"
  for k in ooc orig; do
    mkdir -p "$V/$k"
    for f in _mem.index _mem.index.weights _disk.index; do ln -sfn "$V/s10m$f" "$V/$k/s10m$f"; done
  done
  [ -f "$V/ooc/s10m_partition.bin" ] || /usr/bin/time -v "$MB/my_gp/ooc_mincut" "$V/ooc/s10m" "$V_BASE" uint8 \
    "$V/ooc/work" $T 20 > "$V/ooc/mincut.log" 2>&1
  [ -f "$V/orig/s10m_partition.bin" ] || /usr/bin/time -v "$MB/my_gp/new_mincut" "$V/orig/s10m" "$V_FBASE" \
    > "$V/orig/mincut.log" 2>&1
  for k in ooc orig; do
    d=$V/$k/s10m
    if [ -L "${d}_disk.index" ]; then
      "$RELAYOUT" "$V/s10m_disk.index" "${d}_partition.bin" "${d}_disk_relayout.index" $T > "$V/$k/relayout.log" 2>&1
      rm "${d}_disk.index"
      mv "${d}_disk_relayout.index" "${d}_disk.index"
    fi
    link_shared "$d" "$V_SHARED" "${V_SHARED}_pq_pivots_v5.bin"
    search "$d" "$V_NAV" "$V_Q" "$V_GT" "$V/$k/search.log"
  done
  r_ooc=$(recall_at "$V/ooc/search.log" 100); r_orig=$(recall_at "$V/orig/search.log" 100)
  status "10M recall@L100: ooc_mincut=$r_ooc new_mincut=$r_orig"
  if ! awk -v a="$r_ooc" -v b="$r_orig" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(a>0 && d<=1.0)}'; then
    status "VALIDATION FAILED, stop"; exit 1
  fi
  touch "$V/validate.done"
fi

# ---------------- 2. 1B weighted graph + disk layout ----------------
if [ ! -f "$OUT/build.done" ]; then
  status "1B sharded weighted build start (RAM budget ${RAM_GB} GB)"
  # one process per shard (exit code 2 = shard done, more remain) so swapped-out pages do not pile up
  while :; do
    rc=0
    MARGO_ONE_SHARD_PER_PROCESS=1 /usr/bin/time -v "$MB/tests/my_build_merged_disk_index" uint8 "$BASE" "$P" \
      64 100 "$RAM_GB" $T >> "$OUT/build.log" 2>&1 || rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -eq 2 ] || { status "1B build failed rc=$rc"; exit $rc; }
  done
  touch "$OUT/build.done"
  status "1B build done"
fi

# ---------------- 3. mincut + relayout ----------------
if [ ! -f "${P}_partition.bin" ]; then
  status "1B ooc_mincut start"
  /usr/bin/time -v "$MB/my_gp/ooc_mincut" "$P" "$BASE" uint8 "$OUT/mincut_work" $T 36 >> "$OUT/mincut.log" 2>&1
  status "1B ooc_mincut done"
fi
rm -rf "$OUT/mincut_work"
if [ ! -f "$OUT/relayout.done" ]; then
  status "1B relayout start"
  "$RELAYOUT" "${P}_disk.index" "${P}_partition.bin" "${P}_disk_relayout.index" $T > "$OUT/relayout.log" 2>&1
  # the pre-layout disk index can be regenerated from _mem.index with create_disk_layout
  mv "${P}_disk_relayout.index" "${P}_disk.index"
  touch "$OUT/relayout.done"
  status "1B relayout done"
fi

# ---------------- 4. shared files + smoke test ----------------
link_shared "$P" "$SHARED/sift1b_index" "$SHARED/pipeann/sift1b_pq_pivots.bin"
ln -sfn "$SHARED/sift1b_index_disk.index" "${P}_disk_beam_search.index"
status "1B smoke search start"
search "$P" "$NAV" "$Q" "$GT" "$OUT/smoke.log"
status "1B smoke recall@L50=$(recall_at "$OUT/smoke.log" 50) recall@L100=$(recall_at "$OUT/smoke.log" 100)"
status "ALLDONE"
