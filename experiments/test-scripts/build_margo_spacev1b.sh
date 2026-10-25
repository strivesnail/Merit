#!/usr/bin/env bash
# SPACEV1B MARGO: parallel weighted shards -> merge/disk -> ooc_mincut -> ooc_relayout -> smoke.
# Resumable via markers under $OUT. Reuses DiskANN PQ/samples + Starling nav graph.
# Usage: setsid nohup bash build_margo_spacev1b.sh >> /mnt/raid5/spacev1b/margo/pipeline.out 2>&1 < /dev/null &
set -euo pipefail
export PATH=/home/jianz/miniconda3/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib

MB=/home/jianz/workload/code/baselines/MARGO/build
RELAYOUT=/home/jianz/workload/code/baselines/starling/graph_partition/build/ooc_relayout
REPO=/home/jianz/workload/code/NeighborCache
# 4 parallel shard workers × 12 threads; peak RSS ~10–18GB/shard on 62GB DRAM.
# L=70 + num_rnds=1 targets ≤10h wall clock on 1.4B SPACEV (serial L=100 was ~2–3 days).
WORKERS=${WORKERS:-4}
T=${T:-12}
RAM_GB=${RAM_GB:-24}
L=${L:-70}
R=${R:-64}
export MARGO_NUM_RNDS=${MARGO_NUM_RNDS:-1}

D=/mnt/raid5/spacev1b
BASE=${D}/base.1B.i8bin
DISKANN=${D}/spacev1b_index
STARLING=${D}/starling
NAV=${STARLING}/mem/_index
Q=/mnt/hdd/merit_park/spacev1b/query.bin
GT=/mnt/hdd/merit_park/spacev1b/truth.bin

OUT=${D}/margo
P=${OUT}/spacev1b
STATUS=${OUT}/status.txt
PART_PREFIX=${P}_mem.index_tempFiles

mkdir -p "$OUT"
log() { echo "$(date -Is) $*" | tee -a "$STATUS"; }
need_gb() {
  local avail
  avail=$(df --output=avail -BG /mnt/raid5 | tail -1 | tr -dc 0-9)
  (( avail >= $1 )) || { log "STOP: $2 needs ${1}G free on /mnt/raid5, have ${avail}G"; exit 1; }
}

num_parts() {
  if [[ -f ${PART_PREFIX}_num_parts.txt ]]; then
    tr -dc 0-9 < ${PART_PREFIX}_num_parts.txt
  else
    echo 0
  fi
}

count_done() {
  local nparts=$1 d=0 i
  for ((i=0; i<nparts; i++)); do
    [[ -f ${PART_PREFIX}_subshard-${i}_mem.index.done ]] && d=$((d+1))
  done
  echo "$d"
}

[[ -f ${BASE}.done ]] || { log "STOP: base not ready"; exit 1; }
[[ -f ${D}/verified.done ]] || { log "STOP: DiskANN+Starling not verified yet"; exit 1; }
[[ -f ${NAV} ]] || { log "STOP: missing Starling nav $NAV"; exit 1; }

# ---------------- 1a. parallel weighted shard builds ----------------
if [[ ! -f ${OUT}/build.done ]]; then
  need_gb 700 "MARGO weighted build"
  log "weighted build start (RAM budget ${RAM_GB} GB, T=${T}/worker, WORKERS=${WORKERS}, L=${L}, num_rnds=${MARGO_NUM_RNDS})"

  # Ensure partition exists (single process creates num_parts + id files).
  if [[ ! -f ${PART_PREFIX}_num_parts.txt ]]; then
    log "partition + first shard pass (serial bootstrap)"
    rc=0
    MARGO_ONE_SHARD_PER_PROCESS=1 MARGO_SKIP_MERGE=1 /usr/bin/time -v \
      "$MB/tests/my_build_merged_disk_index" int8 "$BASE" "$P" \
      "$R" "$L" "$RAM_GB" "$T" >> "${OUT}/build.log" 2>&1 || rc=$?
    [[ $rc -eq 0 || $rc -eq 2 ]] || { log "bootstrap failed rc=$rc"; exit "$rc"; }
  fi

  NPARTS=$(num_parts)
  [[ "$NPARTS" -gt 0 ]] || { log "STOP: num_parts=0"; exit 1; }
  log "parallel shard phase: ${NPARTS} shards, $(count_done "$NPARTS") already done"

  worker_loop() {
    local wid=$1 rc
    while :; do
      local done now
      done=$(count_done "$NPARTS")
      [[ "$done" -ge "$NPARTS" ]] && { echo "$(date -Is) worker${wid}: all shards done" >> "${OUT}/worker${wid}.log"; return 0; }
      rc=0
      MARGO_ONE_SHARD_PER_PROCESS=1 MARGO_SKIP_MERGE=1 \
        "$MB/tests/my_build_merged_disk_index" int8 "$BASE" "$P" \
        "$R" "$L" "$RAM_GB" "$T" >> "${OUT}/worker${wid}.log" 2>&1 || rc=$?
      if [[ $rc -eq 0 ]]; then
        return 0
      fi
      if [[ $rc -ne 2 ]]; then
        echo "$(date -Is) worker${wid}: failed rc=$rc" >> "${OUT}/worker${wid}.log"
        return "$rc"
      fi
      now=$(count_done "$NPARTS")
      echo "$(date -Is) worker${wid}: shard checkpoint ${now}/${NPARTS}" >> "${OUT}/worker${wid}.log"
      # If nothing new and others hold locks, brief wait.
      [[ "$now" -eq "$done" ]] && sleep 15
    done
  }

  pids=()
  for ((w=0; w<WORKERS; w++)); do
    worker_loop "$w" &
    pids+=($!)
    log "started worker${w} pid=${pids[-1]}"
    # Stagger starts so they claim different shards and avoid simultaneous peak RSS.
    sleep 20
  done

  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  [[ $fail -eq 0 ]] || { log "STOP: a shard worker failed"; exit 1; }

  DONE_N=$(count_done "$NPARTS")
  [[ "$DONE_N" -eq "$NPARTS" ]] || { log "STOP: only ${DONE_N}/${NPARTS} shards done"; exit 1; }
  log "all ${NPARTS} shards done, starting merge + disk layout"

  # ---------------- 1b. merge + disk layout (single process) ----------------
  # No SKIP_MERGE: skips finished shards, runs weighted merge + create_disk_layout.
  /usr/bin/time -v \
    "$MB/tests/my_build_merged_disk_index" int8 "$BASE" "$P" \
    "$R" "$L" "$RAM_GB" 48 >> "${OUT}/merge.log" 2>&1 \
    || { log "STOP: merge/disk layout failed"; exit 1; }

  [[ -s ${P}_disk.index ]] || { log "STOP: no ${P}_disk.index"; exit 1; }
  rm -f ${P}_mem.index ${P}_mem.index.data ${P}_mem.index.weights 2>/dev/null || true
  touch ${OUT}/build.done
  log "weighted build done $(du -sh ${P}_disk.index | cut -f1)"
fi

# ---------------- 2. ooc_mincut ----------------
if [[ ! -f ${P}_partition.bin ]]; then
  need_gb 200 "MARGO ooc_mincut work"
  mkdir -p ${OUT}/mincut_work
  log "ooc_mincut start"
  /usr/bin/time -v "$MB/my_gp/ooc_mincut" "$P" "$BASE" int8 "${OUT}/mincut_work" 40 40 \
    >> "${OUT}/mincut.log" 2>&1
  [[ -s ${P}_partition.bin ]] || { log "STOP: no partition.bin"; exit 1; }
  rm -rf ${OUT}/mincut_work
  log "ooc_mincut done $(du -sh ${P}_partition.bin | cut -f1)"
fi

# ---------------- 3. relayout ----------------
if [[ ! -f ${OUT}/relayout.done ]]; then
  need_gb 520 "MARGO relayout (temp second copy)"
  log "relayout start"
  "$RELAYOUT" "${P}_disk.index" "${P}_partition.bin" "${P}_disk_relayout.index" 48 \
    > "${OUT}/relayout.log" 2>&1
  mv "${P}_disk_relayout.index" "${P}_disk.index"
  touch ${OUT}/relayout.done
  log "relayout done $(du -sh ${P}_disk.index | cut -f1)"
fi

# ---------------- 4. shared PQ / samples + smoke ----------------
python3 ${REPO}/experiments/scripts/pq_pivots_to_v5.py \
  ${DISKANN}_pq_pivots.bin ${OUT}/pq_pivots_v5.bin
ln -sfn ${DISKANN}_pq_compressed.bin ${P}_pq_compressed.bin
ln -sfn ${OUT}/pq_pivots_v5.bin ${P}_pq_pivots.bin
ln -sfn ${DISKANN}_sample_data.bin ${P}_sample_data.bin
ln -sfn ${DISKANN}_sample_ids.bin ${P}_sample_ids.bin
ln -sfn ${DISKANN}_disk.index ${P}_disk_beam_search.index

if [[ ! -f ${OUT}/smoke.done ]]; then
  log "smoke search start"
  env OMP_NUM_THREADS=20 \
    "$MB/tests/search_disk_index" --data_type int8 --dist_fn l2 --index_path_prefix "$P" \
    --query_file "$Q" --gt_file "$GT" -K 10 -L 50 100 -W 4 -T 20 --num_nodes_to_cache 0 \
    --mem_L 10 --mem_index_path "$NAV" --use_page_search 1 \
    --disk_file_path "${P}_disk.index" --result_path "${OUT}/smoke_res" \
    > "${OUT}/smoke.log" 2>&1
  rm -f ${OUT}/smoke_res_*
  local_smoke=$(grep -E '^ +(50|100) ' ${OUT}/smoke.log | awk '{printf "L=%s QPS=%s recall=%s; ", $1, $3, $NF}')
  log "smoke: ${local_smoke}"
  r100=$(awk '$1==100 {print $NF; exit}' ${OUT}/smoke.log)
  [[ -n "$r100" ]] || { log "VERIFY FAIL: no L=100 row"; exit 1; }
  awk -v r="$r100" 'BEGIN{exit !(r+0 > 0.5)}' || { log "VERIFY FAIL: recall=$r100"; exit 1; }
  log "verify OK recall@L100=$r100"
  touch ${OUT}/smoke.done
fi

touch ${OUT}/done
log "MARGO SPACEV1B ALLDONE"
