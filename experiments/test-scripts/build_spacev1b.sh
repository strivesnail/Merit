#!/usr/bin/env bash
# SPACEV1B (1.4B x 100 int8, L2): DiskANN (same params as SIFT1B) -> smoke search -> Starling.
# Every stage leaves a marker and is skipped on rerun; only this script's own intermediates are removed
# (DiskANN _mem.index after the disk layout exists, Starling partition work dir).
# Usage: setsid nohup bash build_spacev1b.sh >> /mnt/raid5/spacev1b/build.out 2>&1 < /dev/null &
set -euo pipefail
export PATH=/home/jianz/miniconda3/bin:$PATH
export LD_LIBRARY_PATH=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
REPO=/home/jianz/workload/code/NeighborCache
SB=/home/jianz/workload/code/baselines/starling
GPB=${SB}/graph_partition/build
T=${T:-24}
D=/mnt/raid5/spacev1b
BASE=${D}/base.1B.i8bin
IDX=${D}/spacev1b_index
Q=/mnt/hdd/merit_park/spacev1b/query.bin
GT=/mnt/hdd/merit_park/spacev1b/truth.bin
STATUS=${D}/build_status.txt

log() { echo "$(date -Is) $*" | tee -a "${STATUS}"; }
need_gb() {  # <GB> <what> [dir]
  local avail dir=${3:-/mnt/raid5}
  avail=$(df --output=avail -BG "${dir}" | tail -1 | tr -dc 0-9)
  (( avail >= $1 )) || { log "STOP: $2 needs ${1}G, ${dir} has ${avail}G"; exit 1; }
}

log "waiting for download"
while [[ ! -f ${BASE}.done ]]; do
  pgrep -f download_spacev1b.sh > /dev/null || { [[ -f ${BASE}.done ]] || { log "download not running and not done"; exit 1; }; }
  sleep 60
done

# ---------------- DiskANN ----------------
if [[ ! -f ${D}/diskann.done ]]; then
  need_gb 600 "diskann build"
  log "diskann build start (resume)"
  DISKANN_BUILD_RESUME=1 /usr/bin/time -v ${REPO}/diskann/build/apps/build_disk_index --data_type int8 --dist_fn l2 --data_path ${BASE} \
    --index_path_prefix ${IDX} -R 64 -L 100 -B 30 -M 48 -T ${T} >> ${D}/diskann_build.log 2>&1
  [[ -s ${IDX}_disk.index ]] || { log "diskann build produced no disk index"; exit 1; }
  rm -f ${IDX}_mem.index ${IDX}_mem.index.data
  touch ${D}/diskann.done
  log "diskann build done $(du -sh ${IDX}_disk.index | cut -f1)"
fi

if [[ ! -f ${D}/smoke.done ]]; then
  log "diskann smoke search start"
  ${REPO}/diskann/build/apps/search_disk_index --data_type int8 --dist_fn l2 --index_path_prefix ${IDX} \
    --query_file ${Q} --gt_file ${GT} -K 10 -L 50 100 200 -W 4 -T 20 --num_nodes_to_cache 0 \
    --result_path ${D}/smoke_res > ${D}/smoke.log 2>&1
  rm -f ${D}/smoke_res_*
  local_smoke=$(grep -E '^ +(50|100|200) ' ${D}/smoke.log | awk '{printf "L=%s QPS=%s recall=%s; ", $1, $3, $NF}')
  log "diskann smoke: ${local_smoke}"
  # Sanity: L=100 recall must be present and > 0.5 (broken index / wrong dtype would be ~0).
  r100=$(awk '$1==100 {print $NF; exit}' ${D}/smoke.log)
  [[ -n "$r100" ]] || { log "VERIFY FAIL: no L=100 row in smoke.log"; exit 1; }
  awk -v r="$r100" 'BEGIN{exit !(r+0 > 0.5)}' || { log "VERIFY FAIL: diskann L=100 recall=$r100"; exit 1; }
  log "diskann verify OK recall@L100=$r100"
  touch ${D}/smoke.done
fi

# ---------------- Starling (all intermediates on SSD under $D, never HDD) ----------------
OUT=${D}/starling
P=${OUT}/spacev1b
if [[ ! -f ${OUT}/done ]]; then
  # Free DiskANN shard leftovers so partition/relayout work_dir fits on raid5 SSD.
  if [[ -f ${D}/diskann.done ]]; then
    rm -f ${IDX}_mem.index_tempFiles_subshard-* ${IDX}_mem.index ${IDX}_mem.index.data 2>/dev/null || true
  fi
  mkdir -p ${OUT}/work ${OUT}/mem_sample ${OUT}/mem
  log "starling start (work_dir=${OUT}/work on SSD)"
  if [[ ! -f ${P}_partition.bin ]]; then
    need_gb 420 "starling partition" /mnt/raid5
    /usr/bin/time -v ${GPB}/ooc_partitioner --index_file ${IDX}_disk.index --gp_file ${P}_partition.bin \
      --work_dir ${OUT}/work --data_type int8 -T ${T} -L 16 --chunk_nodes 16000000 --rev_budget_gb 24 \
      > ${OUT}/partition.log 2>&1
  fi
  rm -rf ${OUT}/work
  log "starling partition done"
  if [[ ! -f ${OUT}/relayout.done ]]; then
    need_gb 540 "starling relayout" /mnt/raid5
    /usr/bin/time -v ${GPB}/ooc_relayout ${IDX}_disk.index ${P}_partition.bin ${P}_disk.index ${T} > ${OUT}/relayout.log 2>&1
    touch ${OUT}/relayout.done
  fi
  ln -sfn ${IDX}_disk.index ${P}_disk_beam_search.index
  if [[ ! -f ${OUT}/mem/_index ]]; then
    ${SB}/build/tests/utils/gen_random_slice int8 ${BASE} ${OUT}/mem_sample/ 0.01 > ${OUT}/mem_sample/sample.log 2>&1
    /usr/bin/time -v ${SB}/build/tests/build_memory_index --data_type int8 --dist_fn l2 \
      --data_path ${OUT}/mem_sample/ --index_path_prefix ${OUT}/mem/_index -R 48 -L 128 --alpha 1.2 \
      > ${OUT}/mem/build.log 2>&1
  fi
  python3 ${REPO}/experiments/scripts/pq_pivots_to_v5.py ${IDX}_pq_pivots.bin ${OUT}/pq_pivots_v5.bin
  ln -sfn ${IDX}_pq_compressed.bin ${P}_pq_compressed.bin
  ln -sfn ${OUT}/pq_pivots_v5.bin ${P}_pq_pivots.bin
  ln -sfn ${IDX}_sample_data.bin ${P}_sample_data.bin
  ln -sfn ${IDX}_sample_ids.bin ${P}_sample_ids.bin
  touch ${OUT}/done
  log "starling done"
fi

if [[ ! -f ${OUT}/smoke.done ]]; then
  log "starling smoke search start"
  ${SB}/build/tests/search_disk_index --data_type int8 --dist_fn l2 --index_path_prefix ${P} \
    --query_file ${Q} --gt_file ${GT} -K 10 -L 50 100 -W 4 -T 20 --num_nodes_to_cache 0 \
    --mem_L 10 --mem_index_path ${OUT}/mem/_index --use_page_search 1 \
    --disk_file_path ${P}_disk.index --result_path ${OUT}/smoke_res > ${OUT}/smoke.log 2>&1
  rm -f ${OUT}/smoke_res_*
  local_smoke=$(grep -E '^ +(50|100) ' ${OUT}/smoke.log | awk '{printf "L=%s QPS=%s recall=%s; ", $1, $3, $NF}')
  log "starling smoke: ${local_smoke}"
  r100=$(awk '$1==100 {print $NF; exit}' ${OUT}/smoke.log)
  [[ -n "$r100" ]] || { log "VERIFY FAIL: no L=100 row in starling smoke.log"; exit 1; }
  awk -v r="$r100" 'BEGIN{exit !(r+0 > 0.5)}' || { log "VERIFY FAIL: starling L=100 recall=$r100"; exit 1; }
  log "starling verify OK recall@L100=$r100"
  touch ${OUT}/smoke.done
fi

log "DISKANN+STARLING DONE + VERIFIED (MARGO next, needs ~1.2T on one disk)"
touch ${D}/verified.done

