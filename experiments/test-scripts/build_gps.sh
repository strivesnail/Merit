#!/usr/bin/env bash
# Build GPS (Gorgeous) on top of an existing DiskANN disk index and PQ, so that GPS shares
# the Vamana graph and PQ codes with the other systems.
# Usage: DTYPE=uint8 BASE=... DISK_INDEX=... PQ_PREFIX=... OUT=/mnt/raid5/gps/<name> bash build_gps.sh
# Resumable: finished stages are skipped.
set -euo pipefail

GPS=/home/jianz/workload/code/baselines/Gorgeous/build
TOOLS=/home/jianz/workload/code/NeighborCache/experiments/tools
export LD_LIBRARY_PATH=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
: "${DTYPE:?}" "${BASE:?}" "${DISK_INDEX:?}" "${PQ_PREFIX:?}" "${OUT:?}"
T=${T:-24}
SAMPLE_RATE=${SAMPLE_RATE:-0.005}

IDX=${OUT}/index/
PQ=${OUT}/pq/
MEM_SAMPLE=${OUT}/mem_sample/sample
MEM_INDEX=${OUT}/mem_index/
GRAPH=${OUT}/graph/
REP=${OUT}/graph_rep/
STATUS=${OUT}/status.txt
mkdir -p "${IDX}" "${PQ}" "$(dirname "${MEM_SAMPLE}")" "${MEM_INDEX}" "${GRAPH}" "${REP}"

log() { echo "$(date -Is) $*" | tee -a "${STATUS}"; }
run() { /usr/bin/time -v "$@"; }

ln -sf "${DISK_INDEX}" "${IDX}_disk.index"
[[ -f ${PQ}_pq_pivots.bin ]] || python3 /home/jianz/workload/code/NeighborCache/experiments/scripts/pq_pivots_to_v5.py \
  "${PQ_PREFIX}_pq_pivots.bin" "${PQ}_pq_pivots.bin"
[[ -f ${PQ_PREFIX}_sample_data.bin ]] && ln -sf "${PQ_PREFIX}_sample_data.bin" "${IDX}_sample_data.bin"
ln -sf "${PQ_PREFIX}_pq_compressed.bin" "${PQ}_pq_compressed.bin"

if [[ ! -f ${MEM_INDEX}_index ]]; then
  log "nav graph start"
  run ${GPS}/tests/utils/gen_random_slice "${DTYPE}" "${BASE}" "${MEM_SAMPLE}" "${SAMPLE_RATE}" > "${OUT}/mem_sample.log" 2>&1
  run ${GPS}/tests/build_memory_index --data_type "${DTYPE}" --dist_fn l2 --data_path "${MEM_SAMPLE}" \
    --index_path_prefix "${MEM_INDEX}_index" -R 24 -L 128 --alpha 1.2 > "${OUT}/mem_index.log" 2>&1
  log "nav graph done"
fi

# The GPS partitioner needs >100 GB RAM at 100M points; gps_layout writes the same graph-replicated
# file (verified byte-identical on SIFT10M) and a graph-only file in ID order.
if [[ ! -f ${REP}layout.done ]]; then
  log "gps_layout start"
  case ${DTYPE} in uint8|int8) ELEM=1 ;; float) ELEM=4 ;; esac
  run ${TOOLS}/gps_layout "${IDX}_disk.index" "${ELEM}" "${GRAPH}" "${REP}" "${T}" > "${OUT}/gps_layout.log" 2>&1
  touch "${REP}layout.done"
  log "gps_layout done"
fi
log "ALLDONE"
