#!/usr/bin/env bash
# Memory-heavy 100M builds, one at a time (62 GB RAM): Turing100M DiskANN -> GPS SIFT100M ->
# Text2Image100M DiskANN (after its download finishes). Resumable.
# Usage: nohup setsid bash build_queue_100m.sh >> /mnt/raid5/build_queue_100m.out 2>&1 < /dev/null &
set -euo pipefail

REPO=/home/jianz/workload/code/NeighborCache
APPS=${REPO}/diskann/build/apps
SCRIPTS=${REPO}/experiments/test-scripts
LIBS=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
T=${T:-24}
M=${M:-32}
STATUS=/mnt/raid5/build_100m_status.txt

log() { echo "$(date -Is) $*" | tee -a "${STATUS}"; }

build_index() {
  local dtype=$1 data=$2 prefix=$3
  if [[ -f ${prefix}.build.done ]]; then
    log "skip build ${prefix}"
    return
  fi
  rm -f "${prefix}"_mem.index_tempFiles_*
  log "build start ${prefix}"
  env LD_LIBRARY_PATH=${LIBS} OMP_NUM_THREADS=${T} /usr/bin/time -v \
    ${APPS}/build_disk_index --data_type "${dtype}" --dist_fn l2 --data_path "${data}" \
    --index_path_prefix "${prefix}" -R 64 -L 100 -B 4 -M "${M}" -T "${T}" \
    > "${prefix}.build.log" 2>&1
  touch "${prefix}.build.done"
  log "build done ${prefix}"
}

build_index float /mnt/raid5/turing100m/base100m.fbin /mnt/raid5/turing100m/turing100m_index

log "gps sift100m start"
DTYPE=uint8 BASE=${REPO}/data/sift100m/sift_base.u8bin DISK_INDEX=/mnt/raid5/sift100m/sift100m_index_disk.index \
  PQ_PREFIX=/mnt/raid5/sift100m/sift100m_index T=${T} OUT=/mnt/raid5/gps/sift100m bash ${SCRIPTS}/build_gps.sh
log "gps sift100m done"

while [[ ! -f /mnt/raid5/t2i100m/download.done ]]; do sleep 120; done
build_index float /mnt/raid5/t2i100m/base100m.fbin /mnt/raid5/t2i100m/t2i100m_index
log "QUEUE ALLDONE"
