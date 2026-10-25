#!/usr/bin/env bash
# Build DiskANN indexes for Turing100M and Text2Image100M (R=64 L=100, 40-byte PQ).
# Resumable: finished stages are skipped. Usage: nohup setsid bash build_100m_indexes.sh >> /mnt/raid5/build_100m.out 2>&1 &
set -euo pipefail

REPO=/home/jianz/workload/code/NeighborCache
APPS=${REPO}/diskann/build/apps
LIBS=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
T=${T:-24}
M=${M:-32}
STATUS=/mnt/raid5/build_100m_status.txt
T2I_URL=https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I

log() { echo "$(date -Is) $*" | tee -a "${STATUS}"; }

build_index() {
  local dtype=$1 data=$2 prefix=$3
  if [[ -f ${prefix}_disk.index && -f ${prefix}.build.done ]]; then
    log "skip build ${prefix}"
    return
  fi
  log "build start ${prefix}"
  env LD_LIBRARY_PATH=${LIBS} OMP_NUM_THREADS=${T} /usr/bin/time -v \
    ${APPS}/build_disk_index --data_type "${dtype}" --dist_fn l2 --data_path "${data}" \
    --index_path_prefix "${prefix}" -R 64 -L 100 -B 4 -M "${M}" -T "${T}" \
    > "${prefix}.build.log" 2>&1
  touch "${prefix}.build.done"
  log "build done ${prefix}"
}

download_t2i() {
  local dir=/mnt/raid5/t2i100m
  mkdir -p "${dir}"
  if [[ -f ${dir}/download.done ]]; then return; fi
  log "t2i download start"
  local n=100000000 d=200
  local bytes=$((n * d * 4))
  python3 -c "import struct,sys;sys.stdout.buffer.write(struct.pack('<ii',${n},${d}))" > "${dir}/header.bin"
  curl -sS --retry 20 --retry-delay 10 -r 8-$((bytes + 7)) -o "${dir}/body.part" "${T2I_URL}/base.1B.fbin"
  [[ $(stat -c %s "${dir}/body.part") -eq ${bytes} ]]
  cat "${dir}/header.bin" "${dir}/body.part" > "${dir}/base100m.fbin"
  rm -f "${dir}/body.part" "${dir}/header.bin"
  curl -sS --retry 20 -o "${dir}/query.public.100K.fbin" "${T2I_URL}/query.public.100K.fbin"
  touch "${dir}/download.done"
  log "t2i download done"
}

# Stale shards from the two failed Turing builds.
rm -f /mnt/raid5/turing100m/turing100m_index_mem.index_tempFiles_* \
      /mnt/raid5/turing100m/turing100m_index_disk.index_centroids.bin \
      /mnt/raid5/turing100m/turing100m_m40_*

download_t2i &
DL=$!

build_index float /mnt/raid5/turing100m/base100m.fbin /mnt/raid5/turing100m/turing100m_index

wait ${DL}
avail=$(df --output=avail -BG /mnt/raid5 | tail -1 | tr -dc 0-9)
if (( avail < 250 )); then
  log "t2i build waiting: only ${avail}G free on raid5"
  exit 1
fi
build_index float /mnt/raid5/t2i100m/base100m.fbin /mnt/raid5/t2i100m/t2i100m_index
log "ALLDONE"
