#!/usr/bin/env bash
# Migrate essential Merit code + data (no runs/) to jianz@10.64.1.42:/home/jianz/workload
#
# Layout on destination (code / data separated):
#   /home/jianz/workload/
#     code/Merit/          # repo (no data/, no diskann/build/)
#     data/sift1m/
#     data/sift10m/
#     data/sift100m/
#     code/Merit/data -> ../../data   (symlink so scripts keep using Merit/data/...)
#
# Approx size without runs/: sift1m ~7GB + sift10m ~37GB + sift100m ~129GB ≈ 173GB
# (+ code <1GB). Fits a 400GB disk with headroom.
#
# Prerequisites: SSH access to REMOTE from this machine.
#
# Usage:
#   ./experiments/scripts/migrate_essential_to_42.sh           # real sync
#   ./experiments/scripts/migrate_essential_to_42.sh --dry-run # preview
#   DATASETS="sift1m sift10m" ./experiments/scripts/migrate_essential_to_42.sh
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/home/jianz/Merit}"
REMOTE="${REMOTE:-jianz@10.64.1.42}"
DST_ROOT="${DST_ROOT:-/home/jianz/workload}"
DATASETS="${DATASETS:-sift1m sift10m sift100m}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new}"

RSYNC_RSH="ssh ${SSH_OPTS}"
RSYNC_FLAGS=(-aH --info=stats2,progress2 --partial --partial-dir=.rsync-partial -e "${RSYNC_RSH}")
DRY=()
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=(--dry-run)
  echo "[dry-run] no files will be written on remote"
fi

echo "=== Source: ${SRC_ROOT}"
echo "=== Dest:   ${REMOTE}:${DST_ROOT}"
echo "=== Datasets: ${DATASETS}"
echo

echo "=== Local size estimate (excluding runs/) ==="
total_k=0
for ds in ${DATASETS}; do
  if [[ -d "${SRC_ROOT}/data/${ds}" ]]; then
    sz=$(du -sk --exclude=runs "${SRC_ROOT}/data/${ds}" | awk '{print $1}')
    echo "  data/${ds}: $((sz / 1024)) MB"
    total_k=$((total_k + sz))
  else
    echo "  WARN: missing ${SRC_ROOT}/data/${ds}"
  fi
done
code_k=$(du -sk --exclude=data --exclude=diskann/build "${SRC_ROOT}" | awk '{print $1}')
echo "  code (no data/build): $((code_k / 1024)) MB"
echo "  TOTAL ~$(( (total_k + code_k) / 1024 / 1024 )) GB (need headroom on 400GB disk)"
echo

# shellcheck disable=SC2086
ssh ${SSH_OPTS} "${REMOTE}" "mkdir -p '${DST_ROOT}/code' '${DST_ROOT}/data' && df -h '${DST_ROOT}' | tail -1"

echo
echo "=== 1/3 Sync code -> ${DST_ROOT}/code/Merit ==="
rsync "${RSYNC_FLAGS[@]}" "${DRY[@]}" \
  --delete \
  --exclude='/.git/' \
  --exclude='/data/' \
  --exclude='/diskann/build/' \
  --exclude='**/__pycache__/' \
  --exclude='**/*.pyc' \
  --exclude='.rsync-partial/' \
  "${SRC_ROOT}/" "${REMOTE}:${DST_ROOT}/code/Merit/"

echo
echo "=== 2/3 Sync data (no runs/) ==="
for ds in ${DATASETS}; do
  src="${SRC_ROOT}/data/${ds}"
  [[ -d "${src}" ]] || continue
  echo "--- ${ds} ---"
  rsync "${RSYNC_FLAGS[@]}" "${DRY[@]}" \
    --exclude='/runs/' \
    --exclude='runs/**' \
    --exclude='.rsync-partial/' \
    --exclude='*.tar.gz' \
    --exclude='download_*.log' \
    "${src}/" "${REMOTE}:${DST_ROOT}/data/${ds}/"
done

echo
echo "=== 3/3 Wire code/Merit/data -> ../../data ==="
if [[ ${#DRY[@]} -eq 0 ]]; then
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${REMOTE}" bash -s <<EOF
set -euo pipefail
cd '${DST_ROOT}/code/Merit'
if [[ -e data && ! -L data ]]; then
  echo "ERROR: ${DST_ROOT}/code/Merit/data exists and is not a symlink; refuse to overwrite."
  exit 1
fi
ln -sfn ../../data data
ls -ld data
df -h '${DST_ROOT}'
du -sh '${DST_ROOT}/code' '${DST_ROOT}/data'/* 2>/dev/null || true
echo
echo "Done. On remote, rebuild:"
echo "  cd ${DST_ROOT}/code/Merit/diskann && mkdir -p build && cd build && cmake .. && make -j\\\$(nproc) search_disk_index"
EOF
else
  echo "[dry-run] skip symlink + remote df"
fi

echo
echo "All steps finished."
