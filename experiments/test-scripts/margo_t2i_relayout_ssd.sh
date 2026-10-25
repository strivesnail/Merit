#!/usr/bin/env bash
# Redo the MARGO T2I100M relayout with its input on raid5: the HDD sustains only ~340 random 4 KB reads/s.
# Frees raid5 space by moving SIFT100M's MARGO dir and memory index to the HDD (copy, verify size, then remove).
# Output stays on the HDD (written in 16 MB chunks). Afterwards build_baselines_100m.sh finishes the MARGO stage.
set -euo pipefail
OFF=/mnt/hdd/merit_park/raid5_offload/sift100m
M=/mnt/hdd/merit_park/t2i100m/margo
TMP=/mnt/raid5/t2i100m_margo_relayout_in
GPB=/home/jianz/workload/code/baselines/starling/graph_partition/build
LOG=/mnt/hdd/merit_park/raid5_offload/margo_t2i_relayout_ssd.status
mkdir -p "$OFF"
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

move_verified() {  # <src> <dst dir>
  local src=$1 dst=$2/$(basename "$1")
  [[ -L $src ]] && { log "skip move $src (already moved)"; return; }
  log "move $src -> $dst"
  rsync -a "$src" "$2/"
  local a b
  a=$(du -sb --apparent-size "$src" | cut -f1); b=$(du -sb --apparent-size "$dst" | cut -f1)
  [[ $a == "$b" ]] || { log "size mismatch $src ($a) vs $dst ($b)"; exit 1; }
  rm -rf "$src"
  # keep the old path valid: build_baselines_100m.sh skips stages by their done files
  ln -sfn "$dst" "$src"
}

move_verified /mnt/raid5/sift100m/margo "$OFF"
move_verified /mnt/raid5/sift100m/sift100m_index_mem.index "$OFF"
df -h /mnt/raid5 | tail -1 | tee -a "$LOG"

mkdir -p "$TMP"
if [[ ! -f $TMP/copy.done ]]; then
  log "copy MARGO T2I disk index to raid5"
  cp "$M/t2i100m_disk.index" "$TMP/t2i100m_disk.index"
  cmp -n 1048576 "$M/t2i100m_disk.index" "$TMP/t2i100m_disk.index"
  touch "$TMP/copy.done"
fi

log "relayout start (input on raid5)"
rm -f "$M/t2i100m_disk_relayout.index"
"$GPB/ooc_relayout" "$TMP/t2i100m_disk.index" "$M/t2i100m_partition.bin" "$M/t2i100m_disk_relayout.index" 24 \
  > "$M/relayout.log" 2>&1
mv "$M/t2i100m_disk_relayout.index" "$M/t2i100m_disk.index"
touch "$M/relayout.done"
rm -rf "$TMP"
log "relayout done; finishing build_baselines_100m.sh"
cd /home/jianz/workload/code/NeighborCache/experiments/test-scripts
bash build_baselines_100m.sh >> /mnt/raid5/build_baselines_100m.out 2>&1
log "ALL DONE"
