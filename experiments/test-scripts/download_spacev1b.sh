#!/bin/bash
# Downloads the 33 SPACEV1B vector parts (Git LFS) in parallel with resume, verifies sha256, and
# concatenates them into a DiskANN int8 bin (header: npts, dim) at $OUT/base.1B.i8bin.
set -uo pipefail
OUT=${OUT:-/mnt/raid5/spacev1b}
PAR=${PAR:-8}
RAW=https://raw.githubusercontent.com/microsoft/SPTAG/main/datasets/SPACEV1B/vectors.bin
MEDIA=https://media.githubusercontent.com/media/microsoft/SPTAG/main/datasets/SPACEV1B/vectors.bin
mkdir -p $OUT/parts
cd $OUT/parts
log() { echo "$(date -Is) $*" | tee -a $OUT/download.status; }

fetch() {
  local i=$1 f=vectors_$1.bin
  [ -f $f.ok ] && return 0
  curl -s $RAW/$f > $f.ptr || return 1
  local sha=$(awk '/^oid/{sub("sha256:","",$2); print $2}' $f.ptr) size=$(awk '/^size/{print $2}' $f.ptr)
  for t in 1 2 3 4 5 6 7 8 9 10; do
    curl -sL -C - --retry 5 --retry-delay 10 -o $f $MEDIA/$f
    [ "$(stat -c %s $f 2>/dev/null)" = "$size" ] && break
    sleep 30
  done
  if [ "$(sha256sum $f | cut -d' ' -f1)" = "$sha" ]; then touch $f.ok; log "part $i ok"; else log "part $i BAD sha"; rm -f $f; return 1; fi
}
export -f fetch log
export RAW MEDIA OUT
log "download start PAR=$PAR"
seq 1 33 | xargs -P $PAR -I{} bash -c 'fetch {}'
n_ok=$(ls *.ok 2>/dev/null | wc -l)
[ "$n_ok" = 33 ] || { log "only $n_ok/33 parts ok, stopping"; exit 1; }
log "all parts ok"

# vectors_1.bin starts with int32 count, int32 dim; parts are a raw byte split of one file.
[ -f $OUT/base.1B.i8bin.done ] && exit 0
for i in $(seq 1 33); do cat vectors_$i.bin; done > $OUT/base.1B.i8bin
read -r N D < <(/home/jianz/miniconda3/bin/python -c "import numpy as np; print(*np.fromfile('$OUT/base.1B.i8bin', dtype=np.int32, count=2))")
EXP=$((8 + N * D))
SZ=$(stat -c %s $OUT/base.1B.i8bin)
log "concat N=$N D=$D size=$SZ expected=$EXP"
[ "$SZ" = "$EXP" ] || { log "SIZE MISMATCH"; exit 1; }
touch $OUT/base.1B.i8bin.done
rm -f vectors_*.bin
log "DOWNLOAD DONE"
