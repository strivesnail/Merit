#!/usr/bin/env bash
# Stage one dataset's DiskANN + Starling + MARGO search files onto /mnt/nvme_single for fair Fig.1 I/O.
# Usage: DS=sift100m|turing100m|sift1b|sift10m bash stage_fig1_nvme.sh
set -euo pipefail
: "${DS:?}"
NV=/mnt/nvme_single
R5=/mnt/raid5
log() { echo "$(date -Is) [stage $DS] $*"; }

avail_g() { df --output=avail -BG "$NV" | tail -1 | tr -dc 0-9; }

case "$DS" in
  sift100m)
    # already staged by hand; verify
    for f in \
      $NV/sift100m/sift100m_index_disk.index \
      $NV/sift100m/starling/sift100m_disk.index \
      $NV/sift100m/margo/sift100m_disk.index \
      $NV/sift100m/starling/mem/_index; do
      [[ -e $f ]] || { log "MISSING $f"; exit 1; }
    done
    log "ok (DiskANN+Starling+MARGO on NVMe)"
    ;;
  turing100m)
    need=150
    (( $(avail_g) >= need )) || { log "need ~${need}G free, have $(avail_g)G"; exit 1; }
    mkdir -p $NV/turing100m/starling/mem $NV/turing100m/margo
    rsync -a --info=progress2 $R5/turing100m/turing100m_index_disk.index $NV/turing100m/
    rsync -a $R5/turing100m/turing100m_index_pq_compressed.bin $R5/turing100m/turing100m_index_pq_pivots.bin \
      $R5/turing100m/turing100m_index_sample_*.bin $R5/turing100m/turing100m_index_disk.index_* $NV/turing100m/ 2>/dev/null || true
    rsync -a --info=progress2 $R5/turing100m/starling/turing100m_disk.index $NV/turing100m/starling/
    rsync -a $R5/turing100m/starling/mem/ $NV/turing100m/starling/mem/
    rsync -a --info=progress2 $R5/turing100m/margo/turing100m_disk.index $NV/turing100m/margo/
    log "turing staged"
    ;;
  sift1b)
    need=900
    (( $(avail_g) >= need )) || { log "need ~${need}G free for full 1B stage, have $(avail_g)G — park other datasets first"; exit 1; }
    mkdir -p $NV/sift1b/starling/mem $NV/sift1b/margo
    # materialize diskann if currently a symlink
    if [[ -L $NV/sift1b/sift1b_index_disk.index ]]; then
      rm -f $NV/sift1b/sift1b_index_disk.index
      rsync -a --info=progress2 $R5/sift1b/sift1b_index_disk.index $NV/sift1b/
    fi
    if [[ -L $NV/sift1b/starling ]]; then rm -f $NV/sift1b/starling; mkdir -p $NV/sift1b/starling/mem; fi
    rsync -a --info=progress2 $R5/sift1b/starling/sift1b_disk.index $NV/sift1b/starling/ 2>/dev/null || \
      rsync -a --info=progress2 $R5/sift1b/starling_nvme_park/sift1b_disk.index $NV/sift1b/starling/
    rsync -a --info=progress2 $R5/sift1b/margo/sift1b_disk.index $NV/sift1b/margo/
    log "sift1b staged"
    ;;
  sift10m)
    need=20
    (( $(avail_g) >= need )) || { log "need ~${need}G free"; exit 1; }
    mkdir -p $NV/sift10m/starling/mem $NV/sift10m/margo
    log "sift10m: ensure starling/margo built on raid5 then rsync (prep_* scripts)"
    ;;
  *) log "unknown DS"; exit 1 ;;
esac
df -h $NV | tail -1
