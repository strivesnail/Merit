#!/usr/bin/env bash
# Fig.1 one dataset: ×3 shuffled queries, sweep L for DiskANN / BFS / NeighborCache / Starling / MARGO.
# Required env: DS=sift100m|sift1b|sift10m|turing100m
# Optional: LS="128" (motivation-matched) SYSTEMS="diskann bfs neighborcache starling margo"
set -uo pipefail
export PATH=/home/jianz/miniconda3/bin:$PATH
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
NEIGHBORCACHE_BIN=${NEIGHBORCACHE_BIN:-/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index}
STARLING_BIN=${STARLING_BIN:-/home/jianz/workload/code/baselines/starling/build/tests/search_disk_index}
MARGO_BIN=${MARGO_BIN:-/home/jianz/workload/code/baselines/MARGO/build/tests/search_disk_index}

: "${DS:?set DS=sift100m|sift1b|sift10m|turing100m}"
LS=${LS:-"128"}
SYSTEMS=${SYSTEMS:-"diskann bfs neighborcache starling margo"}
THREADS=${THREADS:-16}
W=${W:-4}
K=${K:-10}
SKIP=${SKIP:-3000}
OUTROOT=${OUTROOT:-/mnt/raid5/fig1_x3/${DS}}
WLROOT=${WLROOT:-/mnt/raid5/fig1_x3/workloads}
mkdir -p "$OUTROOT" "$WLROOT"
STATUS=$OUTROOT/status.txt
log() { echo "$(date -Is) [$DS] $*" | tee -a "$STATUS"; }

# Paper-matched DRAM: every cached bar gets 400MB total.
# BFS / N-only: all 400MB in the node cache.
# NeighborCache (N+D): 350MB N-cache + 50MB shared D-page prefetch buffer.
DRAM_MB=${DRAM_MB:-400}
N_MB=${N_MB:-350}
SHARED_MB=${SHARED_MB:-50}
# DiskANN max node len on SIFT uint8 indexes (see search log: "max node len 388").
NODE_BYTES=${NODE_BYTES:-388}
PAGE_BYTES=4096
bfs_nodes_from_mb() { echo $(( ($1 * 1024 * 1024) / NODE_BYTES )); }
ncap_from_mb() { echo $(( ($1 * 1024 * 1024) / NODE_BYTES )); }
shared_pages_from_mb() { echo $(( ($1 * 1024 * 1024) / PAGE_BYTES )); }

case "$DS" in
  sift100m)
    DTYPE=uint8
    # All search indexes staged under NVMe for fair I/O.
    INDEX=/mnt/nvme_single/sift100m/sift100m_index
    [[ -f ${INDEX}_disk.index ]] || INDEX=/home/jianz/workload/data/sift100m/sift100m_index
    QSRC=/home/jianz/workload/data/sift100m/query.public.10K.u8bin
    GTSRC=/home/jianz/workload/data/sift100m/bigann-100M
    STAR_P=/mnt/nvme_single/sift100m/starling/sift100m
    MARGO_P=/mnt/nvme_single/sift100m/margo/sift100m
    NAV=/mnt/nvme_single/sift100m/starling/mem/_index
    DPAGES=${DPAGES:-2000000}         # SSD D-cache cap ~20% of index pages
    UNIQUE=${UNIQUE:-200000}
    ;;
  sift1b)
    DTYPE=uint8
    # Stage DiskANN+Starling+MARGO onto NVMe before running (see stage_fig1_nvme.sh).
    INDEX=/mnt/nvme_single/sift1b/sift1b_index
    QSRC=/mnt/raid5/sift1b/query.public.10K.u8bin
    GTSRC=/mnt/raid5/sift1b/GT.public.1B.ibin
    STAR_P=/mnt/nvme_single/sift1b/starling/sift1b
    MARGO_P=/mnt/nvme_single/sift1b/margo/sift1b
    NAV=/mnt/nvme_single/sift1b/starling/mem/_index
    DPAGES=${DPAGES:-20000000}
    UNIQUE=${UNIQUE:-46929}
    ;;
  sift10m)
    DTYPE=uint8
    INDEX=/mnt/nvme_single/sift10m/sift10m_index
    [[ -f ${INDEX}_disk.index ]] || INDEX=/home/jianz/workload/data/sift10m/sift10m_index
    QSRC=/home/jianz/workload/data/sift10m/query.public.10K.u8bin
    GTSRC=/home/jianz/workload/data/sift10m/bigann-10M
    STAR_P=/mnt/nvme_single/sift10m/starling/sift10m
    MARGO_P=/mnt/nvme_single/sift10m/margo/sift10m
    NAV=/mnt/nvme_single/sift10m/starling/mem/_index
    DPAGES=${DPAGES:-200000}
    UNIQUE=${UNIQUE:-50000}
    ;;
  turing100m)
    DTYPE=float
    INDEX=/mnt/nvme_single/turing100m/turing100m_index
    QSRC=/mnt/raid5/turing100m/query100K.fbin
    GTSRC=/mnt/raid5/turing100m/msturing-gt-100M
    STAR_P=/mnt/nvme_single/turing100m/starling/turing100m
    MARGO_P=/mnt/nvme_single/turing100m/margo/turing100m
    NAV=/mnt/nvme_single/turing100m/starling/mem/_index
    # float nodes are larger; override if known from index meta, else keep 388 until measured.
    DPAGES=${DPAGES:-3333333}
    UNIQUE=${UNIQUE:-200000}
    ;;
  *) log "unknown DS=$DS"; exit 1 ;;
esac

BFS_NODES=${BFS_NODES:-$(bfs_nodes_from_mb "$DRAM_MB")}
NCAP=${NCAP:-$(ncap_from_mb "$N_MB")}
SHARED_PAGES=${SHARED_PAGES:-$(shared_pages_from_mb "$SHARED_MB")}
# neighborcache_memory_gb sizes the N-cache pool; keep it at the 350MB N-slice (not full host RAM).
NEIGHBORCACHE_MEM_GB=${NEIGHBORCACHE_MEM_GB:-$(python3 -c "print($N_MB/1024.0)")}

# ---- workload (override with QFILE=/... GFILE=/... for Zipf etc.) ----
if [[ -z "${QFILE:-}" || -z "${GFILE:-}" ]]; then
  if [[ "$DTYPE" == "uint8" ]]; then
    QFILE=$WLROOT/${DS}_x3_shuffled.u8bin
  else
    QFILE=$WLROOT/${DS}_x3_shuffled.fbin
  fi
  GFILE=$WLROOT/${DS}_x3_shuffled_gt10.ibin
fi
if [[ ! -f "$QFILE" || ! -f "$GFILE" ]]; then
  log "generating ×3 workload"
  python3 /home/jianz/workload/code/NeighborCache/experiments/scripts/make_fig1_x3_workloads.py \
    --tag "$DS" --query "$QSRC" --gt "$GTSRC" --out-dir "$WLROOT" --repeat 3 --max-pool 10000
fi
[[ -f "$QFILE" && -f "$GFILE" ]] || { log "STOP: missing workload $QFILE"; exit 1; }

PIN=(OMP_PROC_BIND=close "OMP_PLACES={0}:$THREADS")
NC_ENV=(NEIGHBORCACHE_NCACHE_SHARDS=128 NEIGHBORCACHE_HEAP_SHARDS=4 NEIGHBORCACHE_NCACHE_SPIN=0
  NEIGHBORCACHE_NCACHE_ADMISSION=adaptive_hop_reject NEIGHBORCACHE_NCACHE_HOP_THRESHOLD=12
  NEIGHBORCACHE_NCACHE_ADAPT_HALF_LIFE_QUERIES=25000 NEIGHBORCACHE_NCACHE_ADAPT_MIN_HOP=8
  NEIGHBORCACHE_NCACHE_GHOST_PERCENT=10 NEIGHBORCACHE_NCACHE_FAST_MISS=1 NEIGHBORCACHE_NCACHE_CLOCK=0
  NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=$SHARED_PAGES)

run_diskann_family() {
  # $1=tag $2=L $3=cache_nodes $4=neighborcache? 0/1
  local tag=$1 L=$2 cache=$3 mode=$4
  local out=$OUTROOT/${tag}_L${L}
  mkdir -p "$out"
  echo "==== START $tag L=$L $(date -Is) ====" > "$out/run.out"
  if [[ "$mode" == "neighborcache" ]]; then
    env LD_LIBRARY_PATH="$LIB" OMP_NUM_THREADS=$THREADS "${PIN[@]}" \
      NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DYNAMIC_3CACHE=1 NEIGHBORCACHE_NCACHE_ONLY=0 \
      NEIGHBORCACHE_NCACHE_COLD=1 \
      NEIGHBORCACHE_CACHE_FROM_WORKLOAD=1 NEIGHBORCACHE_WORKLOAD_UNIQUE_NODES=$UNIQUE \
      NEIGHBORCACHE_NCACHE_WORKLOAD_FRAC=0 NEIGHBORCACHE_DCACHE_WORKLOAD_FRAC=1.0 \
      NEIGHBORCACHE_NCACHE_CAP_NODES=$NCAP NEIGHBORCACHE_DCACHE_CAP=$DPAGES NEIGHBORCACHE_DCACHE_NODE_CAP=$((DPAGES * 10)) \
      NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC=0 NEIGHBORCACHE_REQUIRE_FULL_DCACHE_WARMUP=0 \
      NEIGHBORCACHE_MEASURE_SKIP_QUERIES=$SKIP NEIGHBORCACHE_RESOLVE_PARENT=0 \
      NEIGHBORCACHE_REAL_IO_COACCESS=1 NEIGHBORCACHE_REAL_IO_DIRECT=1 NEIGHBORCACHE_REAL_IO_WINDOW=4 NEIGHBORCACHE_REAL_IO_SAMPLE_QUERIES=64 \
      NEIGHBORCACHE_SEED_T=2 NEIGHBORCACHE_DEFERRED_NCACHE_PAGE_WRITE=1 NEIGHBORCACHE_PENDING_REQUIRE_FULL_PAGE=0 \
      NEIGHBORCACHE_PENDING_BUFFER_BYTES=4194304 "${NC_ENV[@]}" \
      NEIGHBORCACHE_MCACHE_MAX_EDGES=64 NEIGHBORCACHE_DCACHE_NET_GATE=0 \
      NEIGHBORCACHE_DCACHE_WRITER_THREADS=1 NEIGHBORCACHE_DCACHE_COMMIT_THREADS=1 \
      NEIGHBORCACHE_DCACHE_QUERY_PATCH=1 NEIGHBORCACHE_DCACHE_QUERY_PATCH_ASYNC=1 \
      NEIGHBORCACHE_DYN_MEMBIT=1 NEIGHBORCACHE_DYN_LOC_SHARDS=64 \
      "$NEIGHBORCACHE_BIN" --data_type "$DTYPE" --dist_fn l2 --index_path_prefix "$INDEX" \
      --query_file "$QFILE" --gt_file "$GFILE" --recall_at "$K" --search_list "$L" \
      --beamwidth "$W" --num_threads "$THREADS" --num_nodes_to_cache 0 \
      --neighborcache_memory_gb "$NEIGHBORCACHE_MEM_GB" --neighborcache_memory_runtime_admit true \
      --neighborcache_disk_cache_ratio 0.01 --neighborcache_disk_cache_layout bfs \
      --enable_query_sector_cache true \
      --result_path "$out/result" >> "$out/run.out" 2>&1
  else
    env LD_LIBRARY_PATH="$LIB" OMP_NUM_THREADS=$THREADS "${PIN[@]}" NEIGHBORCACHE_USE_RAMFS=0 \
      "$NEIGHBORCACHE_BIN" --data_type "$DTYPE" --dist_fn l2 --index_path_prefix "$INDEX" \
      --query_file "$QFILE" --gt_file "$GFILE" --recall_at "$K" --search_list "$L" \
      --beamwidth "$W" --num_threads "$THREADS" --num_nodes_to_cache "$cache" \
      --result_path "$out/result" >> "$out/run.out" 2>&1
  fi
  echo "==== END $tag L=$L $(date -Is) ====" >> "$out/run.out"
  local row
  row=$(grep -E "^[[:space:]]*$L[[:space:]]+$W[[:space:]]" "$out/run.out" | tail -1)
  log "$tag L=$L $row"
  rm -f "$out"/*.dyn.data "$out"/result_*_dists_float.bin 2>/dev/null || true
}

run_pagesearch() {
  # $1=starling|margo $2=L
  local sys=$1 L=$2
  local bin=$STARLING_BIN p=$STAR_P
  [[ "$sys" == margo ]] && bin=$MARGO_BIN && p=$MARGO_P
  local out=$OUTROOT/${sys}_L${L}
  mkdir -p "$out"
  if [[ ! -f ${p}_disk.index || ! -f $NAV ]]; then
    log "SKIP $sys: missing index ($p) or nav ($NAV)"
    return 0
  fi
  echo "==== START $sys L=$L $(date -Is) ====" > "$out/run.out"
  env LD_LIBRARY_PATH=/home/jianz/miniconda3/lib OMP_NUM_THREADS=$THREADS "${PIN[@]}" \
    "$bin" --data_type "$DTYPE" --dist_fn l2 --index_path_prefix "$p" \
    --query_file "$QFILE" --gt_file "$GFILE" -K "$K" -L "$L" -W "$W" -T "$THREADS" \
    --num_nodes_to_cache 0 --mem_L 10 --mem_index_path "$NAV" \
    --use_page_search 1 --use_ratio 1.0 --disk_file_path "${p}_disk.index" \
    --result_path "$out/result" >> "$out/run.out" 2>&1
  echo "==== END $sys L=$L $(date -Is) ====" >> "$out/run.out"
  local row
  row=$(grep -E "^[[:space:]]*$L[[:space:]]+$W[[:space:]]" "$out/run.out" | tail -1)
  log "$sys L=$L $row"
  rm -f "$out"/result_*_dists_float.bin 2>/dev/null || true
}

log "Fig.1 start OUT=$OUTROOT Q=$QFILE DRAM=${DRAM_MB}MB (BFS=${BFS_NODES}nodes/${DRAM_MB}MB N=${NCAP}nodes/${N_MB}MB shared=${SHARED_PAGES}p/${SHARED_MB}MB neighborcache_gb=$NEIGHBORCACHE_MEM_GB) DPAGES=$DPAGES LS=$LS"
for L in $LS; do
  for s in $SYSTEMS; do
    case $s in
      diskann) run_diskann_family diskann "$L" 0 plain ;;
      bfs) run_diskann_family bfs "$L" "$BFS_NODES" plain ;;
      neighborcache|neighborcache) run_diskann_family neighborcache "$L" 0 neighborcache ;;
      starling) run_pagesearch starling "$L" ;;
      margo) run_pagesearch margo "$L" ;;
      *) log "unknown system $s" ;;
    esac
  done
done
log "Fig.1 DONE $DS"
echo ALLDONE | tee -a "$STATUS"
