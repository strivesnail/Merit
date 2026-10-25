#!/bin/bash
# knn2000 hotspot 0/30/50/70%, split trace: profile on even queries (_A), measure on odd queries (_B).
# All children per seed, shared cache 60000 pages, no N-cache. Baseline: DiskANN with a 240MB BFS cache on _B.
set -u
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/nvme_single/sift100m/sift100m_index
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
R=/mnt/nvme_single/sift100m/runs/dcache_knn2000_split_20261012
P=/mnt/hdd/merit_park/tmp_profiles/knn2000_split_20261012
REPS=${REPS:-3}
mkdir -p $R $P
log() { echo "$(date -Is) $*" | tee -a $R/status.txt; }
base() { local out=$1 q=$2 gt=$3; shift 3; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result "$@" > $out/run.out 2>&1; }
run() { local out=$1 q=$2 gt=$3; shift 3; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=60000 \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result --num_nodes_to_cache 0 "$@" > $out/run.out 2>&1; }
row() { local f=$1/run.out r=$(grep -E '^\s+128\s+4\s' $1/run.out | tail -1)
  echo "QPS=$(echo $r | awk '{print $3}') IOs=$(echo $r | awk '{print $6}') recall=$(echo $r | awk '{print $NF}') $(grep -o 'base_reads=.*' $f | tail -1)"; }
for t in ${TAGS:-x1 x31 x51 x71}; do
  p=$WLD/knn2000_${t}_block_s4_pct
  qa=${p}_A.u8bin; ga=${p}_A_gt10.bin; qb=${p}_B.u8bin; gb=${p}_B_gt10.bin
  base $R/$t/diskann_bfs $qb $gb --num_nodes_to_cache 633402
  log "[$t] diskann_bfs $(row $R/$t/diskann_bfs)"
  [ -f $P/${t}_node_visit.bin ] || run $R/$t/profile_run $qa $ga --enable_access_profile --access_profile_prefix $P/$t
  for fr in ${FRACS:-0.5 1.0}; do
    DC="--neighborcache_profile_prefix $P/$t --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
    b=$R/$t/s${fr}_build
    NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=$fr run $b $qb $gb $DC
    sync; sleep 3
    for rep in $(seq 1 $REPS); do
      NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC=$fr run $R/$t/s${fr}_r$rep $qb $gb $DC --neighborcache_disk_cache_reuse_prefix $b/result
      log "[$t] split seedfrac=$fr rep=$rep $(grep -o 'packed file sectors=[0-9]*' $b/run.out) $(row $R/$t/s${fr}_r$rep)"
    done
    rm -f $b/*_neighborcache_dc.data $b/*_neighborcache_dc.nodes $b/*_neighborcache_dc.seed_*
  done
done
log SPLIT_DONE
