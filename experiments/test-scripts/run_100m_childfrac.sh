#!/bin/bash
# knn2000 hotspot 0/30/50/70%, seed 100%: each seed keeps the hottest 50% (by count) of its profile children,
# no grandchild fill; shared cache 60000 pages, no N-cache.
set -u
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/nvme_single/sift100m/sift100m_index
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
R=/mnt/nvme_single/sift100m/runs/dcache_knn2000_childfrac_20261012
P=/mnt/hdd/merit_park/tmp_profiles/knn2000_seedfrac_allch240_20261012
CF=${CF:-0.5}
mkdir -p $R
log() { echo "$(date -Is) $*" | tee -a $R/status.txt; }
run() { local out=$1 q=$2 gt=$3; shift 3; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_CHILD_FRAC=$CF \
    NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=60000 \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result --num_nodes_to_cache 0 "$@" > $out/run.out 2>&1; }
row() { local f=$1/run.out r=$(grep -E '^\s+128\s+4\s' $1/run.out | tail -1)
  echo "QPS=$(echo $r | awk '{print $3}') recall=$(echo $r | awk '{print $NF}') $(grep -o 'base_reads=.*' $f | tail -1)"; }
for t in ${TAGS:-x1 x31 x51 x71}; do
  q=$WLD/knn2000_${t}_block_s4_pct.u8bin; gt=$WLD/knn2000_${t}_block_s4_pct_gt10.bin
  DC="--neighborcache_profile_prefix $P/$t --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
  b=$R/$t/cf${CF}_build
  run $b $q $gt $DC
  sync; sleep 3
  for rep in 1 2; do
    run $R/$t/cf${CF}_r$rep $q $gt $DC --neighborcache_disk_cache_reuse_prefix $b/result
    log "[$t] childfrac=$CF seedfrac=1.0 rep=$rep $(grep -o 'packed file sectors=[0-9]*' $b/run.out) $(row $R/$t/cf${CF}_r$rep)"
  done
  rm -f $b/*_neighborcache_dc.data $b/*_neighborcache_dc.nodes $b/*_neighborcache_dc.seed_*
done
log CHILDFRAC_DONE
