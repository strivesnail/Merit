#!/bin/bash
# SIFT100M, knn2000 region (+0%/+50% hotspot): shared D-cache page cache size sweep.
# All seeds in D-cache, every seed keeps all profile children (NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1).
# Drop NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 to get the 2-page cap variant.
set -u
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=/mnt/nvme_single/sift100m/sift100m_index
WLD=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
R=/mnt/nvme_single/sift100m/runs/dcache_knn2000_cachecap_20261012
P=/mnt/hdd/merit_park/tmp_profiles/knn2000_cachecap
mkdir -p $R $P
log() { echo "$(date -Is) $*" | tee -a $R/status.txt; }
run() { local out=$1 q=$2 gt=$3 cap=$4; shift 4; mkdir -p $out
  env LD_LIBRARY_PATH=$LIB OMP_NUM_THREADS=20 NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD=1 NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES=2 NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN=1 \
    NEIGHBORCACHE_CACHE_REPLICA_FALLBACK=1 NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=$cap \
    $SEARCH --data_type uint8 --dist_fn l2 --index_path_prefix $INDEX --query_file $q --gt_file $gt --recall_at 10 \
    --search_list 128 --beamwidth 4 --num_threads 20 --result_path $out/result --num_nodes_to_cache 0 "$@" > $out/run.out 2>&1; }
split() { /home/jianz/miniconda3/bin/python - "$1" <<'PY'
import sys, numpy as np
f=sys.argv[1]; h=open(f).readline().strip().split(","); a=np.loadtxt(f,delimiter=",",skiprows=1,ndmin=2)
c={k:i for i,k in enumerate(h)}; q=a[:,c["query_id"]]; r=a[:,c["n_disk_reads"]]
o=r[q<10000].mean(); hot=r[q>=10000].mean() if (q>=10000).any() else float('nan')
print(f"official_reads={o:.2f} hot_reads={hot:.2f}")
PY
}
for t in x1 x51; do
  q=$WLD/knn2000_${t}_block_s4_pct.u8bin; gt=$WLD/knn2000_${t}_block_s4_pct_gt10.bin
  run $R/$t/profile_run $q $gt 5000 --enable_access_profile --access_profile_prefix $P/$t
  DC="--neighborcache_profile_prefix $P/$t --neighborcache_disk_cache_ratio 0.15 --neighborcache_disk_cache_layout directed_seed_replica_pct100"
  run $R/$t/build $q $gt 5000 $DC
  sync; sleep 3
  for cap in 5000 10000 20000 40000 60000 100000 1000000; do
    o=$R/$t/c$cap
    run $o $q $gt $cap $DC --neighborcache_disk_cache_reuse_prefix $R/$t/build/result --dump_query_stats $o/qstats.csv
    r=$(grep -E '^\s+128\s+4\s' $o/run.out | tail -1)
    log "[$t] cap=$cap QPS=$(echo $r | awk '{print $3}') recall=$(echo $r | awk '{print $NF}') $(grep -o 'base_reads=.*' $o/run.out | tail -1) $(split $o/qstats.csv) $(grep -o 'inserts=[0-9]* evictions=[0-9]*' $o/run.out | tail -1)"
  done
  rm -f $R/$t/build/*_neighborcache_dc.data $R/$t/build/*_neighborcache_dc.nodes $R/$t/build/*_neighborcache_dc.seed_* $P/${t}_*.bin
done
log CACHECAP_DONE
