#!/bin/bash
# D-cache only (no N-cache) at several capacities, reusing the profiles recorded by
# run_100m_cluster_hotspot_fixed.sh. Baseline is DiskANN without any cache (the D-cache lives on SSD).
export PATH=/home/jianz/miniconda3/bin:$PATH
set -uo pipefail
SEARCH=/home/jianz/workload/code/NeighborCache/diskann/build/apps/search_disk_index
LIB=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
INDEX=${INDEX:-/mnt/raid5/sift100m/sift100m_index}
WL=/home/jianz/workload/data/sift100m/workloads/cluster_hotspot
SRC=${SRC:-/mnt/raid5/sift100m/runs/cluster_hotspot_fixed_20261010}
ROOT=${ROOT:-/mnt/raid5/sift100m/runs/dcache_only_$(date +%Y%m%d)}
TAGS=${TAGS:-"x1 x2 x10 x50"}
RATIOS=${RATIOS:-"0.001 0.005 0.02"}
LAYOUT=${LAYOUT:-directed_beam}
L=${L:-100}; W=4; THREADS=20
NCACHE=${NCACHE:-0}
mkdir -p "$ROOT"
log() { echo "$(date -Is) $*" | tee -a "$ROOT/status.txt"; }

run() { # $1=out dir $2=query $3=gt, rest = extra args
  local out=$1 q=$2 gt=$3; shift 3
  mkdir -p "$out"
  env LD_LIBRARY_PATH="$LIB" OMP_NUM_THREADS=$THREADS NEIGHBORCACHE_USE_RAMFS=0 NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1 \
    NEIGHBORCACHE_NCACHE_CAP_NODES=$NCACHE \
    "$SEARCH" --data_type uint8 --dist_fn l2 --index_path_prefix "$INDEX" --query_file "$q" --gt_file "$gt" \
    --recall_at 10 --search_list $L --beamwidth $W --num_threads $THREADS --result_path "$out/result" \
    --num_nodes_to_cache "${NNC:-0}" "$@" > "$out/run.out" 2>&1
  [ "${KEEP_DC:-0}" = 1 ] || rm -f "$out"/*.dyn.data "$out"/*_neighborcache_dc.data
  grep -E "^\s+$L\s+$W\s" "$out/run.out" | tail -1 | awk '{printf "QPS=%s IOs=%s recall=%s", $3, $6, $NF}'
}

for t in $TAGS; do
  p=$WL/${WPRE:-cluster${KC:-100}}_${t}${WSUF:-_fixed}
  for mode in ${MODES:-same split}; do
    if [ $mode = same ]; then q=$p.u8bin; gt=${p}_gt10.bin; else q=${p}_B.u8bin; gt=${p}_B_gt10.bin; fi
    prof=$SRC/${mode}_$t/profile
    if [ "${RECORD_PROFILE:-0}" = 1 ]; then
      prof=$ROOT/${mode}_$t/profile
      run "$ROOT/${mode}_$t/profile_run" "$q" "$gt" --enable_access_profile --access_profile_prefix "$prof" > /dev/null
      rm -f "${prof}_node_visit.bin" "${prof}_node_out_heat.bin"
      sync; sleep 3
      log "[$mode $t] profile recorded at L=$L"
    fi
    rs=$RATIOS
    if [ -n "${UNIQUE_FRAC:-}" ]; then
      # Capacity = UNIQUE_FRAC of the distinct nodes in the profile, as a fraction of index pages.
      rs=$(python -c "import numpy as np; x=np.fromfile('${prof}_node_expand.bin',dtype=np.uint64,offset=8); print(f'{${UNIQUE_FRAC}*(x>0).sum()/10/1e7:.6f}')")
    fi
    [ "${SKIP_NOCACHE:-0}" = 1 ] || log "[$mode $t] nocache $(run "$ROOT/${mode}_$t/nocache" "$q" "$gt")"
    # Static N-cache: Top-NCACHE nodes by profile expand count, in memory. The D-cache keeps its own
    # contents (no exclusion) so that it matches the D-cache-only runs.
    ncargs=()
    if [ "$NCACHE" -gt 0 ]; then
      ncargs=(--neighborcache_memory_gb 1 --neighborcache_disk_cache_exclude_memory false)
      log "[$mode $t] bfs_N$NCACHE $(NNC=$NCACHE run "$ROOT/${mode}_$t/bfs_n" "$q" "$gt")"
      log "[$mode $t] ncache_only N=$NCACHE $(run "$ROOT/${mode}_$t/n_only" "$q" "$gt" "${ncargs[@]}" \
        --neighborcache_profile_prefix "$prof")"
    fi
    for f in ${UNIQUE_FRACS:-}; do
      case $f in
        d*) r=${f#d} ;;
        *) r=$(python -c "import numpy as np; x=np.fromfile('${prof}_node_expand.bin',dtype=np.uint64,offset=8); print(f'{${f}*(x>0).sum()/10/1e7:.6f}')") ;;
      esac
      # Build the D-cache file first and flush it; measuring right after the build races with writeback.
      b="$ROOT/${mode}_$t/u${f}_build"
      KEEP_DC=1 run "$b" "$q" "$gt" --neighborcache_profile_prefix "$prof" --neighborcache_disk_cache_ratio "$r" \
        --neighborcache_disk_cache_layout "$LAYOUT" > /dev/null
      sync; sleep 3
      log "[$mode $t] unique=$f d=$r $(run "$ROOT/${mode}_$t/u$f" "$q" "$gt" --neighborcache_profile_prefix "$prof" \
        --neighborcache_disk_cache_ratio "$r" --neighborcache_disk_cache_layout "$LAYOUT" --neighborcache_disk_cache_reuse_prefix "$b/result" \
        ${ncargs[@]+"${ncargs[@]}"})"
      rm -f "$b"/*_neighborcache_dc.data
    done
    [ "${DEL_PROFILE:-0}" = 1 ] && [ "${RECORD_PROFILE:-0}" = 1 ] && rm -f "${prof}"_*.bin
    [ -n "${UNIQUE_FRACS:-}" ] && continue
    for r in $rs; do
      log "[$mode $t] d=$r $(run "$ROOT/${mode}_$t/d$r" "$q" "$gt" --neighborcache_profile_prefix "$prof" \
        --neighborcache_disk_cache_ratio "$r" --neighborcache_disk_cache_layout "$LAYOUT")"
    done
  done
done
log "DCACHE ONLY DONE"
