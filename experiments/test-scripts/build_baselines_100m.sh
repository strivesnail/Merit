#!/usr/bin/env bash
# Starling and MARGO on the 100M datasets with the same pipelines as SIFT1B, plus Text2Image100M ground truth.
# Runs one stage at a time (62 GB RAM); a stage is skipped if done and the queue stops if raid5 lacks space.
# Never deletes existing files; only its own intermediates (gp work dirs, pre-layout MARGO disk index).
# Usage: nohup setsid bash build_baselines_100m.sh >> /mnt/raid5/build_baselines_100m.out 2>&1 < /dev/null &
set -euo pipefail
export PATH=/home/jianz/miniconda3/bin:$PATH
export LD_LIBRARY_PATH=/tmp/neighborcache-link-deps/root/usr/lib/x86_64-linux-gnu:/home/jianz/miniconda3/lib
REPO=/home/jianz/workload/code/NeighborCache
SB=/home/jianz/workload/code/baselines/starling
GPB=${SB}/graph_partition/build
MB=/home/jianz/workload/code/baselines/MARGO/build
T=${T:-24}
STATUS=/mnt/raid5/build_baselines_100m_status.txt

log() { echo "$(date -Is) $*" | tee -a "${STATUS}"; }
need_gb() {  # <GB needed> <what> [dir]
  local avail dir=${3:-/mnt/raid5}
  avail=$(df --output=avail -BG "${dir}" | tail -1 | tr -dc 0-9)
  if (( avail < $1 )); then
    log "STOP: $2 needs ${1}G, ${dir} has ${avail}G"
    exit 1
  fi
}
v5_pivots() {  # <src index prefix> -> path of the 5-section pivot file next to the output
  python3 ${REPO}/experiments/scripts/pq_pivots_to_v5.py "$1_pq_pivots.bin" "$2"
}
link_shared() {  # <prefix> <src index prefix> <v5 pivots>
  ln -sfn "$2_pq_compressed.bin" "$1_pq_compressed.bin"
  ln -sfn "$3" "$1_pq_pivots.bin"
  ln -sfn "$2_sample_data.bin" "$1_sample_data.bin"
  ln -sfn "$2_sample_ids.bin" "$1_sample_ids.bin"
}

starling() {  # <name> <dtype> <base> <src index prefix> <index GB>
  local name=$1 dtype=$2 base=$3 src=$4 gb=$5
  local out=/mnt/raid5/${name}/starling p=/mnt/raid5/${name}/starling/${name}
  [[ -f ${out}/done ]] && { log "skip starling ${name}"; return; }
  need_gb $((gb + 20)) "starling ${name}"
  mkdir -p "${out}/work" "${out}/mem_sample" "${out}/mem"
  log "starling ${name} start"
  [[ -f ${p}_partition.bin ]] || /usr/bin/time -v ${GPB}/ooc_partitioner --index_file "${src}_disk.index" \
    --gp_file "${p}_partition.bin" --work_dir "${out}/work" --data_type "${dtype}" -T ${T} -L 16 \
    --chunk_nodes 16000000 --rev_budget_gb 24 > "${out}/partition.log" 2>&1
  rm -rf "${out}/work"
  [[ -f ${out}/relayout.done ]] || {
    ${GPB}/ooc_relayout "${src}_disk.index" "${p}_partition.bin" "${p}_disk.index" ${T} > "${out}/relayout.log" 2>&1
    touch "${out}/relayout.done"; }
  ln -sfn "${src}_disk.index" "${p}_disk_beam_search.index"
  [[ -f ${out}/mem/_index ]] || {
    ${SB}/build/tests/utils/gen_random_slice "${dtype}" "${base}" "${out}/mem_sample/" 0.01 > "${out}/mem_sample/sample.log" 2>&1
    /usr/bin/time -v ${SB}/build/tests/build_memory_index --data_type "${dtype}" --dist_fn l2 \
      --data_path "${out}/mem_sample/" --index_path_prefix "${out}/mem/_index" -R 48 -L 128 --alpha 1.2 \
      > "${out}/mem/build.log" 2>&1; }
  v5_pivots "${src}" "${out}/pq_pivots_v5.bin"
  link_shared "${p}" "${src}" "${out}/pq_pivots_v5.bin"
  touch "${out}/done"
  log "starling ${name} done"
}

margo() {  # <name> <dtype> <base> <src index prefix> <index GB> <graph GB> [root]
  local name=$1 dtype=$2 base=$3 src=$4 gb=$5 ggb=$6 root=${7:-/mnt/raid5}
  local out=${root}/${name}/margo p=${root}/${name}/margo/${name}
  [[ -f ${out}/done ]] && { log "skip margo ${name}"; return; }
  need_gb $((2 * gb + 2 * ggb + 20)) "margo ${name}" "${root}"
  mkdir -p "${out}"
  log "margo ${name} start"
  if [[ ! -f ${out}/build.done ]]; then
    while :; do
      rc=0
      MARGO_ONE_SHARD_PER_PROCESS=1 /usr/bin/time -v ${MB}/tests/my_build_merged_disk_index "${dtype}" "${base}" "${p}" \
        64 100 24 ${T} >> "${out}/build.log" 2>&1 || rc=$?
      [[ ${rc} -eq 0 ]] && break
      [[ ${rc} -eq 2 ]] || { log "margo ${name} build failed rc=${rc}"; exit ${rc}; }
    done
    touch "${out}/build.done"
  fi
  [[ -f ${p}_partition.bin ]] || /usr/bin/time -v ${MB}/my_gp/ooc_mincut "${p}" "${base}" "${dtype}" "${out}/mincut_work" \
    ${T} 36 >> "${out}/mincut.log" 2>&1
  rm -rf "${out}/mincut_work"
  if [[ ! -f ${out}/relayout.done ]]; then
    ${GPB}/ooc_relayout "${p}_disk.index" "${p}_partition.bin" "${p}_disk_relayout.index" ${T} > "${out}/relayout.log" 2>&1
    mv "${p}_disk_relayout.index" "${p}_disk.index"
    touch "${out}/relayout.done"
  fi
  ln -sfn "${src}_disk.index" "${p}_disk_beam_search.index"
  v5_pivots "${src}" "${out}/pq_pivots_v5.bin"
  link_shared "${p}" "${src}" "${out}/pq_pivots_v5.bin"
  touch "${out}/done"
  log "margo ${name} done"
}

t2i_gt() {
  local d=/mnt/raid5/t2i100m
  [[ -f ${d}/gt.done ]] && return
  log "t2i gt start"
  /usr/bin/time -v ${REPO}/diskann/build/apps/utils/compute_groundtruth --data_type float --dist_fn l2 \
    --base_file ${d}/base100m.fbin --query_file ${d}/query.public.100K.fbin --gt_file ${d}/t2i-gt-100M \
    --K 100 > ${d}/gt.log 2>&1
  touch ${d}/gt.done
  log "t2i gt done"
}

S100=/mnt/raid5/sift100m/sift100m_index
TUR=/mnt/raid5/turing100m/turing100m_index
T2I=/mnt/raid5/t2i100m/t2i100m_index

t2i_gt
starling sift100m uint8 ${REPO}/data/sift100m/sift_base.u8bin ${S100} 41
margo sift100m uint8 ${REPO}/data/sift100m/sift_base.u8bin ${S100} 41 26
starling turing100m float /mnt/raid5/turing100m/base100m.fbin ${TUR} 69
margo turing100m float /mnt/raid5/turing100m/base100m.fbin ${TUR} 69 24
starling t2i100m float /mnt/raid5/t2i100m/base100m.fbin ${T2I} 137
# raid5 cannot hold it; build on the HDD and copy the disk index to NVMe for testing.
margo t2i100m float /mnt/raid5/t2i100m/base100m.fbin ${T2I} 137 26 /mnt/hdd/merit_park
log "ALLDONE"
