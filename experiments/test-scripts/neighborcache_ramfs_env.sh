#!/usr/bin/env bash
# Shared tmpfs (memory filesystem) staging for NEIGHBORCACHE / DiskANN search benchmarks.
#
# Usage (in any test script, after DATA_DIR is set):
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/neighborcache_ramfs_env.sh"
#   PERSIST_DATA_DIR="${DATA_DIR}"
#   neighborcache_ramfs_activate "${PERSIST_DATA_DIR}"
#   DATA_DIR="${NEIGHBORCACHE_RAMFS_ACTIVE_DATA_DIR}"
#   # Keep logs/results on persistent disk, e.g. OUT_DIR="${PERSIST_DATA_DIR}/runs/..."
#
# Disable: NEIGHBORCACHE_USE_RAMFS=0
# Override mount dir: NEIGHBORCACHE_RAMFS_ROOT=/tmp/my_ramfs
# Override tmpfs size when explicit mount is needed: NEIGHBORCACHE_RAMFS_SIZE=32G

neighborcache_ramfs_enabled() {
  [[ "${NEIGHBORCACHE_USE_RAMFS:-1}" != "0" ]]
}

neighborcache_ramfs_root() {
  echo "${NEIGHBORCACHE_RAMFS_ROOT:-/tmp/neighborcache_ramfs_${USER:-neighborcache}}"
}

neighborcache_ramfs_log() {
  echo "NEIGHBORCACHE ramfs: $*" >&2
}

neighborcache_ramfs_on_tmpfs() {
  local path="$1"
  findmnt -T "${path}" -o FSTYPE -n 2>/dev/null | grep -qx tmpfs
}

neighborcache_ramfs_ensure_mount() {
  local root="$1"
  mkdir -p "${root}"
  if neighborcache_ramfs_on_tmpfs "${root}"; then
    neighborcache_ramfs_log "using tmpfs at ${root} ($(df -h "${root}" | tail -1 | awk '{print $2, "total,", $4, "avail"}'))"
    return 0
  fi

  local size="${NEIGHBORCACHE_RAMFS_SIZE:-16G}"
  neighborcache_ramfs_log "mounting tmpfs size=${size} at ${root}"
  if sudo mount -t tmpfs -o "size=${size}" tmpfs "${root}"; then
    neighborcache_ramfs_log "mounted tmpfs at ${root}"
    return 0
  fi

  neighborcache_ramfs_log "WARN: could not mount tmpfs; ${root} stays on $(findmnt -T "${root}" -o FSTYPE -n 2>/dev/null || echo unknown)"
}

neighborcache_ramfs_copy_if_missing() {
  local src="$1" dst="$2"
  [[ -e "${src}" ]] || return 0
  if [[ -e "${dst}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${dst}")"
  neighborcache_ramfs_log "copy $(basename "${src}") ..."
  cp -a "${src}" "${dst}"
}

neighborcache_ramfs_copy_glob_if_missing() {
  local src_dir="$1" dst_dir="$2" pattern="$3"
  shopt -s nullglob
  local f
  for f in "${src_dir}"/${pattern}; do
    neighborcache_ramfs_copy_if_missing "${f}" "${dst_dir}/$(basename "${f}")"
  done
  shopt -u nullglob
}

neighborcache_ramfs_compute_stamp() {
  local src_dir="$1"
  {
    find "${src_dir}" -maxdepth 1 -type f \( \
      -name 'sift1m_index*' -o \
      -name 'sift1m_relayout_index*' -o \
      -name 'sift1m_hotnode_relayout_index*' -o \
      -name 'sift_query.fbin' -o \
      -name 'sift_groundtruth.bin' -o \
      -name 'run2_profile*' -o \
      -name 'relayout_order*' \
      \) -printf '%s %T@ %f\n' 2>/dev/null || true
    if [[ -d "${src_dir}/workloads" ]]; then
      find "${src_dir}/workloads" -type f -printf '%s %T@ %P\n' 2>/dev/null | head -5000 || true
    fi
  } | sort | sha256sum | awk '{print $1}'
}

neighborcache_ramfs_stage_dir() {
  local src_dir="$1" dst_dir="$2"

  mkdir -p "${dst_dir}"

  neighborcache_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'sift1m_index*'
  neighborcache_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'sift100k_index*'
  neighborcache_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'sift100m_index*'
  neighborcache_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'sift1m_relayout_index*'
  neighborcache_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'sift1m_hotnode_relayout_index*'
  neighborcache_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'relayout_order*'
  neighborcache_ramfs_copy_if_missing "${src_dir}/sift_query.fbin" "${dst_dir}/sift_query.fbin"
  neighborcache_ramfs_copy_if_missing "${src_dir}/sift_groundtruth.bin" "${dst_dir}/sift_groundtruth.bin"
  neighborcache_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'run2_profile*'

  if [[ -d "${src_dir}/workloads" ]]; then
    if [[ ! -d "${dst_dir}/workloads" ]]; then
      neighborcache_ramfs_log "copy workloads/ ..."
      cp -a "${src_dir}/workloads" "${dst_dir}/"
    fi
  fi

  if [[ ! -f "${dst_dir}/sift1m_index_disk.index" ]] && ! compgen -G "${dst_dir}/*index_disk.index" >/dev/null; then
    neighborcache_ramfs_log "ERROR: staged dir missing *_disk.index under ${dst_dir}" >&2
    return 1
  fi
}

neighborcache_ramfs_activate() {
  local src_dir="$1"
  src_dir="$(cd "${src_dir}" && pwd)"

  if ! neighborcache_ramfs_enabled; then
    NEIGHBORCACHE_RAMFS_ACTIVE_DATA_DIR="${src_dir}"
    export NEIGHBORCACHE_RAMFS_ACTIVE_DATA_DIR PERSIST_DATA_DIR="${src_dir}"
    neighborcache_ramfs_log "disabled (NEIGHBORCACHE_USE_RAMFS=0); using ${src_dir}"
    return 0
  fi

  local root dst_dir stamp_file need_bytes avail_bytes
  root="$(neighborcache_ramfs_root)"
  neighborcache_ramfs_ensure_mount "${root}"

  dst_dir="${root}/$(basename "${src_dir}")"
  stamp_file="${dst_dir}/.neighborcache_ramfs_stamp"

  need_bytes="$(du -sb "${src_dir}" 2>/dev/null | awk '{print $1}')"
  avail_bytes="$(df -B1 "${root}" 2>/dev/null | tail -1 | awk '{print $4}')"
  if [[ -n "${need_bytes}" && -n "${avail_bytes}" && "${need_bytes}" -gt $((avail_bytes * 9 / 10)) ]]; then
    neighborcache_ramfs_log "WARN: ${src_dir} (~$((need_bytes / 1024 / 1024))MiB) exceeds ~90% tmpfs avail; using persistent storage"
    NEIGHBORCACHE_RAMFS_ACTIVE_DATA_DIR="${src_dir}"
    export NEIGHBORCACHE_RAMFS_ACTIVE_DATA_DIR PERSIST_DATA_DIR="${src_dir}"
    return 0
  fi

  local new_stamp=""
  new_stamp="$(neighborcache_ramfs_compute_stamp "${src_dir}")"
  if [[ -f "${stamp_file}" && "$(cat "${stamp_file}")" == "${new_stamp}" && -f "${dst_dir}/sift1m_index_disk.index" ]]; then
    neighborcache_ramfs_log "reuse staged ${dst_dir}"
  else
    neighborcache_ramfs_log "staging ${src_dir} -> ${dst_dir}"
    rm -rf "${dst_dir}"
    neighborcache_ramfs_stage_dir "${src_dir}" "${dst_dir}"
    echo "${new_stamp}" > "${stamp_file}"
  fi

  NEIGHBORCACHE_RAMFS_ACTIVE_DATA_DIR="${dst_dir}"
  export NEIGHBORCACHE_RAMFS_ACTIVE_DATA_DIR PERSIST_DATA_DIR="${src_dir}"
  neighborcache_ramfs_log "active DATA_DIR=${NEIGHBORCACHE_RAMFS_ACTIVE_DATA_DIR} (persistent ${PERSIST_DATA_DIR})"
}

# Resolve neighborcache_ramfs_env.sh from sibling or NeighborCache tree (for experiments/ wrappers).
neighborcache_ramfs_source_path() {
  local here="$1"
  if [[ -f "${here}/neighborcache_ramfs_env.sh" ]]; then
    echo "${here}/neighborcache_ramfs_env.sh"
    return 0
  fi
  if [[ -f "${here}/../../NeighborCache/experiments/test-scripts/neighborcache_ramfs_env.sh" ]]; then
    echo "${here}/../../NeighborCache/experiments/test-scripts/neighborcache_ramfs_env.sh"
    return 0
  fi
  return 1
}
