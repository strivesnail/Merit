#!/usr/bin/env bash
# Shared tmpfs (memory filesystem) staging for MERIT / DiskANN search benchmarks.
#
# Usage (in any test script, after DATA_DIR is set):
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/merit_ramfs_env.sh"
#   PERSIST_DATA_DIR="${DATA_DIR}"
#   merit_ramfs_activate "${PERSIST_DATA_DIR}"
#   DATA_DIR="${MERIT_RAMFS_ACTIVE_DATA_DIR}"
#   # Keep logs/results on persistent disk, e.g. OUT_DIR="${PERSIST_DATA_DIR}/runs/..."
#
# Disable: MERIT_USE_RAMFS=0
# Override mount dir: MERIT_RAMFS_ROOT=/tmp/my_ramfs
# Override tmpfs size when explicit mount is needed: MERIT_RAMFS_SIZE=32G

merit_ramfs_enabled() {
  [[ "${MERIT_USE_RAMFS:-1}" != "0" ]]
}

merit_ramfs_root() {
  echo "${MERIT_RAMFS_ROOT:-/tmp/merit_ramfs_${USER:-merit}}"
}

merit_ramfs_log() {
  echo "MERIT ramfs: $*" >&2
}

merit_ramfs_on_tmpfs() {
  local path="$1"
  findmnt -T "${path}" -o FSTYPE -n 2>/dev/null | grep -qx tmpfs
}

merit_ramfs_ensure_mount() {
  local root="$1"
  mkdir -p "${root}"
  if merit_ramfs_on_tmpfs "${root}"; then
    merit_ramfs_log "using tmpfs at ${root} ($(df -h "${root}" | tail -1 | awk '{print $2, "total,", $4, "avail"}'))"
    return 0
  fi

  local size="${MERIT_RAMFS_SIZE:-16G}"
  merit_ramfs_log "mounting tmpfs size=${size} at ${root}"
  if sudo mount -t tmpfs -o "size=${size}" tmpfs "${root}"; then
    merit_ramfs_log "mounted tmpfs at ${root}"
    return 0
  fi

  merit_ramfs_log "WARN: could not mount tmpfs; ${root} stays on $(findmnt -T "${root}" -o FSTYPE -n 2>/dev/null || echo unknown)"
}

merit_ramfs_copy_if_missing() {
  local src="$1" dst="$2"
  [[ -e "${src}" ]] || return 0
  if [[ -e "${dst}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${dst}")"
  merit_ramfs_log "copy $(basename "${src}") ..."
  cp -a "${src}" "${dst}"
}

merit_ramfs_copy_glob_if_missing() {
  local src_dir="$1" dst_dir="$2" pattern="$3"
  shopt -s nullglob
  local f
  for f in "${src_dir}"/${pattern}; do
    merit_ramfs_copy_if_missing "${f}" "${dst_dir}/$(basename "${f}")"
  done
  shopt -u nullglob
}

merit_ramfs_compute_stamp() {
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

merit_ramfs_stage_dir() {
  local src_dir="$1" dst_dir="$2"

  mkdir -p "${dst_dir}"

  merit_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'sift1m_index*'
  merit_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'sift100k_index*'
  merit_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'sift100m_index*'
  merit_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'sift1m_relayout_index*'
  merit_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'sift1m_hotnode_relayout_index*'
  merit_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'relayout_order*'
  merit_ramfs_copy_if_missing "${src_dir}/sift_query.fbin" "${dst_dir}/sift_query.fbin"
  merit_ramfs_copy_if_missing "${src_dir}/sift_groundtruth.bin" "${dst_dir}/sift_groundtruth.bin"
  merit_ramfs_copy_glob_if_missing "${src_dir}" "${dst_dir}" 'run2_profile*'

  if [[ -d "${src_dir}/workloads" ]]; then
    if [[ ! -d "${dst_dir}/workloads" ]]; then
      merit_ramfs_log "copy workloads/ ..."
      cp -a "${src_dir}/workloads" "${dst_dir}/"
    fi
  fi

  if [[ ! -f "${dst_dir}/sift1m_index_disk.index" ]] && ! compgen -G "${dst_dir}/*index_disk.index" >/dev/null; then
    merit_ramfs_log "ERROR: staged dir missing *_disk.index under ${dst_dir}" >&2
    return 1
  fi
}

merit_ramfs_activate() {
  local src_dir="$1"
  src_dir="$(cd "${src_dir}" && pwd)"

  if ! merit_ramfs_enabled; then
    MERIT_RAMFS_ACTIVE_DATA_DIR="${src_dir}"
    export MERIT_RAMFS_ACTIVE_DATA_DIR PERSIST_DATA_DIR="${src_dir}"
    merit_ramfs_log "disabled (MERIT_USE_RAMFS=0); using ${src_dir}"
    return 0
  fi

  local root dst_dir stamp_file need_bytes avail_bytes
  root="$(merit_ramfs_root)"
  merit_ramfs_ensure_mount "${root}"

  dst_dir="${root}/$(basename "${src_dir}")"
  stamp_file="${dst_dir}/.merit_ramfs_stamp"

  need_bytes="$(du -sb "${src_dir}" 2>/dev/null | awk '{print $1}')"
  avail_bytes="$(df -B1 "${root}" 2>/dev/null | tail -1 | awk '{print $4}')"
  if [[ -n "${need_bytes}" && -n "${avail_bytes}" && "${need_bytes}" -gt $((avail_bytes * 9 / 10)) ]]; then
    merit_ramfs_log "WARN: ${src_dir} (~$((need_bytes / 1024 / 1024))MiB) exceeds ~90% tmpfs avail; using persistent storage"
    MERIT_RAMFS_ACTIVE_DATA_DIR="${src_dir}"
    export MERIT_RAMFS_ACTIVE_DATA_DIR PERSIST_DATA_DIR="${src_dir}"
    return 0
  fi

  local new_stamp=""
  new_stamp="$(merit_ramfs_compute_stamp "${src_dir}")"
  if [[ -f "${stamp_file}" && "$(cat "${stamp_file}")" == "${new_stamp}" && -f "${dst_dir}/sift1m_index_disk.index" ]]; then
    merit_ramfs_log "reuse staged ${dst_dir}"
  else
    merit_ramfs_log "staging ${src_dir} -> ${dst_dir}"
    rm -rf "${dst_dir}"
    merit_ramfs_stage_dir "${src_dir}" "${dst_dir}"
    echo "${new_stamp}" > "${stamp_file}"
  fi

  MERIT_RAMFS_ACTIVE_DATA_DIR="${dst_dir}"
  export MERIT_RAMFS_ACTIVE_DATA_DIR PERSIST_DATA_DIR="${src_dir}"
  merit_ramfs_log "active DATA_DIR=${MERIT_RAMFS_ACTIVE_DATA_DIR} (persistent ${PERSIST_DATA_DIR})"
}

# Resolve merit_ramfs_env.sh from sibling or Merit tree (for experiments/ wrappers).
merit_ramfs_source_path() {
  local here="$1"
  if [[ -f "${here}/merit_ramfs_env.sh" ]]; then
    echo "${here}/merit_ramfs_env.sh"
    return 0
  fi
  if [[ -f "${here}/../../Merit/experiments/test-scripts/merit_ramfs_env.sh" ]]; then
    echo "${here}/../../Merit/experiments/test-scripts/merit_ramfs_env.sh"
    return 0
  fi
  return 1
}
