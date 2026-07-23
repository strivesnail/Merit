#!/usr/bin/env bash
# SIFT 100M — same as DiskANN / Big-ANN Benchmarks (bigann-100M):
#   https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py
#   BigANNDataset(nb_M=100): first 100M vectors of base.1B.u8bin + public query + GT_100M/bigann-100M
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift100m}"
NB="${NB:-100000000}"
DIM="${DIM:-128}"
ORIG_NB="${ORIG_NB:-1000000000}"

BASE_URL="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann"
GT_URL="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/bigann-100M"

DS_FN="base.1B.u8bin"
QS_FN="query.public.10K.u8bin"
CROP_FN="${DS_FN}.crop_nb_${NB}"

# DiskANN build/search paths (uint8 competition format)
BASE_OUT="${DATA_DIR}/sift_base.u8bin"
QUERY_OUT="${DATA_DIR}/sift_query.u8bin"
GT_OUT="${DATA_DIR}/sift_groundtruth.bin"

LOG="${DATA_DIR}/download_bigann100m.log"

mkdir -p "${DATA_DIR}"

crop_bytes() {
  python3 - <<PY
nb, dim = int(${NB}), int(${DIM})
print(8 + nb * dim)  # uint8 u8bin header + payload
PY
}

patch_crop_header() {
  local f="$1"
  python3 - <<PY
import numpy as np
path = "${f}"
nb, orig, dim = int(${NB}), int(${ORIG_NB}), int(${DIM})
h = np.memmap(path, dtype=np.uint32, mode="r+", shape=2)
assert int(h[0]) == orig and int(h[1]) == dim, (int(h[0]), int(h[1]))
h[0] = nb
del h
print(f"patched header -> npts={nb} dim={dim}")
PY
}

download_crop_base() {
  local crop_path="${DATA_DIR}/${CROP_FN}"
  local need
  need="$(crop_bytes)"
  if [[ -f "${crop_path}" ]] && [[ "$(stat -c%s "${crop_path}")" -eq "${need}" ]]; then
    echo "Cropped base already present: ${crop_path}"
  else
    echo "=== Download cropped base (~$((need / 1024 / 1024 / 1024)) GiB) from ${BASE_URL}/${DS_FN} ==="
    rm -f "${crop_path}.partial"
    wget -c -O "${crop_path}.partial" --quota="${need}" "${BASE_URL}/${DS_FN}"
    mv -f "${crop_path}.partial" "${crop_path}"
    patch_crop_header "${crop_path}"
  fi
  ln -sf "${CROP_FN}" "${BASE_OUT}"
}

download_small() {
  local url="$1" out="$2" name="$3"
  if [[ -f "${out}" ]]; then
    echo "${name} OK: ${out}"
    return 0
  fi
  echo "=== Download ${name} ==="
  wget -c -O "${out}" "${url}"
}

{
  echo "download_sift100m_bigann_competition $(date -Is)"
  echo "DATA_DIR=${DATA_DIR} NB=${NB}"
  echo "Ref: big-ann-benchmarks BigANNDataset(100) / DiskANN T2 BIGANN"

  download_small "${BASE_URL}/${QS_FN}" "${DATA_DIR}/${QS_FN}" "query"
  ln -sf "${QS_FN}" "${QUERY_OUT}"

  download_small "${GT_URL}" "${DATA_DIR}/bigann-100M" "ground truth"
  ln -sf "bigann-100M" "${GT_OUT}"

  download_crop_base

  echo ""
  echo "Done $(date -Is)"
  ls -lh "${DATA_DIR}/${CROP_FN}" "${DATA_DIR}/${QS_FN}" "${DATA_DIR}/bigann-100M"
  echo "Use: --data_type uint8 --dist_fn l2"
  echo "  base=${BASE_OUT} query=${QUERY_OUT} gt=${GT_OUT}"
} 2>&1 | tee -a "${LOG}"
