#!/usr/bin/env bash
# Systematic checks for Mean IO (us) vs Disk Reads: sidecar vs unified, sector cache, run order, reps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DISKANN_BUILD="${DISKANN_BUILD:-$(cd "${SCRIPT_DIR}/../../diskann/build" && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
QUERY_FILE="${QUERY_FILE:-${DATA_DIR}/sift_query.fbin}"
PROFILE="${PROFILE:-${DATA_DIR}/run2_profile_same_trace}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/disk_io_diagnosis_$(date +%Y%m%d_%H%M%S)}"
SEARCH="${DISKANN_BUILD}/apps/search_disk_index"
THREADS="${THREADS:-16}"
L="${L:-100}"
K="${K:-10}"
W="${W:-2}"
REPS="${REPS:-3}"
DISK_RATIO="${DISK_RATIO:-0.1}"
DISK_K_HOPS="${DISK_K_HOPS:-2}"

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
: > "${SUMMARY}"
echo -e "case\trep\tQPS\tMeanLat_us\tMeanIOs\tBasePages\tSidecarPg\tDiskReads\tSectCacheHit\tIoUsPerRead\tMeanIO_us\tMeritDcHit\tRecall" >> "${SUMMARY}"

COMMON=(
  --data_type float --dist_fn l2
  --index_path_prefix "${DATA_DIR}/sift1m_index"
  --query_file "${QUERY_FILE}"
  --gt_file "${DATA_DIR}/sift_groundtruth.bin"
  --recall_at "${K}" --search_list "${L}" --beamwidth "${W}"
  --num_threads "${THREADS}"
  --num_nodes_to_cache 0
)

parse_row() {
  python3 - "$1" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
hdr = None
row = None
for line in text.splitlines():
    if re.search(r"^\s+L\s+Beamwidth", line):
        hdr = line.split()
    m = re.match(r"^\s+100\s+2\s+", line)
    if m:
        row = line.split()
if not hdr or not row:
    sys.exit(1)
# find column indices from header tokens (multi-word names)
cols = []
i = 0
while i < len(hdr):
    if hdr[i] == "Mean" and i + 1 < len(hdr) and hdr[i + 1] == "IO":
        if i + 2 < len(hdr) and hdr[i + 2] == "(us)":
            cols.append(("MeanIO_us", i))
            i += 3
            continue
        cols.append(("MeanIOs", i))
        i += 2
        continue
    if hdr[i] == "Mean" and i + 1 < len(hdr) and hdr[i + 1] == "Latency":
        cols.append(("MeanLat", i)); i += 2; continue
    if hdr[i] == "99.9" and i + 1 < len(hdr) and hdr[i + 1] == "Latency":
        i += 2; continue
    if hdr[i] == "Disk" and i + 1 < len(hdr) and hdr[i + 1] == "Reads":
        cols.append(("DiskReads", i)); i += 2; continue
    if hdr[i] == "IoUs/Read":
        cols.append(("IoUsPerRead", i)); i += 1; continue
    if hdr[i] == "BasePages":
        cols.append(("BasePages", i)); i += 1; continue
    if hdr[i] == "SidecarPg":
        cols.append(("SidecarPg", i)); i += 1; continue
    if hdr[i] == "SectCacheHit":
        cols.append(("SectCacheHit", i)); i += 1; continue
    if hdr[i] == "MeritDcHit":
        cols.append(("MeritDcHit", i)); i += 1; continue
    if hdr[i] == "Recall@10":
        cols.append(("Recall", i)); i += 1; continue
    if hdr[i] == "Beamwidth":
        i += 1; continue
    if hdr[i] in ("L", "QPS", "Dup"):
        if hdr[i] == "QPS":
            cols.append(("QPS", i))
        i += 1
        continue
    i += 1

def val(name, default=""):
    for n, idx in cols:
        if n == name:
            # row aligns with hdr positions only for fixed-width; use offset from QPS
            pass
    # Re-parse: data row fields align with header by splitting fixed columns is fragile.
    # Header order after our app is stable; map by scanning header list linearly to data index.
    pass

# Linear token align: skip L, Beamwidth, then same sequence as hdr after beamwidth
hi = hdr.index("Beamwidth") + 1
di = 2  # after 100 and 2
out = {}
while hi < len(hdr) and di < len(row):
    tok = hdr[hi]
    if tok == "Mean" and hi + 1 < len(hdr) and hdr[hi + 1] == "Latency":
        out["MeanLat"] = row[di]; hi += 2; di += 1; continue
    if tok == "99.9":
        hi += 2; di += 1; continue
    if tok == "Mean" and hi + 1 < len(hdr) and hdr[hi + 1] == "IOs":
        out["MeanIOs"] = row[di]; hi += 2; di += 1; continue
    if tok == "BasePages":
        out["BasePages"] = row[di]; hi += 1; di += 1; continue
    if tok == "SidecarPg":
        out["SidecarPg"] = row[di]; hi += 1; di += 1; continue
    if tok == "Dup":
        hi += 2; di += 1; continue
    if tok == "Disk" and hi + 1 < len(hdr) and hdr[hi + 1] == "Reads":
        out["DiskReads"] = row[di]; hi += 2; di += 1; continue
    if tok == "SectCacheHit":
        out["SectCacheHit"] = row[di]; hi += 1; di += 1; continue
    if tok == "IoUs/Read":
        out["IoUsPerRead"] = row[di]; hi += 1; di += 1; continue
    if tok == "|dSec|":
        hi += 1; di += 1; continue
    if tok == "Jump<=8s%":
        hi += 1; di += 1; continue
    if tok == "MeritDcHit":
        out["MeritDcHit"] = row[di]; hi += 1; di += 1; continue
    if tok == "Mean" and hi + 1 < len(hdr) and hdr[hi + 1] == "Hops":
        hi += 2; di += 1; continue
    if tok == "Mean" and hi + 1 < len(hdr) and hdr[hi + 1] == "IO" and hi + 2 < len(hdr) and hdr[hi + 2] == "(us)":
        out["MeanIO_us"] = row[di]; hi += 3; di += 1; continue
    if tok == "CPU":
        hi += 2; di += 1; continue
    if tok == "Recall@10":
        out["Recall"] = row[di]; hi += 1; di += 1; continue
    if tok == "QPS":
        out["QPS"] = row[di]; hi += 1; di += 1; continue
    hi += 1; di += 1

for k in ("QPS", "MeanLat", "MeanIOs", "BasePages", "SidecarPg", "DiskReads", "SectCacheHit", "IoUsPerRead", "MeanIO_us", "MeritDcHit", "Recall"):
    print(out.get(k, ""))
PY
}

run_one() {
  local case="$1"
  local rep="$2"
  shift 2
  local log="${OUT_DIR}/${case}_r${rep}.out"
  echo "[$(date +%H:%M:%S)] ${case} rep=${rep}"
  "$SEARCH" "$@" >"${log}" 2>&1
  read -r qps lat ios base side dr sc iur mio mdc rec <<< "$(parse_row "${log}")"
  [[ -z "${side}" ]] && side="-"
  [[ -z "${mdc}" ]] && mdc="-"
  echo -e "${case}\t${rep}\t${qps}\t${lat}\t${ios}\t${base}\t${side}\t${dr}\t${sc}\t${iur}\t${mio}\t${mdc}\t${rec}" >> "${SUMMARY}"
}

echo "Disk IO diagnosis -> ${OUT_DIR}"
echo "REPS=${REPS} profile=${PROFILE}"

DISK_COMMON=(
  "${COMMON[@]}"
  --merit_disk_cache_ratio "${DISK_RATIO}"
  --merit_profile_prefix "${PROFILE}"
  --merit_disk_cache_k_hops "${DISK_K_HOPS}"
)

for r in $(seq 1 "${REPS}"); do
  # H2: repeat variance — interleave cases each round
  run_one baseline "${r}" \
    "${COMMON[@]}" --enable_query_sector_cache \
    --result_path "${OUT_DIR}/baseline_r${r}"

  run_one sidecar_qsc "${r}" \
    "${DISK_COMMON[@]}" --enable_query_sector_cache \
    --merit_unified_disk_cache false \
    --result_path "${OUT_DIR}/sidecar_qsc_r${r}"

  run_one unified_qsc "${r}" \
    "${DISK_COMMON[@]}" --enable_query_sector_cache \
    --merit_unified_disk_cache true \
    --result_path "${OUT_DIR}/unified_qsc_r${r}"

  run_one sidecar_no_qsc "${r}" \
    "${DISK_COMMON[@]}" \
    --merit_unified_disk_cache false \
    --result_path "${OUT_DIR}/sidecar_no_qsc_r${r}"
done

# H4: cold page cache (optional — may need sudo)
if [[ "${DROP_CACHES:-0}" == "1" ]]; then
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true
  run_one baseline_cold 1 "${COMMON[@]}" --enable_query_sector_cache --result_path "${OUT_DIR}/baseline_cold"
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true
  run_one sidecar_cold 1 "${DISK_COMMON[@]}" --enable_query_sector_cache \
    --merit_unified_disk_cache false --result_path "${OUT_DIR}/sidecar_cold"
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true
  run_one unified_cold 1 "${DISK_COMMON[@]}" --enable_query_sector_cache \
    --merit_unified_disk_cache true --result_path "${OUT_DIR}/unified_cold"
fi

echo ""
echo "===== aggregate (Mean IO us: mean ± std over reps) ====="
python3 - "${SUMMARY}" <<'PY'
import sys, statistics
from collections import defaultdict
path = sys.argv[1]
rows = []
with open(path) as f:
    hdr = f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 12:
            continue
        case, rep = p[0], p[1]
        def ffloat(x):
            try: return float(x)
            except: return None
        rows.append((case, int(rep), {k: ffloat(v) for k, v in zip(
            ["QPS","MeanLat","MeanIOs","BasePages","SidecarPg","DiskReads","SectCacheHit","IoUsPerRead","MeanIO_us","MeritDcHit","Recall"],
            p[2:])}))

by = defaultdict(list)
for case, rep, m in rows:
    if m["MeanIO_us"] is not None:
        by[case].append(m)

for case in sorted(by.keys()):
    ms = by[case]
    def avg(k):
        xs = [m[k] for m in ms if m.get(k) is not None]
        return statistics.mean(xs) if xs else float('nan')
    def std(k):
        xs = [m[k] for m in ms if m.get(k) is not None]
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0
    print(f"{case:20s}  MeanIO_us={avg('MeanIO_us'):7.1f}±{std('MeanIO_us'):5.1f}  "
          f"DiskRd={avg('DiskReads'):6.2f}  IoUs/Rd={avg('IoUsPerRead'):5.2f}  "
          f"BasePg={avg('BasePages'):6.2f}  SidePg={avg('SidecarPg') if avg('SidecarPg')==avg('SidecarPg') else '-':>6}  "
          f"QPS={avg('QPS'):7.1f}")
PY

echo "Full TSV: ${SUMMARY}"
