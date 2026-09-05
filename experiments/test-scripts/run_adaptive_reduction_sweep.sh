#!/usr/bin/env bash
# Adaptive baseline + two-axis disk-cache reduction sweep (SIFT1M workloads).
set -euo pipefail

export LD_LIBRARY_PATH="${HOME}/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MERIT_USE_RAMFS=0
export MERIT_RECORD_DRIVEN_SEED_ACCESS=1
export MERIT_ADAPTIVE_PARENT_OR_SELF=1
unset MERIT_ADAPTIVE_EXTENT MERIT_DISABLE_MULTIREAD MERIT_PARENT_SINGLE_PAGE MERIT_CACHE_REPLICA_FALLBACK

SEARCH="${SEARCH:-/home/jianz/Merit/diskann/build/apps/search_disk_index}"
DATA="${DATA:-/home/jianz/Merit/data/sift1m}"
OUT="${OUT:-$DATA/runs/adaptive_reduction_sweep}"
L=50; W=4; K=1; THREADS=8

declare -A RATIO=(
  [uniform]=0.906 [small_x]=0.783 [small_y]=0.476 [large_x]=0.871
  [large_y]=0.717 [x_only]=0.875 [x_then_y]=0.841 [y_only]=0.716
)

WORKLOADS=(uniform)
if [[ "${1:-}" == "--all" ]]; then
  WORKLOADS=(uniform small_x small_y large_x large_y x_only x_then_y y_only)
fi

BASE_INDEX="$DATA/sift1m_index_disk.index"
BASE_BYTES=$(stat -c%s "$BASE_INDEX")
FULLCOVER="$DATA/runs/seed_replica_fullcover_workloads"

mean_disk_reads() {
  python3 - "$1" <<'PY'
import csv, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(f"{st.mean(float(r['n_disk_reads']) for r in rows):.2f}")
PY
}

dc_ratio() {
  local f="$1"
  [[ -f "$f" ]] || { echo "NA"; return; }
  local sz
  sz=$(stat -c%s "$f")
  python3 - "$sz" "$BASE_BYTES" <<'PY'
import sys
print(f"{int(sys.argv[1]) / int(sys.argv[2]):.3f}")
PY
}

run_case() {
  local wl="$1" tag="$2" layout="$3" ratio="$4" reuse="${5:-}"
  local case_out="$OUT/$wl/$tag"
  mkdir -p "$case_out"
  local args=(
    --data_type float --dist_fn l2
    --index_path_prefix "$DATA/sift1m_index"
    --query_file "$DATA/workloads/${wl}_10k.fbin"
    --gt_file "$DATA/workloads/${wl}_10k_gt.bin"
    --result_path "$case_out/run"
    --recall_at "$K" --search_list "$L" --beamwidth "$W"
    --num_threads "$THREADS"
    --merit_disk_cache_layout "$layout"
    --merit_disk_cache_ratio "$ratio"
    --dump_query_stats "$case_out/qstats.csv"
  )
  if [[ -n "$reuse" ]]; then
    args+=(
      --merit_disk_cache_reuse_prefix "$reuse"
      --access_profile_prefix "$DATA/workloads/profiles/${wl}_10k"
    )
  else
    args+=(--access_profile_prefix "$DATA/workloads/profiles/${wl}_10k")
  fi
  "$SEARCH" "${args[@]}" >"$case_out/run.out" 2>&1
  local reads dc
  reads=$(mean_disk_reads "$case_out/qstats.csv")
  dc=$(dc_ratio "$case_out/run_merit_dc.data")
  echo "$wl $tag reads=$reads dc=$dc layout=$layout ratio=$ratio" | tee -a "$OUT/summary.log"
}

mkdir -p "$OUT"
: >"$OUT/summary.log"

for wl in "${WORKLOADS[@]}"; do
  r="${RATIO[$wl]}"
  reuse="$FULLCOVER/$wl/seed_replica"

  echo "===== $wl adaptive baseline (reuse full-cover cache) ====="
  run_case "$wl" "adaptive_baseline" "directed_seed_replica_pct100" "$r" "$reuse"

  for br in 0.75 0.65 0.55 0.50; do
    run_case "$wl" "seed_budget_r${br}" "directed_seed_replica_budget_pct100" "$br"
  done

  for br in 0.75 0.65 0.55; do
    run_case "$wl" "seed_benefit_r${br}" "directed_seed_replica_benefit_pct100" "$br"
  done

  for pct in pct90 pct80 pct50; do
    run_case "$wl" "nbr_${pct}" "directed_seed_replica_${pct}" "$r"
  done

  run_case "$wl" "combo_benefit80_r065" "directed_seed_replica_benefit_pct80" "0.65"
  run_case "$wl" "combo_pct90_r075" "directed_seed_replica_pct90" "0.75"
done

echo "Done. See $OUT/summary.log"
