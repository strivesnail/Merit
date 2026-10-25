#!/usr/bin/env bash
# Fig.1 queue: sift100m -> turing100m -> sift1b -> sift10m (skip systems with missing indexes).
set -uo pipefail
cd "$(dirname "$0")"
LOG=/mnt/raid5/fig1_x3/queue.out
mkdir -p /mnt/raid5/fig1_x3
echo "$(date -Is) queue start" | tee -a "$LOG"
# If sift100m already running/done, continue others.
for ds in sift100m turing100m sift1b sift10m; do
  if [[ -f /mnt/raid5/fig1_x3/$ds/done || -f /mnt/raid5/fig1_x3/$ds/status.txt ]] \
     && grep -q 'Fig.1 DONE' /mnt/raid5/fig1_x3/$ds/status.txt 2>/dev/null; then
    echo "$(date -Is) skip done $ds" | tee -a "$LOG"
    continue
  fi
  # avoid double-start if this ds is currently running
  if pgrep -f "DS=$ds .*run_fig1_one" >/dev/null 2>&1 || pgrep -f "run_fig1_one.sh" >/dev/null 2>&1; then
    if [[ -f /mnt/raid5/fig1_x3/$ds/status.txt ]] && ! grep -q 'Fig.1 DONE' /mnt/raid5/fig1_x3/$ds/status.txt; then
      echo "$(date -Is) wait for in-flight $ds" | tee -a "$LOG"
      while ! grep -q 'Fig.1 DONE\|ALLDONE' /mnt/raid5/fig1_x3/$ds/status.txt 2>/dev/null; do sleep 60; done
      continue
    fi
  fi
  echo "$(date -Is) start $ds" | tee -a "$LOG"
  DS=$ds THREADS=16 SKIP=3000 LS="80 100 120 150" \
    SYSTEMS="diskann bfs neighborcache starling margo" \
    bash run_fig1_one.sh >> /mnt/raid5/fig1_x3/$ds/pipeline.out 2>&1 \
    || echo "$(date -Is) FAIL $ds" | tee -a "$LOG"
done
echo "$(date -Is) queue ALLDONE" | tee -a "$LOG"
