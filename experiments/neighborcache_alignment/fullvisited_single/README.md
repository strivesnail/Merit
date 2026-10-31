# Full visited-seed D cache, small private prefetch, and an N-only curve

Artifacts: `/mnt/nvme2/nc_fullvisited_seed_single_20261031/`.

Single query worker, CPU 0; SIFT100M uint8 128D/R64, K=10, L=128, beam=4.
The input is the saved 20K stream (10K uniform + 10K Zipf, shuffled). Each timed
process runs the same stream twice: 20K runtime-cache warmup queries followed by
20K measured queries, using the native `--search_list 128 128` driver. File
caches are cleared with sync/fadvise and mincore verified before each process.
SSD hardware caches are not reset. The full-D point and selected N-only upper
point each have two runs; other N-only curve points are single-run observations.

## Full-D configuration and result

- All 1,338,445 distinct full-vector visited targets from this same workload are
  seeds. They came from 2,907,400 actual uncached base-frontier target acquisitions;
  PQ-only candidates are excluded. All these acquisitions are processed with
  full vectors. The profile's access sum and final IDs/distances are verified.
- Each seed owns one immutable 4 KiB D page: seed first, then up to nine distinct
  **visited** graph neighbors in adjacency order. Full vector and adjacency bytes
  are unchanged. The file has 8,619,889 non-padding record copies in total.
- D is 5,482,270,720 bytes, about 5.11 GiB, on SSD. A sorted seed-ID array uses
  5,353,780 RAM bytes (5.106 MiB) and maps a requested seed to its own D page.
- N=2,048 online LRU nodes; exact record payload=794,624 bytes (0.758 MiB).
- Prefetch=2,560 pages, 10 MiB, one private LRU retained across queries. It uses
  the production `NeighborCacheThreadPageCache`, including the compact resident
  record directory. Any resident page can satisfy its registered member records.
- M heat/seed-selection and dynamic maintenance are disabled. No payload stash,
  D repacking, background writeback or D writes are used. Clean eviction drops
  the prefetch copy. Only the requested target is expanded, not all page members.

| Metric | Full visited D + N + prefetch |
|---|---:|
| QPS, mean of two runs | 585.055 |
| Individual QPS | 588.33, 581.78 |
| N hits / all node accesses | 3.0291% |
| Prefetch hits / all node accesses | 59.7101% |
| D physical page reads / query | 54.16605 |
| Base-index reads during queries | 0 |
| D lookup hits / actual D probes after N/P misses | 100% |
| D coverage including P, after N misses | 100% |
| Estimated additional cache RAM | 17.543 MiB |

The RAM estimate is `(RSS after_search - RSS after_load) - common query delta`.
Full-D raw growth above the loaded PQ/index is 32.410 MiB. The uncached two-pass
run's common query/result/scratch growth is 14.867 MiB, giving 17.543 MiB.
Thus 17.543 MiB includes N, prefetch, the 5.106 MiB read-only D index and resident
metadata/allocator effects. It excludes the common approximately 3.75 GiB loaded
PQ/index footprint and SSD D payload. It is a differential working-set estimate,
not an exact allocation attribution or total process RSS.

## N-only memory curve

N-only uses the same native node pool and online LRU admission, no M/P/D and no
oracle/BFS prefill. Query-local original-page reuse and the warmup/measurement
stream are identical. The common query-memory delta is subtracted for both.

| N-only nodes | QPS | Estimated additional RAM, MiB |
|---:|---:|---:|
| 0 | 308.69 | 0 |
| 32,768 | 341.78 | 14.285 |
| 49,152 | 348.96 | 22.105 |
| 65,536 | 356.41 | 29.891 |
| 524,288 | 486.88 | 232.418 |
| 786,432 | 566.26 | 339.844 |
| 917,504 | 612.35 | 390.385 |
| 1,048,576 | 663.05 | 458.359 |

The observed target crossing is bracketed by 786,432 and 917,504 nodes. The
latter is a verified sufficient capacity, not a claimed exact minimum; its two
QPS values are 612.27 and 612.43. Connected plot segments merely connect measured
points. The green horizontal reference is 585.06 QPS; the vertical reference is
17.54 MiB total additional RAM for the full-D scheme. It is not N's 0.758 MiB
payload alone. `results/nonly_memory_qps.csv` contains the plotted values.

### Uniform memory-grid additions

At the user's request, `densify_nonly.py` retains these eight observations and
adds twelve measured capacities targeting approximately 40, 80, ..., 480 MiB of
N-only RAM. Capacities are chosen prospectively by inverting the existing
node-capacity versus RAM measurements and rounding to 1,024 nodes. Each new point
executes a fresh one-worker process with the same 20K warmup + 20K measured stream.
Actual RSS and QPS, rather than the requested grid or an interpolated QPS, are
plotted. The run manifest records the requested budget and actual node capacity.
Because allocator/hash capacity is discrete, actual RAM spacings are approximate.
The figure is updated after every completed point; incomplete trials are excluded.
The green reference legend is `NC`; its memory and QPS are unchanged.

The completed figure has 20 distinct N-only capacities. Independent repetitions
at nominal 80, 240 and 360 MiB differ by at most 0.183% in QPS, and all access
counts/output checks match. An additional full-D reference check gives 584.38
QPS, 0.115% below the original primary two-run mean; it is retained separately
as a stability control rather than changing the predeclared reference line.
The 360 MiB-grid point measures 361.061 MiB and 585.045 QPS (584.51 / 585.58),
compared with NC's 17.543 MiB and 585.055 QPS: about 20.58 times the cache RAM
at almost identical throughput, in this static same-workload experiment.
The denser data still look approximately linear in the middle of the tested
range. `results/dense/analysis.json` records measured interval slopes and a
diagnostic fit; the fit is never used to generate or smooth plotted points.

## Interpretation and validation

This is a **same-workload offline oracle/static experiment**: the seed set and
neighbor filter use the entire measured workload. It is not a held-out result or
proof that online NMD achieves the same memory efficiency while learning unknown
queries. N-only is online LRU, not an oracle-selected static hot-node baseline.
Compared with the preceding online experiment, M/stash maintenance and the D
index representation also change; the QPS gain cannot be attributed solely to
D coverage. Both old and new results are retained.

The base index is on `/dev/nvme4n1` (`/mnt/nvme_single`); D is on `/dev/nvme2n1`.
The comparison uses the existing configured devices, not a same-device latency
isolation experiment. D preparation is excluded from query throughput.

Validation checks 10,002 D pages / 64,702 complete records against the original
index, including graph-order membership and padding rules; verifies the file is
fully allocated; then checks a separate 1,000-query integration run. Both full
20K measurements and every N-only measurement return bit-identical IDs and float
distances to the saved original-index reference. Query counts also satisfy
`N hits + P hits + physical D reads == all node accesses`, with zero base reads.
No absolute recall claim is made without a ground-truth calculation.

The isolated generator starts from current native source and links the rebuilt
native library, leaving production source and default binaries unchanged. Header,
generated-source and executable hashes, compile/link commands, precise run
manifests, file-cache checks and per-query CSVs remain in the artifact directory.
`results/` saves compact summaries, complete per-run counters, commands and plots.

Scripts run in sequence:

```sh
python experiments/neighborcache_alignment/fullvisited_single/profile.py
g++ -O3 -fopenmp -std=c++17 \
  experiments/neighborcache_alignment/fullvisited_single/build_visited_seed.cpp \
  -o /mnt/nvme2/nc_fullvisited_seed_single_20261031/build_visited_seed
taskset -c 20-23 /mnt/nvme2/nc_fullvisited_seed_single_20261031/build_visited_seed \
  /mnt/nvme_single/sift100m/sift100m_index_disk.index \
  /mnt/nvme2/nc_fullvisited_seed_single_20261031/visited_seed_ids.u32 \
  /mnt/nvme2/nc_fullvisited_seed_single_20261031/visited_seeds.pages
python experiments/neighborcache_alignment/fullvisited_single/build.py
python experiments/neighborcache_alignment/fullvisited_single/run.py
python experiments/neighborcache_alignment/fullvisited_single/match_nonly.py
python experiments/neighborcache_alignment/fullvisited_single/report.py
```

These scripts use the recorded local index/build/reference paths. They never
silently overwrite a completed experiment. The builder refuses an existing D
output. N-only points with complete result JSONs can be reused on resumption.
