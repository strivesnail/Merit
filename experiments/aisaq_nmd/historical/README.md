# AiSAQ NMD interface prototype

Pinned upstream: `kioxia-jp/aisaq-diskann` at
`32881764fad3d2a56b337f17d26cbdd2dca4b554`.

Apply `aisaq.patch` to that checkout and copy `nmd_cache.h` into its `include/`.
The adapter uses the existing PipeANN NMD policy: admission, heat, reverse
neighbor updates, page reconstruction, victim selection, and append writeback
are unchanged. It decodes AiSAQ's 32-bit neighbor count and copies the entire
record, including inline PQ bytes. Environment variables use `AISAQ_NMD_*`.
The source header identity is recorded in `upstream.json`.

The prototype requires full-inline PQ, no ID rearrangement, and at least two
records per 4 KiB sector. Initial tests disable the native static node cache.
It intercepts the already-selected read batch, retaining the original logical
beam selection and original distance/neighbor expansion. D locations are resolved before issuing a mixed batch of D and base-index reads
through Linux libaio. Each completion is checked; failed/short D reads fall back
to the original index and are not counted as D hits. Buffered D reads may still
block inside `io_submit`; this does not promise fully asynchronous storage I/O.
Finite I/O-budget termination has not been validated; use the default unlimited
search I/O budget. `AISAQ_NMD_VERIFY=1` compares each cache hit byte-for-byte with
the original index (validation only, never performance timing).

## Validation run

Artifacts: `/mnt/nvme2/aisaq_nmd_trial_20261027`.
100,000 vectors from SIFT10M; first 1,000 public queries and their fivefold
repetition. This is an integration test, not the previous full-dataset benchmark.
Index: R=32, build L=100, PQ=16 bytes, 32 inline PQ codes, no rearrangement.
Search: K=10, L=64, W=4, vector beam=1, 8 threads.
N=1 MiB, prefetch=1 MiB; empty N/P/D at startup, heat threshold=2; no prefill
or extra warmup. Full-record cache payload is 772 bytes (five records/page).

`run_trial.py` validates output identity and full-record reads; `repeat_trial.py`
performs three repetitions with alternating baseline/NMD order. Both retain
commands, logs and results. All IDs and float distance output files must match
baseline exactly. Scratch D files from repeated runs are removed after logging.

The reported total demand reads include original-index read requests AND NMD D
read requests. They are not physical-device I/O counters. NMD writes are reported
separately. D uses buffered I/O, whereas the base index uses direct I/O: OS page
cache memory is not a strictly controlled budget. Therefore QPS differences do
not establish memory-fair or large-scale improvements. The prototype deliberately
retains the existing online rebuild/write amplification for evaluation.

## Local build recipe

Install dependencies in user-owned directories as needed. The local build used
Conda MKL/Boost/tcmalloc and extracted Ubuntu liburing packages, without sudo.

```sh
export CPATH=/home/jianz/miniconda3/include:/tmp/aisaq-nmd-deps/root/usr/include
export LIBRARY_PATH=/home/jianz/miniconda3/lib:/tmp/aisaq-nmd-deps/root/usr/lib/x86_64-linux-gnu:/tmp/merit-link-deps/root/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=/home/jianz/miniconda3/lib:/tmp/aisaq-nmd-deps/root/usr/lib/x86_64-linux-gnu
/home/jianz/miniconda3/bin/cmake -S /tmp/nmd-candidates-aisaq -B /tmp/nmd-candidates-aisaq/build \
  -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=/home/jianz/miniconda3 \
  -DMKL_PATH=/home/jianz/miniconda3/lib -DMKL_INCLUDE_PATH=/home/jianz/miniconda3/include \
  -DOMP_PATH=/home/jianz/miniconda3/lib -DCMAKE_CXX_STANDARD_LIBRARIES=-latomic
/home/jianz/miniconda3/bin/cmake --build /tmp/nmd-candidates-aisaq/build \
  --target build_disk_index search_disk_index compute_groundtruth -j 4
```

## Historical synchronous prototype: three-run result

Public 1K: demand reads/query 84.70 -> 73.53; median QPS 5322.81 -> 4845.59.
Fivefold 5K: demand reads/query 84.70 -> 70.46; median QPS 5456.12 -> 3781.64.
Recall@10=97.5% throughout; all returned IDs and distances match byte-for-byte.
NMD additionally writes approximately 23.29 and 56.37 pages/query, respectively.
Thus total read+write request count increases; this is not an overall I/O/performance win.

## Batched D-read experiment (2026-10-27)

Artifacts: `/mnt/nvme2/aisaq_nmd_batch_20261027`.
`compare_batch.py` compares the archived synchronous binary, the new batched
binary, and NMD-off using the same index/queries/settings above. Three rotations
of execution order per workload; verification runs are excluded from timing.

| Workload | NMD off QPS | Synchronous NMD QPS | Batched NMD QPS | Batch vs sync |
|---|---:|---:|---:|---:|
| Public 1K | 5411.30 | 4769.88 | 4525.28 | -5.13% |
| Public 1K repeated ×5 | 5450.69 | 3738.91 | 3738.20 | -0.02% |

Values are three-run medians. This change did **not** improve measured QPS.
All 18 timing runs and both byte-verification runs matched baseline IDs and
float distances exactly, with recall@10=97.5%. Both verification workloads
exercised N, prefetch, and D hits. No D read failures occurred.

`test_batch_reader.cpp` passed mixed direct/buffered reads, unaligned buffered
reads, short D reads, invalid descriptors, base read failure, and reuse of the
same AIO context after errors. The reader drains accepted requests and exposes
D completion errors to the cache adapter instead of using incomplete records.

Batch read requests/query were 73.52 / 70.46 (public / ×5), and D write
requests/query were 23.29 / 56.37, virtually unchanged from the synchronous
variant. Admission, rebuilding, eviction, and synchronous buffered writeback
are unchanged. This test does not isolate lock contention or writeback time,
and is not evidence that synchronous D reads were the dominant bottleneck.
The native reader is also changed in the NMD-off run; the synchronous reference
is an archived pre-change binary. This is a whole-patch comparison, not a pure
single-variable lock/I/O microbenchmark. No full SIFT100M rerun was performed.

Build and run the reader regression using the dependency environment above:

```sh
g++ -std=c++17 -DUSE_AVX2 -mavx2 -I/tmp/nmd-candidates-aisaq/include \
  experiments/aisaq_nmd/test_batch_reader.cpp \
  /tmp/nmd-candidates-aisaq/build/src/libdiskann.a \
  -laio -ltcmalloc -lpthread -o /tmp/test_aisaq_batch_reader
/tmp/test_aisaq_batch_reader
python3 experiments/aisaq_nmd/compare_batch.py
```

## Per-page payload lock experiment (2026-10-27)

Artifacts: `/mnt/nvme2/aisaq_nmd_pagelock_20261027`.
The current header uses one mutex per prefetch page. Reads acquire only their
page lock. Updates/rebuilds acquire pool metadata then the selected page lock,
release the metadata lock, and update the payload under the page lock.
Allocation, global minimum-heat victim selection (same seed-ID tie breaker),
seed mapping, and D page numbering still use the global metadata mutex.
There is no partition-local capacity or eviction policy. Eviction snapshots
still take place under both locks; this is not removal of all global contention.
Lock order is metadata -> page, never page -> metadata. Page count and mutex
addresses are fixed before workers start. Global metadata protects seed/seed
heat, while the page mutex protects count, member IDs, per-slot heat and bytes.
NMD admission, heat thresholds, writeback and batched D reading are unchanged.

`compare_pagelock.py` compares the archived batched/global-lock binary with the
new per-page-lock binary and NMD off. Same 100K-vector integration dataset,
public 1K / repeated 5K queries, N=P=1 MiB, 8 threads, three order rotations.

| Workload | Global-lock QPS | Per-page-lock QPS | Change | NMD off QPS |
|---|---:|---:|---:|---:|
| Public 1K | 4358.82 | 4427.16 | +1.57% | 5329.99 |
| Public 1K repeated ×5 | 3558.19 | 2969.31 | -16.55% | 5437.00 |

These are three-run medians. Public per-page runs were 4427.16, 3572.43,
4737.16 QPS: the variation is much larger than the median improvement.
This prototype does not demonstrate a stable throughput gain; do not present
it as a performance fix or extrapolate it to SIFT100M. Read/write request
counts remain essentially unchanged. Prefetch hits are rare in this setup;
extra lock operations and remaining metadata serialization are possible
explanations, not measured attribution of CPU/lock-wait time.

Validation: all 18 timed runs and two full-record verification runs match
baseline IDs and floating-point distances exactly; recall@10=97.5%, no D read
failures. `test_page_concurrency.cpp` verified records across 192,000 expansions
with 16 threads. Its deterministic single-thread run produces identical
cache counters and byte-identical D files in the old/new implementations
(hashes retained in `validation.json`). ThreadSanitizer could not start in this
environment (`unexpected memory mapping`); race freedom is not established by
these checks. Scratch test D files were removed after validation.

Compile the standalone concurrency check with `g++ -std=c++17 -O2 -pthread
-I/tmp/nmd-candidates-aisaq/include experiments/aisaq_nmd/test_page_concurrency.cpp
-o /tmp/test_page_concurrency`, then run `/tmp/test_page_concurrency 16 /tmp/test_page.nmd`.
`NMD_HEADER` can select the archived header to reproduce the deterministic
single-thread policy comparison. Timing results and commands are saved by
`python3 experiments/aisaq_nmd/compare_pagelock.py`.

## Current prefetch policy: sharded LRU (supersedes prior experiments)

PipeANN (`pipeann/include/nmd_cache.h`) and AiSAQ (`nmd_cache.h`) now use the
same page-management methods. Up to 64 independent page shards own their page
payloads, seed map, free slots, and doubly linked LRU list. Seeds map by
`seed % shard_count`; fixed physical page slots map to the corresponding shard.
Existing capacity is divided across shards, retaining the exact total page
count (no capacity increase). Each successful prefetch hit, existing-node page
update, neighbor insertion, or rebuild promotes the page to MRU. Allocation
evicts the owning shard's LRU tail in O(1), independent of seed heat.
Heat remains in admission and selection of neighbors within a page.
The old global minimum-seed-heat scan and experimental per-page mutex array
are removed. D page numbering is atomic because shards can evict concurrently;
D writeback still occurs after releasing the page shard lock. N/M/D admission,
record layouts, and disk-read integration otherwise remain unchanged.

This aligns the prefetch **policy family** with sharded LRU in NeighborCache,
Starling and MARGO; it does not assert that their shard assignment, page layouts,
or all other cache operations are identical. Historical plots and timings above
were obtained with older policies and have NOT been regenerated as LRU results.

Validation artifacts: `/mnt/nvme2/nmd_lru_20261027`.
`test_prefetch_lru.cpp` is compiled against both headers and checks read/update
promotion, eviction of a hot but least-recently-used page, isolation between
shards, and LRU link integrity. Both pass. The existing record-integrity stress
check also passes with 16 threads / 192,000 expansions for each implementation.
AiSAQ search verification uses the existing 100K SIFT integration index, public
1K queries and fivefold repetition; full-record hits and returned IDs/distances
are checked against the uncached reference. This is not a full-dataset QPS rerun.

## Latest alignment work

The current header additionally changes N to default LRU, avoids recreating
existing seeds, follows graph order on initial packing, replaces the last
non-seed slot on incremental fill, and reserves stable D offsets with dirty
writeback. Older benchmark sections above refer to earlier revisions. This
is not yet full NeighborCache equivalence; see
`../neighborcache_alignment/progress.json` for outstanding M/query/D behavior.
