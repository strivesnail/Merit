# Shared online NeighborCache backend

PipeANN, AiSAQ, Starling and MARGO use the actual online NeighborCache cache
implementation in `diskann/src/pq_flash_index.cpp`. The bridge compiles that source
under a private namespace; it does not maintain separate heat thresholds,
seed-selection heaps, prefetch replacement algorithms or D-page policies.

The host adapters translate immutable node records, candidate source IDs, record acquisitions,
query boundaries and I/O completions. Native candidate selection, distance code
and beam logic remain in each engine. Different engines generate different
access streams: sharing cache code does not imply identical cache states or ANN
outputs across engines, or between cached and uncached asynchronous searches.

## Cache behavior

- Demand lookup checks N, then resident seed prefetch, then the reference D routing
  and query-local page machinery, then the host's original index.
- The reference sharded N cache, M scores/heaps/state transitions, graph-order
  seed construction, N/stash lookup, reverse-owner updates, prefetch LRU,
  dirty eviction, D readback, bounded seed replacement and query-end maintenance
  are used directly. No host adapter selects seeds using `heat >= 2`.
- A later admitted graph neighbor can update an existing seed page regardless of
  which parent discovered it. The reference ownership/graph relation decides
  eligibility; the traversed parent is recorded separately for co-access scoring.
- PipeANN and AiSAQ submit demand D reads through their existing asynchronous
  readers. Starling/MARGO retain a separate libaio D batch. Reference maintenance
  may perform blocking reads and writeback; this is not an all-asynchronous claim.
- D demand lookup attempts/hits and physical D reads are separate diagnostics.
  Query-local locations and pages can avoid a fresh routing lookup, so D reads
  must not be used as the denominator of the lookup hit rate.

## Build

Requires Linux, C++17, AVX2, OpenMP, libaio headers/library, and the include
dependencies already used by `diskann` (including MKL headers).

```sh
cmake -S neighborcache_backend -B neighborcache_backend/build -DCMAKE_BUILD_TYPE=Release
cmake --build neighborcache_backend/build -j 4
ctest --test-dir neighborcache_backend/build --output-on-failure
export NEIGHBORCACHE_BACKEND_LIBRARY="$PWD/neighborcache_backend/build/libneighborcache_backend.so"
```

On this machine dependencies are in user-owned locations, with no sudo:

```sh
export CPATH=/tmp/pipeann-inc:/home/jianz/miniconda3/include:/tmp/aisaq-nmd-deps/root/usr/include
export LIBRARY_PATH=/home/jianz/miniconda3/lib:/tmp/aisaq-nmd-deps/root/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=/home/jianz/miniconda3/lib:/tmp/aisaq-nmd-deps/root/usr/lib/x86_64-linux-gnu
```

Rebuild host executables after installing the headers. `NC_BACKEND_ABI` identifies
the interface version; library and headers must be deployed together.

## Configuration and validated scope

PipeANN: `PIPEANN_NMD=1`, `PIPEANN_NMD_PREFILL=0`, `PIPEANN_NMD_NMB`,
`PIPEANN_NMD_PMB`, `PIPEANN_NMD_PATH`. AiSAQ uses the analogous `AISAQ_NMD_*`.
N MiB is converted into a node count; prefetch MiB into 4 KiB pages. These are
payload capacities, not whole-process RSS budgets. Both accept explicit
`NEIGHBORCACHE_NCACHE_CAP_NODES` and `NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES`.
Optional `*_PROFILE` retains the u64 point count + u32 per-node expansion counts.

Starling/MARGO: `NEIGHBORCACHE_NCACHE_CAP_NODES`,
`NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES`, `NEIGHBORCACHE_DCACHE_PREFILL=0`.
Set N capacity to zero to disable their adapter. PipeANN/AiSAQ use their enable
flag to disable. All four use reference `NEIGHBORCACHE_DCACHE_CAP`,
`NEIGHBORCACHE_MCACHE_CAP` and other dynamic-cache settings. D capacity can be
clipped by the reference implementation; consult the startup log rather than
inferring it from old benchmark percentages. Set `NEIGHBORCACHE_THREADS` to the
host's query thread count. Legacy visit-rank/seed-fraction settings do not select
seeds in this online backend. Legacy host-specific prefill flags remain rejected.

For an explicit visited-hot20 experiment, `NEIGHBORCACHE_HOT_PREFILL_PREFIX`
loads canonical `.pages` and `.slots` produced by
`experiments/neighborcache_alignment/build_hot_prefill.py`. Set
`NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC=1`; the reference importer initializes the
real D directory. With online maintenance enabled it also initializes M seed
states, heaps and seed reverse-owner links. Explicit static mode skips those
dynamic structures. N and prefetch start empty. This offline prefill can
read graph neighbors directly from the original index; it is distinct from
online construction using N/stash contents.

Set both `NEIGHBORCACHE_DCACHE_CAP` and
`NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES` to the requested visited-seed count to
avoid the default space limit based on original index pages. With no explicit
space override, the original default limit is unchanged. Set
`NEIGHBORCACHE_FREEZE_MAINTENANCE=0` to continue online maintenance after prefill;
`=1` selects fixed-D measurement: no M heat table, seed replacement heap, or
parent relation table is built. The read-only D location index and member bitset
remain available. It retains N/prefetch LRU. The reference default when this
variable is absent is to freeze when D is already full, so experiments must set
it explicitly. Frozen admission no longer updates seed-page membership/payload.

`NEIGHBORCACHE_BACKEND_REPORT=1` prints the reference cache and sampled lock
statistics after native search timing. These are cumulative event diagnostics,
not physical SSD counters or additive percentages of wall time.

Validated workloads use immutable uint8 SIFT records and the default N admission
policy (MRU insertion/LRU eviction). PipeANN uses ordinary sparse records with
16-bit neighbor count and zero dense-neighbor count; AiSAQ uses full inline PQ,
no ID rearrangement and native static node cache disabled. Starling/MARGO use
their existing page layouts. Records fit wholly inside a 4 KiB page. AiSAQ's
opaque inline PQ bytes are preserved by the record codec.

Dense/filtered or updating PipeANN indices, other record layouts, finite search
I/O budgets, and optional hop/shadow adaptive admission controllers have not
been validated. In particular the bridge currently forwards hop zero, and the
pipelined hosts do not provide equivalent per-query I/O timing for those optional
controllers. The default online LRU experiment does not enable them.

`PIPEANN_NMD_VERIFY=1`, `AISAQ_NMD_VERIFY=1`, or `NEIGHBORCACHE_VERIFY=1` for the
page engines enables bytewise comparison with the original index on every cache
hit. Verification opens/reads the original index and must be disabled for timing.

## Parent provenance (ABI 2)

Candidates carry a source ID through the native queue. `prepare_parent` only sets
request-local provenance for optional routing; it never inserts a parent relation
or updates heat. Once the host has a complete record to process, `expand(raw)`
registers the first acquired parent for that node in the query. This includes
cache hits as well as disk records, not every neighbor scanned or PQ candidate.
Nodes without a known source (roots or incidental page siblings) have no invented
parent. Rejected candidates create no parent-map entries.

Explicit static mode (`NEIGHBORCACHE_FREEZE_MAINTENANCE=1`) skips this map entirely
and skips M allocation/import/query-end updates. Optional parent-based D routing
can still use a source ID without maintaining a relation table. The default
freeze-on-full mode can retain M data initialized earlier for online operation;
use explicit `=1` for the no-M static experiment.

The backend ABI is now 2: rebuild hosts and deploy the matching shared library.
CTest additionally covers candidate source preservation through queue pruning,
no registration on an unsuccessful read, query-local first-parent ownership,
and zero M/parent entries in explicit static mode.

Same-session three-repeat timings and separate profiling are generated by
`run_read_parent.py`, `prepare_read_parent_profile.py`,
`run_read_parent_profile.py` and `report_read_parent.py` under
`experiments/neighborcache_alignment`.

## Validation and fixes

Static prefill also accepts the `NCHOT2P\0` slots magic with the existing six
uint64 header fields and one row of node IDs per 4 KiB page. Slot zero owns
the page. Consecutive rows may repeat the same seed at most twice, forming
a contiguous group; noncontiguous duplicates and larger groups are rejected.
Each page repeats its seed and stores up to `nodes_per_page - 1` neighbors.
The original `NCHOT20\0` format still requires one page per seed.

Grouped input requires `NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC=1`,
`NEIGHBORCACHE_FREEZE_MAINTENANCE=1`, `NEIGHBORCACHE_DCACHE_SECOND_PAGE=1`,
and `NEIGHBORCACHE_DCACHE_MAX_PAGES=2`. Set both D capacity limits to the
actual total page count. The importer publishes immutable span metadata before
queries; the existing IO path reads a two-page span as one 8 KiB request and
installs both pages in prefetch. A one-page group still uses 4 KiB. The
`hot_prefill_groups` test checks a partially filled second page, both pages'
prefetch contents, and a separate one-page group. The D IO report distinguishes
request count from bytes requested.

Record equality does not establish search equivalence for page-search hosts:
Starling/MARGO process incidental neighbors from original index pages, while
their default NC-hit branch processes the requested record. Changing cache
membership can therefore change returned IDs and recall even when every cached
record matches its original bytes. Baseline-versus-NC benchmarks must report
recall; a QPS increase alone is not evidence of transparent cache acceleration.

### Experimental prefetch-page expansion

Rebuilt Starling/MARGO hosts can opt into `NEIGHBORCACHE_EXPAND_PREFETCH_PAGE=1`.
This requires explicit frozen static mode. One-page prefills use
`NEIGHBORCACHE_DCACHE_SECOND_PAGE=0` and `NEIGHBORCACHE_DCACHE_MAX_PAGES=1`.
Grouped `NCHOT2P` prefills use `NEIGHBORCACHE_DCACHE_SECOND_PAGE=1` and
`NEIGHBORCACHE_DCACHE_MAX_PAGES=2`; both the host client and backend must be rebuilt.
On a prefetch hit, the adapter copies the target's resident 4 KiB page and any
resident companion page from the same immutable seed group. A companion that
has been evicted is skipped without issuing disk IO. The target and other valid
unique records are processed by the host; padding, repeated seeds across pages,
and nodes already expanded in the query are skipped. N hits and fresh D reads
remain target-only. If the fallback lookup loses the target page to eviction
before collecting members, only the already copied target is returned.

The optional `nc_prefetch_page_members` export leaves the ABI 2 table unchanged.
Existing hosts retain their behavior with the option disabled. Record verification
also checks the additional page members. The `prefetch_expand` test covers member
decoding, padding deduplication, single-use snapshots, and target-only N/fresh-D
paths. `prefetch_expand_concurrent` checks record bytes with four concurrent
readers and a working set larger than the prefetch capacity, exercising eviction.
`prefetch_expand_groups` checks two-page snapshots larger than a single-page
output buffer, mixed one-/two-page groups, byte equality and duplicate removal;
`prefetch_expand_groups_concurrent` exercises grouped reads and partial resident
snapshots during concurrent eviction.
This is a search-strategy experiment: D seed-page members differ from
original index-page members, so matching baseline recall or returned IDs is not
guaranteed. Report recall together with QPS.

The page-search hosts also check each final result against all IDs already
emitted, rather than only the immediately preceding sorted entry. Equal-distance
entries can interleave repeated IDs, so adjacent-only deduplication can return
fewer than K distinct nodes. This output fix applies with expansion both enabled
and disabled; comparison runs must use the same rebuilt host on both sides.

The CTest suite exercises three codecs with four concurrent threads, 38,400
expansions each, N/prefetch/D reads, full seed capacity, eviction/writeback and
record equality. Two additional tests inject EINTR and partial AIO completions,
with and without the latency profiler.

Integration results and portable external patches are in
[`experiments/neighborcache_alignment`](../experiments/neighborcache_alignment/README.md).
No new full-dataset performance claim or plot is produced by this migration.

The reference code also contains the fixes found by integration testing:
N payload copies under the N shard lock; page copies under the page shard lock;
atomic publication of dirty payload and membership; pinned replacement-slot
protection; validation/copying of dynamic page membership; maintenance reads
prefer the resident dirty page over an older disk copy; AIO completion waits
retry EINTR and accumulate partial completions without resubmitting reads.

Dynamic prefetch dirty-page eviction uses asynchronous writeback. The LRU
victim stays resident and readable while a bounded writer job owns an aligned
snapshot. A full queue or an unfinished write refuses the new admission, with
no disk wait under the prefetch shard lock. Completion clears the dirty version
only if no newer update occurred; at most one writeback per physical page is
in flight. Failed writes leave the page dirty and readable for retry. Pending
writes pin physical pages against reuse. Explicit invalidation drains a pending
write outside the shard lock before discarding the page; ordinary LRU eviction
does not wait. Shutdown drains submitted jobs, not every never-evicted dirty page
(this is an ephemeral cache, not a persistence API).

The native DiskANN `prefetch_async_writeback` test (`UNIT_TEST=ON`) exercises
the real writer with a deliberately paused consumer, a full queue, updates during
writeback, injected write failure/retry, sector invalidation/reuse, shutdown drain,
and four concurrent producers. The backend codec/eviction tests exercise the
same implementation through the shared ABI.

## Thread-private immutable prefetch (2026-10-28)

The native DiskANN/NeighborCache path and all four adapters now accept:

```sh
export NEIGHBORCACHE_FREEZE_MAINTENANCE=1
export NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE=1
export NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY=0
export NEIGHBORCACHE_THREADS=20
export NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=2560
```

`SHARED_PAGE_CACHE_PAGES` remains the **total resident page budget**, even in
private mode. The example gives 20 workers 128 pages (512 KiB) each. Remainders
are assigned to the first registered workers, preserving the exact total.
The native path obtains its worker count from the loaded index; adapters use
`NCConfig.threads`, normally set by `NEIGHBORCACHE_THREADS`.

Each calling worker owns one LRU and page directory. Hot accesses take no shared
cache lock. Pages survive query boundaries by default. Set
`NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY=1` to clear only the calling worker's
prefetch at the outermost query ReadSection. N remains governed by its existing
policy; the immutable D file and routing index remain shared. Query objects may
be pooled, but cache ownership follows the actual calling worker, not the query
object or a hash of the page ID.

This option currently requires **static, immutable D pages**. Missing
`FREEZE_MAINTENANCE=1`, dirty admission, or an attempt to overwrite a resident
page with changed bytes fails explicitly. Independent copies of changing pages
would otherwise require a coherence protocol; online maintenance keeps the
existing shared implementation. Native search, candidate ranking, graph
expansion and distance computations are not changed by this option. Conditional
prefetch-page expansion in Starling/MARGO can still yield different search
outputs when cache residency changes; measure recall, not only QPS.

`get_ptr` pins a page until the current query ReadSection ends, including when
that page is evicted. Pinned scratch and metadata are outside the resident
payload budget. This is an equal resident-capacity comparison, not an equal RSS
claim. The worker pool must be stable over a cache instance's lifetime; registering
more calling threads than configured fails rather than sharing one private LRU.
No ABI change is needed: replace the backend library and pass the environment
flags to PipeANN, AiSAQ, Starling or MARGO. The private-cache header is compiled
into the backend; it does not need to be copied into the host repositories.

Tests cover worker isolation, local reset, pointer lifetime after eviction,
replicated member pages, multi-page expansion, unchanged shared behavior and
rejection of mutable configurations. Integration runs and single-pass QPS/recall
comparisons are in `experiments/neighborcache_alignment/thread_private/`.
