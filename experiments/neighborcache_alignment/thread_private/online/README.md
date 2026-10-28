# Static and online thread-private prefetch validation

This revision replaces the earlier static-only private cache with reusable page slots, a fixed hash table and an indexed LRU. It also supports dynamic D publication, invalidation and asynchronous dirty eviction. N/M policies and host search policies are unchanged. Shared prefetch remains the default; the private flag now works with either explicit freeze setting.

## Semantics

- Each query worker owns its payloads, lookup table and LRU; pages persist across queries unless reset is requested. There is no query-to-query-thread migration of cache ownership.
- Shared per-sector generation metadata invalidates stale copies in other workers. Only the current dirty replica may enqueue writeback. Background maintenance invalidates replicas and directly submits its dirty snapshot rather than consuming a query cache slot.
- Dirty eviction copies into the existing owned writer task and removes the LRU entry without waiting for disk completion or queue space. Overload can drop the cache write; the dirty D image stays unreadable and demand falls back to the base index. Existing physical pins protect reuse and write ordering.
- Short D-mutation metadata locks remain, as do existing N/M/routing locks. This is not a lock-free whole-engine claim. Private hits do not take a shared prefetch lock.
- Resident slots/hash/LRU reuse preallocated storage. Pointer-returning APIs use reusable query scratch for stable snapshots. Scratch, the online generation array, and pending writer tasks are outside the resident payload budget. Static mode does not allocate the online generation array.

## Validation

- Backend: 21/21 CTests, including three concurrent online record layouts (260-byte 16-bit degree, 260-byte 32-bit degree, 772-byte opaque payload), stale replicas, invalidation, asynchronous handoff on query reset, pointer lifetime and 100,000 randomized LRU/hash operations.
- AddressSanitizer/UndefinedBehaviorSanitizer passed the private dynamic invalidation/reset/hash stress test.
- Native DiskANN rebuilt. The actual writer regression passed in shared and private modes, with deliberately paused workers, queue overflow, two outstanding versions of one sector, and injected IO failure. The query-side eviction does not wait for the writer.
- Separate static first-100-query full-record verification passed on all four adapters (SIFT100M Starling/MARGO, SIFT10M PipeANN, SIFT100K AiSAQ).
- Online verification: 100 real public query vectors repeated 50 times = 5,000 queries. Starling/MARGO/PipeANN use SIFT10M and 20 workers; AiSAQ uses SIFT100K and 8 workers. N capacity=1,024 nodes, M=2,048, D cap=512 pages, prefetch=8 pages/worker (32 KiB), no query reset, no prefill, maintenance enabled. Every cache-hit record is compared bytewise with the original index. Timing includes verification and is NOT a QPS result.

| Engine | Prefetch hits | D read requests | Seed opens | Appends | Replacements | Completed writes |
|---|---:|---:|---:|---:|---:|---:|
| starling | 98977 | 24067 | 448 | 222 | 283 | 460 |
| margo | 107373 | 24829 | 454 | 150 | 240 | 466 |
| pipeann | 129940 | 35993 | 465 | 30 | 10 | 466 |
| aisaq | 61302 | 52619 | 382 | 9 | 289 | 590 |

Initial public-query-only coverage trials used 2, 8 or 32 pages/worker. Some did not trigger any prefetch hits, or never evicted a dirty page, and were rejected by coverage assertions despite passing record comparisons. The repeated-query workload above exercises all required paths; the initial logs remain in the artifact directory.

No new QPS claim is made by this validation. Earlier static timing tables describe the previous implementation, not this revision. Bytewise record checks do not establish search-result equivalence for asynchronous or page-expansion hosts.

Artifacts: `/mnt/nvme2/nc_private_dynamic_20261028/`. Successful online logs/commands: `online_verify_completed/<engine>/`; static checks: `static_final/verification/<engine>/`. Build the backend into ROOT/build, then run `python verify_online.py --root ROOT` with the existing local index paths. The native regression is registered under `UNIT_TEST=ON`; this machine also ran it directly against the rebuilt library.
