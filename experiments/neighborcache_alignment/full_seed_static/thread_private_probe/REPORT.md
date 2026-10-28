# Thread-private prefetch experiment

SIFT100M A→B 40K queries (10K uniform + 10K Zipf0.8 shuffled per phase), K=10 L=128 beam=4, 20 workers pinned to CPU 0–19. One run per mode, with sync + file-scoped POSIX_FADV_DONTNEED and mincore verification before each run. N=0 M=0. D is immutable, with all 2,632,133 profiled unique visited nodes as seeds.

Both modes have exactly 2,060 pages = 8,240 KiB = 8.046875 MiB of prefetch payload. Shared mode uses 128 shard LRUs, 16 or 17 pages each, and retains pages across queries. Private mode has one lock-free cache and LRU per worker, exactly 103 pages = 412 KiB per worker, and clears only that worker’s cache before each query. Resident-node owner directories are private; the immutable D data and seed map remain shared. Metadata and the rest of the index are outside this payload budget.

| Mode | QPS | Hit % | D reads/query | Mean latency µs | IO µs/query | Reset µs/query |
|---|---:|---:|---:|---:|---:|---:|
| shared | 7658.49 | 61.32 | 56.25 | 2553.89 | 1535.95 | 0.00 |
| private_reset | 8439.85 | 57.62 | 61.63 | 2294.03 | 1667.12 | 17.54 |

Private-reset QPS change: +10.20%; D reads/query change: +9.56%.

Overall wall-clock QPS includes query resets. Per-query search latency excludes reset. Reset time is summed across workers; it must not be divided by wall time and interpreted as a serial overhead fraction.

Private-reset per-query visits, hits and reads exactly match the single-thread 103-page reset run. Maximum loads/query = 103. All 40K×10 result IDs match the existing reference in both modes; this is not a new ground-truth recall measurement.

This comparison changes cache ownership, number of LRUs, locking and query-boundary retention together. It evaluates the requested combined design; it cannot isolate a single cause such as lock contention. Dynamic query scheduling also changes which worker receives each query. One run per configuration does not provide confidence intervals.

Unit tests: thread isolation even for identical record IDs, clearing one worker while another retains its page, replica cleanup during LRU eviction, reinsert after reset, exact capacity allocation/remainder, 20-worker stress, insufficient capacity rejection. Existing shared-cache LRU, remainder, capacity-clamp and concurrent tests also pass.

Artifacts: method.json, summary.json, validation.json, comparison.csv; raw command, cache-clear evidence, output, per-query statistics, reference result IDs and timeline under runs/.

Timing interpretation: mean search latency minus recorded IO time changes from 1017.94 to 626.91 µs/query (-38.41%). This residual is not a direct lock-time measurement; it includes other CPU work, cache management, and unaccounted waits. Increased reads and IO time alongside reduced total latency point to an improvement outside recorded IO, but do not quantify individual causes.
