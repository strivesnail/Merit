# Lazy M edge-list descriptors

The logical M cache remains 4,000,000 slots. Edge-list container descriptors
are created as slots first become live, under the existing shard lock. Slot
IDs, capacity, heat, edge contents and eviction policy are unchanged. Growth is
capped at the configured shard capacity and clear/init frees old descriptors.
`NEIGHBORCACHE_MCACHE_EAGER_EDGE_SLOTS=1` restores eager allocation for comparison.

## Measured comparison

One cold run per mode using the same binary; only the allocation switch differs.
SIFT1B; public 10K queries repeated ten times and shuffled, 100K queries total;
K=10, L=128, beam=4. Ten query workers use CPUs 0–9 and background workers use
CPUs 20–23. N=366,218 nodes; persistent private prefetch=1.25 MiB per worker
(12.5 MiB total); M=4M slots; D=2M pages. All caches start empty, with verified
file-cache eviction and no warmup or prefill. Node heat gate=100%.

| Metric | Eager | Lazy |
|---|---:|---:|
| Live M records | 1,403,632 | 1,403,632 |
| Allocated edge-list descriptors (MiB) | 91.553 | 48.000 |
| Additional RSS after cache initialization (MiB) | 704.156 | 610.211 |
| Additional RSS after 100K queries (MiB) | 1139.813 | 1093.770 |
| QPS | 3225.87 | 3229.96 |
| N hits / all target accesses | 27.409% | 27.424% |
| Prefetch hits / all target accesses | 27.744% | 27.750% |
| Demand disk pages / query | 63.669 | 63.638 |

Descriptor storage saves exactly 43.553 MiB in this run. Observed additional
RSS after the workload saves about 46.043 MiB; RSS includes allocator and other
cache state and is not identical to allocated descriptor bytes. Additional RSS
subtracts each process's RSS after loading the common index, so common PQ/index
memory is excluded. It includes N, M, prefetch and the other cache directories.
The 400K node-equivalent payload budget used elsewhere counts only N+prefetch.

QPS is essentially unchanged; one run per mode does not establish a speedup.
Memory savings depend on M occupancy and shrink as M fills. Both modes return
bit-identical IDs and distances, Recall@10=94.94%, and 14,748,600 full-vector
target accesses.

## Validation and artifacts

Native tests cover empty allocation, growth, full occupancy, slot reuse,
clear/init, concurrent snapshots during same-shard growth, membership bitmaps
and concurrent readers. Shared/private prefetch regressions and the common
backend's 25 tests pass. The common backend supplies Starling, MARGO, PipeANN
and AiSAQ; separate host throughput sweeps were not run.

Raw logs, binaries, cache-clear checks and per-run command hashes are in
`/mnt/nvme2/nc_mcache_lazy_slots_20261101`.

Regenerate the memory figure from the included measurements:

```sh
python analyze.py --output-dir /tmp/mcache-lazy-plots
```
