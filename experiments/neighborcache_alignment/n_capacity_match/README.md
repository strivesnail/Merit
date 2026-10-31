# NMD prefetch plateau and an N-only throughput match

This experiment changes N capacity alone in online native NeighborCache. M is
131,072 entries, D is 65,536 SSD pages / 256 MiB, private prefetch is 2,560 pages /
10 MiB, stash capacity is 424,742 nodes, and node heat sampling is 1/10. Query
worker 0 is pinned to CPU 0; background workers use CPUs 20-23. The canonical
accelerator has 1,048,576 slots; private page zero-tail storage is enabled.

The workload is the saved SIFT100M public 10K uniform + corresponding 10K Zipf
queries, shuffled, K=10, L=128, beam=4. The current native executable is used
without changes to its search/cache algorithms. Its existing multi-L option is
invoked as `--search_list 128 128`: pass one warms the runtime caches with 20K
queries, and pass two measures another 20K queries from the same ordered stream.
Final result files and the CSV are from pass two. These are warmed-workload
measurements, not the older cold-start QPS values.

Before every process, sync/fadvise/mincore clears and verifies the same index
and query files. Hardware SSD caching is not reset. Both schemes enable the
same query-local original-page reuse; only NMD enables cross-query seed
prefetch, M, D and its auxiliary structures. N-only uses the same online node
pool, cold admission and replacement policy, with no BFS prefill, M, D or
prefetch. Every measured run verifies returned IDs and distances against the
saved uncached-reference output. This establishes equal answers for this query
stream, not an absolute recall number for an independent ground-truth set.

The saved `protocol.json` records the selection rule before observing results:
start N at 1,024 nodes and double it. After at least four levels, select the
first of three consecutive levels if the next two doublings improve neither
prefetch/all-accesses nor prefetch/N-misses by more than 0.5 percentage points.
The conditional check prevents a flat all-access curve caused only by N taking
hits away from prefetch from being mistaken for saturation. Confirm the three
plateau points with a second independent process each. If confirmation fails,
continue the sweep before drawing a conclusion.

Next increase N-only capacity until measured QPS reaches the selected NMD QPS,
and refine the observed bracket. N-only and NMD use the same warmed query stream,
not a hot-node trace precomputed from test answers. A crossing in this finite
experiment is not proof of the globally minimum N-only memory requirement.

Hit fractions use **all node accesses** as the main denominator; the secondary
conditional fraction excludes N hits. D demand probes/hits and physical D reads
are recorded separately. The main memory observation is process RSS after search
minus RSS after common index/PQ loading. For comparisons, subtract the no-cache
run's corresponding delta to remove common query-result/scratch allocations.
This estimate includes N, M, all prefetch payload/metadata, D routing indices,
queues, payload stash and allocator effects; it is not nominal N+P payload.
D SSD bytes and common PQ/index memory are excluded. A small configured N does
not imply a small NMD resident footprint.

Run on this machine:

```sh
/home/jianz/miniconda3/bin/python experiments/neighborcache_alignment/n_capacity_match/run.py
/home/jianz/miniconda3/bin/python experiments/neighborcache_alignment/n_capacity_match/plot.py
```

Defaults use existing local query, index, reference and artifact paths. Pass
`--root NEW_OUTPUT_DIR` to preserve a separate experiment. Results already saved
as per-point `result.json` are reused, never replaced. Incomplete directories
must be archived separately before retrying. Source binaries and inputs are
hashed in the per-run command manifests; full query CSVs and logs stay in the
artifact directory `/mnt/nvme2/nc_n_capacity_match_20261031/`.

The intended memory-efficiency narrative is a hypothesis. Report the measured
crossing even if N-only reaches the target using less RAM than NMD.

## Independent prefetch reduction

`prefetch_sweep.py` holds N at 2,048 nodes and reduces the single private LRU
from 10 MiB to 5, 2, 1 and 0.5 MiB, with all other cache/search settings and the
same two-pass protocol retained. It reuses the two verified 10 MiB runs, executes
one process per smaller capacity, checks output IDs/distances, and plots only
prefetch hits divided by all accesses. CSV additionally retains N hit rate and
the N-miss-conditional prefetch fraction. Results are saved separately under
`/mnt/nvme2/nc_prefetch_reduction_single_20261031/`.

The N payload is exactly 2,048 * 388 = 794,624 bytes (0.7578125 MiB), from the
runtime payload-capacity counter. The `after_ncache - after_load` RSS delta
also includes prefetch metadata initialized earlier, so it must not be presented
as an isolated measurement of N-cache RAM.

Session status: 10, 5, 2 and 1 MiB prefetch results completed; 0.5 MiB was
interrupted when the user selected 10 MiB and requested full visited-seed D.
The original online-NMD 352-QPS target was bracketed by N-only capacities
49,152 and 65,536, without claiming an exact minimum. The subsequent full-D
experiment and its newly matched N-only curve are saved under
[fullvisited_single](../fullvisited_single/README.md). `results/status.json`
records these redirects; incomplete trials never enter the plotted data.
