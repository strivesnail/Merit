# Thread-private prefetch across adapters

This opt-in mode is implemented in `diskann/include/neighborcache_thread_page_cache.h`
and selected by `NeighborCacheSharedPageCache`. Native DiskANN uses it directly;
Starling, MARGO, PipeANN and AiSAQ use it through the common ABI-2 backend.
The adapters do not need another host search-policy change for this switch.

```sh
export NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE=1
export NEIGHBORCACHE_FREEZE_MAINTENANCE=1
export NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY=0
export NEIGHBORCACHE_THREADS=20
export NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=2560
```

The budget is total resident page payload, divided among the configured workers:
this example gives 128 pages / 512 KiB to each worker. D and its routing index
remain shared (immutable in the example above); each worker owns its page map, LRU and payload.
Pages persist across queries by default. Set RESET_EACH_QUERY=1 to clear only the
calling worker at its next query boundary. The subsequent online implementation
also supports FREEZE_MAINTENANCE=0: versioned D metadata invalidates stale private
replicas, and dirty eviction detaches an asynchronous write without waiting for IO.
Resident page slots/hash/LRU are now preallocated and reused. See
[online validation](online/README.md) and the backend README for exact semantics.
Private mode requires a stable worker pool; query scratch and metadata are not
included in the page payload budget. Shared mode remains the default.

Build and test instructions are in [the backend README](../../../neighborcache_backend/README.md).
External host revisions and patches are listed in [adapters.json](../adapters.json).
Already integrated ABI-2 hosts can load the new backend without another rebuild.

## Historical static experiments

- [REPORT.md](REPORT.md) and [comparison.csv](comparison.csv): new same-budget
  shared/private comparisons, one run per mode. **Results are mixed, not a general
  speedup.** Each engine retains its existing N/D configuration and workload;
  datasets differ across engines. Recall/output comparisons are included.
- [Earlier full-seed experiment](../full_seed_static/thread_private_probe/README.md):
  the original thread-private implementation, with no N/M and all profiled visited
  nodes as seeds. Its separate `NC_FULLSEED_*` flags and resident-node lookup
  implementation are not the general backend mode above.

The runner ships the exact source configurations under `configurations/`, with
machine-local dataset, prefill and executable paths. Provision those artifacts
or edit the configurations before running elsewhere. Benchmarks are deliberately
not automatic tests and never silently reuse existing results:

```sh
python run.py --root /mnt/nvme2/new_private_comparison
python verify.py --root /mnt/nvme2/new_private_comparison
python summarize.py --root /mnt/nvme2/new_private_comparison
```

Build the backend at ROOT/build/libneighborcache_backend.so before invoking the
runner. CPUs 0–19 (AiSAQ 0–7) are reserved for queries, 22–23 for the controller.
File cache is cleared and checked before every timed run. Full-record verification
runs separately and its timing is excluded. Per-run commands, binary hashes,
logs, IDs and cache-clear checks remain under ROOT/runs and ROOT/verification.

The published result files describe the completed run at
`/mnt/nvme2/nc_private_adapters_20261028`. Native DiskANN and the backend were
rebuilt; 17 backend CTests and four adapter full-record checks passed. The original
shared/online adapter migration was pending in this workspace and is included
as a prerequisite of the private mode, rather than pretending the private header
alone would work against the previous remote checkout.
