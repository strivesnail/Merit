# Cache maintenance trials

Both patches are experimental and apply independently to production baseline
`ac3256d`. Neither is enabled in production: the trials did not demonstrate a
throughput improvement. The native executable and installed common backend
were retained/restored to their verified baseline hashes.

## Protocol

SIFT1B, public 10K queries repeated ten times and shuffled (100K queries).
K=10, L=128, beam=4; 10 query workers on CPUs 0–9 and background workers on
CPUs 20–23. N=366,218 nodes; persistent private prefetch=1.25 MiB per worker
(12.5 MiB total); M=4M entries; D=2M pages. The 400K node-equivalent budget
counts N+prefetch payload only, excluding M, directories and common PQ data.
Each process starts with empty N/M/P/D, no warmup or prefill, and verified
file-cache eviction. Heat gate=100%. Legacy query-patch, optional ID-only
preflight, lean checks and N-hit supplementation are disabled.

Each initial trial used before/after/after/before order. A second before/after
pair was run for directory updates after an I/O-latency shift. No run was
discarded or adjusted. `results.json` contains all ten 100K runs.

## Candidate 1: reuse resident records during seed preparation

`patches/seed_prepare_merge.patch` captures complete resident records during
candidate qualification and reuses them for final page construction. It retains
the W/fetched-member eligibility and D fallback for uncaptured members. A native
regression evicts the acquired records from N before P publication and checks
the published vector and adjacency bytes.

Two-run mean QPS: **3229.69 before, 3221.56 after (-0.25%)**. Prefetch hits as a
fraction of all target accesses: 27.661% before, 27.716% after. This did not
demonstrate a speed gain, so the patch was not promoted.

## Candidate 2: refresh a republished page's directory entries in one lock

`patches/prefetch_directory_update.patch` unlinks only removed members and
combines erase/add for retained members into one shard-locked refresh. It keeps
newest-page lookup priority, updates the canonical route once and preserves
other seed routes. Native regressions check actual page republish, slot changes,
duplicate prevention, canonical lookup and fallback routes.

| Pair | Before QPS | After QPS |
|---|---:|---:|
| First | 3224.70 | 3218.62 |
| Second (I/O drift) | 2472.77 | 2765.18 |
| Repeat | 3218.46 | 3212.62 |

The first and repeat comparisons show essentially unchanged throughput. The
second pair cannot support a speedup claim: observed NVMe read completion latency
rose from about 57 us in the first runs to 66 us in `after_2` and 77 us in
`before_2`. Search I/O time rose with it while remaining query time stayed near
1.1 ms. No other benchmark process was observed; post-run SSD temperature was
about 37 C. The firmware/device-level cause was not established. Device counters
cover all NVMe1 traffic and are sampled once per second.

The patch was not promoted because a throughput improvement was not established.
The plot displays all six directory-trial runs individually, rather than using
an aggregate that would mix different I/O conditions.

## Validation and artifacts

All runs returned bit-identical IDs and distances, Recall@10=94.94%, and
14,748,600 full-vector target accesses. Common backend tests passed 25/25 for
each candidate; native shared/private and optional-mode regressions also passed.
No independent host-adapter throughput sweep was performed.

Raw logs, binaries, telemetry and captured experimental sources are under:

- `/mnt/nvme2/nc_seed_prepare_merge_20261101`
- `/mnt/nvme2/nc_prefetch_directory_update_20261101`

The plot can be regenerated with `python plot.py --output-dir <directory>`.
It separates the original search I/O timer from remaining query wall time;
the latter includes computation, scheduling, locks and maintenance outside that
timer, and is not CPU-only time.
