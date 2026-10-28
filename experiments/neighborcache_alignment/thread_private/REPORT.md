# Thread-private immutable prefetch: adapter comparison

One timed run per mode, 2026-10-28. Compare modes within each row; datasets differ across rows.

| Engine | Dataset | Shared QPS | Private QPS | Change | Recall@10 (%) shared → private | Identical top-10 rows |
|---|---|---:|---:|---:|---|---:|
| diskann | SIFT100M | 5050.12 | 5088.27 | +0.76% | 98.4110 → 98.4110 | 20000/20000 |
| starling | SIFT100M | 6144.76 | 5215.14 | -15.13% | 98.7685 → 98.7695 | 19980/20000 |
| margo | SIFT100M | 6245.70 | 5679.71 | -9.06% | 98.3890 → 98.3890 | 19984/20000 |
| pipeann | SIFT10M | 8980.51 | 9066.81 | +0.96% | 97.3280 → 97.3120 | 9905/10000 |
| aisaq | SIFT100K | 4539.09 | 4396.75 | -3.14% | 97.5000 → 97.5000 | 1000/1000 |

## Interpretation

The general shared-backend integration does not reproduce a universal private-cache speedup. DiskANN and PipeANN change by less than 1%; one run cannot distinguish these changes from measurement noise. Starling and MARGO slow down despite slightly fewer D reads; AiSAQ also issues more D reads. These counters alone do not locate the CPU cost. No lock-time or allocator-cost attribution is established here.

This experiment retains each engine’s existing N cache, partial static D prefill, and search policy. It is not the earlier full-seed, N=0/M=0 standalone experiment. The general cache wrapper uses different lookup and page-lifetime machinery. Do not compare its absolute QPS directly with that earlier sweep.

Recall was recomputed from returned IDs and ground truth with ties at the tenth distance. DiskANN and AiSAQ outputs are identical across modes. Other engines have output differences; equal rounded recall is not proof of identical results. Starling/MARGO have prefetch-page expansion enabled, whose work can change with residency. PipeANN uses asynchronous candidate processing; the cause of its small recall difference is not isolated by this experiment.

## Configuration and validation

- Private caches retain pages between queries. Each worker gets 128 × 4 KiB = 512 KiB. Shared mode has exactly the same total resident payload capacity: 10 MiB for 20 threads; 4 MiB for AiSAQ’s 8 threads.
- DiskANN/Starling/MARGO: SIFT100M, shuffled 10K+Zipf (20K queries), T=20, L=128, W=4, K=10. N capacity is 212,371 nodes. Static D is reused from each engine’s existing configuration.
- PipeANN: SIFT10M public 10K, T=20, L=128, W=4, K=10, N=31 MiB. AiSAQ: SIFT100K public 1K, T=8, L=64, W=4, K=10, N=1 MiB.
- Query CPUs pinned to 0–19 (AiSAQ 0–7). File cache cleared before every run using sync + POSIX_FADV_DONTNEED, checked with mincore. This is not a device-cache reset.
- Fixed D, no M maintenance or writes. Adapter counters confirm M/opens/appends/replaces/writes = 0. Shared/private pairs use identical prefill and search parameters. Query scratch and metadata are outside the payload budget.
- 17/17 backend CTests passed, including isolation, LRU eviction, query reset, pointer lifetime, immutable-page guards, and concurrent prefetch expansion. Native DiskANN rebuilt successfully.
- Separate first-100-query runs passed original-index full-record verification on Starling, MARGO, PipeANN, and AiSAQ. Verification timing is excluded. This does not establish complete search equivalence.

Raw commands, binary hashes, cache-clear checks, output IDs and logs: `/mnt/nvme2/nc_private_adapters_20261028/runs/`.
Independent verification: `/mnt/nvme2/nc_private_adapters_20261028/verification/`.
