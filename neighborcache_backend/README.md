# Shared NeighborCache backend

PipeANN, AiSAQ, Starling and MARGO load `libneighborcache_backend.so` through the C interface in `include/nc_backend.h`. DiskANN uses the same C++ implementations directly. The node cache copies complete fixed-width host records, including PipeANN's packed neighbor counts and AiSAQ's inline PQ payload.

## Build and load

```sh
cmake -S neighborcache_backend -B neighborcache_backend/build -DCMAKE_BUILD_TYPE=Release
cmake --build neighborcache_backend/build -j
ctest --test-dir neighborcache_backend/build --output-on-failure
export NEIGHBORCACHE_BACKEND_LIBRARY="$PWD/neighborcache_backend/build/libneighborcache_backend.so"
```

| Engine | Enable NC | Host integration |
| --- | --- | --- |
| PipeANN | `PIPEANN_NMD=1` | In-tree PipeANN search adapter |
| AiSAQ | `AISAQ_NMD=1` | [Patch and installation](../experiments/aisaq_nmd/README.md) |
| Starling / MARGO | Positive `NEIGHBORCACHE_NCACHE_CAP_NODES` | [Pinned revisions and patches](../experiments/neighborcache_alignment/adapters.json) |

PipeANN and AiSAQ accept a fresh cache output prefix via `PIPEANN_NMD_PATH` or `AISAQ_NMD_PATH`. The page adapter uses the index filename with a `.nc.dyn` suffix. All engines support a fresh `MERIT_D_ODD` file on a second SSD for alternating D pages. The [path cache guide](../diskann/README-path-seed.md) describes sizing and writer settings.

## Query callbacks

1. Begin a query. Resolve each requested node with its discovery parent. A resident hit copies the raw record into the caller's buffer.
2. A D lookup can return a read ticket. For compatible tickets on one page, the host can read the page once and complete each ticket. On a miss the host reads its original index.
3. Pass each acquired record and its parent to the expansion callback before the host mutates its neighbor list. Query completion groups the observed paths and schedules writes.

Use `PIPEANN_NMD_VERIFY=1`, `AISAQ_NMD_VERIFY=1` or the page adapter's `NEIGHBORCACHE_VERIFY=1` to compare served records with the original index during correctness checks.
