# SPDK Backend

PipeANN supports SPDK as an alternative I/O backend. Compared to `libaio` and `uring`:

* **Pros:** stable (tail) latency, high throughput, best multi-SSD scalability.
* **Cons:** reduced portability, need for dedicated SSDs for vector index.

SPDK backend targets **(tail-)latency-critical** workloads and **high-throughput, multi-SSD deployments**.

Search and update commands are identical to the regular [C++ Interface](cpp-interface.md);
only the I/O engine and device setup differ.

## Setup

Setup proceeds in four steps: (1) build SPDK, (2) prepare SSDs, (3) configure
the stripe, (4) rebuild PipeANN against SPDK. Before starting, make sure the
base PipeANN build with the `io_uring` backend works (see [C++ Interface](cpp-interface.md)).

### 1. Build SPDK

Clone SPDK under `third_party` and build it:

```bash
cd /path/to/PipeANN

git clone https://github.com/spdk/spdk.git third_party/spdk
cd third_party/spdk
git submodule update --init --recursive
sudo bash scripts/pkgdep.sh
./configure
make -j$(nproc)
```

### 2. Prepare SSDs

Verify that all target NVMe devices share the same LBA format:

```bash
nvme list
```

Then bind them to SPDK:

```bash
cd /path/to/PipeANN/third_party/spdk
sudo PCI_ALLOWED="0000:66:00.0 0000:67:00.0 0000:68:00.0 0000:e4:00.0" \
    scripts/setup.sh
# To unbind: scripts/setup.sh reset
```

### 3. Configure the Stripe

Edit `spdk_bdevs.json` at the PipeANN repository root. The order of entries
defines the stripe order PipeANN uses across the devices.

```json
{
  "ssds": [
    "0000:66:00.0",
    "0000:67:00.0",
    "0000:68:00.0",
    "0000:e4:00.0"
  ],
  "hugedir": "/dev/hugepages"
}
```

### 4. Build PipeANN with SPDK

```bash
cd /path/to/PipeANN
rm -rf build
mkdir build && cd build
cmake .. -DIO_ENGINE=spdk
make -j$(nproc)
```

CMake should print a line similar to:

```text
-- Using SPDK for vector I/O & liburing for attr I/O, fastest but require user-space bdev.
```

## Running PipeANN with SPDK

Search and update commands stay the same as the regular C++ interface. Two
caveats specific to the SPDK backend:

- **Root privileges required.** Run via `sudo -E` or as `root`.
- **Working directory must contain `spdk_bdevs.json`.** Invoke binaries from
  the repo root (`build/tests/search_disk_index ...`), not `cd build; tests/search_disk_index`.

## Example

Experimental setup:

* CPU: 2 x Intel Xeon Gold 6330 (56 cores, 112 threads)
* SSD: 4 x Intel Optane P5800X NVMe SSDs

We use 50-105 threads for throughput runs and 1 thread for latency runs.

### SIFT100M


Latency run:

```bash
sudo build/tests/search_disk_index uint8 /mnt/nvme2/indices/bigann/100m \
  1 32 \
  /mnt/nvme/data/bigann/bigann_query.bbin \
  /mnt/nvme/data/bigann/100M_gt.bin \
  10 l2 pq 2 10 \
  10 10 30 50 80 100 150 200 300
```

Throughput run:

```bash
sudo build/tests/search_disk_index uint8 /mnt/nvme2/indices/bigann/100m \
  50 32 \
  /mnt/nvme/data/bigann/bigann_query.bbin \
  /mnt/nvme/data/bigann/100M_gt.bin \
  10 l2 pq 2 10 \
  10 10 30 50 80 100 150 200 300
```

Abbreviated output:

```text
SPDK initialized: 4 pollers, LBA size 512 bytes
     L   I/O Width         QPS  AvgLat(us)     P99 Lat   Mean IOs   Recall@10
================================================================================
Latency using 1 thread:
    30          32     2315.65      336.15      443.00       36.07       89.89
    50          32     1647.64      491.98      601.00       55.36       95.59
    80          32     1146.62      730.52      868.00       84.81       98.19
   100          32      954.00      889.73     1046.00      104.52       98.81
   150          32      816.44     1068.70     1483.00      154.02       99.48
   200          32      658.33     1343.15     1615.00      203.57       99.72
   300          32      421.23     2147.07     3473.00      302.82       99.89

Throughput using 50 threads:
    30          32   103720.21      422.83      657.00       36.49       90.06
    50          32    67558.15      648.18     1557.00       55.56       95.64
    80          32    48001.59      927.00     1375.00       84.91       98.21
   100          32    39163.71     1115.88     1477.00      104.59       98.82
   150          32    27593.48     1611.91     2063.00      154.07       99.48
   200          32    20221.23     2197.26     3580.00      203.61       99.72
   300          32    14480.85     3107.67     4203.00      302.85       99.89
```

### SIFT1B

The SIFT1B index prefix is `/mnt/nvme2/indices/SIFT1B/1B`. The query file and
search parameters match SIFT100M; the ground truth file is
`/mnt/nvme/data/bigann/truth.bin`.

Latency run:

```bash
sudo build/tests/search_disk_index uint8 /mnt/nvme2/indices/SIFT1B/1B \
  1 32 \
  /mnt/nvme/data/bigann/bigann_query.bbin \
  /mnt/nvme/data/bigann/truth.bin \
  10 l2 pq 2 10 \
  10 10 30 50 80 100 150 200 300
```

Throughput run:

```bash
sudo build/tests/search_disk_index uint8 /mnt/nvme2/indices/SIFT1B/1B \
  105 32 \
  /mnt/nvme/data/bigann/bigann_query.bbin \
  /mnt/nvme/data/bigann/truth.bin \
  10 l2 pq 2 10 \
  10 10 30 50 80 100 150 200 300
```

Abbreviated output:

```text
SPDK initialized: 4 pollers, LBA size 512 bytes
     L   I/O Width         QPS  AvgLat(us)     P99 Lat   Mean IOs   Recall@10
================================================================================
Latency using 1 thread:
    30          32     1506.21      566.43      870.00       36.90       84.04
    50          32     1069.88      817.22     1177.00       56.03       91.65
    80          32      773.55     1152.05     1651.00       85.29       95.93
   100          32      861.59     1037.76     1462.00      104.99       97.22
   150          32      643.82     1397.63     1908.00      154.41       98.67
   200          32      472.69     1928.17     2820.00      203.95       99.25
   300          32      300.58     3087.57     4413.00      303.24       99.67

Throughput using 105 threads:
    30          32    92349.23     1025.05     1650.00       37.20       84.09
    45          32    71916.16     1332.71     2046.00       51.47       90.42  
    50          32    64324.28     1497.93     2440.00       56.32       91.69
    80          32    43168.75     2232.29     3619.00       85.65       95.94
   100          32    35296.38     2733.85     4165.00      105.11       97.23
   150          32    24913.51     3933.83     8087.00      154.59       98.67
   200          32    18654.20     5209.41     8322.00      204.15       99.25
   300          32    12988.08     7589.64    12322.00      303.32       99.67
```

## Implementation Details

### Data Layout

The `aio` and `io_uring` backends read the on-disk index as regular files; a
multi-SSD deployment typically spreads one or more index files across filesystem
paths backed by different drives.

The SPDK backend keeps only `{index_prefix}_disk.index` on the raw SPDK SSDs and
serves its reads/writes directly. Other artifacts — attribute indexes, PQ
vectors, etc. — remain on the filesystem and are accessed via the `uring`
backend.

Placement of the disk index on SPDK SSDs works as follows:
- PipeANN opens the regular `{index_prefix}_disk.index` file once and streams it
  directly to the SPDK target.
- The target is a RAID-0-style stripe across the PCIe NVMe devices listed in
  `spdk_bdevs.json`.
- The copy is cached on the raw devices. If the index path, SSD list, and
  stripe size all match the recorded marker, subsequent runs print
  `SPDK index already copied` and skip the copy.

Attribute indexes are not yet placed on the SPDK stripe; filtered-search
attribute reads still go through `io_uring` against the original files.

### Thread Model

The SPDK backend uses a poller-based threading model: PipeANN spawns one poller
thread per configured SSD. Each poller owns its SSD's queue pair and runs a
tight loop submitting I/O commands and reaping completions on that device.

Each search thread communicates with every poller via a pair of software queues:

- `SQ` (submission queue): the search thread pushes read/write requests to the
  poller owning the target SSD.
- `CQ` (completion queue): the poller hands completed requests back to the
  originating search thread.

This isolates NVMe queue-pair ownership inside the poller threads while letting
search threads issue requests to every striped SSD without touching NVMe
completions directly.

CPU affinity follows the SSD order declared in `spdk_bdevs.json`:

- pollers are pinned starting at CPU `0`; poller `i` is pinned to CPU `i`.
- search threads and the insert background thread (if enabled) are pinned to
  CPUs immediately after the pollers.

For example, with four SSDs, pollers occupy CPUs `0..3` and the remaining
threads start at CPU `4`.
