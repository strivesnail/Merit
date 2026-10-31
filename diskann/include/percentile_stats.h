// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <fstream>
#include <functional>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann
{
struct QueryStats
{
    float total_us = 0; // total time to process query in micros
    float io_us = 0;    // total time spent in IO
    float cpu_us = 0;   // total time spent in CPU

    unsigned n_4k = 0;         // # of 4kB reads
    unsigned n_8k = 0;         // # of 8kB reads
    unsigned n_12k = 0;        // # of 12kB reads
    unsigned n_ios = 0; // disk-tier node fetches (one per base frontier node or NEIGHBORCACHE disk cache node, + reorder)
    unsigned n_unique_sectors = 0;      // distinct base _disk.index 4KB sectors (frontier path only)
    unsigned n_unique_neighborcache_sectors = 0; // distinct NEIGHBORCACHE disk cache 4KB sectors touched in one query
    unsigned read_size = 0;    // total # of bytes read
    unsigned n_cmps_saved = 0; // # cmps saved
    unsigned n_cmps = 0;       // # cmps
    unsigned n_cache_hits = 0; // # cache_hits
    unsigned n_hops = 0;       // # search hops
    unsigned n_sector_cache_hits = 0; // query-local sector cache hits (no disk read)
    unsigned n_disk_reads = 0;        // actual disk read operations (frontier misses)
    unsigned n_neighborcache_dc_hits = 0;     // nodes served from NEIGHBORCACHE disk-cache file
    unsigned n_neighborcache_sector_cache_hits = 0; // subset of n_sector_cache_hits served from cached D-cache pages
    unsigned n_neighborcache_disk_reads = 0;        // subset of n_disk_reads issued to the D-cache file
    unsigned n_neighborcache_shared_cross_hits = 0; // subset of n_neighborcache_sector_cache_hits on pages read by other queries
    unsigned n_neighborcache_mem_evictions = 0; // runtime evictions from NEIGHBORCACHE memory pool
    unsigned n_neighborcache_dyn_hits = 0;      // nodes served from runtime-flushed d-cache pages
    // Beam-time classification (one increment per expanded beam neighbor):
    // n-cache → n_cache_hits; resident prefetch records → n_neighborcache_prefetch_hits;
    // d-cache directory (seed page not yet in query prefetch buffer) → n_neighborcache_dcache_seed_hits.
    unsigned n_neighborcache_prefetch_hits = 0;     // direct seed-prefetch hit, excluding query-local D-page reuse
    unsigned n_neighborcache_dcache_seed_hits = 0;  // d-cache directory hit (triggers seed-page serve)
    // Actual dynamic directory probes for demand beam nodes, once per node request.
    // Excludes skipped lookup paths and internal maintenance/speculative queries.
    unsigned n_neighborcache_dcache_demand_probes = 0;
    unsigned n_neighborcache_dcache_demand_hits = 0;
    // Child of a D-cache seed. on_page means that child is stored on the parent's seed page.
    unsigned n_dparent_nbrs = 0;
    unsigned n_dparent_nbr_on_page = 0;
    unsigned n_beam_lookups = 0;            // process_beam_neighbor invocations
    unsigned n_neighborcache_dyn_flushes = 0;   // seed pages written at query end / n-cache evict
    unsigned n_neighborcache_dyn_pages = 0;     // committed dynamic d-cache pages after this query
    unsigned n_neighborcache_dyn_disk_reads = 0; // physical 4KB reads from the dynamic d-cache file
    unsigned n_neighborcache_base_pages_avoided = 0; // base pages not read because dynamic d-cache served them
    unsigned n_neighborcache_dcache_probe = 0;   // query sampled while the dynamic d-cache gate was off
    unsigned n_neighborcache_mcache_hits = 0;    // expanded nodes whose metadata was already resident
    unsigned n_neighborcache_mcache_misses = 0;  // expanded nodes inserted into the metadata cache
    unsigned n_neighborcache_multiread_ios = 0;   // disk reads spanning nsectors>1 (pct80 seed groups)
    unsigned n_neighborcache_io_avoided = 0;      // continuation sectors not re-read (grouped under multiread)
    unsigned n_neighborcache_setcover_grouped = 0; // nodes pulled into set-cover via multiread span
    unsigned n_neighborcache_finalize_skipped = 0; // neighborcache pending nodes dropped in finalize (sec_buf null)
    // Consecutive disk-read sector id gaps (only recorded with query sector cache + stats)
    uint64_t sum_abs_sector_jump = 0;
    unsigned n_sector_jump_samples = 0;
    unsigned n_sector_jump_le8 = 0; // jumps with abs delta <= 8 sectors (32 KiB)
    uint64_t max_sector_jump = 0;   // max consecutive |Δsector| in this query
    uint64_t sum_intra_batch_spread = 0;
    unsigned n_intra_batch_spread_samples = 0;

    static constexpr size_t DISK_SECTOR_BUCKETS = 32;
    unsigned n_disk_read_batches = 0;
    float sum_batch_io_us = 0.f;
    unsigned n_batches_size1 = 0;
    float sum_batch_us_size1 = 0.f;
    unsigned n_batches_size2 = 0;
    float sum_batch_us_size2 = 0.f;
    unsigned n_batches_size_other = 0;
    float sum_batch_us_size_other = 0.f;
    std::array<unsigned, DISK_SECTOR_BUCKETS> disk_sector_bucket{};
    unsigned n_disk_sector_bucket_samples = 0;

    // Populated when base-frontier recording is enabled (diagnostics).
    std::vector<uint32_t> base_frontier_nodes;

    struct HopFrontierRecord
    {
        unsigned hop = 0;
        std::vector<uint32_t> neighborcache_nodes;
        std::vector<uint32_t> base_nodes;
        struct PhysicalRead
        {
            bool neighborcache = false;
            uint64_t sector = 0;
            uint16_t nsectors = 1;
        };
        std::vector<PhysicalRead> physical_reads;
    };
    // Populated when hop-frontier recording is enabled (diagnostics).
    std::vector<HopFrontierRecord> hop_frontier_trace;
};

template <typename T>
inline T get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                              const std::function<T(const QueryStats &)> &member_fn)
{
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++)
    {
        vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
}

template <typename T>
inline double get_mean_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
{
    double avg = 0;
    for (uint64_t i = 0; i < len; i++)
    {
        avg += (double)member_fn(stats[i]);
    }
    return avg / len;
}
} // namespace diskann
