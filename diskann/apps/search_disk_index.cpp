// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "hotness_profiler.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"
#include "relayout_utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results)
{
    diskann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(8) << percentiles[s] << "%";
    }
    diskann::cout << std::endl;
    diskann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(9) << results[s];
    }
    diskann::cout << std::endl;
}

template <typename T, typename LabelT = uint32_t>
int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false,
                      const bool enable_access_profile = false, const std::string &access_profile_prefix = "",
                      const std::string &relayout_order_prefix = "", const bool enable_query_sector_cache = false,
                      const double merit_memory_gb = 0.0, const std::string &merit_profile_prefix = "",
                      const double merit_host_memory_gb = 0.0, const double merit_memory_reserve_gb = 8.0,
                      const double merit_disk_cache_ratio = 0.0, const bool merit_disk_cache_exclude_memory = true,
                      const bool merit_memory_runtime_admit = false,
                      const double merit_evict_memory_gb = 0.0, const double merit_evict_disk_ratio = 0.0,
                      const bool merit_unified_disk_cache = false, uint32_t merit_disk_cache_k_hops = 1,
                      const std::string &merit_disk_cache_layout = "directed_beam",
                      const std::string &merit_disk_cache_reuse_prefix = "",
                      const std::string &dump_query_stats_path = "",
                      const std::string &dump_base_nodes_path = "",
                      const std::string &dump_hop_trace_path = "")
{
    diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        diskann::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // load query bin
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndex<T, LabelT>(reader, metric));

    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

    if (res != 0)
    {
        return res;
    }

    if (enable_access_profile)
    {
        _pFlashIndex->enable_access_profile(true);
        diskann::cout << "Access profiling enabled. Output prefix: " << access_profile_prefix << std::endl;
    }

    if (enable_query_sector_cache)
    {
        _pFlashIndex->enable_query_sector_cache(true);
        diskann::cout << "Query-local sector cache enabled (repeat sector reads avoid disk)." << std::endl;
    }

    if (!dump_base_nodes_path.empty())
    {
        _pFlashIndex->enable_base_frontier_recording(true);
        diskann::cout << "Base-frontier node recording enabled -> " << dump_base_nodes_path << std::endl;
    }

    if (!dump_hop_trace_path.empty())
    {
        _pFlashIndex->enable_hop_frontier_recording(true);
        diskann::cout << "Hop-frontier trace recording enabled -> " << dump_hop_trace_path << std::endl;
    }

    std::vector<uint32_t> relayout_order;
    uint64_t relayout_nnodes_per_sector = 0;
    const bool remap_recall_ids = !relayout_order_prefix.empty();
    if (remap_recall_ids)
    {
        if (diskann::load_relayout_order(relayout_order_prefix, relayout_order, relayout_nnodes_per_sector) != 0)
        {
            diskann::cerr << "Failed to load relayout order from prefix " << relayout_order_prefix << std::endl;
            return -1;
        }
        diskann::cout << "Relayout ID remap enabled for recall (order size=" << relayout_order.size() << ")."
                      << std::endl;
    }

    std::vector<uint32_t> node_list;
    const bool dyn3_cache = (std::getenv("MERIT_DYNAMIC_3CACHE") != nullptr &&
                             std::strcmp(std::getenv("MERIT_DYNAMIC_3CACHE"), "0") != 0);
    const bool ncache_only = (std::getenv("MERIT_NCACHE_ONLY") != nullptr &&
                              std::strcmp(std::getenv("MERIT_NCACHE_ONLY"), "0") != 0);
    if (merit_memory_gb > 0.0)
    {
        uint64_t max_nodes = 0;
        std::string budget_report;
        if (_pFlashIndex->plan_merit_memory_cache(merit_memory_gb, merit_host_memory_gb, merit_memory_reserve_gb,
                                                  num_threads, max_nodes, budget_report) != 0)
        {
            diskann::cerr << budget_report << std::endl;
            return -1;
        }
        if (const char *cap_env = std::getenv("MERIT_NCACHE_CAP_NODES"))
        {
            const uint64_t requested_cap = std::strtoull(cap_env, nullptr, 10);
            if (requested_cap > 0 && max_nodes > requested_cap)
            {
                max_nodes = requested_cap;
                diskann::cout << "MERIT N-cache exact node cap: " << max_nodes << std::endl;
            }
        }
        diskann::cout << budget_report << std::endl;

        if (dyn3_cache || ncache_only)
        {
            diskann::cout << "MERIT Memory cache: BFS around medoid, " << max_nodes << " nodes"
                          << (ncache_only && !dyn3_cache ? " (n-cache only)." : " (DiskANN-style).") << std::endl;
            _pFlashIndex->cache_bfs_levels(max_nodes, node_list);
            if (_pFlashIndex->load_merit_memory_pool("", node_list) != 0)
            {
                diskann::cerr << "Failed to load MERIT memory pool from BFS list." << std::endl;
                return -1;
            }
        }
        else
        {
            if (merit_profile_prefix.empty())
            {
                diskann::cerr << "Error: --merit_memory_gb > 0 requires --merit_profile_prefix (Run2 node_expand)."
                              << std::endl;
                return -1;
            }
            if (num_nodes_to_cache > 0)
            {
                diskann::cout << "Note: --merit_memory_gb set; ignoring --num_nodes_to_cache=" << num_nodes_to_cache
                              << " (BFS medoid cache)." << std::endl;
            }
            if (_pFlashIndex->build_merit_memory_node_list(merit_profile_prefix, max_nodes, node_list) != 0)
            {
                diskann::cerr << "Failed to build MERIT memory node list from profile " << merit_profile_prefix
                              << std::endl;
                return -1;
            }
            diskann::cout << "MERIT Memory cache: loading Top-" << node_list.size()
                          << " nodes into MERIT memory pool." << std::endl;
            if (_pFlashIndex->load_merit_memory_pool(merit_profile_prefix, node_list) != 0)
            {
                diskann::cerr << "Failed to load MERIT memory pool." << std::endl;
                return -1;
            }
        }
        _pFlashIndex->enable_merit_memory_runtime_admit(merit_memory_runtime_admit);
        if (merit_memory_runtime_admit)
        {
            diskann::cout << "MERIT memory pool: runtime admit/evict enabled (prefer num_threads=1)." << std::endl;
        }
    }
    else
    {
        diskann::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;
        _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
        // if (num_nodes_to_cache > 0)
        //     _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
        //     num_threads, node_list);
        _pFlashIndex->load_cache_list(node_list);
    }
    node_list.clear();
    node_list.shrink_to_fit();

    if (merit_disk_cache_ratio > 0.0)
    {
        const std::string profile =
            !merit_profile_prefix.empty()
                ? merit_profile_prefix
                : (!access_profile_prefix.empty() ? access_profile_prefix : std::string(""));
        if (profile.empty() && merit_disk_cache_layout != "bfs")
        {
            diskann::cerr << "Error: --merit_disk_cache_ratio > 0 requires --merit_profile_prefix "
                             "(or --access_profile_prefix)."
                          << std::endl;
            return -1;
        }
        uint64_t dc_nodes = 0;
        std::string dc_report;
        if (_pFlashIndex->plan_merit_disk_cache(merit_disk_cache_ratio, dc_nodes, dc_report,
                                                diskann::disk_cache_layout_allows_replicas(merit_disk_cache_layout)) !=
            0)
        {
            diskann::cerr << dc_report << std::endl;
            return -1;
        }
        if (const char *cap_env = std::getenv("MERIT_DCACHE_NODE_CAP"))
        {
            const uint64_t requested_cap = std::strtoull(cap_env, nullptr, 10);
            if (requested_cap > 0)
            {
                dc_nodes = requested_cap;
                diskann::cout << "MERIT d-cache exact node cap: " << dc_nodes << std::endl;
            }
        }
        diskann::cout << dc_report << std::endl;
        diskann::cout << "MERIT disk-cache layout=" << merit_disk_cache_layout << " k_hops=" << merit_disk_cache_k_hops
                      << std::endl;
        const std::string dc_out = result_output_prefix;
        const uint64_t dc_rank_skip =
            merit_disk_cache_exclude_memory ? _pFlashIndex->merit_memory_cached_count() : 0;
        if (dc_rank_skip > 0)
        {
            diskann::cout << "MERIT disk-cache: rank_skip=" << dc_rank_skip
                          << " (exclude memory-tier nodes from disk cache)." << std::endl;
        }
        if (!merit_disk_cache_reuse_prefix.empty())
        {
            diskann::cout << "MERIT disk-cache: reusing disk cache at prefix " << merit_disk_cache_reuse_prefix << std::endl;
            if (_pFlashIndex->load_merit_disk_cache_from_prefix(merit_disk_cache_reuse_prefix) != 0)
            {
                diskann::cerr << "Failed to load MERIT disk cache from " << merit_disk_cache_reuse_prefix << std::endl;
                return -1;
            }
        }
        else if (_pFlashIndex->build_and_load_merit_disk_cache(profile, dc_nodes, dc_out, dc_rank_skip,
                                                               merit_unified_disk_cache, merit_disk_cache_k_hops,
                                                               merit_disk_cache_layout) != 0)
        {
            diskann::cerr << "Failed to build/load MERIT disk cache." << std::endl;
            return -1;
        }
        const char *disable_sc = std::getenv("MERIT_DISABLE_QUERY_SECTOR_CACHE");
        if (disable_sc != nullptr && std::strcmp(disable_sc, "0") != 0)
        {
            diskann::cout << "MERIT disk-cache: query-local sector cache disabled (MERIT_DISABLE_QUERY_SECTOR_CACHE)."
                          << std::endl;
        }
        else
        {
            _pFlashIndex->enable_query_sector_cache(true);
            diskann::cout << "MERIT disk-cache: query-local sector cache enabled (base + disk cache sectors)."
                          << std::endl;
        }
        if (std::getenv("MERIT_VERIFY_DC") != nullptr)
        {
            if (_pFlashIndex->verify_merit_disk_cache_against_base() != 0)
                diskann::cerr << "MERIT verify: sidecar vs base mismatches detected." << std::endl;
        }
    }

    {
        const char *dyn_env = std::getenv("MERIT_DYNAMIC_3CACHE");
        if (dyn_env != nullptr && std::strcmp(dyn_env, "0") != 0)
        {
            _pFlashIndex->enable_merit_dynamic_3cache(true, result_output_prefix);
            diskann::cout << "MERIT dynamic 3-cache enabled (n-cache LRU + m-cache + runtime d-cache flush)."
                          << std::endl;
        }
    }

    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP)
    {
        if (file_exists(warmup_query_file))
        {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        }
        else
        {
            warmup_num = (std::min)((uint32_t)150000, (uint32_t)15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++)
            {
                for (uint32_t d = 0; d < warmup_dim; d++)
                {
                    warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                }
            }
        }
        diskann::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)warmup_num; i++)
        {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        diskann::cout << "..done" << std::endl;
    }

    const std::string merit_profile_for_reload =
        !merit_profile_prefix.empty() ? merit_profile_prefix : access_profile_prefix;
    if (merit_evict_memory_gb > 0.0)
    {
        if (merit_profile_for_reload.empty())
        {
            diskann::cerr << "Error: --merit_evict_memory_gb requires --merit_profile_prefix." << std::endl;
            return -1;
        }
        uint64_t mem_evicted = 0;
        std::string mem_report;
        diskann::cout << "=== MERIT memory batch eviction (reload to " << merit_evict_memory_gb << " GB) ==="
                      << std::endl;
        if (_pFlashIndex->reload_merit_memory_cache(merit_profile_for_reload, merit_evict_memory_gb,
                                                    merit_host_memory_gb, merit_memory_reserve_gb, num_threads,
                                                    mem_evicted, mem_report) != 0)
            return -1;
    }
    if (merit_evict_disk_ratio > 0.0)
    {
        if (merit_profile_for_reload.empty())
        {
            diskann::cerr << "Error: --merit_evict_disk_ratio requires --merit_profile_prefix." << std::endl;
            return -1;
        }
        uint64_t dc_evicted = 0;
        std::string dc_reload_report;
        diskann::cout << "=== MERIT disk-cache batch eviction (reload ratio " << merit_evict_disk_ratio << ") ==="
                      << std::endl;
        const uint64_t dc_skip =
            merit_disk_cache_exclude_memory ? _pFlashIndex->merit_memory_cached_count() : 0;
        if (_pFlashIndex->reload_merit_disk_cache(merit_profile_for_reload, merit_evict_disk_ratio,
                                                  result_output_prefix, dc_skip, dc_evicted, dc_reload_report,
                                                  merit_unified_disk_cache, merit_disk_cache_k_hops,
                                                  merit_disk_cache_layout) != 0)
            return -1;
    }

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "BasePages";
    if (merit_disk_cache_ratio > 0.0)
        diskann::cout << std::setw(16) << "DiskCachePg";
    diskann::cout << std::setw(16) << "Dup IOs";
    if (enable_query_sector_cache)
    {
        diskann::cout << std::setw(16) << "Disk Reads" << std::setw(16) << "SectCacheHit" << std::setw(16)
                      << "IoUs/Read" << std::setw(16) << "|dSec|" << std::setw(16) << "Jump<=8s%";
    }
    if (merit_disk_cache_ratio > 0.0)
    {
        diskann::cout << std::setw(16) << "MeritDcHit";
    }
    diskann::cout << std::setw(16) << "Mean Hops" << std::setw(16) << "Mean IO (us)" << std::setw(16) << "CPU (s)";
    if (calc_recall_flag)
    {
        diskann::cout << std::setw(16) << recall_string << std::endl;
    }
    else
        diskann::cout << std::endl;
    diskann::cout << "=================================================================="
                     "================================================================="
                  << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    double best_recall = 0.0;
    bool dynamic_full_warmup_done = false;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        if (beamwidth <= 0)
        {
            diskann::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        }
        else
            optimized_beamwidth = beamwidth;

        const char *full_warmup_env = std::getenv("MERIT_REQUIRE_FULL_DCACHE_WARMUP");
        if (!dynamic_full_warmup_done && _pFlashIndex->merit_dynamic_3cache_enabled() &&
            full_warmup_env != nullptr && std::strcmp(full_warmup_env, "0") != 0)
        {
            const char *max_env = std::getenv("MERIT_FULL_WARMUP_MAX_QUERIES");
            const uint64_t max_warmup_queries =
                max_env == nullptr ? 100000000ULL : std::strtoull(max_env, nullptr, 10);
            const char *batch_env = std::getenv("MERIT_FULL_WARMUP_BATCH_QUERIES");
            const uint64_t requested_batch =
                batch_env == nullptr ? 10000ULL : std::strtoull(batch_env, nullptr, 10);
            const uint64_t warmup_batch = std::max<uint64_t>(1, requested_batch);
            std::vector<uint64_t> warmup_ids(static_cast<size_t>(warmup_batch));
            std::vector<float> warmup_dists(static_cast<size_t>(warmup_batch));
            uint64_t completed = 0;
            uint64_t next_report = 1000000;
            const auto warmup_start = std::chrono::high_resolution_clock::now();
            diskann::cout << "MERIT full D-cache warmup: target="
                          << _pFlashIndex->merit_dynamic_page_capacity() << " pages" << std::endl;
            while (!_pFlashIndex->merit_dynamic_disk_full() && completed < max_warmup_queries)
            {
                const uint64_t current_batch =
                    std::min<uint64_t>(warmup_batch, max_warmup_queries - completed);
#pragma omp parallel for schedule(dynamic, 1)
                for (int64_t i = 0; i < static_cast<int64_t>(current_batch); ++i)
                {
                    const size_t query_index =
                        static_cast<size_t>((completed + static_cast<uint64_t>(i)) % query_num);
                    _pFlashIndex->cached_beam_search(
                        query + query_index * query_aligned_dim, 1, L,
                        warmup_ids.data() + i, warmup_dists.data() + i,
                        optimized_beamwidth, use_reorder_data, nullptr);
                }
                completed += current_batch;
                if (completed >= next_report || _pFlashIndex->merit_dynamic_disk_full())
                {
                    diskann::cout << "MERIT full D-cache warmup progress: queries=" << completed
                                  << " pages=" << _pFlashIndex->merit_dynamic_page_count() << "/"
                                  << _pFlashIndex->merit_dynamic_page_capacity() << std::endl;
                    next_report += 1000000;
                }
            }
            if (!_pFlashIndex->merit_dynamic_disk_full())
            {
                diskann::cerr << "MERIT full D-cache warmup failed after " << completed
                              << " queries: pages=" << _pFlashIndex->merit_dynamic_page_count() << "/"
                              << _pFlashIndex->merit_dynamic_page_capacity() << std::endl;
                return -1;
            }
            const auto warmup_end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> warmup_elapsed = warmup_end - warmup_start;
            diskann::cout << "MERIT full D-cache warmup complete: queries=" << completed
                          << " seconds=" << warmup_elapsed.count()
                          << "; measured query phase starts now." << std::endl;
            _pFlashIndex->merit_dynamic_begin_measurement();
            dynamic_full_warmup_done = true;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new diskann::QueryStats[query_num];

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            if (!filtered_search)
            {
                _pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, L,
                                                 query_result_ids_64.data() + (i * recall_at),
                                                 query_result_dists[test_id].data() + (i * recall_at),
                                                 optimized_beamwidth, use_reorder_data, stats + i);
            }
            else
            {
                LabelT label_for_search;
                if (query_filters.size() == 1)
                { // one label for all queries
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[0]);
                }
                else
                { // one label for each query
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[i]);
                }
                _pFlashIndex->cached_beam_search(
                    query + (i * query_aligned_dim), recall_at, L, query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, true, label_for_search,
                    use_reorder_data, stats + i);
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
                                                   query_num, recall_at);

        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_99 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.99, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_pages = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_unique_sectors; });

        auto mean_disk_cache_pages = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_unique_merit_sectors; });

        auto mean_dup_ios = diskann::get_mean_stats<uint32_t>(stats, query_num, [](const diskann::QueryStats &stats) {
            const unsigned unique_pages = stats.n_unique_sectors + stats.n_unique_merit_sectors;
            return stats.n_ios > unique_pages ? stats.n_ios - unique_pages : 0u;
        });

        auto mean_hops = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                           [](const diskann::QueryStats &stats) { return stats.n_hops; });

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_io_us = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.io_us; });

        double recall = 0;
        if (calc_recall_flag)
        {
            if (remap_recall_ids)
            {
                std::vector<uint32_t> mapped_ids(query_num * recall_at);
                for (size_t qi = 0; qi < query_num; qi++)
                {
                    for (uint32_t ki = 0; ki < recall_at; ki++)
                    {
                        const uint32_t new_id = query_result_ids[test_id][qi * recall_at + ki];
                        mapped_ids[qi * recall_at + ki] =
                            (new_id < relayout_order.size()) ? relayout_order[new_id] : new_id;
                    }
                }
                recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                   mapped_ids.data(), recall_at, recall_at);
            }
            else
            {
                recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                   query_result_ids[test_id].data(), recall_at, recall_at);
            }
            best_recall = std::max(recall, best_recall);
        }

        diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_99 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_pages;
        if (merit_disk_cache_ratio > 0.0)
            diskann::cout << std::setw(16) << mean_disk_cache_pages;
        diskann::cout << std::setw(16) << mean_dup_ios;
        if (enable_query_sector_cache)
        {
            auto mean_disk_reads = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_disk_reads; });
            auto mean_sector_cache_hits = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_sector_cache_hits; });
            auto mean_io_us_per_disk_read = diskann::get_mean_stats<float>(stats, query_num, [](const diskann::QueryStats &s) {
                return s.n_disk_reads > 0 ? (s.io_us / (float)s.n_disk_reads) : 0.f;
            });
            auto mean_abs_sector_jump = diskann::get_mean_stats<double>(stats, query_num, [](const diskann::QueryStats &s) {
                return s.n_sector_jump_samples > 0 ? ((double)s.sum_abs_sector_jump / (double)s.n_sector_jump_samples)
                                                   : 0.0;
            });
            auto mean_local_jump_pct = diskann::get_mean_stats<double>(stats, query_num, [](const diskann::QueryStats &s) {
                return s.n_sector_jump_samples > 0
                           ? (100.0 * (double)s.n_sector_jump_le8 / (double)s.n_sector_jump_samples)
                           : 0.0;
            });
            diskann::cout << std::setw(16) << mean_disk_reads << std::setw(16) << mean_sector_cache_hits
                          << std::setw(16) << mean_io_us_per_disk_read << std::setw(16) << mean_abs_sector_jump
                          << std::setw(16) << mean_local_jump_pct;
        }
        if (merit_disk_cache_ratio > 0.0)
        {
            auto mean_merit_dc = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_merit_dc_hits; });
            diskann::cout << std::setw(16) << mean_merit_dc;
        }
        diskann::cout << std::setw(16) << mean_hops << std::setw(16) << mean_io_us << std::setw(16) << mean_cpuus;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << recall;
        }
        diskann::cout << std::endl;
        {
            auto mean_disk_reads_all = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &s) { return s.n_disk_reads; });
            auto mean_dyn_hits = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &s) { return s.n_merit_dyn_hits; });
            auto mean_dyn_flushes = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &s) { return s.n_merit_dyn_flushes; });
            auto mean_dyn_reads = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &s) { return s.n_merit_dyn_disk_reads; });
            auto mean_base_avoided = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &s) { return s.n_merit_base_pages_avoided; });
            uint64_t mcache_hits = 0;
            uint64_t mcache_misses = 0;
            for (size_t qi = 0; qi < query_num; qi++)
            {
                mcache_hits += stats[qi].n_merit_mcache_hits;
                mcache_misses += stats[qi].n_merit_mcache_misses;
            }
            if (_pFlashIndex->merit_dynamic_3cache_enabled())
            {
                diskann::cout << "      dynamic3: mean_disk_reads=" << mean_disk_reads_all
                              << " mean_dyn_hits=" << mean_dyn_hits
                              << " mean_dyn_reads=" << mean_dyn_reads
                              << " mean_base_pages_avoided=" << mean_base_avoided
                              << " net_pages_saved=" << (mean_base_avoided - mean_dyn_reads)
                              << " mean_dyn_flushes=" << mean_dyn_flushes
                              << " mcache_hit_rate="
                              << (mcache_hits + mcache_misses == 0
                                      ? 0.0
                                      : 100.0 * static_cast<double>(mcache_hits) /
                                            static_cast<double>(mcache_hits + mcache_misses))
                              << "%";
                if (calc_recall_flag)
                    diskann::cout << " recall@" << recall_at << "=" << recall;
                diskann::cout << std::endl;
                _pFlashIndex->print_merit_dynamic_3cache_stats();
            }
            _pFlashIndex->print_merit_memory_cache_stats();
        }
        if (merit_disk_cache_ratio > 0.0)
        {
            auto mean_multiread = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &s) { return s.n_merit_multiread_ios; });
            auto mean_io_avoided = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &s) { return s.n_merit_io_avoided; });
            auto mean_scover_grp = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &s) { return s.n_merit_setcover_grouped; });
            auto mean_finalize_skipped = diskann::get_mean_stats<uint32_t>(
                stats, query_num, [](const diskann::QueryStats &s) { return s.n_merit_finalize_skipped; });
            diskann::cout << "      pct80 IO: multiread_ios=" << mean_multiread << " io_avoided=" << mean_io_avoided
                          << " setcover_grouped=" << mean_scover_grp
                          << " finalize_skipped=" << mean_finalize_skipped << std::endl;
        }
        if (enable_query_sector_cache)
        {
            auto mean_max_jump = diskann::get_mean_stats<double>(
                stats, query_num, [](const diskann::QueryStats &s) { return (double)s.max_sector_jump; });
            auto mean_intra_batch_spread = diskann::get_mean_stats<double>(stats, query_num, [](const diskann::QueryStats &s) {
                return s.n_intra_batch_spread_samples > 0
                           ? ((double)s.sum_intra_batch_spread / (double)s.n_intra_batch_spread_samples)
                           : 0.0;
            });
            auto mean_abs_sector_jump = diskann::get_mean_stats<double>(stats, query_num, [](const diskann::QueryStats &s) {
                return s.n_sector_jump_samples > 0 ? ((double)s.sum_abs_sector_jump / (double)s.n_sector_jump_samples)
                                                   : 0.0;
            });

            uint64_t tot_batches = 0, tot_batches1 = 0, tot_batches2 = 0, tot_batches_other = 0;
            uint64_t tot_disk_reads = 0;
            double sum_batch_us1 = 0, sum_batch_us2 = 0, sum_batch_us_other = 0;
            std::array<uint64_t, diskann::QueryStats::DISK_SECTOR_BUCKETS> global_bucket{};
            uint64_t tot_bucket_samples = 0;
            for (size_t qi = 0; qi < query_num; qi++)
            {
                tot_batches += stats[qi].n_disk_read_batches;
                tot_batches1 += stats[qi].n_batches_size1;
                tot_batches2 += stats[qi].n_batches_size2;
                tot_batches_other += stats[qi].n_batches_size_other;
                tot_disk_reads += stats[qi].n_disk_reads;
                sum_batch_us1 += stats[qi].sum_batch_us_size1;
                sum_batch_us2 += stats[qi].sum_batch_us_size2;
                sum_batch_us_other += stats[qi].sum_batch_us_size_other;
                for (size_t b = 0; b < diskann::QueryStats::DISK_SECTOR_BUCKETS; b++)
                    global_bucket[b] += stats[qi].disk_sector_bucket[b];
                tot_bucket_samples += stats[qi].n_disk_sector_bucket_samples;
            }

            const double mean_batch_size = tot_batches > 0 ? ((double)tot_disk_reads / (double)tot_batches) : 0.0;
            const double batch_us_size1 = tot_batches1 > 0 ? (sum_batch_us1 / (double)tot_batches1) : 0.0;
            const double batch_us_size2 = tot_batches2 > 0 ? (sum_batch_us2 / (double)tot_batches2) : 0.0;
            const double us_per_read_in_batch2 =
                tot_batches2 > 0 ? (sum_batch_us2 / (2.0 * (double)tot_batches2)) : 0.0;
            const double us_per_read_in_batch1 = batch_us_size1;

            diskann::cout << "      locality: mean max|dSec|=" << mean_max_jump
                          << " intra-batch spread(sectors)=" << mean_intra_batch_spread
                          << " mean |dSec|~MiB=" << (mean_abs_sector_jump * 4096.0 / (1024.0 * 1024.0)) << std::endl;
            diskann::cout << "      batch IO: batches=" << tot_batches << " mean_batch_size=" << mean_batch_size
                          << " size1 n=" << tot_batches1 << " us/batch=" << batch_us_size1
                          << " us/read=" << us_per_read_in_batch1 << " size2 n=" << tot_batches2
                          << " us/batch=" << batch_us_size2 << " us/read(allocated)=" << us_per_read_in_batch2;
            if (tot_batches_other > 0)
                diskann::cout << " size>2 n=" << tot_batches_other;
            diskann::cout << std::endl;

            std::string hist_path = result_output_prefix + "_" + std::to_string(L) + "_disk_sector_hist.txt";
            std::ofstream hist_out(hist_path);
            if (hist_out.is_open())
            {
                hist_out << "# disk read sector id histogram (" << diskann::QueryStats::DISK_SECTOR_BUCKETS
                         << " buckets over graph sector index)\n";
                hist_out << "# tot_disk_reads=" << tot_disk_reads << " tot_batches=" << tot_batches
                         << " mean_batch_size=" << mean_batch_size << "\n";
                hist_out << "# batch_us size1=" << batch_us_size1 << " size2=" << batch_us_size2
                         << " us_per_read_batch2=" << us_per_read_in_batch2 << "\n";
                for (size_t b = 0; b < diskann::QueryStats::DISK_SECTOR_BUCKETS; b++)
                {
                    const double pct =
                        tot_bucket_samples > 0 ? (100.0 * (double)global_bucket[b] / (double)tot_bucket_samples) : 0.0;
                    hist_out << "bucket " << b << " count " << global_bucket[b] << " pct " << pct << "\n";
                }
                hist_out.close();
                diskann::cout << "      sector histogram written to " << hist_path << std::endl;
            }
        }
        if (!dump_query_stats_path.empty())
        {
            std::ofstream qout(dump_query_stats_path);
            if (qout.is_open())
            {
                qout << "query_id,total_us,io_us,cpu_us,n_ios,n_base_pages,n_disk_cache_pages,n_merit_hits,"
                        "n_mem_hits,n_disk_reads,n_sector_cache_hits,n_hops,max_sector_jump,"
                        "n_merit_dyn_hits,n_merit_dyn_flushes,n_merit_dyn_pages,n_merit_dyn_disk_reads,"
                        "n_merit_base_pages_avoided,n_merit_dcache_probe,"
                        "n_merit_mcache_hits,n_merit_mcache_misses\n";
                for (size_t qi = 0; qi < query_num; qi++)
                {
                    const auto &s = stats[qi];
                    qout << qi << ',' << s.total_us << ',' << s.io_us << ',' << s.cpu_us << ',' << s.n_ios << ','
                         << s.n_unique_sectors << ',' << s.n_unique_merit_sectors << ',' << s.n_merit_dc_hits << ','
                         << s.n_cache_hits << ',' << s.n_disk_reads << ',' << s.n_sector_cache_hits << ','
                         << s.n_hops << ',' << s.max_sector_jump << ',' << s.n_merit_dyn_hits << ','
                         << s.n_merit_dyn_flushes << ',' << s.n_merit_dyn_pages << ','
                         << s.n_merit_dyn_disk_reads << ',' << s.n_merit_base_pages_avoided << ','
                         << s.n_merit_dcache_probe << ',' << s.n_merit_mcache_hits << ','
                         << s.n_merit_mcache_misses << '\n';
                }
                qout.close();
                diskann::cout << "      per-query stats written to " << dump_query_stats_path << std::endl;
            }
        }
        if (!dump_base_nodes_path.empty())
        {
            std::vector<uint64_t> node_expand;
            std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> unused_edges;
            const std::string profile_for_dump =
                merit_profile_prefix.empty() ? access_profile_prefix : merit_profile_prefix;
            if (!profile_for_dump.empty() &&
                diskann::HotnessProfiler::load(profile_for_dump, node_expand, unused_edges) != 0)
            {
                diskann::cerr << "Warning: failed to load profile for base-node dump from " << profile_for_dump
                              << std::endl;
                node_expand.clear();
            }

            std::ofstream bout(dump_base_nodes_path);
            if (bout.is_open())
            {
                bout << "query_id,node_id,node_expand,in_mem_pool,in_disk_cache_map\n";
                for (size_t qi = 0; qi < query_num; qi++)
                {
                    for (const uint32_t nid : stats[qi].base_frontier_nodes)
                    {
                        const uint64_t expand =
                            (nid < node_expand.size()) ? node_expand[nid] : static_cast<uint64_t>(0);
                        bout << qi << ',' << nid << ',' << expand << ','
                             << (_pFlashIndex->merit_mem_pool_contains(nid) ? 1 : 0) << ','
                             << (_pFlashIndex->merit_dc_map_contains(nid) ? 1 : 0) << '\n';
                    }
                }
                bout.close();
                diskann::cout << "      base-frontier nodes written to " << dump_base_nodes_path << std::endl;
            }
        }
        if (!dump_hop_trace_path.empty())
        {
            std::ofstream hout(dump_hop_trace_path);
            if (hout.is_open())
            {
                hout << "query_id,iteration,record_type,source,node_id,sector,nsectors\n";
                for (size_t qi = 0; qi < query_num; qi++)
                {
                    for (const auto &rec : stats[qi].hop_frontier_trace)
                    {
                        for (uint32_t nid : rec.merit_nodes)
                            hout << qi << ',' << rec.hop << ",frontier,merit," << nid << ",,\n";
                        for (uint32_t nid : rec.base_nodes)
                            hout << qi << ',' << rec.hop << ",frontier,base," << nid << ",,\n";
                        for (const auto &read : rec.physical_reads)
                            hout << qi << ',' << rec.hop << ",physical_read,"
                                 << (read.merit ? "merit" : "base") << ",," << read.sector << ','
                                 << read.nsectors << '\n';
                    }
                }
                hout.close();
                diskann::cout << "      hop-frontier trace written to " << dump_hop_trace_path << std::endl;
            }
        }
        delete[] stats;
    }

    diskann::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
            continue;

        std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    }

    if (enable_access_profile)
    {
        if (access_profile_prefix.empty())
        {
            diskann::cerr << "Error: --access_profile_prefix required when --enable_access_profile is set"
                          << std::endl;
            diskann::aligned_free(query);
            if (warmup != nullptr)
                diskann::aligned_free(warmup);
            return -1;
        }
        _pFlashIndex->print_access_profile_cdf();
        if (_pFlashIndex->save_access_profile(access_profile_prefix) != 0)
        {
            diskann::cerr << "Failed to save access profile." << std::endl;
            diskann::aligned_free(query);
            if (warmup != nullptr)
                diskann::aligned_free(warmup);
            return -1;
        }
    }

    diskann::aligned_free(query);
    if (warmup != nullptr)
        diskann::aligned_free(warmup);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path_prefix, query_file, gt_file, filter_label,
        label_type, query_filters_file;
    uint32_t num_threads, K, W, num_nodes_to_cache, search_io_limit;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    bool enable_access_profile = false;
    std::string access_profile_prefix;
    std::string relayout_order_prefix;
    bool enable_query_sector_cache = false;
    double merit_memory_gb = 0.0;
    std::string merit_profile_prefix;
    double merit_host_memory_gb = 0.0;
    double merit_memory_reserve_gb = 8.0;
    double merit_disk_cache_ratio = 0.0;
    bool merit_disk_cache_exclude_memory = true;
    bool merit_memory_runtime_admit = false;
    double merit_evict_memory_gb = 0.0;
    double merit_evict_disk_ratio = 0.0;
    bool merit_unified_disk_cache = false;
    uint32_t merit_disk_cache_k_hops = 1;
    std::string merit_disk_cache_layout = "directed_beam";
    std::string merit_disk_cache_reuse_prefix;
    std::string dump_query_stats_path;
    std::string dump_base_nodes_path;
    std::string dump_hop_trace_path;
    float fail_if_recall_below = 0.0f;

    po::options_description desc{
        program_options_utils::make_program_description("search_disk_index", "Searches on-disk DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path_prefix)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                                       program_options_utils::BEAMWIDTH);
        optional_configs.add_options()("num_nodes_to_cache", po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                                       program_options_utils::NUMBER_OF_NODES_TO_CACHE);
        optional_configs.add_options()(
            "search_io_limit",
            po::value<uint32_t>(&search_io_limit)->default_value(std::numeric_limits<uint32_t>::max()),
            "Max #IOs for search.  Default value: uint32::max()");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("use_reorder_data", po::bool_switch()->default_value(false),
                                       "Include full precision data in the index. Use only in "
                                       "conjuction with compressed data on SSD.  Default value: false");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        optional_configs.add_options()("enable_access_profile", po::bool_switch()->default_value(false),
                                       "Enable node/edge access profiling during search.");
        optional_configs.add_options()("access_profile_prefix",
                                       po::value<std::string>(&access_profile_prefix)->default_value(std::string("")),
                                       "Output prefix for access profile files (Run2).");
        optional_configs.add_options()("relayout_order_prefix",
                                       po::value<std::string>(&relayout_order_prefix)->default_value(std::string("")),
                                       "Prefix for relayout order files; remaps result IDs to original point IDs for "
                                       "recall on permuted indexes.");
        optional_configs.add_options()("enable_query_sector_cache", po::bool_switch()->default_value(false),
                                       "Cache disk sectors within each query; repeat reads use memory not O_DIRECT.");
        optional_configs.add_options()(
            "merit_memory_gb", po::value<double>(&merit_memory_gb)->default_value(0.0),
            "MERIT Memory cache budget in GB (0=disabled). Selects Top-N nodes by Run2 expand counts.");
        optional_configs.add_options()(
            "merit_profile_prefix", po::value<std::string>(&merit_profile_prefix)->default_value(std::string("")),
            "Run2 access profile prefix (required when merit_memory_gb > 0).");
        optional_configs.add_options()(
            "merit_host_memory_gb", po::value<double>(&merit_host_memory_gb)->default_value(0.0),
            "Host DRAM size in GB for budget check (0=auto-detect from /proc/meminfo).");
        optional_configs.add_options()(
            "merit_memory_reserve_gb", po::value<double>(&merit_memory_reserve_gb)->default_value(8.0),
            "GB reserved for OS/other (not available to MERIT cache). Default: 8.");
        optional_configs.add_options()(
            "merit_disk_cache_ratio", po::value<double>(&merit_disk_cache_ratio)->default_value(0.0),
            "MERIT Disk Cache size as a fraction of base _disk.index bytes (e.g. 0.1 => 10%). "
            "0=disabled. For edge_replica, values >1.0 are allowed (space-for-time replicas).");
        optional_configs.add_options()(
            "merit_disk_cache_exclude_memory", po::value<bool>(&merit_disk_cache_exclude_memory)->default_value(true),
            "If true, disk-cache uses expand ranks after memory tier (no duplicate nodes).");
        optional_configs.add_options()(
            "merit_memory_runtime_admit", po::value<bool>(&merit_memory_runtime_admit)->default_value(false),
            "During search, promote base/disk nodes into MERIT pool (evict coldest). Risky with num_threads>1.");
        optional_configs.add_options()(
            "merit_evict_memory_gb", po::value<double>(&merit_evict_memory_gb)->default_value(0.0),
            "After warmup, batch-reload MERIT memory to this GB (evicts colder nodes). 0=skip.");
        optional_configs.add_options()(
            "merit_evict_disk_ratio", po::value<double>(&merit_evict_disk_ratio)->default_value(0.0),
            "After warmup, batch-rebuild disk-cache at this base ratio (evicts from map). 0=skip.");
        optional_configs.add_options()(
            "merit_unified_disk_cache", po::value<bool>(&merit_unified_disk_cache)->default_value(false),
            "Append MERIT disk tier to a copy of _disk.index; one fd + merged io_uring batch per hop.");
        optional_configs.add_options()(
            "merit_disk_cache_k_hops", po::value<uint32_t>(&merit_disk_cache_k_hops)->default_value(1),
            "MERIT disk cache node order: 0 = flat Top-N by node_expand; >0 = k-hop page packing (node/edge) or beam width (frontier/directed_beam).");
        optional_configs.add_options()(
            "merit_disk_cache_layout", po::value<std::string>(&merit_disk_cache_layout)->default_value("directed_beam"),
            "MERIT disk cache packing: flat(A) | node(B) | edge(C) | frontier(D) | directed_beam(E) | directed_beam_pct80(E_pct80) | directed_child_only_pct100(scheme_i) | directed_child_replica_pct100(scheme_i_unlimited) | directed_seed_replica_pct100(seed_replica) | parent(P) | edge_dir | edge_star | dir_edge_star | directed_beam_hybrid | directed_star | d | e | e_pct80.");
        optional_configs.add_options()(
            "merit_disk_cache_reuse_prefix", po::value<std::string>(&merit_disk_cache_reuse_prefix)->default_value(""),
            "Load existing disk cache from prefix_merit_dc.{data,nodes} instead of repacking.");
        optional_configs.add_options()(
            "dump_query_stats", po::value<std::string>(&dump_query_stats_path)->default_value(""),
            "Write per-query QueryStats CSV after search (for tail-latency analysis).");
        optional_configs.add_options()(
            "dump_base_nodes_path", po::value<std::string>(&dump_base_nodes_path)->default_value(""),
            "Write base-frontier node ids per query (query_id,node_id,node_expand,in_mem_pool,in_disk_cache_map).");
        optional_configs.add_options()(
            "dump_hop_trace_path", po::value<std::string>(&dump_hop_trace_path)->default_value(""),
            "Write per-hop merit/base frontier nodes (query_id,hop,source,node_id).");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (vm["use_reorder_data"].as<bool>())
            use_reorder_data = true;
        if (vm["enable_access_profile"].as<bool>())
            enable_access_profile = true;
        if (vm["enable_query_sector_cache"].as<bool>())
            enable_query_sector_cache = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) && (metric == diskann::Metric::INNER_PRODUCT))
    {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    if (use_reorder_data && data_type != std::string("float"))
    {
        std::cout << "Error: Reorder data for reordering currently only "
                     "supported for float data type."
                  << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "")
    {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    std::vector<std::string> query_filters;
    if (filter_label != "")
    {
        query_filters.push_back(filter_label);
    }
    else if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

    try
    {
        if (!query_filters.empty() && label_type == "ushort")
        {
            if (data_type == std::string("float"))
                return search_disk_index<float, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    enable_access_profile, access_profile_prefix, relayout_order_prefix, enable_query_sector_cache,
                    merit_memory_gb, merit_profile_prefix, merit_host_memory_gb, merit_memory_reserve_gb,
                    merit_disk_cache_ratio, merit_disk_cache_exclude_memory, merit_memory_runtime_admit,
                    merit_evict_memory_gb, merit_evict_disk_ratio, merit_unified_disk_cache, merit_disk_cache_k_hops,
                    merit_disk_cache_layout, merit_disk_cache_reuse_prefix, dump_query_stats_path, dump_base_nodes_path,
                    dump_hop_trace_path);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    enable_access_profile, access_profile_prefix, relayout_order_prefix, enable_query_sector_cache,
                    merit_memory_gb, merit_profile_prefix, merit_host_memory_gb, merit_memory_reserve_gb,
                    merit_disk_cache_ratio, merit_disk_cache_exclude_memory, merit_memory_runtime_admit,
                    merit_evict_memory_gb, merit_evict_disk_ratio, merit_unified_disk_cache, merit_disk_cache_k_hops,
                    merit_disk_cache_layout, merit_disk_cache_reuse_prefix, dump_query_stats_path, dump_base_nodes_path,
                    dump_hop_trace_path);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    enable_access_profile, access_profile_prefix, relayout_order_prefix, enable_query_sector_cache,
                    merit_memory_gb, merit_profile_prefix, merit_host_memory_gb, merit_memory_reserve_gb,
                    merit_disk_cache_ratio, merit_disk_cache_exclude_memory, merit_memory_runtime_admit,
                    merit_evict_memory_gb, merit_evict_disk_ratio, merit_unified_disk_cache, merit_disk_cache_k_hops,
                    merit_disk_cache_layout, merit_disk_cache_reuse_prefix, dump_query_stats_path, dump_base_nodes_path,
                    dump_hop_trace_path);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("float"))
                return search_disk_index<float>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                fail_if_recall_below, query_filters, use_reorder_data,
                                                enable_access_profile, access_profile_prefix, relayout_order_prefix,
                                                enable_query_sector_cache, merit_memory_gb, merit_profile_prefix,
                                                merit_host_memory_gb, merit_memory_reserve_gb, merit_disk_cache_ratio,
                                                merit_disk_cache_exclude_memory, merit_memory_runtime_admit,
                                                merit_evict_memory_gb, merit_evict_disk_ratio, merit_unified_disk_cache, merit_disk_cache_k_hops,
                    merit_disk_cache_layout, merit_disk_cache_reuse_prefix, dump_query_stats_path, dump_base_nodes_path,
                    dump_hop_trace_path);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                 num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                 fail_if_recall_below, query_filters, use_reorder_data,
                                                 enable_access_profile, access_profile_prefix, relayout_order_prefix,
                                                 enable_query_sector_cache, merit_memory_gb, merit_profile_prefix,
                                                 merit_host_memory_gb, merit_memory_reserve_gb, merit_disk_cache_ratio,
                                                 merit_disk_cache_exclude_memory, merit_memory_runtime_admit,
                                                 merit_evict_memory_gb, merit_evict_disk_ratio, merit_unified_disk_cache,
                                                 merit_disk_cache_k_hops, merit_disk_cache_layout,
                                                 merit_disk_cache_reuse_prefix, dump_query_stats_path, dump_base_nodes_path,
                                                 dump_hop_trace_path);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                  num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                  fail_if_recall_below, query_filters, use_reorder_data,
                                                  enable_access_profile, access_profile_prefix, relayout_order_prefix,
                                                  enable_query_sector_cache, merit_memory_gb, merit_profile_prefix,
                                                  merit_host_memory_gb, merit_memory_reserve_gb, merit_disk_cache_ratio,
                                                  merit_disk_cache_exclude_memory, merit_memory_runtime_admit,
                                                  merit_evict_memory_gb, merit_evict_disk_ratio, merit_unified_disk_cache,
                                                  merit_disk_cache_k_hops, merit_disk_cache_layout,
                                                 merit_disk_cache_reuse_prefix, dump_query_stats_path, dump_base_nodes_path,
                                                 dump_hop_trace_path);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}