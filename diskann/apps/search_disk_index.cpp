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
#include <omp.h>
#include <pthread.h>
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

#ifndef _WINDOWS
// NEIGHBORCACHE_QUERY_CPUS="0-9" pins OpenMP thread i to the i-th listed core.
static void neighborcache_pin_this_query_thread()
{
    thread_local bool pinned = false;
    if (pinned)
        return;
    pinned = true;
    const char *spec = std::getenv("NEIGHBORCACHE_QUERY_CPUS");
    if (spec == nullptr || *spec == '\0')
        return;
    std::vector<int> cpus;
    const char *p = spec;
    while (*p != '\0')
    {
        char *end = nullptr;
        const long lo = std::strtol(p, &end, 10);
        long hi = lo;
        if (end != nullptr && *end == '-')
            hi = std::strtol(end + 1, &end, 10);
        if (end == p)
            break;
        for (long c = lo; c <= hi; ++c)
            if (c >= 0 && c < CPU_SETSIZE)
                cpus.push_back(static_cast<int>(c));
        p = end;
        if (*p == ',')
            ++p;
        else if (*p != '\0')
            break;
    }
    if (cpus.empty())
        return;
    const int tid = omp_get_thread_num();
    const int cpu = cpus[static_cast<size_t>(tid) % cpus.size()];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0)
        diskann::cout << "NEIGHBORCACHE pin query thread=" << tid << " cpu=" << cpu << std::endl;
}
#endif

static void log_rss(const char *stage)
{
#ifndef _WINDOWS
    std::ifstream status("/proc/self/status");
    std::string line, rss, hwm;
    while (std::getline(status, line))
    {
        if (line.rfind("VmRSS:", 0) == 0)
            rss = line.substr(6);
        else if (line.rfind("VmHWM:", 0) == 0)
            hwm = line.substr(6);
    }
    diskann::cout << "MEM_STAGE " << stage << " rss=" << rss << " hwm=" << hwm << std::endl;
#endif
}

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
                      const double neighborcache_memory_gb = 0.0,
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
    log_rss("after_load");

    if (enable_access_profile)
    {
        _pFlashIndex->enable_access_profile(true);
        diskann::cout << "Access profiling enabled. Output prefix: " << access_profile_prefix << std::endl;
    }

    if (enable_query_sector_cache)
    {
        _pFlashIndex->enable_query_sector_cache(true);
        diskann::cout << "Query sector statistics enabled." << std::endl;
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
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache,node_list);
    _pFlashIndex->load_cache_list(node_list);
    const char *cap=std::getenv("NEIGHBORCACHE_NCACHE_CAP_NODES");
    uint64_t nodes=cap ? std::stoull(cap) : 0;
    if(!nodes && neighborcache_memory_gb>0) {
        std::string report;
        if(_pFlashIndex->plan_neighborcache_memory_cache(neighborcache_memory_gb,0,8,num_threads,nodes,report)!=0)
            throw std::runtime_error(report);
    }
    if(nodes) _pFlashIndex->enable_path_cache(nodes,result_output_prefix);
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

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "BasePages";
    diskann::cout << std::setw(16) << "Dup IOs";
    if (enable_query_sector_cache)
    {
        diskann::cout << std::setw(16) << "Disk Reads" << std::setw(16) << "SectCacheHit" << std::setw(16)
                      << "IoUs/Read" << std::setw(16) << "|dSec|" << std::setw(16) << "Jump<=8s%";
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

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new diskann::QueryStats[query_num];

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        // Optional cold-start throughput trace. Each worker writes only its own
        // query slot; file IO and sorting are deferred until after the search.
        const char *timeline_path = std::getenv("NEIGHBORCACHE_QUERY_TIMELINE");
        const bool trace_timeline = timeline_path != nullptr && timeline_path[0] != '\0';
        std::vector<int64_t> completed_ns(trace_timeline ? query_num : 0);
        std::vector<int64_t> d_pages_snapshot(trace_timeline ? query_num : 0, -1);
        const uint64_t initial_d_pages = trace_timeline ? _pFlashIndex->path_pages() : 0;
        const auto timeline_start = std::chrono::steady_clock::now();
        const char *query_schedule = std::getenv("NEIGHBORCACHE_QUERY_SCHEDULE");
        const bool fixed_query_workers = query_schedule != nullptr && std::strcmp(query_schedule, "static") == 0;
        omp_set_schedule(fixed_query_workers ? omp_sched_static : omp_sched_dynamic, 1);
        std::vector<int> query_workers(dump_query_stats_path.empty() ? 0 : query_num);
        diskann::cout << "Query scheduling: " << (fixed_query_workers ? "static,1" : "dynamic,1") << std::endl;
        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(runtime)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
#ifndef _WINDOWS
            neighborcache_pin_this_query_thread();
#endif
            if (!query_workers.empty())
                query_workers[i] = omp_get_thread_num();
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
            if (trace_timeline)
            {
                completed_ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - timeline_start).count();
                // Sparse, approximate cache-growth checkpoints, outside the
                // completion timestamp. This does not synchronize query batches.
                if ((i + 1) % 1000 == 0)
                    d_pages_snapshot[i] = _pFlashIndex->path_pages();
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());
        if (trace_timeline)
        {
            const std::string path = std::string(timeline_path) + ".L" + std::to_string(L) + ".csv";
            std::ofstream trace(path);
            if (!trace)
                throw std::runtime_error("Cannot write query timeline: " + path);
            trace << "query_id,completed_ns,d_pages_snapshot\n";
            for (uint64_t i = 0; i < query_num; ++i)
                trace << i << ',' << completed_ns[i] << ',' << d_pages_snapshot[i] << '\n';
            trace.close();
            if (!trace)
                throw std::runtime_error("Failed writing query timeline: " + path);
            diskann::cout << "Query timeline: initial_d_pages=" << initial_d_pages
                          << " final_d_pages=" << _pFlashIndex->path_pages()
                          << " queries=" << query_num << " file=" << path << std::endl;
        }

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
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_unique_neighborcache_sectors; });

        auto mean_dup_ios = diskann::get_mean_stats<uint32_t>(stats, query_num, [](const diskann::QueryStats &stats) {
            const unsigned unique_pages = stats.n_unique_sectors + stats.n_unique_neighborcache_sectors;
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
        diskann::cout << std::setw(16) << mean_hops << std::setw(16) << mean_io_us << std::setw(16) << mean_cpuus;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << recall;
        }
        diskann::cout << std::endl;
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
                qout << "query_id,total_us,io_us,cpu_us,n_ios,n_base_pages,n_disk_cache_pages,n_mem_hits,"
                        "n_disk_reads,n_hops,max_sector_jump,n_neighborcache_dyn_hits,n_neighborcache_prefetch_hits,"
                        "n_neighborcache_dcache_seed_hits,n_beam_lookups,n_neighborcache_dyn_disk_reads,n_cmps,"
                        "n_neighborcache_dcache_demand_probes,n_neighborcache_dcache_demand_hits,query_worker\n";
                for (size_t qi = 0; qi < query_num; qi++)
                {
                    const auto &s = stats[qi];
                    qout << qi << ',' << s.total_us << ',' << s.io_us << ',' << s.cpu_us << ',' << s.n_ios << ','
                         << s.n_unique_sectors << ',' << s.n_unique_neighborcache_sectors << ',' << s.n_cache_hits << ','
                         << s.n_disk_reads << ',' << s.n_hops << ',' << s.max_sector_jump << ','
                         << s.n_neighborcache_dyn_hits << ',' << s.n_neighborcache_prefetch_hits << ','
                         << s.n_neighborcache_dcache_seed_hits << ',' << s.n_beam_lookups << ','
                         << s.n_neighborcache_dyn_disk_reads << ',' << s.n_cmps << ','
                         << s.n_neighborcache_dcache_demand_probes << ',' << s.n_neighborcache_dcache_demand_hits
                         << ',' << query_workers[qi] << '\n';
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
                access_profile_prefix;
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
                bout << "query_id,node_id,node_expand,in_ncache\n";
                for (size_t qi = 0; qi < query_num; qi++)
                {
                    for (const uint32_t nid : stats[qi].base_frontier_nodes)
                    {
                        const uint64_t expand =
                            (nid < node_expand.size()) ? node_expand[nid] : static_cast<uint64_t>(0);
                        bout << qi << ',' << nid << ',' << expand << ','
                             << (_pFlashIndex->node_cached(nid) ? 1 : 0) << '\n';
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
                        for (uint32_t nid : rec.neighborcache_nodes)
                            hout << qi << ',' << rec.hop << ",frontier,neighborcache," << nid << ",,\n";
                        for (uint32_t nid : rec.base_nodes)
                            hout << qi << ',' << rec.hop << ",frontier,base," << nid << ",,\n";
                        for (const auto &read : rec.physical_reads)
                            hout << qi << ',' << rec.hop << ",physical_read,"
                                 << (read.neighborcache ? "neighborcache" : "base") << ",," << read.sector << ','
                                 << read.nsectors << '\n';
                    }
                }
                hout.close();
                diskann::cout << "      hop-frontier trace written to " << dump_hop_trace_path << std::endl;
            }
        }
        delete[] stats;
    }

    log_rss("after_search");
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
    double neighborcache_memory_gb = 0.0;
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
        optional_configs.add_options()("neighborcache_memory_gb",po::value<double>(&neighborcache_memory_gb)->default_value(0.0),
            "Enable the shared path cache with this N-cache budget in GiB.");
        optional_configs.add_options()(
            "dump_query_stats", po::value<std::string>(&dump_query_stats_path)->default_value(""),
            "Write per-query QueryStats CSV after search (for tail-latency analysis).");
        optional_configs.add_options()(
            "dump_base_nodes_path", po::value<std::string>(&dump_base_nodes_path)->default_value(""),
            "Write base-frontier node ids per query (query_id,node_id,node_expand,in_ncache).");
        optional_configs.add_options()(
            "dump_hop_trace_path", po::value<std::string>(&dump_hop_trace_path)->default_value(""),
            "Write per-hop neighborcache/base frontier nodes (query_id,hop,source,node_id).");

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
                    neighborcache_memory_gb, dump_query_stats_path, dump_base_nodes_path,
                    dump_hop_trace_path);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    enable_access_profile, access_profile_prefix, relayout_order_prefix, enable_query_sector_cache,
                    neighborcache_memory_gb, dump_query_stats_path, dump_base_nodes_path,
                    dump_hop_trace_path);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    enable_access_profile, access_profile_prefix, relayout_order_prefix, enable_query_sector_cache,
                    neighborcache_memory_gb, dump_query_stats_path, dump_base_nodes_path,
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
                                                enable_query_sector_cache, neighborcache_memory_gb, dump_query_stats_path, dump_base_nodes_path,
                    dump_hop_trace_path);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                 num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                 fail_if_recall_below, query_filters, use_reorder_data,
                                                 enable_access_profile, access_profile_prefix, relayout_order_prefix,
                                                 enable_query_sector_cache, neighborcache_memory_gb, dump_query_stats_path, dump_base_nodes_path,
                                                 dump_hop_trace_path);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                  num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                  fail_if_recall_below, query_filters, use_reorder_data,
                                                  enable_access_profile, access_profile_prefix, relayout_order_prefix,
                                                  enable_query_sector_cache, neighborcache_memory_gb, dump_query_stats_path, dump_base_nodes_path,
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
