// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "common_includes.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <list>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq.h"
#include "utils.h"
#include "windows_customizations.h"
#include "hotness_profiler.h"
#include "neighborcache_memory_pool.h"
#include "neighborcache_metadata_cache.h"
#include "neighborcache_shared_page_cache.h"
#include "neighborcache_lock_metrics.h"
#include "scratch.h"
#include "relayout_utils.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann
{

struct NeighborCacheSpinLock
{
    void lock()
    {
        uint32_t spins = 0;
        while (_flag.test_and_set(std::memory_order_acquire))
            pause(spins);
    }
    void unlock()
    {
        _flag.clear(std::memory_order_release);
    }
    std::atomic_flag _flag = ATOMIC_FLAG_INIT;

    static void pause(uint32_t &spins)
    {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
        if (++spins == 128)
        {
            std::this_thread::yield();
            spins = 0;
        }
    }
};

// Spinning reader-writer lock. Readers share. A waiting writer blocks new
// readers, then takes the lock once the current readers leave. After a short
// spin it yields, so a holder that was descheduled does not pin the core.
struct NeighborCacheSpinRwLock
{
    static constexpr uint32_t kWriter = 1u << 31;
    static constexpr uint32_t kWriterWait = 1u << 30;

    void lock()
    {
        uint32_t spins = 0;
        for (;;)
        {
            _state.fetch_or(kWriterWait, std::memory_order_acq_rel);
            uint32_t expected = kWriterWait;
            if (_state.compare_exchange_weak(expected, kWriterWait | kWriter, std::memory_order_acq_rel,
                                             std::memory_order_acquire))
                return;
            NeighborCacheSpinLock::pause(spins);
        }
    }

    void unlock()
    {
        _state.fetch_and(~(kWriter | kWriterWait), std::memory_order_release);
    }

    void lock_shared()
    {
        uint32_t spins = 0;
        for (;;)
        {
            uint32_t state = _state.load(std::memory_order_acquire);
            if ((state & (kWriter | kWriterWait)) == 0 &&
                _state.compare_exchange_weak(state, state + 1, std::memory_order_acq_rel, std::memory_order_acquire))
                return;
            NeighborCacheSpinLock::pause(spins);
        }
    }

    void unlock_shared()
    {
        _state.fetch_sub(1, std::memory_order_release);
    }

    std::atomic<uint32_t> _state{0};
};


inline constexpr uint8_t NEIGHBORCACHE_DC_SEED_MAGIC = 0x4D;
inline constexpr uint8_t NEIGHBORCACHE_DC_FLAG_TOTAL_PAGES_MASK = 0xFF;

template <typename T, typename LabelT = uint32_t> class PQFlashIndex
{
  public:
    DISKANN_DLLEXPORT PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader,
                                   diskann::Metric metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT ~PQFlashIndex();

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix);
#else
    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *index_prefix);
#endif

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                   const char *index_filepath, const char *pivots_filepath,
                                                   const char *compressed_filepath);
#else
    DISKANN_DLLEXPORT int load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                   const char *pivots_filepath, const char *compressed_filepath);
#endif

    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &node_list);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                   uint64_t l_search, uint64_t beamwidth,
                                                                   uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                   std::vector<uint32_t> &node_list);
#else
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                   uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                   uint32_t num_threads,
                                                                   std::vector<uint32_t> &node_list);
#endif

    DISKANN_DLLEXPORT void cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                            const bool shuffle = false, double max_fraction = 0.1);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_reorder_data = false, QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_filter, const LabelT &filter_label,
                                              const bool use_reorder_data = false, QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const uint32_t io_limit, const bool use_reorder_data = false,
                                              QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                              uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                              const bool use_filter, const LabelT &filter_label,
                                              const uint32_t io_limit, const bool use_reorder_data = false,
                                              QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT LabelT get_converted_label(const std::string &filter_label);

    DISKANN_DLLEXPORT uint32_t range_search(const T *query1, const double range, const uint64_t min_l_search,
                                            const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                            std::vector<float> &distances, const uint64_t min_beam_width,
                                            QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT uint64_t get_data_dim();

    std::shared_ptr<AlignedFileReader> &reader;

    DISKANN_DLLEXPORT diskann::Metric get_metric();

    //
    // node_ids: input list of node_ids to be read
    // coord_buffers: pointers to pre-allocated buffers that coords need to copied to. If null, dont copy.
    // nbr_buffers: pre-allocated buffers to copy neighbors into
    //
    // returns a vector of bool one for each node_id: true if read is success, else false
    //
    DISKANN_DLLEXPORT std::vector<bool> read_nodes(const std::vector<uint32_t> &node_ids,
                                                   std::vector<T *> &coord_buffers,
                                                   std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers);

    DISKANN_DLLEXPORT std::vector<std::uint8_t> get_pq_vector(std::uint64_t vid);
    DISKANN_DLLEXPORT uint64_t get_num_points();
    DISKANN_DLLEXPORT uint64_t get_max_degree();

    DISKANN_DLLEXPORT void enable_access_profile(bool enable);
    DISKANN_DLLEXPORT void reset_access_profile();
    DISKANN_DLLEXPORT void enable_query_sector_cache(bool enable);
    DISKANN_DLLEXPORT void print_neighborcache_shared_cache_stats() const;
    DISKANN_DLLEXPORT void enable_base_frontier_recording(bool enable);
    DISKANN_DLLEXPORT void enable_hop_frontier_recording(bool enable);
    DISKANN_DLLEXPORT int save_access_profile(const std::string &output_prefix) const;
    DISKANN_DLLEXPORT void print_access_profile_cdf() const;

    DISKANN_DLLEXPORT uint64_t bytes_per_cached_node() const;
    DISKANN_DLLEXPORT uint64_t estimate_baseline_resident_bytes(uint32_t num_threads) const;
    DISKANN_DLLEXPORT int plan_neighborcache_memory_cache(double neighborcache_memory_gb, double host_memory_gb,
                                                  double reserve_gb, uint32_t num_threads,
                                                  uint64_t &out_max_nodes, std::string &report) const;
    DISKANN_DLLEXPORT int build_neighborcache_memory_node_list(const std::string &profile_prefix, uint64_t max_nodes,
                                                      std::vector<uint32_t> &node_list,
                                                      uint64_t rank_skip = 0) const;
    DISKANN_DLLEXPORT int build_neighborcache_disk_node_list(const std::string &profile_prefix, uint64_t max_nodes,
                                                     uint64_t memory_tier_exclude_count, uint32_t k_hops,
                                                     const std::string &layout, std::vector<uint32_t> &node_list,
                                                     std::vector<SeedPageGroup> *seed_page_groups = nullptr);

    DISKANN_DLLEXPORT uint64_t neighborcache_memory_cached_count() const;
    DISKANN_DLLEXPORT void print_neighborcache_memory_cache_stats() const;
    DISKANN_DLLEXPORT bool neighborcache_mem_pool_contains(uint32_t node_id) const;
    DISKANN_DLLEXPORT bool neighborcache_dc_map_contains(uint32_t node_id) const;
    DISKANN_DLLEXPORT int verify_neighborcache_disk_cache_against_base(uint64_t max_reports = 20);
    DISKANN_DLLEXPORT void clear_neighborcache_memory_cache();
    DISKANN_DLLEXPORT int load_neighborcache_memory_pool(const std::string &profile_prefix,
                                                  std::vector<uint32_t> &node_list);
    DISKANN_DLLEXPORT void enable_neighborcache_memory_runtime_admit(bool enable);
    DISKANN_DLLEXPORT void enable_neighborcache_dynamic_3cache(bool enable, const std::string &flush_prefix = "");
    DISKANN_DLLEXPORT bool neighborcache_dynamic_3cache_enabled() const;
    DISKANN_DLLEXPORT void print_neighborcache_dynamic_3cache_stats() const;
    // Block until co-read pages already handed to the commit thread are on disk.
    // Search calls this after the QPS timer stops.
    DISKANN_DLLEXPORT void flush_neighborcache_coread_writes();
    DISKANN_DLLEXPORT uint64_t neighborcache_dynamic_flush_count() const;
    DISKANN_DLLEXPORT uint64_t neighborcache_dynamic_page_count() const;
    DISKANN_DLLEXPORT uint64_t neighborcache_dynamic_page_capacity() const;
    DISKANN_DLLEXPORT bool neighborcache_dynamic_disk_full() const;
    DISKANN_DLLEXPORT void neighborcache_dynamic_begin_measurement();
    DISKANN_DLLEXPORT uint64_t neighborcache_dynamic_clean_seeds() const;
    DISKANN_DLLEXPORT uint64_t neighborcache_dynamic_partial_seeds() const;
    DISKANN_DLLEXPORT int reload_neighborcache_memory_cache(const std::string &profile_prefix, double neighborcache_memory_gb,
                                                    double host_memory_gb, double reserve_gb, uint32_t num_threads,
                                                    uint64_t &evicted_nodes, std::string &report);

    DISKANN_DLLEXPORT int plan_neighborcache_disk_cache(double base_ratio, uint64_t &out_max_nodes, std::string &report,
                                                bool allow_replica_slots = false) const;
    DISKANN_DLLEXPORT int build_and_load_neighborcache_disk_cache(const std::string &profile_prefix, uint64_t max_nodes,
                                                          const std::string &output_prefix, uint64_t rank_skip = 0,
                                                          bool unified_single_file = false, uint32_t k_hops = 2,
                                                          const std::string &layout = "node");
    DISKANN_DLLEXPORT int load_neighborcache_disk_cache_from_prefix(const std::string &output_prefix);
    DISKANN_DLLEXPORT int reload_neighborcache_disk_cache(const std::string &profile_prefix, double base_ratio,
                                                  const std::string &output_prefix, uint64_t rank_skip,
                                                  uint64_t &evicted_nodes, std::string &report,
                                                  bool unified_single_file = false, uint32_t k_hops = 2,
                                                  const std::string &layout = "node");

  protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(uint64_t nthreads, uint64_t visited_reserve = 4096);

    DISKANN_DLLEXPORT void set_universal_label(const LabelT &label);

  private:
    DISKANN_DLLEXPORT inline bool point_has_label(uint32_t point_id, LabelT label_id);
    std::unordered_map<std::string, LabelT> load_label_map(std::basic_istream<char> &infile);
    DISKANN_DLLEXPORT void parse_label_file(std::basic_istream<char> &infile, size_t &num_pts_labels);
    DISKANN_DLLEXPORT void get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                   uint32_t &num_total_labels);
    DISKANN_DLLEXPORT void generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                  const uint32_t nthreads);
    void reset_stream_for_reading(std::basic_istream<char> &infile);

    // sector # on disk where node_id is present with in the graph part
    DISKANN_DLLEXPORT uint64_t get_node_sector(uint64_t node_id);

    // ptr to start of the node
    DISKANN_DLLEXPORT char *offset_to_node(char *sector_buf, uint64_t node_id);

    // returns region of `node_buf` containing [NNBRS][NBR_ID(uint32_t)]
    DISKANN_DLLEXPORT uint32_t *offset_to_node_nhood(char *node_buf);

    // returns region of `node_buf` containing [COORD(T)]
    DISKANN_DLLEXPORT T *offset_to_node_coords(char *node_buf);

    struct NeighborCacheDiskLoc
    {
        uint32_t sector = 0;
        uint16_t slot = 0;
        uint16_t nsectors = 1;
    };

    struct NeighborCacheReadPending
    {
        uint32_t id = 0;
        char *sec_buf = nullptr;
        NeighborCacheDiskLoc loc;
        // Start sector of the IO buffer in sec_buf (multi-page seed group reads).
        uint32_t io_base_sector = 0;
        // Number of contiguous 4KB pages in the single physical IO.
        uint16_t io_nsectors = 1;
        bool dynamic_page_pinned = false;
        bool from_query_cache = false;
        // Node bytes served in place from the shared page cache (valid for the query's read section).
        const char *node_ptr = nullptr;
    };

    void prepare_neighborcache_disk_cache_io(const std::vector<uint32_t> &neighborcache_ids, SSDQueryScratch<T> *query_scratch,
                                  char *sector_scratch, uint64_t &sector_scratch_idx, size_t num_sectors_per_node,
                                  std::vector<NeighborCacheReadPending> &pending, std::vector<AlignedRead> &neighborcache_io,
                                  std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout_groups,
                                  QueryStats *stats, uint32_t &num_ios);

    void complete_neighborcache_disk_cache_io(SSDQueryScratch<T> *query_scratch, std::vector<NeighborCacheReadPending> &pending,
                                   const std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout_groups);

    void finalize_neighborcache_pending_nodes(const std::vector<NeighborCacheReadPending> &pending,
                                      const std::vector<uint32_t> &neighborcache_order, SSDQueryScratch<T> *query_scratch,
                                      char *sector_scratch, uint64_t &sector_scratch_idx,
                                      std::vector<std::pair<uint32_t, char *>> &frontier_nhoods,
                                      QueryStats *stats = nullptr);

    // index info for multi-node sectors
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    //
    // index info for multi-sector nodes
    // nhood of node `i` is in sector: [i * DIV_ROUND_UP(_max_node_len, SECTOR_LEN)]
    // offset in sector: [0]
    //
    // Common info
    // coords start at ofsset
    // #nbrs of node `i`: *(unsigned*) (offset + disk_bytes_per_point)
    // nbrs of node `i` : (unsigned*) (offset + disk_bytes_per_point + 1)

    uint64_t _max_node_len = 0;
    uint64_t _nnodes_per_sector = 0; // 0 for multi-sector nodes, >0 for multi-node sectors
    uint64_t _max_degree = 0;

    // Data used for searching with re-order vectors
    uint64_t _ndims_reorder_vecs = 0;
    uint64_t _reorder_data_start_sector = 0;
    uint64_t _nvecs_per_sector = 0;

    diskann::Metric metric = diskann::Metric::L2;

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    float _max_base_norm = 0.0f;

    // data info
    uint64_t _num_points = 0;
    uint64_t _num_frozen_points = 0;
    uint64_t _frozen_location = 0;
    uint64_t _data_dim = 0;
    uint64_t _aligned_dim = 0;
    uint64_t _disk_bytes_per_point = 0; // Number of bytes

    std::string _disk_index_file;
    uint64_t _base_disk_index_bytes = 0;
    std::vector<std::pair<uint32_t, uint32_t>> _node_visit_counter;

    // PQ data
    // _n_chunks = # of chunks ndims is split into
    // data: char * _n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * _n_chunks]
    uint8_t *data = nullptr;
    uint64_t _n_chunks;
    FixedChunkPQTable _pq_table;

    // distance comparator
    std::shared_ptr<Distance<T>> _dist_cmp;
    std::shared_ptr<Distance<float>> _dist_cmp_float;

    // for very large datasets: we use PQ even for the disk resident index
    bool _use_disk_index_pq = false;
    uint64_t _disk_pq_n_chunks = 0;
    FixedChunkPQTable _disk_pq_table;

    // medoid/start info

    // graph has one entry point by default,
    // we can optionally have multiple starting points
    uint32_t *_medoids = nullptr;
    // defaults to 1
    size_t _num_medoids;
    // by default, it is empty. If there are multiple
    // centroids, we pick the medoid corresponding to the
    // closest centroid as the starting point of search
    float *_centroid_data = nullptr;

    // nhood_cache; the uint32_t in nhood_Cache are offsets into nhood_cache_buf
    unsigned *_nhood_cache_buf = nullptr;
    tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>> _nhood_cache;

    // coord_cache; The T* in coord_cache are offsets into coord_cache_buf
    T *_coord_cache_buf = nullptr;
    tsl::robin_map<uint32_t, T *> _coord_cache;

    std::unique_ptr<NeighborCacheMemoryPool<T>> _neighborcache_mem_pool;
    // Runtime try_admit during search (eviction); off until pool lookup stability verified.
    bool _neighborcache_mem_runtime_admit = false;

    // NEIGHBORCACHE disk-cache: packed side file; location = (sector_id, slot_in_sector)
    std::shared_ptr<AlignedFileReader> _neighborcache_disk_reader;
    tsl::robin_map<uint32_t, std::vector<NeighborCacheDiskLoc>> _neighborcache_dc_map;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _neighborcache_dc_seed_page_nbrs;
    // seed -> member -> loc on that seed's page(s); used for parent/seed-first IO routing
    tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, NeighborCacheDiskLoc>> _neighborcache_dc_seed_member_loc;
    tsl::robin_map<uint32_t, NeighborCacheDiskLoc> _neighborcache_dc_seed_canonical_loc;
    std::string _neighborcache_dc_path;
    uint64_t _neighborcache_dc_num_nodes = 0;
    bool _neighborcache_unified_disk = false;
    uint64_t _neighborcache_region_byte_offset = 0;
    bool _neighborcache_seed_only_layout = false;
    bool _neighborcache_seed_only_expand = false;
    bool _neighborcache_seed_first_lookup = false;
    bool _neighborcache_child_only_layout = false;

    static constexpr uint32_t NEIGHBORCACHE_DYN_SECTOR_BASE = 0x80000000u;
    static constexpr uint32_t NEIGHBORCACHE_DYN_INVALID_PAGE = std::numeric_limits<uint32_t>::max();

    static constexpr size_t NEIGHBORCACHE_DYN_LOC_MAX_SHARDS = 128;
    struct NeighborCacheDynLocShard
    {
        mutable std::shared_mutex mu;
        tsl::robin_map<uint32_t, std::vector<NeighborCacheDiskLoc>> locations;
    };

    enum class NeighborCacheNodeState : uint8_t
    {
        NonSeed = 0,
        ReadyToInsertion = 1,
        Seed = 2,
        ReadyToDeletion = 3
    };

    struct NeighborCachePendingFlush
    {
        uint32_t seed_id = 0;
        NeighborCacheMetadataCache::Snapshot snap;
        std::vector<uint32_t> member_ids;
        std::vector<std::vector<T>> member_coords;
        std::vector<std::vector<uint32_t>> member_nbrs;
    };

    struct NeighborCacheReadyPair
    {
        uint32_t insertion_id = std::numeric_limits<uint32_t>::max();
        uint32_t deletion_id = std::numeric_limits<uint32_t>::max();
        uint32_t insertion_slot = NeighborCacheMetadataCache::kInvalid;
        uint32_t deletion_slot = NeighborCacheMetadataCache::kInvalid;
        uint32_t created_epoch = 0;
        uint16_t retained_members = 0;
        NeighborCachePendingFlush snap;
        bool committing = false;
        bool queued = false;
    };

    struct NeighborCachePayloadStashEntry
    {
        std::vector<T> coords;
        std::vector<uint32_t> nbrs;
        std::list<uint32_t>::iterator lru_it;
    };

    bool _neighborcache_dyn_enabled = false;
    // After begin_measurement with a full D-cache, keep D lookups but stop
    // M-cache/pair/heap maintenance that almost never flushes.
    bool _neighborcache_freeze_maintenance = false;
    // Read-only membership filter built when maintenance is frozen (100M-bit).
    std::vector<uint64_t> _neighborcache_dyn_member_bits;
    bool _neighborcache_dcache_net_gate_enabled = false;
    uint32_t _neighborcache_dcache_probe_period = 100;
    std::atomic<bool> _neighborcache_dcache_gate_active{true};
    std::atomic<uint64_t> _neighborcache_dcache_query_issued{0};
    std::atomic<uint64_t> _neighborcache_dcache_query_completed{0};
    std::atomic<uint64_t> _neighborcache_dcache_enabled_queries{0};
    std::atomic<uint64_t> _neighborcache_dcache_avoided_pages{0};
    std::atomic<uint64_t> _neighborcache_dcache_physical_reads{0};
    std::atomic<uint64_t> _neighborcache_dcache_served_nodes{0};
    std::atomic<uint64_t> _neighborcache_dcache_page_writes{0};
    std::atomic<uint64_t> _neighborcache_dcache_gate_transitions{0};
    double _neighborcache_dcache_net_write_weight = 0.0;
    double _neighborcache_dcache_base_read_us = 40.0;
    double _neighborcache_dcache_read_us = 50.0;
    double _neighborcache_dcache_write_us = 56.0;
    double _neighborcache_dcache_cpu_us = 20.0;
    double _neighborcache_dcache_gate_off_us = 0.0;
    double _neighborcache_dcache_gate_on_us = 10.0;
    mutable std::mutex _neighborcache_dcache_gate_mu;
    uint64_t _neighborcache_dcache_last_enabled_queries = 0;
    uint64_t _neighborcache_dcache_last_avoided_pages = 0;
    uint64_t _neighborcache_dcache_last_physical_reads = 0;
    uint64_t _neighborcache_dcache_last_served_nodes = 0;
    uint64_t _neighborcache_dcache_last_page_writes = 0;
    uint32_t _neighborcache_dcache_low_windows = 0;
    uint32_t _neighborcache_dcache_high_windows = 0;
    uint32_t _neighborcache_dcache_inactive_stages = 0;
    // Build a candidate page from N-cache-resident members and defer copying
    // payloads until write time. An optional bounded write-back buffer can
    // retain only candidate-page members after N-cache eviction.
    bool _neighborcache_deferred_ncache_page_write = false;
    // Pin a new seed's N-cache neighbors until its page is written, then demote them.
    bool _neighborcache_seed_pin_neighbors = false;
    // Scans take a shared lock so query threads can read together. Pin, unpin,
    // and release take the exclusive lock.
    mutable NeighborCacheSpinRwLock _neighborcache_seed_pin_mu;
    tsl::robin_map<uint32_t, tsl::robin_set<uint32_t>> _neighborcache_seed_pin_wait;
    tsl::robin_map<uint32_t, tsl::robin_set<uint32_t>> _neighborcache_seed_pin_held;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _neighborcache_seed_pin_nbrs;
    NeighborCacheSpinLock _neighborcache_seed_pin_seen_mu;
    tsl::robin_map<uint32_t, uint32_t> _neighborcache_seed_pin_seen_held;
    tsl::robin_map<uint32_t, uint32_t> _neighborcache_seed_pin_slot;
    uint32_t _neighborcache_seed_pin_age_queries = 192;
    uint32_t _neighborcache_seed_pin_writes_per_query = 2;
    std::atomic<uint32_t> _neighborcache_seed_pin_budget_epoch{0};
    std::atomic<uint32_t> _neighborcache_seed_pin_budget_used{0};
    std::atomic<uint64_t> _neighborcache_seed_pin_forced{0};
    std::atomic<uint64_t> _neighborcache_seed_pin_cold_writes{0};
    std::atomic<uint64_t> _neighborcache_seed_pin_early_writes{0};
    // Separate LRU of forming seed pages. Not the shared disk-page prefetch.
    // Slot 0 is the seed. Up to 9 neighbors follow. On by default, 80MB.
    static constexpr uint32_t kNeighborCacheSeedPageSlots = 10;
    static constexpr uint32_t kNeighborCacheSeedPageShards = 64;
    static constexpr uint64_t kNeighborCacheSeedPageDefaultPages = 80ull * 256ull;
    struct NeighborCacheSeedPage
    {
        uint32_t seed_id = 0;
        uint8_t count = 0;
        uint32_t ids[kNeighborCacheSeedPageSlots]{};
        uint64_t touch[kNeighborCacheSeedPageSlots]{};
        std::vector<T> coords[kNeighborCacheSeedPageSlots];
        std::vector<uint32_t> nbrs[kNeighborCacheSeedPageSlots];
        std::list<uint32_t>::iterator lru_it{};
    };
    struct NeighborCacheSeedPageShard
    {
        mutable NeighborCacheSpinLock mu;
        std::list<uint32_t> lru;
        tsl::robin_map<uint32_t, NeighborCacheSeedPage> pages;
        uint32_t cap = 1;
    };
    struct NeighborCacheSeedPageOwnerShard
    {
        mutable NeighborCacheSpinLock mu;
        // Graph neighbor (and the seed id) -> seeds whose live page lists it.
        tsl::robin_map<uint32_t, std::vector<uint32_t>> by_node;
    };
    bool _neighborcache_seed_page_lru = false;
    // Shared prefetch holds assembled seed pages. The seed-page LRU stays off.
    // A dirty page is written to the D-cache only when the prefetch evicts it.
    bool _neighborcache_seed_prefetch = false;
    std::unique_ptr<std::atomic<uint8_t>[]> _neighborcache_prefetch_dirty;
    std::mutex _neighborcache_seed_lock_mu;
    tsl::robin_map<uint32_t, std::unique_ptr<NeighborCacheSpinLock>> _neighborcache_seed_locks;
    uint32_t _neighborcache_seed_page_cap = 0;
    std::vector<std::unique_ptr<NeighborCacheSeedPageShard>> _neighborcache_seed_page_shards;
    std::vector<std::unique_ptr<NeighborCacheSeedPageOwnerShard>> _neighborcache_seed_page_owners;
    std::atomic<uint64_t> _neighborcache_seed_page_stamp{1};
    std::atomic<uint64_t> _neighborcache_seed_page_opens{0};
    std::atomic<uint64_t> _neighborcache_seed_page_open_miss{0};
    std::atomic<uint64_t> _neighborcache_seed_page_open_nbrs{0};
    std::atomic<uint64_t> _neighborcache_seed_page_appends{0};
    std::atomic<uint64_t> _neighborcache_seed_page_replaces{0};
    std::atomic<uint64_t> _neighborcache_seed_page_touches{0};
    std::atomic<uint64_t> _neighborcache_seed_page_hits{0};
    std::atomic<uint64_t> _neighborcache_seed_page_evicts{0};
    std::atomic<uint64_t> _neighborcache_seed_page_enqueued{0};
    std::atomic<uint64_t> _neighborcache_seed_page_dropped{0};
    std::atomic<uint64_t> _neighborcache_seed_page_written{0};
    std::atomic<uint64_t> _neighborcache_seed_page_skipped{0};
    std::atomic<uint64_t> _neighborcache_seed_page_failed{0};
    std::atomic<uint64_t> _neighborcache_seed_page_nbr_sum{0};
    std::atomic<uint32_t> _neighborcache_seed_page_inflight{0};
    bool _neighborcache_dcache_family = false;
    uint64_t _neighborcache_dcache_family_cap = 0;
    std::mutex _neighborcache_family_mu;
    std::list<uint32_t> _neighborcache_dcache_lru;
    tsl::robin_map<uint32_t, std::list<uint32_t>::iterator> _neighborcache_dcache_lru_pos;
    // A node may sit on several seed pages, so it can belong to more than one family.
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _neighborcache_family_of;
    tsl::robin_map<uint32_t, uint32_t> _neighborcache_family_page;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _neighborcache_family_members;
    std::atomic<uint64_t> _neighborcache_family_formed{0};
    std::atomic<uint64_t> _neighborcache_family_expanded{0};
    std::atomic<uint64_t> _neighborcache_family_dissolved{0};
    std::atomic<uint64_t> _neighborcache_family_dropped{0};
    std::atomic<uint64_t> _neighborcache_family_idle{0};
    std::atomic<uint64_t> _neighborcache_family_coalesced{0};
    std::atomic<uint32_t> _neighborcache_family_inflight{0};
    // Pack this search's disk reads onto one page, instead of N-cache eviction victims.
    bool _neighborcache_coread_pack = false;
    std::atomic<uint64_t> _neighborcache_coread_pages{0};
    std::atomic<uint64_t> _neighborcache_coread_hop2_pages{0};
    std::atomic<uint64_t> _neighborcache_coread_enqueued{0};
    std::atomic<uint64_t> _neighborcache_coread_write_dropped{0};
    bool _neighborcache_dcache_evict_ncache_on_commit = false;
    std::atomic<uint64_t> _neighborcache_dcache_ncache_erases{0};
    bool _neighborcache_real_io_coaccess = true;
    uint32_t _neighborcache_real_io_window = 4;
    uint32_t _neighborcache_real_io_sample_queries = 64;
    uint32_t _neighborcache_real_io_max_partners = 64;
    std::atomic<uint64_t> _neighborcache_real_io_queries{0};
    std::atomic<uint64_t> _neighborcache_real_io_sampled_queries{0};
    std::atomic<uint64_t> _neighborcache_real_io_nodes{0};
    std::atomic<uint64_t> _neighborcache_real_io_pairs{0};
    // Only pairs that are graph neighbors (either adjacency list holds the other) become co-access edges.
    bool _neighborcache_real_io_direct = true;
    std::atomic<uint64_t> _neighborcache_real_io_window_pairs{0};
    std::atomic<uint64_t> _neighborcache_real_io_adjacent_pairs{0};
    NeighborCacheMetadataCache _neighborcache_mcache;
    // Members are stored densely by physical page to avoid one heap-allocated
    // vector per live d-cache page.
    std::vector<uint32_t> _neighborcache_dyn_page_members;
    // Second page of a seed. INVALID if this page has no partner.
    // Partners are adjacent physical pages so one read covers both.
    std::vector<uint32_t> _neighborcache_dyn_page_sibling;
    std::vector<uint8_t> _neighborcache_dyn_page_secondary;
    // Contiguous on-demand span. Length is stored on every page of the span.
    std::vector<uint8_t> _neighborcache_dyn_page_span_len;
    std::vector<uint32_t> _neighborcache_dyn_page_span_base;
    std::vector<uint16_t> _neighborcache_dyn_page_member_count;
    uint32_t _neighborcache_dyn_members_per_page = 1;
    // Member count of each seed's first successful page write. Same-seed rewrites keep it.
    tsl::robin_map<uint32_t, uint16_t> _neighborcache_seed_first_fill;
    std::array<std::unique_ptr<NeighborCacheDynLocShard>, NEIGHBORCACHE_DYN_LOC_MAX_SHARDS> _neighborcache_dyn_loc_shards;
    size_t _neighborcache_dyn_loc_shard_count = 64;
    std::unique_ptr<std::atomic<uint64_t>[]> _neighborcache_canon_slots;
    size_t _neighborcache_canon_mask = 0;
    std::atomic<uint8_t> _neighborcache_canon_authoritative{1};
    std::vector<uint32_t> _neighborcache_dyn_free;
    uint64_t _neighborcache_dyn_committed_pages = 0;
    uint64_t _neighborcache_dyn_page_cap = 4096;
    uint64_t _neighborcache_dyn_physical_cap = 0;
    uint32_t _neighborcache_dyn_next_page = 0;
    std::unique_ptr<std::atomic<uint32_t>[]> _neighborcache_dyn_page_gen;
    std::unique_ptr<std::atomic<uint32_t>[]> _neighborcache_dyn_page_readers;
    std::unique_ptr<std::atomic<uint8_t>[]> _neighborcache_dyn_page_retired;
    std::unique_ptr<std::atomic<uint32_t>[]> _neighborcache_span_reads;
    std::unique_ptr<std::atomic<uint32_t>[]> _neighborcache_span_hits;
    std::unique_ptr<std::atomic<uint32_t>[]> _neighborcache_span_decisions;
    bool _neighborcache_span_adaptive = false;
    float _neighborcache_span_min_hits = 1.0f;
    uint32_t _neighborcache_span_explore = 16;
    mutable std::atomic<uint64_t> _neighborcache_span_full_reads{0};
    mutable std::atomic<uint64_t> _neighborcache_span_primary_only{0};
    std::mutex _neighborcache_dyn_reclaim_mu;
    std::vector<uint32_t> _neighborcache_dyn_reclaim;
    std::string _neighborcache_dyn_path;
    std::shared_ptr<AlignedFileReader> _neighborcache_dyn_reader;
    int _neighborcache_dyn_write_fd = -1;
    std::fstream _neighborcache_dyn_write_stream;
    std::mutex _neighborcache_dyn_writer_mu;
    uint64_t _neighborcache_dyn_flush_count = 0;

    // Dedicated D-cache page writer: all pwrite/fdatasync to _neighborcache_dyn_write_fd
    // (and Windows stream page writes) run only on this thread.
    enum class NeighborCacheDynWriteOp : uint8_t
    {
        Write = 0,
        Sync = 1
    };
    struct NeighborCacheDynWriteJob
    {
        NeighborCacheDynWriteOp op = NeighborCacheDynWriteOp::Write;
        off_t offset = 0;
        size_t nbytes = 0;
        std::unique_ptr<char, void (*)(void *)> data{nullptr, ::free};
        uint64_t enqueued_ns = 0;
        std::mutex mu;
        std::condition_variable cv;
        bool done = false;
        bool ok = false;
    };
    std::vector<std::thread> _neighborcache_dyn_writer_threads;
    size_t _neighborcache_dyn_writer_thread_count = 1;
    std::mutex _neighborcache_dyn_write_q_mu;
    std::condition_variable _neighborcache_dyn_write_q_cv;
    std::deque<std::shared_ptr<NeighborCacheDynWriteJob>> _neighborcache_dyn_write_q;
    size_t _neighborcache_dyn_write_q_cap = 64;
    std::atomic<bool> _neighborcache_dyn_writer_stop{true};
    bool _neighborcache_dyn_writer_started = false;
    std::atomic<uint64_t> _neighborcache_dyn_writer_enqueued{0};
    std::atomic<uint64_t> _neighborcache_dyn_writer_queue_ns{0};
    std::atomic<uint64_t> _neighborcache_dyn_writer_io_ns{0};
    std::atomic<uint64_t> _neighborcache_dyn_writer_completed{0};
    std::atomic<uint64_t> _neighborcache_dyn_writer_failed{0};
    std::thread _neighborcache_refresh_thread;
    std::atomic<bool> _neighborcache_refresh_thread_stop{true};
    bool _neighborcache_refresh_thread_on = false;
    void neighborcache_refresh_thread_loop();
    void neighborcache_dyn_writer_loop();
    void start_neighborcache_dyn_writer();
    void stop_neighborcache_dyn_writer();
    std::shared_ptr<NeighborCacheDynWriteJob> neighborcache_dyn_writer_submit(NeighborCacheDynWriteOp op, off_t offset, const void *data,
                                                             size_t nbytes);
    bool neighborcache_dyn_writer_submit_and_wait(NeighborCacheDynWriteOp op, off_t offset, const void *data, size_t nbytes);
    // Co-read: one commit thread allocates and hands the page to a writer.
    // Locations are published only after that pwrite finishes.
    bool neighborcache_dyn_coread_submit_page(NeighborCachePendingFlush &pf, uint32_t &page_idx,
                                      std::shared_ptr<NeighborCacheDynWriteJob> &job);
    bool neighborcache_dyn_coread_complete_page(NeighborCachePendingFlush &pf, uint32_t page_idx, bool write_ok);
    size_t neighborcache_dyn_loc_shard_index(uint32_t node_id) const;
    NeighborCacheDynLocShard &neighborcache_dyn_loc_shard(uint32_t node_id);
    const NeighborCacheDynLocShard &neighborcache_dyn_loc_shard(uint32_t node_id) const;
    void neighborcache_dyn_loc_clear_unlocked();
    void neighborcache_dyn_loc_add_unlocked(uint32_t node_id, const NeighborCacheDiskLoc &loc);
    void neighborcache_dyn_loc_erase_sector_unlocked(uint32_t node_id, uint32_t sector);
    // One atomic word per slot: tag in the low 32 bits, page/slot in the high bits.
    // Query threads load it directly. 1 = live loc, 0 = not in the D-cache, -1 = use the locked map.
    void neighborcache_canon_reset();
    void neighborcache_canon_publish(uint32_t node_id, const NeighborCacheDiskLoc &loc);
    void neighborcache_canon_clear_node(uint32_t node_id);
    int neighborcache_canon_lookup(uint32_t node_id, NeighborCacheDiskLoc &out) const;
    void neighborcache_dyn_bump_page_gen(uint32_t page_idx);
    bool neighborcache_query_page_prefetch_current(const SSDQueryScratch<T> *query_scratch, uint32_t page_idx) const;
    void neighborcache_query_page_prefetch_remember(SSDQueryScratch<T> *query_scratch, uint32_t page_idx) const;
    bool neighborcache_dyn_seed_page_unlocked(uint32_t seed_id, uint32_t &page_idx) const;
    size_t neighborcache_dc_loc_count_unlocked(uint32_t node_id) const;
    bool neighborcache_dc_loc_at_unlocked(uint32_t node_id, size_t index, NeighborCacheDiskLoc &loc) const;
    bool neighborcache_dc_loc_contains_unlocked(uint32_t node_id, const NeighborCacheDiskLoc &loc) const;
    bool neighborcache_dc_loc_directory_empty_unlocked() const;
    template <typename Fn> void neighborcache_dc_for_each_loc_unlocked(uint32_t node_id, Fn &&fn) const
    {
        if (_neighborcache_dyn_enabled)
        {
            const NeighborCacheDynLocShard &shard = neighborcache_dyn_loc_shard(node_id);
            NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::DynamicShared, false);
            const auto it = shard.locations.find(node_id);
            if (it == shard.locations.end())
                return;
            size_t index = 0;
            for (const NeighborCacheDiskLoc &loc : it.value())
                fn(loc, index++);
            return;
        }
        const auto it = _neighborcache_dc_map.find(node_id);
        if (it == _neighborcache_dc_map.end())
            return;
        for (size_t index = 0; index < it->second.size(); ++index)
            fn(it->second[index], index);
    }
    struct NeighborCacheHeapEntry
    {
        float score = 0;
        uint32_t slot_id = NeighborCacheMetadataCache::kInvalid;
    };
    enum class NeighborCacheHeapKind : uint8_t
    {
        None = 0,
        Max = 1,
        Min = 2
    };
    static constexpr size_t NEIGHBORCACHE_MAX_HEAP_SHARDS = 32;
    static constexpr size_t NEIGHBORCACHE_DEFAULT_HEAP_SHARDS = 4;
    struct alignas(64) NeighborCacheHeapShard
    {
        mutable std::mutex mu;
        std::vector<NeighborCacheHeapEntry> max_heap;
        std::vector<NeighborCacheHeapEntry> min_heap;
    };
    std::array<NeighborCacheHeapShard, NEIGHBORCACHE_MAX_HEAP_SHARDS> _neighborcache_heap_shards;
    size_t _neighborcache_heap_shard_count = NEIGHBORCACHE_DEFAULT_HEAP_SHARDS;
    std::vector<uint32_t> _neighborcache_heap_position;
    std::vector<uint8_t> _neighborcache_heap_kind;
    std::unique_ptr<std::atomic<uint8_t>[]> _neighborcache_node_state;
    uint64_t _neighborcache_node_state_capacity = 0;
    tsl::robin_map<uint32_t, NeighborCacheReadyPair> _neighborcache_ready_pairs; // insertion slot -> pair
    static constexpr size_t NEIGHBORCACHE_PENDING_MEMBER_SHARDS = 64;
    struct NeighborCachePendingMemberShard
    {
        mutable NeighborCacheSpinRwLock mu;
        tsl::robin_map<uint32_t, std::vector<uint32_t>> slots;
    };
    std::array<std::unique_ptr<NeighborCachePendingMemberShard>, NEIGHBORCACHE_PENDING_MEMBER_SHARDS>
        _neighborcache_pending_member_shards;
    tsl::robin_map<uint32_t, uint32_t> _neighborcache_deletion_to_pending;
    uint64_t _neighborcache_reserved_free_pages = 0;
    std::atomic<uint32_t> _neighborcache_pending_pair_count{0};
    std::atomic<uint32_t> _neighborcache_deletion_pair_count{0};
    std::atomic<uint32_t> _neighborcache_query_epoch{0};
    uint32_t _neighborcache_pending_max_age_queries = 65536;
    uint32_t _neighborcache_pending_check_interval = 2048;
    uint32_t _neighborcache_pending_force_batch = 4;
    size_t _neighborcache_pending_pair_cap = 4096;
    std::atomic<bool> _neighborcache_refresh_needed{false};
    std::atomic<bool> _neighborcache_refresh_running{false};
    static constexpr size_t NEIGHBORCACHE_PENDING_PAIR_CAP_MAX = 65536;
    std::atomic<float> _neighborcache_score_unit{1.0f};
    // Forward decay shared by node scores and edge heat: increments add
    // unit = 2^(accesses_since_cycle_start / half_life). A cycle ends (everything divided by unit, unit back to 1)
    // when unit reaches NEIGHBORCACHE_SCORE_UNIT_MAX, or early once unit >= 2 if an edge heat saturated.
    std::atomic<uint64_t> _neighborcache_decay_query_count{0};
    std::atomic<uint64_t> _neighborcache_decay_access_count{0};
    std::atomic<uint64_t> _neighborcache_decay_cycle_start{0};
    std::atomic<bool> _neighborcache_decay_started{false};
    std::mutex _neighborcache_decay_mu;
    double _neighborcache_decay_half_life_accesses = 2.8e6;
    std::atomic<uint64_t> _neighborcache_decay_cycles{0};
    std::atomic<uint64_t> _neighborcache_decay_early_cycles{0};
    static constexpr float NEIGHBORCACHE_SCORE_UNIT_MAX = 8.0f;
    uint64_t _neighborcache_pair_refresh = 0;
    uint64_t _neighborcache_heap_writes = 0;
    uint64_t _neighborcache_heap_deletes = 0;
    uint64_t _neighborcache_ncache_write_trig = 0;
    mutable NeighborCacheSpinRwLock _neighborcache_pair_mu;
    static constexpr uint32_t NEIGHBORCACHE_FETCHED_PER_PARENT = 8;
    struct NeighborCacheRecentChildren
    {
        std::array<uint32_t, NEIGHBORCACHE_FETCHED_PER_PARENT> ids;

        NeighborCacheRecentChildren()
        {
            ids.fill(NeighborCacheMetadataCache::kInvalid);
        }

        void touch(uint32_t child)
        {
            size_t count = 0;
            size_t pos = ids.size();
            while (count < ids.size() && ids[count] != NeighborCacheMetadataCache::kInvalid)
            {
                if (ids[count] == child)
                    pos = count;
                ++count;
            }
            if (pos < count)
            {
                for (size_t i = pos + 1; i < count; ++i)
                    ids[i - 1] = ids[i];
                ids[count - 1] = child;
                return;
            }
            if (count < ids.size())
            {
                ids[count] = child;
                if (count + 1 < ids.size())
                    ids[count + 1] = NeighborCacheMetadataCache::kInvalid;
                return;
            }
            for (size_t i = 1; i < count; ++i)
                ids[i - 1] = ids[i];
            ids[count - 1] = child;
        }

        size_t size() const
        {
            size_t count = 0;
            while (count < ids.size() && ids[count] != NeighborCacheMetadataCache::kInvalid)
                ++count;
            return count;
        }

        bool empty() const
        {
            return ids[0] == NeighborCacheMetadataCache::kInvalid;
        }

        const uint32_t *begin() const
        {
            return ids.data();
        }
        const uint32_t *end() const
        {
            return ids.data() + size();
        }
        size_t capacity() const
        {
            return ids.size();
        }
        void clear()
        {
            ids[0] = NeighborCacheMetadataCache::kInvalid;
        }
    };
    uint64_t _neighborcache_stash_cap = 16384;
    tsl::robin_map<uint32_t, NeighborCachePayloadStashEntry> _neighborcache_payload_stash;
    std::list<uint32_t> _neighborcache_stash_lru;
    // Optional write-back buffer for deferred D-cache pages. It retains full
    // payloads of pending-page members after N-cache eviction, but does not
    // participate in N-cache replacement.
    uint64_t _neighborcache_pending_buffer_cap_bytes = 0;
    uint64_t _neighborcache_pending_buffer_payload_bytes = 0;
    bool _neighborcache_pending_require_full_page = true;
    uint32_t _neighborcache_pending_buffer_ready_percent = 50;
    uint16_t _neighborcache_pending_buffer_ready_min_members = 2;
    std::atomic<uint64_t> _neighborcache_pending_buffer_inserts{0};
    std::atomic<uint64_t> _neighborcache_pending_buffer_ready_writes{0};
    std::atomic<uint64_t> _neighborcache_pending_buffer_pressure_writes{0};
    std::atomic<uint64_t> _neighborcache_pending_buffer_drops{0};
    std::atomic<uint64_t> _neighborcache_pending_member_refreshes{0};
    std::atomic<uint64_t> _neighborcache_pending_became_full{0};
    // Same-query D-page patch: fold neighbors onto a seed page served this query.
    bool _neighborcache_dcache_no_seed_replace = false;
    // When a page is full, keep old neighbors and write overflow onto later pages.
    bool _neighborcache_dcache_second_page = false;
    uint8_t _neighborcache_dcache_max_pages = 1;
    std::atomic<uint64_t> _neighborcache_dcache_pair_ios{0};
    uint64_t _neighborcache_dcache_freeze_after = 0;
    uint64_t _neighborcache_member_snapshot_at = 0;
    std::atomic<bool> _neighborcache_member_snapshot_done{false};
    tsl::robin_set<uint32_t> _neighborcache_member_at_snapshot;
    std::atomic<uint64_t> _neighborcache_member_snapshot_slots{0};
    std::atomic<uint64_t> _neighborcache_prefetch_from_early{0};
    std::atomic<uint64_t> _neighborcache_prefetch_from_late{0};
    std::atomic<uint64_t> _neighborcache_early_hop_hist[128];
    std::atomic<uint64_t> _neighborcache_late_hop_hist[128];
    bool _neighborcache_dcache_query_patch = true;
    // Rewrite a page when it is read: keep the records in the read buffer, and
    // fill free slots with neighbors whose payloads are already in N-cache.
    bool _neighborcache_repack_on_read = false;
    bool _neighborcache_dcache_query_patch_async = true;
    uint32_t _neighborcache_dcache_query_patch_max = 4;
    std::atomic<uint64_t> _neighborcache_dcache_query_patch_trig{0};
    std::atomic<uint64_t> _neighborcache_dcache_query_patch_ok{0};
    std::atomic<uint64_t> _neighborcache_dcache_query_patch_drop{0};
    struct NeighborCachePatchJob
    {
        uint32_t seed_id = 0;
        std::vector<uint32_t> new_ids;
        std::vector<uint32_t> prior_members;
        std::vector<std::vector<uint32_t>> prior_extra;
        uint32_t replacement_seed = NEIGHBORCACHE_DYN_INVALID_PAGE;
        // Payloads already copied from the sector just read plus N-cache.
        bool ready = false;
        NeighborCachePendingFlush ready_flush;
    };
    std::mutex _neighborcache_patch_q_mu;
    std::condition_variable _neighborcache_patch_q_cv;
    std::deque<std::shared_ptr<NeighborCachePatchJob>> _neighborcache_patch_q;
    size_t _neighborcache_patch_q_cap = 64;
    uint32_t _neighborcache_patch_batch_us = 100;
    bool _neighborcache_patch_worker_no_disk = false;
    std::thread _neighborcache_patch_thread;
    std::atomic<bool> _neighborcache_patch_stop{true};
    bool _neighborcache_patch_started = false;
    tsl::robin_map<uint32_t, uint32_t> _neighborcache_patch_pin_count;
    uint32_t _neighborcache_monitor_warmup_queries = 0;
    uint32_t _neighborcache_monitor_interval_queries = 1000000;
    std::vector<tsl::robin_set<uint32_t>> _neighborcache_pending_buffer_buckets;
    std::vector<NeighborCacheRecentChildren> _neighborcache_parent_fetched_by_slot;
    std::vector<uint32_t> _neighborcache_parent_fetched_owner;
    struct alignas(64) NeighborCachePaddedMutex
    {
        std::mutex mu;
    };
    static constexpr size_t NEIGHBORCACHE_PARENT_FETCHED_STRIPES = 64;
    mutable std::array<NeighborCachePaddedMutex, NEIGHBORCACHE_PARENT_FETCHED_STRIPES> _neighborcache_parent_fetched_mu;
    std::mutex &neighborcache_parent_fetched_mu(uint32_t slot_id) const
    {
        return _neighborcache_parent_fetched_mu[slot_id & (NEIGHBORCACHE_PARENT_FETCHED_STRIPES - 1)].mu;
    }
    void neighborcache_mark_refresh_needed()
    {
        if (!_neighborcache_refresh_needed.load(std::memory_order_relaxed))
            _neighborcache_refresh_needed.store(true, std::memory_order_release);
    }
    mutable std::mutex _neighborcache_stash_mu;
    // Reader-writer lock (std::shared_mutex). Lookup/copy take shared; page
    // install takes exclusive. Not a spin/seqlock: robin_map rehash is not
    // safe to read mid-write. 4KB memcpy is done outside this lock.
    mutable std::shared_mutex _neighborcache_dyn_mu;

    bool neighborcache_dyn_is_sector(uint32_t sector) const
    {
        return sector >= NEIGHBORCACHE_DYN_SECTOR_BASE;
    }
    bool neighborcache_dyn_page_live(uint32_t page_idx) const
    {
        return page_idx < _neighborcache_dyn_page_member_count.size() &&
               _neighborcache_dyn_page_member_count[page_idx] != 0;
    }
    std::vector<uint32_t> neighborcache_dyn_now_page_ids(uint32_t seed_id, const NeighborCacheMetadataCache::Snapshot &snap,
                                                 bool ncache_only = false) const;
    bool neighborcache_dyn_hotter_mismatch(const uint32_t *page_members, size_t page_member_count,
                                   const std::vector<uint32_t> &now_ids) const;
    void neighborcache_dyn_invalidate_seed_page(uint32_t seed_id);
    void neighborcache_dyn_retire_secondary_unlocked(uint32_t page_idx);
    uint32_t neighborcache_dyn_alloc_page_unlocked();
    void neighborcache_dyn_retire_page_unlocked(uint32_t page_idx);
    bool neighborcache_span_should_read_full(uint32_t base, uint32_t span) const;
    void neighborcache_span_carry_stats(uint32_t from_page, uint32_t to_page);
    void neighborcache_dyn_drain_reclaim_unlocked();
    uint32_t neighborcache_dyn_alloc_contiguous_unlocked(uint32_t count);
    void neighborcache_dyn_maybe_mark_seed(uint32_t node_id);
    void neighborcache_dyn_note_touch(const NeighborCacheMetadataCache::TouchResult &tr, SSDQueryScratch<T> *query_scratch);
    void neighborcache_dyn_on_mcache_evict(uint32_t node_id, uint32_t slot_id);
    bool neighborcache_dyn_ncache_evictable(uint32_t node_id) const;
    bool neighborcache_dyn_heap_better_unlocked(const NeighborCacheHeapEntry &a, const NeighborCacheHeapEntry &b,
                                        NeighborCacheHeapKind kind) const;
    void neighborcache_dyn_heap_swap_unlocked(std::vector<NeighborCacheHeapEntry> &heap, size_t a, size_t b);
    void neighborcache_dyn_heap_sift_up_unlocked(std::vector<NeighborCacheHeapEntry> &heap, size_t index, NeighborCacheHeapKind kind);
    void neighborcache_dyn_heap_sift_down_unlocked(std::vector<NeighborCacheHeapEntry> &heap, size_t index, NeighborCacheHeapKind kind);
    void neighborcache_dyn_heap_insert_unlocked(const NeighborCacheHeapEntry &entry, NeighborCacheHeapKind kind);
    size_t neighborcache_dyn_heap_shard_index(uint32_t slot_id) const;
    NeighborCacheHeapShard &neighborcache_dyn_heap_shard(uint32_t slot_id);
    const NeighborCacheHeapShard &neighborcache_dyn_heap_shard(uint32_t slot_id) const;
    std::vector<NeighborCacheHeapEntry> &neighborcache_dyn_heap_for(NeighborCacheHeapShard &shard, NeighborCacheHeapKind kind);
    const std::vector<NeighborCacheHeapEntry> &neighborcache_dyn_heap_for(const NeighborCacheHeapShard &shard,
                                                          NeighborCacheHeapKind kind) const;
    NeighborCacheHeapKind neighborcache_dyn_heap_kind_unlocked(uint32_t slot_id) const;
    void neighborcache_dyn_set_heap_kind_unlocked(uint32_t slot_id, NeighborCacheHeapKind kind);
    void neighborcache_dyn_heap_erase_unlocked(uint32_t slot_id);
    void neighborcache_dyn_heap_apply_unlocked(uint32_t slot_id, float score, NeighborCacheHeapKind desired);
    void neighborcache_dyn_heap_upsert_unlocked(uint32_t slot_id, float score);
    void neighborcache_dyn_clean_heap_unlocked(NeighborCacheHeapShard &shard, NeighborCacheHeapKind kind);
    bool neighborcache_dyn_best_heap_entry(NeighborCacheHeapKind kind, NeighborCacheHeapEntry &entry, size_t &shard_index);
    void neighborcache_dyn_flush_dirty_heap(SSDQueryScratch<T> *query_scratch);
    NeighborCacheNodeState neighborcache_dyn_state_unlocked(uint32_t slot_id) const;
    void neighborcache_dyn_set_state_unlocked(uint32_t node_id, uint32_t slot_id, NeighborCacheNodeState st);
    NeighborCachePendingMemberShard &neighborcache_pending_member_shard(uint32_t node_id);
    const NeighborCachePendingMemberShard &neighborcache_pending_member_shard(uint32_t node_id) const;
    std::vector<uint32_t> neighborcache_pending_member_slots(uint32_t node_id) const;
    bool neighborcache_pending_member_contains(uint32_t node_id) const;
    void neighborcache_pending_member_add(uint32_t node_id, uint32_t insertion_slot);
    void neighborcache_pending_member_remove(uint32_t node_id, uint32_t insertion_slot);
    void neighborcache_pending_member_clear();
    void neighborcache_dyn_clear_ready_pair_unlocked(uint32_t insertion_slot, bool restore_states = true);
    bool neighborcache_dyn_disk_is_full() const;
    bool neighborcache_dcache_updates_frozen() const;
    bool neighborcache_dyn_build_pending_flush(uint32_t seed_id, const NeighborCacheMetadataCache::Snapshot &snap,
                                       NeighborCachePendingFlush &pf) const;
    bool neighborcache_dyn_refresh_pending_members(uint32_t insertion_slot);
    void neighborcache_dyn_maybe_refresh_pair();
    void neighborcache_dyn_pin_seed_neighbors(uint32_t seed_id, uint32_t insertion_slot);
    void neighborcache_dyn_pin_admitted_neighbor(uint32_t node_id);
    void neighborcache_dyn_release_seed_neighbor_holds(uint32_t seed_id);
    bool neighborcache_dyn_pin_take_write_budget(uint32_t budget_epoch);
    void neighborcache_dyn_pin_refund_write_budget(uint32_t budget_epoch);
    void neighborcache_dyn_enqueue_seed_holding_node(uint32_t node_id, QueryStats *stats);
    uint32_t neighborcache_dyn_flush_fullest_pinned_seed();
    void neighborcache_dyn_write_settled_pinned_seeds(uint32_t query_epoch, QueryStats *stats);
    uint32_t neighborcache_seed_page_shard_index(uint32_t id) const;
    void neighborcache_seed_page_link(uint32_t seed_id, const std::vector<uint32_t> &nbrs);
    void neighborcache_seed_page_unlink(uint32_t seed_id, const std::vector<uint32_t> &nbrs);
    void neighborcache_seed_page_enqueue(NeighborCachePendingFlush pf);
    void neighborcache_seed_page_flush_evicted(NeighborCacheSeedPage &page);
    void neighborcache_seed_page_open(uint32_t seed_id);
    void neighborcache_seed_page_note(uint32_t node_id);
    bool neighborcache_seed_page_lookup(uint32_t node_id, std::vector<T> &coords, std::vector<uint32_t> &nbrs);
    // Assembled seed pages live in the shared prefetch. No second page LRU.
    // A write locks only this seed; readers of the prefetch take no lock.
    NeighborCacheSpinLock &neighborcache_seed_write_lock(uint32_t seed_id);
    bool neighborcache_seed_prefetch_publish(uint32_t seed_id, const std::vector<uint32_t> &ids,
                                     const std::vector<std::vector<T>> &coords,
                                     const std::vector<std::vector<uint32_t>> &nbrs);
    void neighborcache_seed_prefetch_note(uint32_t node_id);
    bool neighborcache_seed_prefetch_lookup(uint32_t node_id, std::vector<T> &coords, std::vector<uint32_t> &nbrs);
    // Payload is resident on a seed page. Does not touch the page LRU or the hit counter.
    bool neighborcache_seed_page_contains(uint32_t node_id);
    void neighborcache_seed_page_commit(NeighborCachePendingFlush pf);
    void neighborcache_dyn_request_refresh();
    bool neighborcache_dyn_commit_ready_pair(uint32_t insertion_slot);
    enum class NeighborCacheCommitSource : uint8_t
    {
        Evict = 0,
        Buffered = 1,
        Expired = 2
    };
    struct NeighborCachePairCommitJob
    {
        uint32_t insertion_slot = NeighborCacheMetadataCache::kInvalid;
        NeighborCacheCommitSource source = NeighborCacheCommitSource::Evict;
        uint64_t enqueued_ns = 0;
    };
    // Co-read pages built on the query thread. One commit thread hands each
    // page to a writer thread and publishes the location after pwrite.
    struct NeighborCacheCoreadWriteJob
    {
        NeighborCachePendingFlush pf;
        bool hop2 = false;
    };
    std::mutex _neighborcache_commit_q_mu;
    std::condition_variable _neighborcache_commit_q_cv;
    std::deque<NeighborCachePairCommitJob> _neighborcache_commit_q;
    std::deque<NeighborCacheCoreadWriteJob> _neighborcache_coread_write_q;
    std::deque<NeighborCachePendingFlush> _neighborcache_seed_page_q;
    size_t _neighborcache_seed_page_q_cap = 16384;
    // New family members only. The commit thread appends them to the live page.
    // Queue order is seed ids. A second expansion of the same seed merges into the
    // pending flush so the page is read and written once.
    std::deque<uint32_t> _neighborcache_family_q;
    tsl::robin_map<uint32_t, NeighborCachePendingFlush> _neighborcache_family_pending;
    size_t _neighborcache_family_q_cap = 16384;
    size_t _neighborcache_coread_write_q_cap = 262144;
    NeighborCacheSpinLock _neighborcache_coread_pending_mu;
    tsl::robin_set<uint32_t> _neighborcache_coread_pending;
    tsl::robin_set<uint32_t> _neighborcache_commit_pending_slots;
    std::vector<std::thread> _neighborcache_commit_threads;
    size_t _neighborcache_commit_thread_count = 1;
    size_t _neighborcache_commit_q_cap = 256;
    bool _neighborcache_commit_inline = false;
    std::atomic<bool> _neighborcache_commit_stop{true};
    bool _neighborcache_commit_started = false;
    std::atomic<uint64_t> _neighborcache_commit_enqueued{0};
    std::atomic<uint64_t> _neighborcache_commit_coalesced{0};
    std::atomic<uint64_t> _neighborcache_commit_dropped{0};
    std::atomic<uint64_t> _neighborcache_commit_completed{0};
    std::atomic<uint64_t> _neighborcache_commit_failed{0};
    std::array<std::atomic<uint64_t>, 6> _neighborcache_commit_fail_why{};
    std::atomic<uint64_t> _neighborcache_commit_queue_ns{0};
    std::atomic<uint64_t> _neighborcache_commit_work_ns{0};
    std::array<std::atomic<uint64_t>, 3> _neighborcache_commit_source_enqueued{};
    std::array<std::atomic<uint64_t>, 3> _neighborcache_commit_source_completed{};
    bool neighborcache_dyn_enqueue_ready_pair(uint32_t insertion_slot, NeighborCacheCommitSource source);
    // handed_off: the commit thread will write and count the page.
    // Returns false when the page was neither queued nor written.
    bool neighborcache_dyn_enqueue_coread_write(NeighborCachePendingFlush &pf, bool hop2, bool &handed_off);
    bool neighborcache_coread_id_pending(uint32_t id);
    void neighborcache_dyn_commit_loop();
    void start_neighborcache_dyn_commit_workers();
    void stop_neighborcache_dyn_commit_workers();
    void neighborcache_dyn_wait_for_commits();
    void neighborcache_dyn_commit_expired_pairs(uint32_t query_epoch, QueryStats *stats);
    uint32_t neighborcache_dyn_admit_node(uint32_t node_id, const char *node_disk_buf, QueryStats *stats,
                                  SSDQueryScratch<T> *query_scratch, uint32_t search_hop);
    void neighborcache_dyn_note_base_load(uint32_t node_id, uint32_t parent, const char *node_disk_buf,
                                  SSDQueryScratch<T> *query_scratch);
    bool neighborcache_dyn_note_patch_candidate(uint32_t node_id, uint32_t parent, SSDQueryScratch<T> *query_scratch);
    void neighborcache_dyn_apply_query_patches(QueryStats *stats, SSDQueryScratch<T> *query_scratch);
    std::vector<uint32_t> neighborcache_dyn_assemble_patch_members(
        uint32_t seed_id, const std::vector<uint32_t> &new_ids, const std::vector<uint32_t> &prior_members,
        const std::vector<std::vector<uint32_t>> &prior_extra = {},
        std::vector<std::vector<uint32_t>> *extra_out = nullptr) const;
    void neighborcache_dyn_patch_pin(const std::vector<uint32_t> &member_ids);
    void neighborcache_dyn_patch_unpin(const std::vector<uint32_t> &member_ids);
    void neighborcache_dyn_patch_loop();
    void start_neighborcache_dyn_patch_worker();
    void stop_neighborcache_dyn_patch_worker();
    bool neighborcache_dyn_enqueue_patch(uint32_t seed_id, std::vector<uint32_t> new_ids,
                                 std::vector<uint32_t> prior_members,
                                 std::vector<std::vector<uint32_t>> prior_extra, uint32_t replacement_seed);
    bool neighborcache_dyn_seed_members_equal(uint32_t seed_id, const std::vector<uint32_t> &expected) const;
    bool neighborcache_dyn_enqueue_ready_repack(NeighborCachePendingFlush flush, std::vector<uint32_t> prior_members);
    void neighborcache_dyn_repack_read_page(SSDQueryScratch<T> *query_scratch, uint32_t sector, const char *sector_base,
                                    QueryStats *stats);
    void neighborcache_dyn_record_real_io_coaccess(SSDQueryScratch<T> *query_scratch);
    bool neighborcache_dyn_has_payload(uint32_t node_id) const;
    bool neighborcache_dyn_copy_member_payload(uint32_t node_id, std::vector<T> &coords, std::vector<uint32_t> &nbrs,
                                       bool allow_disk = true) const;
    bool neighborcache_dyn_extract_overlay_payload(uint32_t node_id, std::vector<T> &coords,
                                           std::vector<uint32_t> &nbrs) const;
    int neighborcache_dyn_import_static_prefill();
    bool neighborcache_dyn_buffer_evicted_member(uint32_t node_id, bool &inserted);
    uint32_t neighborcache_dyn_best_buffered_pair(bool require_ready) const;
    void neighborcache_dyn_release_unreferenced_buffer_members(const std::vector<uint32_t> &member_ids);
    void neighborcache_dyn_trim_pending_buffer();
    void neighborcache_dyn_stash_evict_unlocked();
    void neighborcache_dyn_release_page_pin(uint32_t sector);
    void neighborcache_dyn_on_ncache_evict(uint32_t node_id, QueryStats *stats);
    void neighborcache_family_note_read(uint32_t sector);
    void neighborcache_family_on_evict(uint32_t node_id);
    void neighborcache_family_link_unlocked(uint32_t node_id, uint32_t seed_id);
    void neighborcache_family_unlink_node_unlocked(uint32_t node_id, uint32_t seed_id);
    void neighborcache_family_erase_unlocked(uint32_t seed_id);
    void neighborcache_family_touch_unlocked(uint32_t node_id);
    bool neighborcache_family_copy_cached(uint32_t node_id, std::vector<T> &coords, std::vector<uint32_t> &nbrs);
    bool neighborcache_family_cached(uint32_t node_id);
    bool neighborcache_family_load_nbrs(uint32_t node_id, std::vector<uint32_t> &nbrs);
    void neighborcache_family_enqueue(uint32_t seed_id);
    void neighborcache_family_commit(NeighborCachePendingFlush pf);
    void neighborcache_dyn_prepare_query(SSDQueryScratch<T> *query_scratch);
    void neighborcache_dyn_update_dcache_gate(QueryStats *stats, SSDQueryScratch<T> *query_scratch);
    void neighborcache_dyn_pack_coread(SSDQueryScratch<T> *query_scratch);
    void neighborcache_dyn_on_query_end(QueryStats *stats, SSDQueryScratch<T> *query_scratch);
    void neighborcache_dyn_log_monitor(uint32_t query_epoch) const;
    bool neighborcache_dyn_commit_one(NeighborCachePendingFlush &pf, uint32_t replacement_seed = NEIGHBORCACHE_DYN_INVALID_PAGE);
    bool neighborcache_dyn_commit_span(std::vector<NeighborCachePendingFlush> &pages, uint32_t replacement_seed);

    void neighborcache_get_expand_neighbors(uint32_t expand_id, char *node_disk_buf, const uint32_t *&out_nbrs,
                                    uint64_t &out_nnbrs) const;

    void update_neighborcache_seed_first_lookup_flag();

    bool neighborcache_disk_cache_lookup_hit(uint32_t node_id, SSDQueryScratch<T> *query_scratch) const;
    bool neighborcache_dyn_query_local_dcache_hit(uint32_t node_id, SSDQueryScratch<T> *query_scratch) const;
    // Register all members of a dyn page into the query-local prefetch map so
    // same-hop / later-hop co-members count as prefetch instead of another seed hit.
    void neighborcache_pack_dyn_page_into_query_prefetch(uint32_t sector, SSDQueryScratch<T> *query_scratch) const;

    bool neighborcache_resolve_disk_cache_loc(uint32_t node_id, const SSDQueryScratch<T> *query_scratch,
                                      NeighborCacheDiskLoc &out_loc) const;

    bool neighborcache_loc_in_query_cache(const SSDQueryScratch<T> *query_scratch, uint32_t base_sector,
                                  uint16_t nsectors) const;
    bool neighborcache_use_shared_cache(uint32_t sector) const
    {
        return _neighborcache_shared_cache.enabled() && (_neighborcache_shared_cache_dyn || !neighborcache_dyn_is_sector(sector));
    }
    void neighborcache_shared_cache_erase_dyn(uint32_t page_idx, uint32_t count) const
    {
        if (!_neighborcache_shared_cache_dyn || page_idx == NEIGHBORCACHE_DYN_INVALID_PAGE)
            return;
        for (uint32_t i = 0; i < count; ++i)
            _neighborcache_shared_cache.erase(static_cast<uint32_t>(NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx + i));
    }
    bool neighborcache_shared_cache_admit_dyn(uint32_t page_idx, const char *page) const
    {
        if (!_neighborcache_shared_cache_dyn || !_neighborcache_shared_cache.enabled() || page == nullptr ||
            page_idx == NEIGHBORCACHE_DYN_INVALID_PAGE)
            return false;
        return _neighborcache_shared_cache.put(static_cast<uint32_t>(NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx), page, true);
    }
    void neighborcache_prefetch_mark_dirty(uint32_t page_idx);
    static void neighborcache_prefetch_evict_hook(void *ctx, uint32_t key, const char *page);
    void neighborcache_prefetch_on_evict(uint32_t key, const char *page);
    // Copies [base, base+nsectors) from the shared or query-local cache into dst; all or nothing.
    bool neighborcache_cache_get_range(SSDQueryScratch<T> *query_scratch, uint32_t base, uint16_t nsectors, char *dst);
    // Pointer to the packed node at (sector, slot) inside the shared page cache, or nullptr on miss.
    // A node straddling two sectors is stitched into `spill` (one sector) and `spilled` is set.
    const char *neighborcache_shared_node_ptr(uint32_t sector, uint32_t slot, char *spill, bool &spilled);
    void neighborcache_cache_put(SSDQueryScratch<T> *query_scratch, uint32_t sector, const char *src);

    mutable NeighborCacheSharedPageCache _neighborcache_shared_cache;
    // Online D-cache pages are rewritten after reuse, so they need erase-on-alloc and per-query pins.
    bool _neighborcache_shared_cache_dyn = false;

    // thread-specific scratch
    ConcurrentQueue<SSDThreadData<T> *> _thread_data;
    uint64_t _max_nthreads;
    bool _load_flag = false;
    bool _count_visited_nodes = false;
    HotnessProfiler _hotness_profiler;
    bool _query_sector_cache_enabled = false;
    bool _record_base_frontier = false;
    bool _record_hop_frontier = false;
    bool _reorder_data_exists = false;
    uint64_t _reoreder_data_offset = 0;

    // filter support
    uint32_t *_pts_to_label_offsets = nullptr;
    uint32_t *_pts_to_label_counts = nullptr;
    LabelT *_pts_to_labels = nullptr;
    std::unordered_map<LabelT, std::vector<uint32_t>> _filter_to_medoid_ids;
    bool _use_universal_label = false;
    LabelT _universal_filter_label;
    tsl::robin_set<uint32_t> _dummy_pts;
    tsl::robin_set<uint32_t> _has_dummy_pts;
    tsl::robin_map<uint32_t, uint32_t> _dummy_to_real_map;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _real_to_dummy_map;
    std::unordered_map<std::string, LabelT> _label_map;

#ifdef EXEC_ENV_OLS
    // Set to a larger value than the actual header to accommodate
    // any additions we make to the header. This is an outer limit
    // on how big the header can be.
    static const int HEADER_SIZE = defaults::SECTOR_LEN;
    char *getHeaderBytes();
#endif
};
} // namespace diskann
