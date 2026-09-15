// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "common_includes.h"
#include <limits>
#include <list>
#include <fstream>
#include <mutex>
#include <queue>
#include <set>
#include <shared_mutex>

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq.h"
#include "utils.h"
#include "windows_customizations.h"
#include "hotness_profiler.h"
#include "merit_memory_pool.h"
#include "merit_metadata_cache.h"
#include "scratch.h"
#include "relayout_utils.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann
{

inline constexpr uint8_t MERIT_DC_SEED_MAGIC = 0x4D;
inline constexpr uint8_t MERIT_DC_FLAG_TOTAL_PAGES_MASK = 0xFF;

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
                                            const bool shuffle = false);

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
    DISKANN_DLLEXPORT void enable_base_frontier_recording(bool enable);
    DISKANN_DLLEXPORT void enable_hop_frontier_recording(bool enable);
    DISKANN_DLLEXPORT int save_access_profile(const std::string &output_prefix) const;
    DISKANN_DLLEXPORT void print_access_profile_cdf() const;

    // MERIT Memory cache: size in bytes per cached node entry (coords + nhood).
    DISKANN_DLLEXPORT uint64_t bytes_per_cached_node() const;
    // Estimate DRAM already needed by index (PQ, pivots, thread scratch, etc.).
    DISKANN_DLLEXPORT uint64_t estimate_baseline_resident_bytes(uint32_t num_threads) const;
    // Validate requested cache GB against residual host memory; fill report and max node count.
    // Returns 0 on success, -1 if request exceeds residual budget.
    DISKANN_DLLEXPORT int plan_merit_memory_cache(double merit_memory_gb, double host_memory_gb,
                                                  double reserve_gb, uint32_t num_threads,
                                                  uint64_t &out_max_nodes, std::string &report) const;
    // Select Top-N nodes by Run2 node_expand counts (N limited by plan).
    // rank_skip: skip the hottest rank_skip nodes (for disk tier after memory).
    DISKANN_DLLEXPORT int build_merit_memory_node_list(const std::string &profile_prefix, uint64_t max_nodes,
                                                      std::vector<uint32_t> &node_list,
                                                      uint64_t rank_skip = 0) const;
    // Hot-node + k-hop page order (relayout semantics); disk cache only, max_nodes cap, exclude memory-tier ids.
    DISKANN_DLLEXPORT int build_merit_disk_node_list(const std::string &profile_prefix, uint64_t max_nodes,
                                                     uint64_t memory_tier_exclude_count, uint32_t k_hops,
                                                     const std::string &layout, std::vector<uint32_t> &node_list,
                                                     std::vector<SeedPageGroup> *seed_page_groups = nullptr) const;

    DISKANN_DLLEXPORT uint64_t merit_memory_cached_count() const;
    DISKANN_DLLEXPORT bool merit_mem_pool_contains(uint32_t node_id) const;
    DISKANN_DLLEXPORT bool merit_dc_map_contains(uint32_t node_id) const;
    // Compare every MERIT disk-cache node against base _disk.index (coords + nbr list).
    DISKANN_DLLEXPORT int verify_merit_disk_cache_against_base(uint64_t max_reports = 20);
    DISKANN_DLLEXPORT void clear_merit_memory_cache();
    // Load MERIT dynamic pool (Top-N from profile); does not use DiskANN _nhood_cache.
    DISKANN_DLLEXPORT int load_merit_memory_pool(const std::string &profile_prefix,
                                                  std::vector<uint32_t> &node_list);
    DISKANN_DLLEXPORT void enable_merit_memory_runtime_admit(bool enable);
    // Runtime n-cache LRU + m-cache + d-cache seed flush (MERIT_DYNAMIC_3CACHE).
    DISKANN_DLLEXPORT void enable_merit_dynamic_3cache(bool enable, const std::string &flush_prefix = "");
    DISKANN_DLLEXPORT bool merit_dynamic_3cache_enabled() const;
    DISKANN_DLLEXPORT void print_merit_dynamic_3cache_stats() const;
    DISKANN_DLLEXPORT uint64_t merit_dynamic_flush_count() const;
    DISKANN_DLLEXPORT uint64_t merit_dynamic_clean_seeds() const;
    DISKANN_DLLEXPORT uint64_t merit_dynamic_partial_seeds() const;
    // Shrink/grow memory cache; evicted = nodes removed vs previous load.
    DISKANN_DLLEXPORT int reload_merit_memory_cache(const std::string &profile_prefix, double merit_memory_gb,
                                                    double host_memory_gb, double reserve_gb, uint32_t num_threads,
                                                    uint64_t &evicted_nodes, std::string &report);

    // MERIT Disk Cache: ratio of base _disk.index size (e.g. 0.1 => 10% of base bytes).
    DISKANN_DLLEXPORT int plan_merit_disk_cache(double base_ratio, uint64_t &out_max_nodes, std::string &report,
                                                bool allow_replica_slots = false) const;
    // Build side file from ranked expand nodes; rank_skip excludes hottest (memory tier).
    DISKANN_DLLEXPORT int build_and_load_merit_disk_cache(const std::string &profile_prefix, uint64_t max_nodes,
                                                          const std::string &output_prefix, uint64_t rank_skip = 0,
                                                          bool unified_single_file = false, uint32_t k_hops = 2,
                                                          const std::string &layout = "node");
    // Load existing disk cache from prefix_merit_dc.{data,nodes} without repacking.
    DISKANN_DLLEXPORT int load_merit_disk_cache_from_prefix(const std::string &output_prefix);
    // Rebuild disk cache (batch eviction); evicted = nodes dropped vs previous map.
    DISKANN_DLLEXPORT int reload_merit_disk_cache(const std::string &profile_prefix, double base_ratio,
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

    struct MeritDiskLoc
    {
        uint32_t sector = 0;
        uint16_t slot = 0;
        uint16_t nsectors = 1;
    };

    struct MeritReadPending
    {
        uint32_t id = 0;
        char *sec_buf = nullptr;
        MeritDiskLoc loc;
        // Start sector of the IO buffer in sec_buf (multi-page seed group reads).
        uint32_t io_base_sector = 0;
        // Number of contiguous 4KB pages in the single physical IO.
        uint16_t io_nsectors = 1;
        bool dynamic_page_pinned = false;
    };

    void prepare_merit_disk_cache_io(const std::vector<uint32_t> &merit_ids, SSDQueryScratch<T> *query_scratch,
                                  char *sector_scratch, uint64_t &sector_scratch_idx, size_t num_sectors_per_node,
                                  std::vector<MeritReadPending> &pending, std::vector<AlignedRead> &merit_io,
                                  std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout_groups,
                                  QueryStats *stats, uint32_t &num_ios);

    void complete_merit_disk_cache_io(SSDQueryScratch<T> *query_scratch, std::vector<MeritReadPending> &pending,
                                   const std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout_groups);

    void finalize_merit_pending_nodes(const std::vector<MeritReadPending> &pending,
                                      const std::vector<uint32_t> &merit_order, SSDQueryScratch<T> *query_scratch,
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

    std::unique_ptr<MeritMemoryPool<T>> _merit_mem_pool;
    // Runtime try_admit during search (eviction); off until pool lookup stability verified.
    bool _merit_mem_runtime_admit = false;

    // MERIT disk-cache: packed side file; location = (sector_id, slot_in_sector)
    std::shared_ptr<AlignedFileReader> _merit_disk_reader;
    tsl::robin_map<uint32_t, std::vector<MeritDiskLoc>> _merit_dc_map;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _merit_dc_seed_page_nbrs;
    // seed -> member -> loc on that seed's page(s); used for parent/seed-first IO routing
    tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, MeritDiskLoc>> _merit_dc_seed_member_loc;
    tsl::robin_map<uint32_t, MeritDiskLoc> _merit_dc_seed_canonical_loc;
    std::string _merit_dc_path;
    uint64_t _merit_dc_num_nodes = 0;
    bool _merit_unified_disk = false;
    uint64_t _merit_region_byte_offset = 0;
    bool _merit_seed_only_layout = false;
    bool _merit_seed_only_expand = false;
    bool _merit_seed_first_lookup = false;
    bool _merit_child_only_layout = false;

    static constexpr uint32_t MERIT_DYN_SECTOR_BASE = 0x80000000u;
    static constexpr uint32_t MERIT_DYN_INVALID_PAGE = std::numeric_limits<uint32_t>::max();

    struct MeritSeedDirEntry
    {
        std::vector<uint32_t> page_members;
        uint32_t page_id = MERIT_DYN_INVALID_PAGE;
    };

    enum class MeritNodeState : uint8_t
    {
        NonSeed = 0,
        ReadyToInsertion = 1,
        Seed = 2,
        ReadyToDeletion = 3
    };

    struct MeritPendingFlush
    {
        uint32_t seed_id = 0;
        MeritMetadataCache::Snapshot snap;
        std::vector<uint32_t> member_ids;
        std::vector<std::vector<T>> member_coords;
        std::vector<std::vector<uint32_t>> member_nbrs;
    };

    struct MeritReadyPair
    {
        uint32_t insertion_id = std::numeric_limits<uint32_t>::max();
        uint32_t deletion_id = std::numeric_limits<uint32_t>::max();
        uint32_t insertion_slot = MeritMetadataCache::kInvalid;
        uint32_t deletion_slot = MeritMetadataCache::kInvalid;
        MeritPendingFlush snap;
        bool committing = false;
    };

    struct MeritPayloadStashEntry
    {
        std::vector<T> coords;
        std::vector<uint32_t> nbrs;
        std::list<uint32_t>::iterator lru_it;
    };

    bool _merit_dyn_enabled = false;
    MeritMetadataCache _merit_mcache;
    tsl::robin_map<uint32_t, MeritSeedDirEntry> _merit_seed_dir;
    std::vector<uint32_t> _merit_dyn_free;
    tsl::robin_set<uint32_t> _merit_dyn_retired;
    tsl::robin_map<uint32_t, uint32_t> _merit_page_to_seed;
    uint64_t _merit_dyn_page_cap = 4096;
    uint64_t _merit_dyn_physical_cap = 0;
    uint32_t _merit_dyn_next_page = 0;
    std::unique_ptr<std::atomic<uint32_t>[]> _merit_dyn_page_readers;
    std::string _merit_dyn_path;
    std::shared_ptr<AlignedFileReader> _merit_dyn_reader;
    int _merit_dyn_write_fd = -1;
    std::fstream _merit_dyn_write_stream;
    std::mutex _merit_dyn_writer_mu;
    uint64_t _merit_dyn_flush_count = 0;
    struct MeritHeapEntry
    {
        float score = 0;
        uint32_t slot_id = MeritMetadataCache::kInvalid;
        uint32_t generation = 0;
    };
    struct MeritMaxHeapCmp
    {
        bool operator()(const MeritHeapEntry &a, const MeritHeapEntry &b) const
        {
            return a.score != b.score ? a.score < b.score : a.slot_id < b.slot_id;
        }
    };
    struct MeritMinHeapCmp
    {
        bool operator()(const MeritHeapEntry &a, const MeritHeapEntry &b) const
        {
            return a.score != b.score ? a.score > b.score : a.slot_id > b.slot_id;
        }
    };
    std::priority_queue<MeritHeapEntry, std::vector<MeritHeapEntry>, MeritMaxHeapCmp> _merit_max_heap;
    std::priority_queue<MeritHeapEntry, std::vector<MeritHeapEntry>, MeritMinHeapCmp> _merit_min_heap;
    std::vector<uint32_t> _merit_heap_generation;
    tsl::robin_map<uint32_t, float> _merit_heap_score;
    tsl::robin_map<uint32_t, MeritNodeState> _merit_node_state;
    tsl::robin_map<uint32_t, MeritReadyPair> _merit_ready_pairs; // insertion slot -> pair
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _merit_member_to_pending;
    tsl::robin_map<uint32_t, uint32_t> _merit_deletion_to_pending;
    uint64_t _merit_reserved_free_pages = 0;
    std::atomic<uint32_t> _merit_pending_pair_count{0};
    std::atomic<bool> _merit_refresh_needed{false};
    std::atomic<bool> _merit_refresh_running{false};
    static constexpr size_t MERIT_PENDING_PAIR_CAP = 4096;
    std::atomic<float> _merit_score_unit{1.0f};
    uint32_t _merit_score_stage = 0;
    uint32_t _merit_queries_in_stage = 0;
    bool _merit_decay_started = false;
    static constexpr uint32_t MERIT_QUERIES_PER_SCORE_STAGE = 1000;
    static constexpr uint32_t MERIT_SCORE_STAGES_PER_CYCLE = 100;
    static constexpr float MERIT_SCORE_CYCLE_MAX = 16.0f;
    uint64_t _merit_pair_refresh = 0;
    uint64_t _merit_heap_writes = 0;
    uint64_t _merit_heap_deletes = 0;
    uint64_t _merit_ncache_write_trig = 0;
    mutable std::mutex _merit_score_mu; // score update/decay/commit ordering
    mutable std::mutex _merit_heap_mu;
    static constexpr uint32_t MERIT_FETCHED_PER_PARENT = 8;
    uint64_t _merit_stash_cap = 16384;
    tsl::robin_map<uint32_t, MeritPayloadStashEntry> _merit_payload_stash;
    std::list<uint32_t> _merit_stash_lru;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> _merit_parent_fetched;
    mutable std::mutex _merit_stash_mu;
    // Reader-writer lock (std::shared_mutex). Lookup/copy take shared; page
    // install takes exclusive. Not a spin/seqlock: robin_map rehash is not
    // safe to read mid-write. 4KB memcpy is done outside this lock.
    mutable std::shared_mutex _merit_dyn_mu;

    bool merit_dyn_is_sector(uint32_t sector) const
    {
        return sector >= MERIT_DYN_SECTOR_BASE;
    }
    std::vector<uint32_t> merit_dyn_now_page_ids(uint32_t seed_id, const MeritMetadataCache::Snapshot &snap) const;
    bool merit_dyn_hotter_mismatch(const std::vector<uint32_t> &page_members,
                                   const std::vector<uint32_t> &now_ids) const;
    void merit_dyn_invalidate_seed_page(uint32_t seed_id);
    uint32_t merit_dyn_alloc_page_unlocked();
    void merit_dyn_maybe_mark_seed(uint32_t node_id);
    void merit_dyn_note_touch(const MeritMetadataCache::TouchResult &tr, SSDQueryScratch<T> *query_scratch);
    void merit_dyn_on_mcache_evict(uint32_t node_id, uint32_t slot_id);
    void merit_dyn_heap_erase_unlocked(uint32_t slot_id);
    void merit_dyn_heap_upsert_unlocked(uint32_t slot_id, float score);
    void merit_dyn_clean_max_heap_unlocked();
    void merit_dyn_clean_min_heap_unlocked();
    void merit_dyn_rebuild_heaps_unlocked();
    void merit_dyn_flush_dirty_heap(SSDQueryScratch<T> *query_scratch);
    MeritNodeState merit_dyn_state_unlocked(uint32_t slot_id) const;
    void merit_dyn_set_state_unlocked(uint32_t node_id, uint32_t slot_id, MeritNodeState st);
    void merit_dyn_clear_ready_pair_unlocked(uint32_t insertion_slot, bool restore_states = true);
    bool merit_dyn_disk_is_full() const;
    bool merit_dyn_build_pending_flush(uint32_t seed_id, const MeritMetadataCache::Snapshot &snap,
                                       MeritPendingFlush &pf) const;
    void merit_dyn_maybe_refresh_pair();
    void merit_dyn_request_refresh();
    bool merit_dyn_commit_ready_pair(uint32_t insertion_slot);
    uint32_t merit_dyn_admit_node(uint32_t node_id, const char *node_disk_buf, QueryStats *stats,
                                  SSDQueryScratch<T> *query_scratch);
    void merit_dyn_note_base_load(uint32_t node_id, uint32_t parent, const char *node_disk_buf,
                                  SSDQueryScratch<T> *query_scratch);
    bool merit_dyn_has_payload(uint32_t node_id) const;
    bool merit_dyn_copy_member_payload(uint32_t node_id, std::vector<T> &coords, std::vector<uint32_t> &nbrs) const;
    bool merit_dyn_extract_overlay_payload(uint32_t node_id, std::vector<T> &coords,
                                           std::vector<uint32_t> &nbrs) const;
    void merit_dyn_stash_evict_unlocked();
    void merit_dyn_release_page_pin(uint32_t sector);
    void merit_dyn_on_ncache_evict(uint32_t node_id, QueryStats *stats);
    void merit_dyn_on_query_end(QueryStats *stats, SSDQueryScratch<T> *query_scratch);
    bool merit_dyn_commit_one(MeritPendingFlush &pf, uint32_t replacement_seed = MERIT_DYN_INVALID_PAGE);

    void merit_get_expand_neighbors(uint32_t expand_id, char *node_disk_buf, const uint32_t *&out_nbrs,
                                    uint64_t &out_nnbrs) const;

    void update_merit_seed_first_lookup_flag();

    bool merit_disk_cache_lookup_hit(uint32_t node_id, const SSDQueryScratch<T> *query_scratch) const;

    bool merit_resolve_disk_cache_loc(uint32_t node_id, const SSDQueryScratch<T> *query_scratch,
                                      MeritDiskLoc &out_loc) const;

    bool merit_loc_in_query_cache(const SSDQueryScratch<T> *query_scratch, uint32_t base_sector,
                                  uint16_t nsectors) const;

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
