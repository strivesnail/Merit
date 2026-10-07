// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <sstream>
#include <thread>

#include "timer.h"
#include "pq.h"
#include "pq_scratch.h"
#include "pq_flash_index.h"
#include "cosine_similarity.h"
#include "defaults.h"
#include "relayout_utils.h"
#include "merit_lock_metrics.h"

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#include <pthread.h>
#include <sched.h>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace
{
inline size_t merit_thread_shard()
{
    static std::atomic<size_t> next{0};
    thread_local size_t id = next.fetch_add(1, std::memory_order_relaxed);
    return id;
}
class MeritShardedCounter
{
  public:
    static constexpr size_t kShards = 64;
    MeritShardedCounter &operator++()
    {
        fetch_add(1);
        return *this;
    }
    MeritShardedCounter &operator++(int)
    {
        fetch_add(1);
        return *this;
    }
    MeritShardedCounter &operator+=(uint64_t v)
    {
        fetch_add(v);
        return *this;
    }
    void fetch_add(uint64_t v, std::memory_order = std::memory_order_relaxed)
    {
        _slots[merit_thread_shard() & (kShards - 1)].v.fetch_add(v, std::memory_order_relaxed);
    }
    MeritShardedCounter &operator=(uint64_t v)
    {
        for (auto &s : _slots)
            s.v.store(0, std::memory_order_relaxed);
        _slots[0].v.store(v, std::memory_order_relaxed);
        return *this;
    }
    uint64_t load(std::memory_order = std::memory_order_relaxed) const
    {
        uint64_t sum = 0;
        for (const auto &s : _slots)
            sum += s.v.load(std::memory_order_relaxed);
        return sum;
    }
    operator uint64_t() const
    {
        return load();
    }

  private:
    struct alignas(64) Slot
    {
        std::atomic<uint64_t> v{0};
    };
    std::array<Slot, kShards> _slots{};
};
struct MeritDynProbe
{
    MeritShardedCounter ncache_hit;
    MeritShardedCounter ncache_miss;
    MeritShardedCounter resolve_call;
    MeritShardedCounter parent_hit;
    MeritShardedCounter parent_absent;
    MeritShardedCounter parent_no_page;
    MeritShardedCounter parent_not_member;
    MeritShardedCounter self_hit;
    MeritShardedCounter map_hit;
    MeritShardedCounter resolve_miss;
    MeritShardedCounter miss_but_in_map;
    MeritShardedCounter qualify_ok;
    MeritShardedCounter qualify_no_ncache;
    MeritShardedCounter qualify_no_meta;
    MeritShardedCounter qualify_heat;
    MeritShardedCounter qualify_smin;
    MeritShardedCounter qualify_half;
    MeritShardedCounter pair_set;
    MeritShardedCounter pair_skip_max_on_disk;
    MeritShardedCounter pair_skip_count;
    MeritShardedCounter write_trig;
    MeritShardedCounter timeout_trig;
    MeritShardedCounter timeout_ok;
    MeritShardedCounter write_ok;
    MeritShardedCounter delete_ok;

    void reset()
    {
        ncache_hit = 0; ncache_miss = 0; resolve_call = 0;
        parent_hit = 0; parent_absent = 0; parent_no_page = 0; parent_not_member = 0;
        self_hit = 0; map_hit = 0; resolve_miss = 0; miss_but_in_map = 0;
        qualify_ok = 0; qualify_no_ncache = 0; qualify_no_meta = 0; qualify_heat = 0; qualify_smin = 0; qualify_half = 0;
        pair_set = 0; pair_skip_max_on_disk = 0; pair_skip_count = 0;
        write_trig = 0; timeout_trig = 0; timeout_ok = 0; write_ok = 0; delete_ok = 0;
    }
};
MeritDynProbe g_dyn_probe;

// MERIT_BG_CPUS="20-23" (or "20,21,22") pins background threads; unset leaves inherited affinity.
inline void merit_pin_background_thread()
{
#ifndef _WINDOWS
    const char *spec = std::getenv("MERIT_BG_CPUS");
    if (spec == nullptr || *spec == '\0')
        return;
    cpu_set_t set;
    CPU_ZERO(&set);
    const char *p = spec;
    while (*p != '\0')
    {
        char *end = nullptr;
        const long lo = std::strtol(p, &end, 10);
        long hi = lo;
        if (end != nullptr && *end == '-')
            hi = std::strtol(end + 1, &end, 10);
        for (long c = lo; c <= hi; ++c)
            if (c >= 0 && c < CPU_SETSIZE)
                CPU_SET(static_cast<int>(c), &set);
        p = end;
        if (*p == ',')
            ++p;
        else if (*p != '\0')
            break;
    }
    if (CPU_COUNT(&set) > 0)
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

inline bool &merit_is_refresh_thread()
{
    thread_local bool is_refresh = false;
    return is_refresh;
}

enum MeritSection : int
{
    kSecBeam = 0,
    kSecNcLookup,
    kSecPrefLookup,
    kSecDcLookup,
    kSecPack,
    kSecMcacheExpand,
    kSecIoPrep,
    kSecPostIo,
    kSecFinalize,
    kSecMaps,
    kSecExpand,
    kSecAdmit,
    kSecNoteBase,
    kSecExactDist,
    kSecCachedLookup,
    kSecQendPins,
    kSecQendDyn,
    kSecQendNcache,
    kSecResult,
    kSecAdmitShould,
    kSecAdmitEvictHook,
    kSecAdmitTry,
    kSecAdmitTail,
    kSecNoteStash,
    kSecNotePatch,
    kSecMcOnExpand,
    kSecMcNoteTouch,
    kSecCachedNbrLoop,
    kSecQendPatch,
    kSecQendRealIo,
    kSecQendHeap,
    kSecQendGate,
    kSecEvictPeek,
    kSecEvictPair,
    kSecEvictHeap,
    kSecEvictRefresh,
    kSecCount
};
static const char *kMeritSectionNames[kSecCount] = {
    "beam_total",   "ncache_lookup", "prefetch_lookup", "dcache_lookup", "pack",          "mcache_expand",
    "io_prep",      "post_io",       "finalize",        "hop_maps",      "expand_total",  "ncache_admit",
    "note_base",    "exact_dist",    "cached_coord",    "qend_pins",     "qend_dyn",      "qend_ncache",
    "result_sort",  "admit_should",  "admit_evict_hook", "admit_try",    "admit_tail",    "note_stash",
    "note_patch",   "mc_on_expand",  "mc_note_touch",   "cached_nbr_loop", "qend_patch",  "qend_real_io",
    "qend_heap",    "qend_gate",     "evict_peek",      "evict_pair",   "evict_heap",    "evict_refresh"};
struct MeritSectionAcc
{
    uint64_t ns[kSecCount] = {};
    uint64_t queries = 0;
};
inline bool merit_section_on()
{
    static const bool on = [] {
        const char *v = std::getenv("MERIT_SECTION_PROFILE");
        return v != nullptr && std::strcmp(v, "0") != 0;
    }();
    return on;
}
inline std::mutex &merit_section_mu()
{
    static std::mutex mu;
    return mu;
}
inline std::vector<MeritSectionAcc *> &merit_section_registry()
{
    static std::vector<MeritSectionAcc *> reg;
    return reg;
}
inline MeritSectionAcc &merit_section_acc()
{
    thread_local MeritSectionAcc *acc = [] {
        auto *a = new MeritSectionAcc();
        std::lock_guard<std::mutex> lock(merit_section_mu());
        merit_section_registry().push_back(a);
        return a;
    }();
    return *acc;
}
inline uint64_t merit_section_now()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
struct MeritSectionTimer
{
    int id;
    uint64_t t0;
    explicit MeritSectionTimer(int i) : id(i), t0(merit_section_on() ? merit_section_now() : 0)
    {
    }
    ~MeritSectionTimer()
    {
        if (t0 != 0)
            merit_section_acc().ns[id] += merit_section_now() - t0;
    }
};
#define MERIT_SEC_CAT2(a, b) a##b
#define MERIT_SEC_CAT(a, b) MERIT_SEC_CAT2(a, b)
#define MERIT_SECTION(id) MeritSectionTimer MERIT_SEC_CAT(merit_sec_, __LINE__)(id)

inline uint64_t merit_disk_append_cap_for_full_coverage(const std::vector<uint64_t> &node_expand,
                                                        const std::unordered_set<uint32_t> &exclude_ids,
                                                        const std::vector<uint32_t> &node_list, uint64_t max_nodes)
{
    std::unordered_set<uint32_t> in_list(node_list.begin(), node_list.end());
    uint64_t missing = 0;
    for (uint32_t id = 0; id < static_cast<uint32_t>(node_expand.size()); ++id)
    {
        if (exclude_ids.find(id) != exclude_ids.end())
            continue;
        if (node_expand[id] == 0)
            continue;
        if (in_list.find(id) != in_list.end())
            continue;
        missing++;
    }
    if (missing == 0)
        return max_nodes;
    return node_list.size() + missing;
}

inline void write_merit_dc_seed_tail(char *slot_dst, uint64_t max_node_len, uint64_t disk_bytes_per_point,
                                     uint32_t nnbrs, uint8_t total_pages)
{
    const uint64_t used = disk_bytes_per_point + static_cast<uint64_t>(nnbrs + 1) * sizeof(uint32_t);
    if (used + 4 > max_node_len)
        return;
    slot_dst[used] = static_cast<char>(diskann::MERIT_DC_SEED_MAGIC);
    slot_dst[used + 1] = static_cast<char>(total_pages & diskann::MERIT_DC_FLAG_TOTAL_PAGES_MASK);
    slot_dst[used + 2] = 0;
    slot_dst[used + 3] = 0;
}

inline void save_seed_pages_sidecar(const std::string &path, const std::vector<std::pair<uint32_t, uint8_t>> &entries)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const uint32_t n = static_cast<uint32_t>(entries.size());
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const auto &entry : entries)
    {
        out.write(reinterpret_cast<const char *>(&entry.first), sizeof(entry.first));
        out.write(reinterpret_cast<const char *>(&entry.second), sizeof(entry.second));
    }
}

// Sector cache stores data in robin_map-owned arrays; map insert/rehash can invalidate
// pointers into those arrays. Active node buffers must live in per-query sector_scratch.
template <typename PendingT>
inline void merit_fallback_unserved_to_base(const std::vector<uint32_t> &merit_ids,
                                            const std::vector<PendingT> &pending,
                                            const std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout,
                                            std::vector<uint32_t> &frontier)
{
    tsl::robin_set<uint32_t> served;
    served.reserve(pending.size());
    for (const auto &p : pending)
    {
        if (p.sec_buf != nullptr)
            served.insert(p.id);
    }
    for (const auto &kv : disk_fanout)
    {
        for (size_t idx : kv.second)
        {
            if (idx < pending.size())
                served.insert(pending[idx].id);
        }
    }
    for (uint32_t id : merit_ids)
    {
        if (served.find(id) == served.end())
            frontier.push_back(id);
    }
}

inline char *copy_to_sector_scratch(char *sector_scratch, uint64_t &sector_scratch_idx, const char *src,
                                    size_t len)
{
    char *buf = sector_scratch + sector_scratch_idx * diskann::defaults::SECTOR_LEN;
    sector_scratch_idx++;
    memcpy(buf, src, len);
    return buf;
}

inline void reorder_frontier_nhoods_by_expand_order(
    const std::vector<uint32_t> &expand_order, std::vector<std::pair<uint32_t, char *>> &frontier_nhoods)
{
    if (expand_order.empty() || frontier_nhoods.size() <= 1)
        return;
    std::unordered_map<uint32_t, char *> buf_by_id;
    buf_by_id.reserve(frontier_nhoods.size());
    for (const auto &fn : frontier_nhoods)
        buf_by_id[fn.first] = fn.second;
    std::vector<std::pair<uint32_t, char *>> ordered;
    ordered.reserve(frontier_nhoods.size());
    for (uint32_t id : expand_order)
    {
        const auto it = buf_by_id.find(id);
        if (it != buf_by_id.end())
            ordered.emplace_back(id, it->second);
    }
    if (ordered.size() == frontier_nhoods.size())
        frontier_nhoods = std::move(ordered);
}

inline int load_seed_pages_sidecar(const std::string &path, std::vector<std::pair<uint32_t, uint8_t>> &entries)
{
    entries.clear();
    if (!file_exists(path))
        return 0;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        return -1;
    uint32_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in)
        return -1;
    entries.resize(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        in.read(reinterpret_cast<char *>(&entries[i].first), sizeof(entries[i].first));
        in.read(reinterpret_cast<char *>(&entries[i].second), sizeof(entries[i].second));
        if (!in)
            return -1;
    }
    return 0;
}

inline void save_seed_nbrs_sidecar(const std::string &path,
                                   const tsl::robin_map<uint32_t, std::vector<uint32_t>> &seed_nbrs)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const uint32_t n = static_cast<uint32_t>(seed_nbrs.size());
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const auto &kv : seed_nbrs)
    {
        const uint32_t seed = kv.first;
        const uint32_t cnt = static_cast<uint32_t>(kv.second.size());
        out.write(reinterpret_cast<const char *>(&seed), sizeof(seed));
        out.write(reinterpret_cast<const char *>(&cnt), sizeof(cnt));
        if (cnt > 0)
            out.write(reinterpret_cast<const char *>(kv.second.data()), cnt * sizeof(uint32_t));
    }
}

inline int load_seed_nbrs_sidecar(const std::string &path, tsl::robin_map<uint32_t, std::vector<uint32_t>> &seed_nbrs)
{
    seed_nbrs.clear();
    if (!file_exists(path))
        return 0;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        return -1;
    uint32_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in)
        return -1;
    for (uint32_t i = 0; i < n; ++i)
    {
        uint32_t seed = 0;
        uint32_t cnt = 0;
        in.read(reinterpret_cast<char *>(&seed), sizeof(seed));
        in.read(reinterpret_cast<char *>(&cnt), sizeof(cnt));
        if (!in)
            return -1;
        std::vector<uint32_t> nbrs(cnt);
        if (cnt > 0)
        {
            in.read(reinterpret_cast<char *>(nbrs.data()), cnt * sizeof(uint32_t));
            if (!in)
                return -1;
        }
        seed_nbrs[seed] = std::move(nbrs);
    }
    return 0;
}

template <typename MeritDiskLocT>
inline void save_seed_member_locs_sidecar(
    const std::string &path,
    const tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, MeritDiskLocT>> &seed_member_loc)
{
    uint32_t count = 0;
    for (const auto &kv : seed_member_loc)
        count += static_cast<uint32_t>(kv.second.size());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const auto &kv : seed_member_loc)
    {
        const uint32_t seed_id = kv.first;
        for (const auto &member : kv.second)
        {
            const uint32_t member_id = member.first;
            const MeritDiskLocT &loc = member.second;
            out.write(reinterpret_cast<const char *>(&seed_id), sizeof(seed_id));
            out.write(reinterpret_cast<const char *>(&member_id), sizeof(member_id));
            out.write(reinterpret_cast<const char *>(&loc.sector), sizeof(loc.sector));
            out.write(reinterpret_cast<const char *>(&loc.slot), sizeof(loc.slot));
            out.write(reinterpret_cast<const char *>(&loc.nsectors), sizeof(loc.nsectors));
        }
    }
}

template <typename MeritDiskLocT>
inline int load_seed_member_locs_sidecar(
    const std::string &path, tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, MeritDiskLocT>> &seed_member_loc,
    tsl::robin_map<uint32_t, MeritDiskLocT> &seed_canonical_loc)
{
    seed_member_loc.clear();
    seed_canonical_loc.clear();
    if (!file_exists(path))
        return 0;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        return -1;
    uint32_t count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (!in)
        return -1;
    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t seed_id = 0;
        uint32_t member_id = 0;
        MeritDiskLocT loc;
        in.read(reinterpret_cast<char *>(&seed_id), sizeof(seed_id));
        in.read(reinterpret_cast<char *>(&member_id), sizeof(member_id));
        in.read(reinterpret_cast<char *>(&loc.sector), sizeof(loc.sector));
        in.read(reinterpret_cast<char *>(&loc.slot), sizeof(loc.slot));
        in.read(reinterpret_cast<char *>(&loc.nsectors), sizeof(loc.nsectors));
        if (!in)
            return -1;
        seed_member_loc[seed_id][member_id] = loc;
        if (seed_id == member_id)
            seed_canonical_loc[seed_id] = loc;
    }
    return 0;
}

static std::vector<std::pair<uint32_t, uint32_t>> load_merit_copack_pairs()
{
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    const char *path = std::getenv("MERIT_COPACK_PAIRS_FILE");
    if (path == nullptr || path[0] == '\0')
        return pairs;
    std::ifstream in(path);
    if (!in.is_open())
    {
        diskann::cerr << "MERIT copack: failed to open " << path << std::endl;
        return pairs;
    }
    uint32_t a = 0;
    uint32_t b = 0;
    while (in >> a >> b)
    {
        if (a > b)
            std::swap(a, b);
        pairs.emplace_back(a, b);
    }
    diskann::cout << "MERIT copack: loaded " << pairs.size() << " seed pairs from " << path << std::endl;
    return pairs;
}

struct MeritHopIoProbe
{
    std::atomic<uint64_t> batches[2]{};
    std::atomic<uint64_t> batch_ns[2]{};
    std::atomic<uint64_t> reads[2]{};
    std::atomic<uint64_t> bytes[2]{};
    std::atomic<uint64_t> multi_reads{0};
    std::atomic<uint64_t> multi_bytes{0};
};
inline MeritHopIoProbe &merit_hop_io_probe()
{
    static MeritHopIoProbe probe;
    return probe;
}

inline void merit_record_hop_io(const std::vector<AlignedRead> &merit_reqs, const std::vector<AlignedRead> &base_reqs,
                                float us)
{
    uint64_t bytes = 0, multi = 0, multi_bytes = 0;
    for (const auto &r : merit_reqs)
    {
        bytes += r.len;
        if (r.len > diskann::defaults::SECTOR_LEN)
        {
            ++multi;
            multi_bytes += r.len;
        }
    }
    for (const auto &r : base_reqs)
        bytes += r.len;
    auto &p = merit_hop_io_probe();
    const int k = multi > 0 ? 1 : 0;
    p.batches[k].fetch_add(1, std::memory_order_relaxed);
    p.batch_ns[k].fetch_add(static_cast<uint64_t>(us * 1000.0f), std::memory_order_relaxed);
    p.reads[k].fetch_add(merit_reqs.size() + base_reqs.size(), std::memory_order_relaxed);
    p.bytes[k].fetch_add(bytes, std::memory_order_relaxed);
    p.multi_reads.fetch_add(multi, std::memory_order_relaxed);
    p.multi_bytes.fetch_add(multi_bytes, std::memory_order_relaxed);
}

inline float issue_merit_and_base_disk_reads(std::shared_ptr<AlignedFileReader> &base_reader,
                                            std::shared_ptr<AlignedFileReader> &merit_reader, IOContext &base_ctx,
                                            std::vector<AlignedRead> &merit_reqs, std::vector<AlignedRead> &base_reqs,
                                            diskann::Timer &io_timer, diskann::QueryStats *stats)
{
    if (merit_reqs.empty() && base_reqs.empty())
        return 0.f;

    io_timer.reset();
#ifndef _WINDOWS
    if (!merit_reqs.empty() && !base_reqs.empty() && merit_reader != nullptr)
    {
        auto *base_linux = dynamic_cast<LinuxAlignedFileReader *>(base_reader.get());
        auto *merit_linux = dynamic_cast<LinuxAlignedFileReader *>(merit_reader.get());
        if (base_linux != nullptr && merit_linux != nullptr)
        {
            std::vector<FdAlignedRead> multi;
            multi.reserve(merit_reqs.size() + base_reqs.size());
            const int merit_fd = merit_linux->get_file_desc();
            const int base_fd = base_linux->get_file_desc();
            for (auto &r : merit_reqs)
                multi.emplace_back(merit_fd, r.offset, r.len, r.buf);
            for (auto &r : base_reqs)
                multi.emplace_back(base_fd, r.offset, r.len, r.buf);
            base_reader->read_multi(multi, base_ctx);
            const float us = (float)io_timer.elapsed();
            if (stats != nullptr)
                stats->io_us += us;
            merit_record_hop_io(merit_reqs, base_reqs, us);
            return us;
        }
    }
#endif
    if (!merit_reqs.empty() && merit_reader != nullptr)
        merit_reader->read(merit_reqs, merit_reader->get_ctx());
    if (!base_reqs.empty())
        base_reader->read(base_reqs, base_ctx);
    const float us = (float)io_timer.elapsed();
    if (stats != nullptr)
        stats->io_us += us;
    merit_record_hop_io(merit_reqs, base_reqs, us);
    return us;
}
} // namespace

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) (((uint64_t)(id)) / _nvecs_per_sector + _reorder_data_start_sector)

namespace
{
inline void record_disk_read_batch_stats(diskann::QueryStats *stats, float batch_us,
                                         const std::vector<uint64_t> &miss_sector_ids, uint64_t max_graph_sector)
{
    const size_t bs = miss_sector_ids.size();
    stats->n_disk_read_batches++;
    stats->sum_batch_io_us += batch_us;
    if (bs == 1)
    {
        stats->n_batches_size1++;
        stats->sum_batch_us_size1 += batch_us;
    }
    else if (bs == 2)
    {
        stats->n_batches_size2++;
        stats->sum_batch_us_size2 += batch_us;
    }
    else
    {
        stats->n_batches_size_other++;
        stats->sum_batch_us_size_other += batch_us;
    }
    for (uint64_t sid : miss_sector_ids)
    {
        unsigned b = 0;
        if (max_graph_sector > 0)
        {
            b = (unsigned)((sid * diskann::QueryStats::DISK_SECTOR_BUCKETS) / max_graph_sector);
            if (b >= diskann::QueryStats::DISK_SECTOR_BUCKETS)
                b = (unsigned)(diskann::QueryStats::DISK_SECTOR_BUCKETS - 1);
        }
        stats->disk_sector_bucket[b]++;
    }
    stats->n_disk_sector_bucket_samples += (unsigned)miss_sector_ids.size();
}
} // namespace

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) ((((uint64_t)(id)) % _nvecs_per_sector) * _data_dim * sizeof(float))

namespace diskann
{

template <typename T, typename LabelT>
PQFlashIndex<T, LabelT>::PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader, diskann::Metric m)
    : reader(fileReader), metric(m), _thread_data(nullptr)
{
    for (auto &shard : _merit_dyn_loc_shards)
        shard = std::make_unique<MeritDynLocShard>();
    for (auto &shard : _merit_pending_member_shards)
        shard = std::make_unique<MeritPendingMemberShard>();
    diskann::Metric metric_to_invoke = m;
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT)
    {
        if (std::is_floating_point<T>::value)
        {
            diskann::cout << "Since data is floating point, we assume that it has been appropriately pre-processed "
                             "(normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we "
                             "shall invoke an l2 distance function."
                          << std::endl;
            metric_to_invoke = diskann::Metric::L2;
        }
        else
        {
            diskann::cerr << "WARNING: Cannot normalize integral data types."
                          << " This may result in erroneous results or poor recall."
                          << " Consider using L2 distance with integral data types." << std::endl;
        }
    }

    this->_dist_cmp.reset(diskann::get_distance_function<T>(metric_to_invoke));
    this->_dist_cmp_float.reset(diskann::get_distance_function<float>(metric_to_invoke));
}

template <typename T, typename LabelT> PQFlashIndex<T, LabelT>::~PQFlashIndex()
{
    stop_merit_dyn_patch_worker();
    stop_merit_dyn_commit_workers();
    stop_merit_dyn_writer();
#ifndef EXEC_ENV_OLS
    if (data != nullptr)
    {
        delete[] data;
    }
#endif

    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    // delete backing bufs for nhood and coord cache
    if (_nhood_cache_buf != nullptr)
    {
        delete[] _nhood_cache_buf;
        diskann::aligned_free(_coord_cache_buf);
    }

    if (_merit_disk_reader)
    {
        _merit_disk_reader->close();
        _merit_disk_reader.reset();
    }
    if (_merit_dyn_reader)
    {
        _merit_dyn_reader->close();
        _merit_dyn_reader.reset();
    }
    if (_merit_dyn_write_stream.is_open())
        _merit_dyn_write_stream.close();
#ifndef _WINDOWS
    if (_merit_dyn_write_fd >= 0)
    {
        ::close(_merit_dyn_write_fd);
        _merit_dyn_write_fd = -1;
    }
#endif
#ifndef _WINDOWS
    if (_merit_dyn_write_fd >= 0)
    {
        ::close(_merit_dyn_write_fd);
        _merit_dyn_write_fd = -1;
    }
#endif
    _merit_dc_map.clear();

    if (_load_flag)
    {
        diskann::cout << "Clearing scratch" << std::endl;
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        manager.destroy();
        this->reader->deregister_all_threads();
        reader->close();
    }
    if (_pts_to_label_offsets != nullptr)
    {
        delete[] _pts_to_label_offsets;
    }
    if (_pts_to_label_counts != nullptr)
    {
        delete[] _pts_to_label_counts;
    }
    if (_pts_to_labels != nullptr)
    {
        delete[] _pts_to_labels;
    }
    if (_medoids != nullptr)
    {
        delete[] _medoids;
    }
}

template <typename T, typename LabelT> inline uint64_t PQFlashIndex<T, LabelT>::get_node_sector(uint64_t node_id)
{
    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                       : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));
}

template <typename T, typename LabelT>
inline char *PQFlashIndex<T, LabelT>::offset_to_node(char *sector_buf, uint64_t node_id)
{
    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);
}

template <typename T, typename LabelT> inline uint32_t *PQFlashIndex<T, LabelT>::offset_to_node_nhood(char *node_buf)
{
    return (unsigned *)(node_buf + _disk_bytes_per_point);
}

template <typename T, typename LabelT> inline T *PQFlashIndex<T, LabelT>::offset_to_node_coords(char *node_buf)
{
    return (T *)(node_buf);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::setup_thread_data(uint64_t nthreads, uint64_t visited_reserve)
{
    diskann::cout << "Setting up thread-specific contexts for nthreads: " << nthreads << std::endl;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int)nthreads)
    for (int64_t thread = 0; thread < (int64_t)nthreads; thread++)
    {
#pragma omp critical
        {
            SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, visited_reserve);
            this->reader->register_thread();
            data->ctx = this->reader->get_ctx();
            this->_thread_data.push(data);
        }
    }
    _load_flag = true;
}

template <typename T, typename LabelT>
std::vector<bool> PQFlashIndex<T, LabelT>::read_nodes(const std::vector<uint32_t> &node_ids,
                                                      std::vector<T *> &coord_buffers,
                                                      std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers)
{
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(node_ids.size(), true);

    char *buf = nullptr;
    auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    alloc_aligned((void **)&buf, node_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);

    // create read requests
    for (size_t i = 0; i < node_ids.size(); ++i)
    {
        auto node_id = node_ids[i];

        AlignedRead read;
        read.len = num_sectors * defaults::SECTOR_LEN;
        read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
        read.offset = get_node_sector(node_id) * defaults::SECTOR_LEN;
        read_reqs.push_back(read);
    }

    // borrow thread data and issue reads
    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;
    reader->read(read_reqs, ctx);

    // copy reads into buffers
    for (uint32_t i = 0; i < read_reqs.size(); i++)
    {
#if defined(_WINDOWS) && defined(USE_BING_INFRA) // this block is to handle failed reads in
                                                 // production settings
        if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS)
        {
            retval[i] = false;
            continue;
        }
#endif

        char *node_buf = offset_to_node((char *)read_reqs[i].buf, node_ids[i]);

        if (coord_buffers[i] != nullptr)
        {
            T *node_coords = offset_to_node_coords(node_buf);
            memcpy(coord_buffers[i], node_coords, _disk_bytes_per_point);
        }

        if (nbr_buffers[i].second != nullptr)
        {
            uint32_t *node_nhood = offset_to_node_nhood(node_buf);
            auto num_nbrs = *node_nhood;
            nbr_buffers[i].first = num_nbrs;
            memcpy(nbr_buffers[i].second, node_nhood + 1, num_nbrs * sizeof(uint32_t));
        }
    }

    aligned_free(buf);

    return retval;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::load_cache_list(std::vector<uint32_t> &node_list)
{
    diskann::cout << "Loading the cache list into memory.." << std::flush;
    size_t num_cached_nodes = node_list.size();

    // Allocate space for neighborhood cache
    _nhood_cache_buf = new uint32_t[num_cached_nodes * (_max_degree + 1)];
    memset(_nhood_cache_buf, 0, num_cached_nodes * (_max_degree + 1));

    // Allocate space for coordinate cache
    size_t coord_cache_buf_len = num_cached_nodes * _aligned_dim;
    diskann::alloc_aligned((void **)&_coord_cache_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(_coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);

        // Copy offset into buffers to read into
        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            nodes_to_read.push_back(node_list[node_idx]);
            coord_buffers.push_back(_coord_cache_buf + node_idx * _aligned_dim);
            nbr_buffers.emplace_back(0, _nhood_cache_buf + node_idx * (_max_degree + 1));
        }

        // issue the reads
        auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

        // check for success and insert into the cache.
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (read_status[i] == true)
            {
                _coord_cache.insert(std::make_pair(nodes_to_read[i], coord_buffers[i]));
                _nhood_cache.insert(std::make_pair(nodes_to_read[i], nbr_buffers[i]));
            }
        }
    }
    diskann::cout << "..done." << std::endl;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                      uint64_t l_search, uint64_t beamwidth,
                                                                      uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#else
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                      uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                      uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#endif
    if (num_nodes_to_cache >= this->_num_points)
    {
        // for small num_points and big num_nodes_to_cache, use below way to get the node_list quickly
        node_list.resize(this->_num_points);
        for (uint32_t i = 0; i < this->_num_points; ++i)
        {
            node_list[i] = i;
        }
        return;
    }

    this->_count_visited_nodes = true;
    this->_node_visit_counter.clear();
    this->_node_visit_counter.resize(this->_num_points);
    for (uint32_t i = 0; i < _node_visit_counter.size(); i++)
    {
        this->_node_visit_counter[i].first = i;
        this->_node_visit_counter[i].second = 0;
    }

    uint64_t sample_num, sample_dim, sample_aligned_dim;
    T *samples;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(sample_bin))
    {
        diskann::load_aligned_bin<T>(files, sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#else
    if (file_exists(sample_bin))
    {
        diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#endif
    else
    {
        diskann::cerr << "Sample bin file not found. Not generating cache." << std::endl;
        return;
    }

    std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
    std::vector<float> tmp_result_dists(sample_num, 0);

    bool filtered_search = false;
    std::vector<LabelT> random_query_filters(sample_num);
    if (_filter_to_medoid_ids.size() != 0)
    {
        filtered_search = true;
        generate_random_labels(random_query_filters, (uint32_t)sample_num, nthreads);
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t)sample_num; i++)
    {
        auto &label_for_search = random_query_filters[i];
        // run a search on the sample query with a random label (sampled from base label distribution), and it will
        // concurrently update the node_visit_counter to track most visited nodes. The last false is to not use the
        // "use_reorder_data" option which enables a final reranking if the disk index itself contains only PQ data.
        cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search, tmp_result_ids_64.data() + i,
                           tmp_result_dists.data() + i, beamwidth, filtered_search, label_for_search, false);
    }

    std::sort(this->_node_visit_counter.begin(), _node_visit_counter.end(),
              [](std::pair<uint32_t, uint32_t> &left, std::pair<uint32_t, uint32_t> &right) {
                  return left.second > right.second;
              });
    node_list.clear();
    node_list.shrink_to_fit();
    num_nodes_to_cache = std::min(num_nodes_to_cache, this->_node_visit_counter.size());
    node_list.reserve(num_nodes_to_cache);
    for (uint64_t i = 0; i < num_nodes_to_cache; i++)
    {
        node_list.push_back(this->_node_visit_counter[i].first);
    }
    this->_count_visited_nodes = false;

    diskann::aligned_free(samples);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                               const bool shuffle, double max_fraction)
{
    std::random_device rng;
    std::mt19937 urng(rng());

    tsl::robin_set<uint32_t> node_set;

    max_fraction = std::max(0.0, std::min(1.0, max_fraction));
    const uint64_t max_nodes = (uint64_t)(std::round(this->_num_points * max_fraction));
    if (num_nodes_to_cache > max_nodes)
    {
        diskann::cout << "Reducing nodes to cache from: " << num_nodes_to_cache << " to: " << max_nodes << " ("
                      << (max_fraction * 100.0) << " percent of total nodes:" << this->_num_points << ")"
                      << std::endl;
        num_nodes_to_cache = max_nodes == 0 ? 1 : max_nodes;
    }
    diskann::cout << "Caching " << num_nodes_to_cache << "..." << std::endl;

    std::unique_ptr<tsl::robin_set<uint32_t>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<uint32_t>>();
    prev_level = std::make_unique<tsl::robin_set<uint32_t>>();

    for (uint64_t miter = 0; miter < _num_medoids && cur_level->size() < num_nodes_to_cache; miter++)
    {
        cur_level->insert(_medoids[miter]);
    }

    if ((_filter_to_medoid_ids.size() > 0) && (cur_level->size() < num_nodes_to_cache))
    {
        for (auto &x : _filter_to_medoid_ids)
        {
            for (auto &y : x.second)
            {
                cur_level->insert(y);
                if (cur_level->size() == num_nodes_to_cache)
                    break;
            }
            if (cur_level->size() == num_nodes_to_cache)
                break;
        }
    }

    uint64_t lvl = 1;
    uint64_t prev_node_set_size = 0;
    while ((node_set.size() + cur_level->size() < num_nodes_to_cache) && cur_level->size() != 0)
    {
        // swap prev_level and cur_level
        std::swap(prev_level, cur_level);
        // clear cur_level
        cur_level->clear();

        std::vector<uint32_t> nodes_to_expand;

        for (const uint32_t &id : *prev_level)
        {
            if (node_set.find(id) != node_set.end())
            {
                continue;
            }
            node_set.insert(id);
            nodes_to_expand.push_back(id);
        }

        if (shuffle)
            std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);
        else
            std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

        diskann::cout << "Level: " << lvl << std::flush;
        bool finish_flag = false;

        uint64_t BLOCK_SIZE = 1024;
        uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
        for (size_t block = 0; block < nblocks && !finish_flag; block++)
        {
            diskann::cout << "." << std::flush;
            size_t start = block * BLOCK_SIZE;
            size_t end = (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());

            std::vector<uint32_t> nodes_to_read;
            std::vector<T *> coord_buffers(end - start, nullptr);
            std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;

            for (size_t cur_pt = start; cur_pt < end; cur_pt++)
            {
                nodes_to_read.push_back(nodes_to_expand[cur_pt]);
                nbr_buffers.emplace_back(0, new uint32_t[_max_degree + 1]);
            }

            // issue read requests
            auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

            // process each nhood buf
            for (uint32_t i = 0; i < read_status.size(); i++)
            {
                if (read_status[i] == false)
                {
                    continue;
                }
                else
                {
                    uint32_t nnbrs = nbr_buffers[i].first;
                    uint32_t *nbrs = nbr_buffers[i].second;

                    // explore next level
                    for (uint32_t j = 0; j < nnbrs && !finish_flag; j++)
                    {
                        if (node_set.find(nbrs[j]) == node_set.end())
                        {
                            cur_level->insert(nbrs[j]);
                        }
                        if (cur_level->size() + node_set.size() >= num_nodes_to_cache)
                        {
                            finish_flag = true;
                        }
                    }
                }
                delete[] nbr_buffers[i].second;
            }
        }

        diskann::cout << ". #nodes: " << node_set.size() - prev_node_set_size
                      << ", #nodes thus far: " << node_set.size() << std::endl;
        prev_node_set_size = node_set.size();
        lvl++;
    }

    assert(node_set.size() + cur_level->size() == num_nodes_to_cache || cur_level->size() == 0);

    node_list.clear();
    node_list.reserve(node_set.size() + cur_level->size());
    for (auto node : node_set)
        node_list.push_back(node);
    for (auto node : *cur_level)
        node_list.push_back(node);

    diskann::cout << "Level: " << lvl << std::flush;
    diskann::cout << ". #nodes: " << node_list.size() - prev_node_set_size << ", #nodes thus far: " << node_list.size()
                  << std::endl;
    diskann::cout << "done" << std::endl;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::use_medoids_data_as_centroids()
{
    if (_centroid_data != nullptr)
        aligned_free(_centroid_data);
    alloc_aligned(((void **)&_centroid_data), _num_medoids * _aligned_dim * sizeof(float), 32);
    std::memset(_centroid_data, 0, _num_medoids * _aligned_dim * sizeof(float));

    diskann::cout << "Loading centroid data from medoids vector data of " << _num_medoids << " medoid(s)" << std::endl;

    std::vector<uint32_t> nodes_to_read;
    std::vector<T *> medoid_bufs;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs;

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        nodes_to_read.push_back(_medoids[cur_m]);
        medoid_bufs.push_back(new T[_data_dim]);
        nbr_bufs.emplace_back(0, nullptr);
    }

    auto read_status = read_nodes(nodes_to_read, medoid_bufs, nbr_bufs);

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        if (read_status[cur_m] == true)
        {
            if (!_use_disk_index_pq)
            {
                for (uint32_t i = 0; i < _data_dim; i++)
                    _centroid_data[cur_m * _aligned_dim + i] = medoid_bufs[cur_m][i];
            }
            else
            {
                _disk_pq_table.inflate_vector((uint8_t *)medoid_bufs[cur_m], (_centroid_data + cur_m * _aligned_dim));
            }
        }
        else
        {
            throw ANNException("Unable to read a medoid", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        delete[] medoid_bufs[cur_m];
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_random_labels(std::vector<LabelT> &labels, const uint32_t num_labels,
                                                     const uint32_t nthreads)
{
    std::random_device rd;
    labels.clear();
    labels.resize(num_labels);

    uint64_t num_total_labels = _pts_to_label_offsets[_num_points - 1] + _pts_to_label_counts[_num_points - 1];
    std::mt19937 gen(rd());
    if (num_total_labels == 0)
    {
        std::stringstream stream;
        stream << "No labels found in data. Not sampling random labels ";
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::uniform_int_distribution<uint64_t> dis(0, num_total_labels - 1);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < num_labels; i++)
    {
        uint64_t rnd_loc = dis(gen);
        labels[i] = (LabelT)_pts_to_labels[rnd_loc];
    }
}

template <typename T, typename LabelT>
std::unordered_map<std::string, LabelT> PQFlashIndex<T, LabelT>::load_label_map(std::basic_istream<char> &map_reader)
{
    std::unordered_map<std::string, LabelT> string_to_int_mp;
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (LabelT)std::stoul(token);
        string_to_int_mp[label_str] = token_as_num;
    }
    return string_to_int_mp;
}

template <typename T, typename LabelT>
LabelT PQFlashIndex<T, LabelT>::get_converted_label(const std::string &filter_label)
{
    if (_label_map.find(filter_label) != _label_map.end())
    {
        return _label_map[filter_label];
    }
    if (_use_universal_label)
    {
        return _universal_filter_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::reset_stream_for_reading(std::basic_istream<char> &infile)
{
    infile.clear();
    infile.seekg(0);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::get_label_file_metadata(const std::string &fileContent, uint32_t &num_pts,
                                                      uint32_t &num_total_labels)
{
    num_pts = 0;
    num_total_labels = 0;

    size_t file_size = fileContent.length();

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = fileContent.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = fileContent.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label
            {
                next_lbl_pos = next_pos;
            }

            num_total_labels++;

            lbl_pos = next_lbl_pos + 1;
        }

        cur_pos = next_pos + 1;

        num_pts++;
    }

    diskann::cout << "Labels file metadata: num_points: " << num_pts << ", #total_labels: " << num_total_labels
                  << std::endl;
}

template <typename T, typename LabelT>
inline bool PQFlashIndex<T, LabelT>::point_has_label(uint32_t point_id, LabelT label_id)
{
    uint32_t start_vec = _pts_to_label_offsets[point_id];
    uint32_t num_lbls = _pts_to_label_counts[point_id];
    bool ret_val = false;
    for (uint32_t i = 0; i < num_lbls; i++)
    {
        if (_pts_to_labels[start_vec + i] == label_id)
        {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::parse_label_file(std::basic_istream<char> &infile, size_t &num_points_labels)
{
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();

    std::string buffer(file_size, ' ');

    infile.seekg(0, std::ios::beg);
    infile.read(&buffer[0], file_size);

    std::string line;
    uint32_t line_cnt = 0;

    uint32_t num_pts_in_label_file;
    uint32_t num_total_labels;
    get_label_file_metadata(buffer, num_pts_in_label_file, num_total_labels);

    _pts_to_label_offsets = new uint32_t[num_pts_in_label_file];
    _pts_to_label_counts = new uint32_t[num_pts_in_label_file];
    _pts_to_labels = new LabelT[num_total_labels];
    uint32_t labels_seen_so_far = 0;

    std::string label_str;
    size_t cur_pos = 0;
    size_t next_pos = 0;
    while (cur_pos < file_size && cur_pos != std::string::npos)
    {
        next_pos = buffer.find('\n', cur_pos);
        if (next_pos == std::string::npos)
        {
            break;
        }

        _pts_to_label_offsets[line_cnt] = labels_seen_so_far;
        uint32_t &num_lbls_in_cur_pt = _pts_to_label_counts[line_cnt];
        num_lbls_in_cur_pt = 0;

        size_t lbl_pos = cur_pos;
        size_t next_lbl_pos = 0;
        while (lbl_pos < next_pos && lbl_pos != std::string::npos)
        {
            next_lbl_pos = buffer.find(',', lbl_pos);
            if (next_lbl_pos == std::string::npos) // the last label in the whole file
            {
                next_lbl_pos = next_pos;
            }

            if (next_lbl_pos > next_pos) // the last label in one line, just read to the end
            {
                next_lbl_pos = next_pos;
            }

            label_str.assign(buffer.c_str() + lbl_pos, next_lbl_pos - lbl_pos);
            if (label_str[label_str.length() - 1] == '\t') // '\t' won't exist in label file?
            {
                label_str.erase(label_str.length() - 1);
            }

            LabelT token_as_num = (LabelT)std::stoul(label_str);
            _pts_to_labels[labels_seen_so_far++] = (LabelT)token_as_num;
            num_lbls_in_cur_pt++;

            // move to next label
            lbl_pos = next_lbl_pos + 1;
        }

        // move to next line
        cur_pos = next_pos + 1;

        if (num_lbls_in_cur_pt == 0)
        {
            diskann::cout << "No label found for point " << line_cnt << std::endl;
            exit(-1);
        }

        line_cnt++;
    }

    num_points_labels = line_cnt;
    reset_stream_for_reading(infile);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::set_universal_label(const LabelT &label)
{
    _use_universal_label = true;
    _universal_filter_label = label;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load(MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix)
{
#else
template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::load(uint32_t num_threads, const char *index_prefix)
{
#endif
    std::string pq_table_bin = std::string(index_prefix) + "_pq_pivots.bin";
    std::string pq_compressed_vectors = std::string(index_prefix) + "_pq_compressed.bin";
    std::string _disk_index_file = std::string(index_prefix) + "_disk.index";
#ifdef EXEC_ENV_OLS
    return load_from_separate_paths(files, num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#else
    return load_from_separate_paths(num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#endif
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                      const char *index_filepath, const char *pivots_filepath,
                                                      const char *compressed_filepath)
{
#else
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                      const char *pivots_filepath, const char *compressed_filepath)
{
#endif
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_vectors = compressed_filepath;
    std::string _disk_index_file = index_filepath;
    std::string medoids_file = std::string(_disk_index_file) + "_medoids.bin";
    std::string centroids_file = std::string(_disk_index_file) + "_centroids.bin";

    std::string labels_file = std ::string(_disk_index_file) + "_labels.txt";
    std::string labels_to_medoids = std ::string(_disk_index_file) + "_labels_to_medoids.txt";
    std::string dummy_map_file = std ::string(_disk_index_file) + "_dummy_map.txt";
    std::string labels_map_file = std ::string(_disk_index_file) + "_labels_map.txt";
    size_t num_pts_in_label_file = 0;

    size_t pq_file_dim, pq_file_num_centroids;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#endif

    this->_disk_index_file = _disk_index_file;
    _base_disk_index_bytes = get_file_size(_disk_index_file);

    if (pq_file_num_centroids != 256)
    {
        diskann::cout << "Error. Number of PQ centroids is not 256. Exiting." << std::endl;
        return -1;
    }

    this->_data_dim = pq_file_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->_disk_bytes_per_point = this->_data_dim * sizeof(T);
    this->_aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint8_t>(files, pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#else
    diskann::load_bin<uint8_t>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#endif

    this->_num_points = npts_u64;
    this->_n_chunks = nchunks_u64;
    this->_hotness_profiler.init(npts_u64);
#ifdef EXEC_ENV_OLS
    if (files.fileExists(labels_file))
    {
        FileContent &content_labels = files.getContent(labels_file);
        std::stringstream infile(std::string((const char *)content_labels._content, content_labels._size));
#else
    if (file_exists(labels_file))
    {
        std::ifstream infile(labels_file, std::ios::binary);
        if (infile.fail())
        {
            throw diskann::ANNException(std::string("Failed to open file ") + labels_file, -1);
        }
#endif
        parse_label_file(infile, num_pts_in_label_file);
        assert(num_pts_in_label_file == this->_num_points);

#ifndef EXEC_ENV_OLS
        infile.close();
#endif

#ifdef EXEC_ENV_OLS
        FileContent &content_labels_map = files.getContent(labels_map_file);
        std::stringstream map_reader(std::string((const char *)content_labels_map._content, content_labels_map._size));
#else
        std::ifstream map_reader(labels_map_file);
#endif
        _label_map = load_label_map(map_reader);

#ifndef EXEC_ENV_OLS
        map_reader.close();
#endif

#ifdef EXEC_ENV_OLS
        if (files.fileExists(labels_to_medoids))
        {
            FileContent &content_labels_to_meoids = files.getContent(labels_to_medoids);
            std::stringstream medoid_stream(
                std::string((const char *)content_labels_to_meoids._content, content_labels_to_meoids._size));
#else
        if (file_exists(labels_to_medoids))
        {
            std::ifstream medoid_stream(labels_to_medoids);
            assert(medoid_stream.is_open());
#endif
            std::string line, token;

            _filter_to_medoid_ids.clear();
            try
            {
                while (std::getline(medoid_stream, line))
                {
                    std::istringstream iss(line);
                    uint32_t cnt = 0;
                    std::vector<uint32_t> medoids;
                    LabelT label;
                    while (std::getline(iss, token, ','))
                    {
                        if (cnt == 0)
                            label = (LabelT)std::stoul(token);
                        else
                            medoids.push_back((uint32_t)stoul(token));
                        cnt++;
                    }
                    _filter_to_medoid_ids[label].swap(medoids);
                }
            }
            catch (std::system_error &e)
            {
                throw FileException(labels_to_medoids, e, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        std::string univ_label_file = std ::string(_disk_index_file) + "_universal_label.txt";

#ifdef EXEC_ENV_OLS
        if (files.fileExists(univ_label_file))
        {
            FileContent &content_univ_label = files.getContent(univ_label_file);
            std::stringstream universal_label_reader(
                std::string((const char *)content_univ_label._content, content_univ_label._size));
#else
        if (file_exists(univ_label_file))
        {
            std::ifstream universal_label_reader(univ_label_file);
            assert(universal_label_reader.is_open());
#endif
            std::string univ_label;
            universal_label_reader >> univ_label;
#ifndef EXEC_ENV_OLS
            universal_label_reader.close();
#endif
            LabelT label_as_num = (LabelT)std::stoul(univ_label);
            set_universal_label(label_as_num);
        }

#ifdef EXEC_ENV_OLS
        if (files.fileExists(dummy_map_file))
        {
            FileContent &content_dummy_map = files.getContent(dummy_map_file);
            std::stringstream dummy_map_stream(
                std::string((const char *)content_dummy_map._content, content_dummy_map._size));
#else
        if (file_exists(dummy_map_file))
        {
            std::ifstream dummy_map_stream(dummy_map_file);
            assert(dummy_map_stream.is_open());
#endif
            std::string line, token;

            while (std::getline(dummy_map_stream, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t dummy_id;
                uint32_t real_id;
                while (std::getline(iss, token, ','))
                {
                    if (cnt == 0)
                        dummy_id = (uint32_t)stoul(token);
                    else
                        real_id = (uint32_t)stoul(token);
                    cnt++;
                }
                _dummy_pts.insert(dummy_id);
                _has_dummy_pts.insert(real_id);
                _dummy_to_real_map[dummy_id] = real_id;

                if (_real_to_dummy_map.find(real_id) == _real_to_dummy_map.end())
                    _real_to_dummy_map[real_id] = std::vector<uint32_t>();

                _real_to_dummy_map[real_id].emplace_back(dummy_id);
            }
#ifndef EXEC_ENV_OLS
            dummy_map_stream.close();
#endif
            diskann::cout << "Loaded dummy map" << std::endl;
        }
    }

#ifdef EXEC_ENV_OLS
    _pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    _pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

    diskann::cout << "Loaded PQ centroids and in-memory compressed vectors. #points: " << _num_points
                  << " #dim: " << _data_dim << " #aligned_dim: " << _aligned_dim << " #chunks: " << _n_chunks
                  << std::endl;

    if (_n_chunks > MAX_PQ_CHUNKS)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                  "PQ data does not exceed "
               << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::string disk_pq_pivots_path = this->_disk_index_file + "_pq_pivots.bin";
#ifdef EXEC_ENV_OLS
    if (files.fileExists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(files, disk_pq_pivots_path.c_str(), 0);
#else
    if (file_exists(disk_pq_pivots_path))
    {
        _use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        _disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
#endif
        _disk_pq_n_chunks = _disk_pq_table.get_num_chunks();
        _disk_bytes_per_point =
            _disk_pq_n_chunks * sizeof(uint8_t); // revising disk_bytes_per_point since DISK PQ is used.
        diskann::cout << "Disk index uses PQ data compressed down to " << _disk_pq_n_chunks << " bytes per point."
                      << std::endl;
    }

// read index metadata
#ifdef EXEC_ENV_OLS
    // This is a bit tricky. We have to read the header from the
    // disk_index_file. But  this is now exclusively a preserve of the
    // DiskPriorityIO class. So, we need to estimate how many
    // bytes are needed to store the header and read in that many using our
    // 'standard' aligned file reader approach.
    reader->open(_disk_index_file);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

    char *bytes = getHeaderBytes();
    ContentBuf buf(bytes, HEADER_SIZE);
    std::basic_istream<char> index_metadata(&buf);
#else
    std::ifstream index_metadata(_disk_index_file, std::ios::binary);
#endif

    uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
                     // metadata, nc should be 1)
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    uint64_t disk_nnodes;
    uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    if (disk_nnodes != _num_points)
    {
        diskann::cout << "Mismatch in #points for compressed data file and disk "
                         "index file: "
                      << disk_nnodes << " vs " << _num_points << std::endl;
        return -1;
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, _max_node_len);
    READ_U64(index_metadata, _nnodes_per_sector);
    _max_degree = ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;

    if (_max_degree > defaults::MAX_GRAPH_DEGREE)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max graph degree (R) does "
                  "not exceed "
               << defaults::MAX_GRAPH_DEGREE << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->_num_frozen_points);
    uint64_t file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->_num_frozen_points == 1)
        this->_frozen_location = file_frozen_id;
    if (this->_num_frozen_points == 1)
    {
        diskann::cout << " Detected frozen point in index at location " << this->_frozen_location
                      << ". Will not output it at search time." << std::endl;
    }

    READ_U64(index_metadata, this->_reorder_data_exists);
    if (this->_reorder_data_exists)
    {
        if (this->_use_disk_index_pq == false)
        {
            throw ANNException("Reordering is designed for used with disk PQ "
                               "compression option",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        READ_U64(index_metadata, this->_reorder_data_start_sector);
        READ_U64(index_metadata, this->_ndims_reorder_vecs);
        READ_U64(index_metadata, this->_nvecs_per_sector);
    }

    diskann::cout << "Disk-Index File Meta-data: ";
    diskann::cout << "# nodes per sector: " << _nnodes_per_sector;
    diskann::cout << ", max node len (bytes): " << _max_node_len;
    diskann::cout << ", max node degree: " << _max_degree << std::endl;

#ifdef EXEC_ENV_OLS
    delete[] bytes;
#else
    index_metadata.close();
#endif

#ifndef EXEC_ENV_OLS
    // open AlignedFileReader handle to index_file
    std::string index_fname(_disk_index_file);
    reader->open(index_fname);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

#endif

#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(files, norm_file, medoids_file, _medoids, _num_medoids, tmp_dim);
#else
    if (file_exists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(medoids_file, _medoids, _num_medoids, tmp_dim);
#endif

        if (tmp_dim != 1)
        {
            std::stringstream stream;
            stream << "Error loading medoids file. Expected bin format of m times "
                      "1 vector of uint32_t."
                   << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#ifdef EXEC_ENV_OLS
        if (!files.fileExists(centroids_file))
        {
#else
        if (!file_exists(centroids_file))
        {
#endif
            diskann::cout << "Centroid data file not found. Using corresponding vectors "
                             "for the medoids "
                          << std::endl;
            use_medoids_data_as_centroids();
        }
        else
        {
            size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
            diskann::load_aligned_bin<float>(files, centroids_file, _centroid_data, num_centroids, tmp_dim,
                                             aligned_tmp_dim);
#else
            diskann::load_aligned_bin<float>(centroids_file, _centroid_data, num_centroids, tmp_dim, aligned_tmp_dim);
#endif
            if (aligned_tmp_dim != _aligned_dim || num_centroids != _num_medoids)
            {
                std::stringstream stream;
                stream << "Error loading centroids data file. Expected bin format "
                          "of "
                          "m times data_dim vector of float, where m is number of "
                          "medoids "
                          "in medoids file.";
                diskann::cerr << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
    }
    else
    {
        _num_medoids = 1;
        _medoids = new uint32_t[1];
        _medoids[0] = (uint32_t)(medoid_id_on_file);
        use_medoids_data_as_centroids();
    }

    std::string norm_file = std::string(_disk_index_file) + "_max_base_norm.bin";

#ifdef EXEC_ENV_OLS
    if (files.fileExists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(files, norm_val, dumr, dumc);
#else
    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
#endif
        this->_max_base_norm = norm_val[0];
        diskann::cout << "Setting re-scaling factor of base vectors to " << this->_max_base_norm << std::endl;
        delete[] norm_val;
    }
    diskann::cout << "done.." << std::endl;
    return 0;
}

#ifdef USE_BING_INFRA
bool getNextCompletedRequest(std::shared_ptr<AlignedFileReader> &reader, IOContext &ctx, size_t size,
                             int &completedIndex)
{
    if ((*ctx.m_pRequests)[0].m_callback)
    {
        bool waitsRemaining = false;
        long completeCount = ctx.m_completeCount;
        do
        {
            for (int i = 0; i < size; i++)
            {
                auto ithStatus = (*ctx.m_pRequestsStatus)[i];
                if (ithStatus == IOContext::Status::READ_SUCCESS)
                {
                    completedIndex = i;
                    return true;
                }
                else if (ithStatus == IOContext::Status::READ_WAIT)
                {
                    waitsRemaining = true;
                }
            }

            // if we didn't find one in READ_SUCCESS, wait for one to complete.
            if (waitsRemaining)
            {
                WaitOnAddress(&ctx.m_completeCount, &completeCount, sizeof(completeCount), 100);
                // this assumes the knowledge of the reader behavior (implicit
                // contract). need better factoring?
            }
        } while (waitsRemaining);

        completedIndex = -1;
        return false;
    }
    else
    {
        reader->wait(ctx, completedIndex);
        return completedIndex != -1;
    }
}
#endif

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, std::numeric_limits<uint32_t>::max(),
                       use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, use_filter, filter_label,
                       std::numeric_limits<uint32_t>::max(), use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats)
{
    LabelT dummy_filter = 0;
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, false, dummy_filter, io_limit,
                       use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats)
{

    uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__,
                           __LINE__);

    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->pq_scratch();

    // reset query scratch
    query_scratch->reset();
    if (_merit_dyn_enabled)
        merit_dyn_prepare_query(query_scratch);
    if (_query_sector_cache_enabled)
    {
        query_scratch->sector_cache.clear();
        query_scratch->merit_sector_cache.clear();
        query_scratch->disk_read_sector_order.clear();
    }

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float query_norm = 0;
    T *aligned_query_T = query_scratch->aligned_query_T();
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    // normalization step. for cosine, we simply normalize the query
    // for mips, we normalize the first d-1 dims, and add a 0 for last dim, since an extra coordinate was used to
    // convert MIPS to L2 search
    if (metric == diskann::Metric::INNER_PRODUCT || metric == diskann::Metric::COSINE)
    {
        uint64_t inherent_dim = (metric == diskann::Metric::COSINE) ? this->_data_dim : (uint64_t)(this->_data_dim - 1);
        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = query1[i];
            query_norm += query1[i] * query1[i];
        }
        if (metric == diskann::Metric::INNER_PRODUCT)
            aligned_query_T[this->_data_dim - 1] = 0;

        query_norm = std::sqrt(query_norm);

        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = (T)(aligned_query_T[i] / query_norm);
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }
    else
    {
        for (size_t i = 0; i < this->_data_dim; i++)
        {
            aligned_query_T[i] = query1[i];
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }

    // pointers to buffers for data
    T *data_buf = query_scratch->coord_scratch;
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    uint64_t &sector_scratch_idx = query_scratch->sector_idx;
    const uint64_t num_sectors_per_node =
        _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // query <-> PQ chunk centers distances
    _pq_table.preprocess_query(query_rotated); // center the query and rotate if
                                               // we have a rotation matrix
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                                            float *dists_out) {
        diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);
    };
    auto profile_on_expand = [this, query_scratch, stats](uint32_t node_id) {
        query_scratch->merit_expand_count++;
        if (this->_merit_dyn_enabled && query_scratch->merit_dcache_maintenance_enabled)
        {
            const bool force_update =
                this->_merit_dcache_net_gate_enabled && query_scratch->merit_dcache_probe;
            const bool suppress_update =
                this->_merit_dcache_net_gate_enabled && !query_scratch->merit_dcache_lookup_enabled;
            const auto decision =
                this->_merit_mcache.plan_expand_update(node_id, force_update, suppress_update);
            if (stats != nullptr && this->_merit_mcache.adaptive_update_enabled())
            {
                if (decision.observed_hit)
                    stats->n_merit_mcache_hits++;
                else
                    stats->n_merit_mcache_misses++;
            }
            if (decision.update)
            {
                MeritMetadataCache::TouchResult tr;
                {
                    MERIT_SECTION(kSecMcOnExpand);
                    tr = this->_merit_mcache.on_expand(node_id, this->_merit_score_unit);
                }
                if (stats != nullptr && !this->_merit_mcache.adaptive_update_enabled() && tr.present)
                {
                    if (tr.inserted)
                        stats->n_merit_mcache_misses++;
                    else
                        stats->n_merit_mcache_hits++;
                }
                MERIT_SECTION(kSecMcNoteTouch);
                this->merit_dyn_note_touch(tr, query_scratch);
            }
        }
        if (!this->_hotness_profiler.enabled())
            return;
        this->_hotness_profiler.on_node_expand(node_id);
        auto parent_it = query_scratch->profile_parent.find(node_id);
        if (parent_it != query_scratch->profile_parent.end())
            this->_hotness_profiler.on_directed_edge(parent_it->second, node_id);
    };
    auto profile_on_first_visit = [this, query_scratch](uint32_t id, uint32_t parent) {
        query_scratch->profile_parent.insert({id, parent});
        if (this->_merit_dyn_enabled && query_scratch->merit_dcache_maintenance_enabled &&
            !this->_merit_real_io_coaccess)
        {
            bool touched = false;
            {
                const auto tr = this->_merit_mcache.on_edge(
                    parent, id, this->_merit_score_unit.load(std::memory_order_relaxed));
                touched = tr.present;
                if (touched)
                    this->merit_dyn_note_touch(tr, query_scratch);
            }
            if (touched)
                this->merit_mark_refresh_needed();
        }
        this->_hotness_profiler.on_node_visit(id);
    };
    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<uint64_t> &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;

    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();
    if (!use_filter)
    {
        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
        {
            float cur_expanded_dist =
                _dist_cmp_float->compare(query_float, _centroid_data + _aligned_dim * cur_m, (uint32_t)_aligned_dim);
            if (cur_expanded_dist < best_dist)
            {
                best_medoid = _medoids[cur_m];
                best_dist = cur_expanded_dist;
            }
        }
    }
    else
    {
        if (_filter_to_medoid_ids.find(filter_label) != _filter_to_medoid_ids.end())
        {
            const auto &medoid_ids = _filter_to_medoid_ids[filter_label];
            for (uint64_t cur_m = 0; cur_m < medoid_ids.size(); cur_m++)
            {
                // for filtered index, we dont store global centroid data as for unfiltered index, so we use PQ distance
                // as approximation to decide closest medoid matching the query filter.
                compute_dists(&medoid_ids[cur_m], 1, dist_scratch);
                float cur_expanded_dist = dist_scratch[0];
                if (cur_expanded_dist < best_dist)
                {
                    best_medoid = medoid_ids[cur_m];
                    best_dist = cur_expanded_dist;
                }
            }
        }
        else
        {
            throw ANNException("Cannot find medoid for specified filter.", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);
    this->_hotness_profiler.on_node_visit(best_medoid);

    uint32_t cmps = 0;
    uint32_t hops = 0;
    uint32_t num_ios = 0;
    uint32_t while_iteration = 0;

    const bool same_seed_beam =
        (_merit_dyn_enabled || !_merit_dc_map.empty()) && (std::getenv("MERIT_SAME_SEED_BEAM") != nullptr &&
                                   std::strcmp(std::getenv("MERIT_SAME_SEED_BEAM"), "0") != 0);
    const bool seed_batch_expand =
        (_merit_dyn_enabled || !_merit_dc_map.empty()) && !same_seed_beam &&
        (std::getenv("MERIT_SEED_BATCH_EXPAND") != nullptr &&
         std::strcmp(std::getenv("MERIT_SEED_BATCH_EXPAND"), "0") != 0);
    size_t same_seed_beam_pool = std::max<uint64_t>(beam_width, 4 * beam_width);
    size_t seed_batch_expand_max = 16;
    float seed_batch_expand_dist_eps = 0.0f;
    if (same_seed_beam)
    {
        const char *pool_env = std::getenv("MERIT_SAME_SEED_BEAM_POOL");
        if (pool_env != nullptr)
            same_seed_beam_pool = static_cast<size_t>(std::strtoul(pool_env, nullptr, 10));
    }
    if (seed_batch_expand)
    {
        const char *max_env = std::getenv("MERIT_SEED_BATCH_EXPAND_MAX");
        if (max_env != nullptr)
            seed_batch_expand_max = static_cast<size_t>(std::strtoul(max_env, nullptr, 10));
        const char *eps_env = std::getenv("MERIT_SEED_BATCH_EXPAND_DIST_EPS");
        if (eps_env != nullptr)
            seed_batch_expand_dist_eps = std::strtof(eps_env, nullptr);
    }
    const bool record_driven_seed_access =
        (std::getenv("MERIT_RECORD_DRIVEN_SEED_ACCESS") != nullptr &&
         std::strcmp(std::getenv("MERIT_RECORD_DRIVEN_SEED_ACCESS"), "0") != 0);
    auto merit_io_seed = [query_scratch, record_driven_seed_access](uint32_t node_id) -> uint32_t {
        if (record_driven_seed_access)
            return node_id;
        const auto parent_it = query_scratch->profile_parent.find(node_id);
        if (parent_it != query_scratch->profile_parent.end())
            return parent_it->second;
        return node_id;
    };

    // cleared every iteration
        const size_t hop_reserve = seed_batch_expand ? l_search : (2 * beam_width);
        std::vector<uint32_t> frontier;
        frontier.reserve(hop_reserve);
        std::vector<uint32_t> merit_frontier;
        merit_frontier.reserve(hop_reserve);
        std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
        frontier_nhoods.reserve(hop_reserve);
        std::vector<AlignedRead> frontier_read_reqs;
        frontier_read_reqs.reserve(hop_reserve);
        std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_nhoods;
        cached_nhoods.reserve(hop_reserve);
        std::vector<uint32_t> expand_order;
        expand_order.reserve(hop_reserve);

        while (retset.has_unexpanded_node() && num_ios < io_limit)
        {
            const uint32_t current_iteration = while_iteration++;
            // clear iteration state
            frontier.clear();
            merit_frontier.clear();
            frontier_nhoods.clear();
            frontier_read_reqs.clear();
            cached_nhoods.clear();
            expand_order.clear();
            sector_scratch_idx = 0;
            float beam_max_dist = 0.0f;
            // find new beam
            auto process_beam_neighbor = [&](const Neighbor &nbr) {
                beam_max_dist = std::max(beam_max_dist, nbr.distance);
                expand_order.push_back(nbr.id);
                if (stats != nullptr)
                    stats->n_beam_lookups++;
                bool mem_hit = false;
                if (_merit_mem_pool != nullptr && _merit_mem_pool->active())
                {
                    MERIT_SECTION(kSecNcLookup);
                    _merit_mem_pool->shadow_access(nbr.id, current_iteration);
                    T *coord_ptr = nullptr;
                    std::pair<uint32_t, uint32_t *> nh_pair;
                    if (_merit_mem_pool->lookup(nbr.id, coord_ptr, nh_pair, true))
                    {
                        cached_nhoods.push_back(std::make_pair(nbr.id, nh_pair));
                        mem_hit = true;
                        g_dyn_probe.ncache_hit++;
                        if (stats != nullptr)
                            stats->n_cache_hits++;
                    }
                }
                if (!mem_hit && !_nhood_cache.empty())
                {
                    auto iter = _nhood_cache.find(nbr.id);
                    if (iter != _nhood_cache.end())
                    {
                        cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                        mem_hit = true;
                        if (stats != nullptr)
                            stats->n_cache_hits++;
                    }
                }
                if (!mem_hit)
                {
                    g_dyn_probe.ncache_miss++;
                    // Prefetch buffer = same-query members packed when a seed page was served.
                    bool pref_hit = false;
                    {
                        MERIT_SECTION(kSecPrefLookup);
                        pref_hit = merit_dyn_query_local_dcache_hit(nbr.id, query_scratch);
                    }
                    bool dc_hit = false;
                    if (!pref_hit)
                    {
                        MERIT_SECTION(kSecDcLookup);
                        dc_hit = merit_disk_cache_lookup_hit(nbr.id, query_scratch);
                    }
                    if (pref_hit)
                    {
                        if (stats != nullptr)
                            stats->n_merit_prefetch_hits++;
                        if (_merit_member_snapshot_done.load(std::memory_order_acquire))
                        {
                            const uint32_t hop = current_iteration < 128 ? current_iteration : 127;
                            if (_merit_member_at_snapshot.find(nbr.id) != _merit_member_at_snapshot.end())
                            {
                                _merit_prefetch_from_early.fetch_add(1, std::memory_order_relaxed);
                                _merit_early_hop_hist[hop].fetch_add(1, std::memory_order_relaxed);
                            }
                            else
                            {
                                _merit_prefetch_from_late.fetch_add(1, std::memory_order_relaxed);
                                _merit_late_hop_hist[hop].fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                        merit_frontier.push_back(nbr.id);
                    }
                    else if (dc_hit)
                    {
                        // First touch of this page in the query → seed hit; immediately expose
                        // siblings in the query-local prefetch map so same-hop co-members are
                        // counted as prefetch (IO still coalesced later).
                        if (query_scratch != nullptr)
                        {
                            const auto lit = query_scratch->merit_dcache_loc_cache.find(nbr.id);
                            if (lit != query_scratch->merit_dcache_loc_cache.end())
                            {
                                const uint32_t sector = static_cast<uint32_t>(lit->second >> 32);
                                MERIT_SECTION(kSecPack);
                                merit_pack_dyn_page_into_query_prefetch(sector, query_scratch);
                            }
                        }
                        if (stats != nullptr)
                            stats->n_merit_dcache_seed_hits++;
                        merit_frontier.push_back(nbr.id);
                    }
                    else
                    {
                        frontier.push_back(nbr.id);
                        if (_record_base_frontier && stats != nullptr)
                            stats->base_frontier_nodes.push_back(nbr.id);
                    }
                }
                {
                    MERIT_SECTION(kSecMcacheExpand);
                    profile_on_expand(nbr.id);
                }
                if (this->_count_visited_nodes)
                {
                    reinterpret_cast<std::atomic<uint32_t> &>(this->_node_visit_counter[nbr.id].second).fetch_add(1);
                }
            };

            {
            MERIT_SECTION(kSecBeam);
            if (same_seed_beam)
            {
                const std::vector<Neighbor> beam_batch = retset.select_beam_seed_biased<tsl::robin_set<uint32_t>>(
                    beam_width, same_seed_beam_pool, merit_io_seed);
                for (const Neighbor &nbr : beam_batch)
                    process_beam_neighbor(nbr);
            }
            else
            {
                uint32_t num_seen = 0;
                while (retset.has_unexpanded_node() && frontier.size() + merit_frontier.size() < beam_width &&
                       num_seen < beam_width)
                {
                    Neighbor nbr = retset.closest_unexpanded();
                    num_seen++;
                    process_beam_neighbor(nbr);
                }
            }

            if (seed_batch_expand && !expand_order.empty())
            {
                tsl::robin_set<uint32_t> beam_ids(expand_order.begin(), expand_order.end());
                tsl::robin_set<uint32_t> active_seeds;
                for (uint32_t id : expand_order)
                    active_seeds.insert(merit_io_seed(id));

                std::vector<Neighbor> extras =
                    retset.collect_unexpanded_with_io_seeds<tsl::robin_set<uint32_t>, tsl::robin_set<uint32_t>>(
                        merit_io_seed, active_seeds, beam_ids);
                const float dist_limit = beam_max_dist + seed_batch_expand_dist_eps;
                extras.erase(
                    std::remove_if(extras.begin(), extras.end(),
                                   [dist_limit](const Neighbor &n) { return n.distance > dist_limit; }),
                    extras.end());
                extras.erase(std::remove_if(extras.begin(), extras.end(),
                                            [&](const Neighbor &n) {
                                                return !merit_disk_cache_lookup_hit(n.id, query_scratch);
                                            }),
                             extras.end());
                if (seed_batch_expand_max > 0 && extras.size() > seed_batch_expand_max)
                    extras.resize(seed_batch_expand_max);
                for (const Neighbor &nbr : extras)
                {
                    retset.mark_expanded(nbr.id);
                    process_beam_neighbor(nbr);
                }
            }
            }

            const size_t read_len = num_sectors_per_node * defaults::SECTOR_LEN;

            if (_record_hop_frontier && stats != nullptr && (!merit_frontier.empty() || !frontier.empty()))
            {
                QueryStats::HopFrontierRecord rec;
                rec.hop = current_iteration;
                rec.merit_nodes = merit_frontier;
                rec.base_nodes = frontier;
                stats->hop_frontier_trace.push_back(std::move(rec));
            }

            if (_hotness_profiler.enabled())
            {
                if (_merit_dyn_enabled || !_merit_dc_map.empty())
                {
                    if (!merit_frontier.empty())
                        _hotness_profiler.on_merit_hop_frontier(merit_frontier);
                }
                else if (!frontier.empty())
                {
                    _hotness_profiler.on_merit_hop_frontier(frontier);
                }
            }

            if (_merit_unified_disk && (!merit_frontier.empty() || !frontier.empty()))
            {
                if (stats != nullptr)
                    stats->n_hops++;

                std::vector<AlignedRead> combined_reqs;
                combined_reqs.reserve(merit_frontier.size() + frontier.size());
                std::vector<uint64_t> miss_sector_ids;
                miss_sector_ids.reserve(frontier.size());
                std::vector<std::pair<uint64_t, char *>> base_miss_bufs;
                base_miss_bufs.reserve(frontier.size());

                std::vector<MeritReadPending> merit_pending;
                std::vector<AlignedRead> merit_io;
                std::unordered_map<uint32_t, std::vector<size_t>> merit_disk_fanout;

                prepare_merit_disk_cache_io(merit_frontier, query_scratch, sector_scratch, sector_scratch_idx,
                                         num_sectors_per_node, merit_pending, merit_io, merit_disk_fanout, stats,
                                         num_ios);
                merit_fallback_unserved_to_base(merit_frontier, merit_pending, merit_disk_fanout, frontier);

                for (const auto &mr : merit_io)
                    combined_reqs.push_back(mr);

                for (uint64_t i = 0; i < frontier.size(); i++)
                {
                    const auto id = frontier[i];
                    std::pair<uint32_t, char *> fnhood;
                    fnhood.first = id;
                    const uint64_t sec = get_node_sector((size_t)id);
                    query_scratch->read_sectors.insert(sec);

                    bool served_from_cache = false;
                    if (_query_sector_cache_enabled && read_len <= defaults::SECTOR_LEN)
                    {
                        const auto cache_it = query_scratch->sector_cache.find(sec);
                        if (cache_it != query_scratch->sector_cache.end())
                        {
                            fnhood.second = copy_to_sector_scratch(sector_scratch, sector_scratch_idx,
                                                                     cache_it->second.data(), read_len);
                            if (stats != nullptr)
                            {
                                stats->n_sector_cache_hits++;
                                stats->n_4k++;
                                stats->n_ios++;
                            }
                            num_ios++;
                            served_from_cache = true;
                        }
                    }
                    if (!served_from_cache)
                    {
                        fnhood.second =
                            sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                        sector_scratch_idx++;
                        combined_reqs.emplace_back(sec * defaults::SECTOR_LEN, read_len, fnhood.second);
                        miss_sector_ids.push_back(sec);
                        base_miss_bufs.emplace_back(sec, fnhood.second);
                        if (stats != nullptr)
                        {
                            query_scratch->disk_read_sector_order.push_back(sec);
                            stats->n_4k++;
                            stats->n_ios++;
                            stats->n_disk_reads++;
                            if (_record_hop_frontier && !stats->hop_frontier_trace.empty())
                                stats->hop_frontier_trace.back().physical_reads.push_back({false, sec, 1});
                        }
                        num_ios++;
                    }
                    frontier_nhoods.push_back(fnhood);
                }

                if (!combined_reqs.empty())
                {
                    std::vector<size_t> order(combined_reqs.size());
                    std::iota(order.begin(), order.end(), 0);
                    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                        return combined_reqs[a].offset < combined_reqs[b].offset;
                    });
                    std::vector<AlignedRead> sorted_reqs;
                    sorted_reqs.reserve(combined_reqs.size());
                    for (size_t idx : order)
                        sorted_reqs.push_back(combined_reqs[idx]);
                    combined_reqs = std::move(sorted_reqs);

                    io_timer.reset();
#ifdef USE_BING_INFRA
                    reader->read(combined_reqs, ctx, true);
#else
                    reader->read(combined_reqs, ctx);
#endif
                    if (stats != nullptr)
                    {
                        const float batch_us = (float)io_timer.elapsed();
                        stats->io_us += batch_us;
                        if (!miss_sector_ids.empty())
                        {
                            const uint64_t max_graph_sector =
                                (_nnodes_per_sector > 0 && _num_points > 0)
                                    ? (1 + (_num_points - 1) / _nnodes_per_sector)
                                    : 1;
                            record_disk_read_batch_stats(stats, batch_us, miss_sector_ids, max_graph_sector);
                            if (miss_sector_ids.size() >= 2)
                            {
                                uint64_t mn = miss_sector_ids[0];
                                uint64_t mx = miss_sector_ids[0];
                                for (uint64_t sid : miss_sector_ids)
                                {
                                    mn = std::min(mn, sid);
                                    mx = std::max(mx, sid);
                                }
                                stats->sum_intra_batch_spread += (mx - mn);
                                stats->n_intra_batch_spread_samples++;
                            }
                        }
                    }
                    if (_query_sector_cache_enabled && read_len <= defaults::SECTOR_LEN)
                    {
                        for (const auto &bm : base_miss_bufs)
                            memcpy(query_scratch->sector_cache[bm.first].data(), bm.second, read_len);
                    }
                }

                complete_merit_disk_cache_io(query_scratch, merit_pending, merit_disk_fanout);
                finalize_merit_pending_nodes(merit_pending, merit_frontier, query_scratch, sector_scratch,
                                           sector_scratch_idx, frontier_nhoods, stats);
            }
            else
            {
            std::vector<MeritReadPending> merit_pending;
            std::vector<AlignedRead> merit_io;
            std::unordered_map<uint32_t, std::vector<size_t>> merit_disk_fanout;
            std::vector<uint64_t> miss_sector_ids;
            miss_sector_ids.reserve(frontier.size());

            uint64_t io_prep_t0 = merit_section_on() ? merit_section_now() : 0;
            if (!merit_frontier.empty() && (_merit_disk_reader || _merit_dyn_enabled))
            {
                prepare_merit_disk_cache_io(merit_frontier, query_scratch, sector_scratch, sector_scratch_idx,
                                         num_sectors_per_node, merit_pending, merit_io, merit_disk_fanout, stats,
                                         num_ios);
            }
            merit_fallback_unserved_to_base(merit_frontier, merit_pending, merit_disk_fanout, frontier);
            auto io_prep_done = [&]() {
                if (io_prep_t0 != 0)
                {
                    merit_section_acc().ns[kSecIoPrep] += merit_section_now() - io_prep_t0;
                    io_prep_t0 = 0;
                }
            };

            // read nhoods of frontier ids
            if (!frontier.empty())
            {
                const size_t read_len = num_sectors_per_node * defaults::SECTOR_LEN;

            if (_query_sector_cache_enabled && read_len <= defaults::SECTOR_LEN)
            {
                miss_sector_ids.reserve(frontier.size());
                for (uint64_t i = 0; i < frontier.size(); i++)
                {
                    auto id = frontier[i];
                    std::pair<uint32_t, char *> fnhood;
                    fnhood.first = id;
                    const uint64_t sec = get_node_sector((size_t)id);
                    query_scratch->read_sectors.insert(sec);

                    const auto cache_it = query_scratch->sector_cache.find(sec);
                    if (cache_it != query_scratch->sector_cache.end())
                    {
                        fnhood.second = copy_to_sector_scratch(sector_scratch, sector_scratch_idx,
                                                                 cache_it->second.data(), read_len);
                        if (stats != nullptr)
                        {
                            stats->n_sector_cache_hits++;
                            stats->n_4k++;
                            stats->n_ios++;
                        }
                        num_ios++;
                    }
                    else
                    {
                        fnhood.second =
                            sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                        sector_scratch_idx++;
                        frontier_read_reqs.emplace_back(sec * defaults::SECTOR_LEN, read_len, fnhood.second);
                        miss_sector_ids.push_back(sec);
                        if (stats != nullptr)
                        {
                            query_scratch->disk_read_sector_order.push_back(sec);
                            stats->n_4k++;
                            stats->n_ios++;
                            stats->n_disk_reads++;
                            if (_record_hop_frontier && !stats->hop_frontier_trace.empty())
                                stats->hop_frontier_trace.back().physical_reads.push_back({false, sec, 1});
                        }
                        num_ios++;
                    }
                    frontier_nhoods.push_back(fnhood);
                }
                io_prep_done();
                if (!merit_io.empty() || !frontier_read_reqs.empty())
                {
                    if (stats != nullptr)
                        stats->n_hops++;
                    const float batch_us = issue_merit_and_base_disk_reads(reader, (_merit_dyn_enabled ? _merit_dyn_reader : _merit_disk_reader), ctx, merit_io,
                                                                           frontier_read_reqs, io_timer, stats);
                    if (stats != nullptr && !frontier_read_reqs.empty())
                    {
                        const uint64_t max_graph_sector =
                            (_nnodes_per_sector > 0 && _num_points > 0)
                                ? (1 + (_num_points - 1) / _nnodes_per_sector)
                                : 1;
                        record_disk_read_batch_stats(stats, batch_us, miss_sector_ids, max_graph_sector);
                        if (miss_sector_ids.size() >= 2)
                        {
                            uint64_t mn = miss_sector_ids[0];
                            uint64_t mx = miss_sector_ids[0];
                            for (uint64_t sid : miss_sector_ids)
                            {
                                mn = std::min(mn, sid);
                                mx = std::max(mx, sid);
                            }
                            stats->sum_intra_batch_spread += (mx - mn);
                            stats->n_intra_batch_spread_samples++;
                        }
                    }
                    MERIT_SECTION(kSecPostIo);
                    for (size_t ri = 0; ri < frontier_read_reqs.size(); ri++)
                        memcpy(query_scratch->sector_cache[miss_sector_ids[ri]].data(),
                               frontier_read_reqs[ri].buf, read_len);
                    complete_merit_disk_cache_io(query_scratch, merit_pending, merit_disk_fanout);
                }
            }
            else
            {
                for (uint64_t i = 0; i < frontier.size(); i++)
                {
                    auto id = frontier[i];
                    std::pair<uint32_t, char *> fnhood;
                    fnhood.first = id;
                    fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                    sector_scratch_idx++;
                    frontier_nhoods.push_back(fnhood);
                    const uint64_t sec = get_node_sector((size_t)id);
                    query_scratch->read_sectors.insert(sec);
                    frontier_read_reqs.emplace_back(sec * defaults::SECTOR_LEN, read_len, fnhood.second);
                    if (stats != nullptr)
                    {
                        stats->n_4k++;
                        stats->n_ios++;
                        stats->n_disk_reads++;
                        if (_record_hop_frontier && !stats->hop_frontier_trace.empty())
                            stats->hop_frontier_trace.back().physical_reads.push_back({false, sec, 1});
                    }
                    num_ios++;
                }
                if (!merit_io.empty() || !frontier_read_reqs.empty())
                {
                    if (stats != nullptr)
                        stats->n_hops++;
                    issue_merit_and_base_disk_reads(reader, (_merit_dyn_enabled ? _merit_dyn_reader : _merit_disk_reader), ctx, merit_io, frontier_read_reqs,
                                                    io_timer, stats);
                    complete_merit_disk_cache_io(query_scratch, merit_pending, merit_disk_fanout);
                }
            }
            }
            io_prep_done();
            if (frontier.empty() && !merit_io.empty())
            {
                if (stats != nullptr)
                    stats->n_hops++;
                issue_merit_and_base_disk_reads(reader, (_merit_dyn_enabled ? _merit_dyn_reader : _merit_disk_reader), ctx, merit_io, frontier_read_reqs, io_timer,
                                                stats);
                MERIT_SECTION(kSecPostIo);
                complete_merit_disk_cache_io(query_scratch, merit_pending, merit_disk_fanout);
            }

            if (!merit_pending.empty())
            {
                MERIT_SECTION(kSecFinalize);
                finalize_merit_pending_nodes(merit_pending, merit_frontier, query_scratch, sector_scratch,
                                             sector_scratch_idx, frontier_nhoods, stats);
            }

            }
            uint64_t maps_t0 = merit_section_on() ? merit_section_now() : 0;
            reorder_frontier_nhoods_by_expand_order(expand_order, frontier_nhoods);

            std::unordered_map<uint32_t, std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_by_id;
            cached_by_id.reserve(cached_nhoods.size());
            for (auto &c : cached_nhoods)
                cached_by_id.emplace(c.first, c);

            std::unordered_map<uint32_t, char *> disk_by_id;
            disk_by_id.reserve(frontier_nhoods.size());
            for (auto &fn : frontier_nhoods)
                disk_by_id[fn.first] = fn.second;
            if (maps_t0 != 0)
                merit_section_acc().ns[kSecMaps] += merit_section_now() - maps_t0;

#ifdef USE_BING_INFRA
        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods)
        {
            T *node_fp_coords_copy = nullptr;
            std::pair<uint32_t, uint32_t *> nh_dummy;
            if (_merit_mem_pool == nullptr || !_merit_mem_pool->active() ||
                !_merit_mem_pool->lookup(cached_nhood.first, node_fp_coords_copy, nh_dummy))
            {
                auto global_cache_iter = _coord_cache.find(cached_nhood.first);
                if (global_cache_iter == _coord_cache.end())
                    continue;
                node_fp_coords_copy = global_cache_iter->second;
            }
            float cur_expanded_dist;
            if (!_use_disk_index_pq)
            {
                cur_expanded_dist = _dist_cmp->compare(aligned_query_T, node_fp_coords_copy, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)node_fp_coords_copy);
                else
                    cur_expanded_dist = _disk_pq_table.l2_distance( // disk_pq does not support OPQ yet
                        query_float, (uint8_t *)node_fp_coords_copy);
            }
            full_retset.push_back(Neighbor((uint32_t)cached_nhood.first, cur_expanded_dist));

            uint64_t nnbrs = cached_nhood.second.first;
            uint32_t *node_nbrs = cached_nhood.second.second;

            // compute node_nbrs <-> query dists in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            // process prefetched nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    profile_on_first_visit(id, cached_nhood.first);
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }
        }
        // process each frontier nhood - compute distances to unvisited nodes
        int completedIndex = -1;
        long requestCount = static_cast<long>(frontier_read_reqs.size());
        while (requestCount > 0 && getNextCompletedRequest(reader, ctx, requestCount, completedIndex))
        {
            assert(completedIndex >= 0);
            auto &frontier_nhood = frontier_nhoods[completedIndex];
            (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
            char *node_disk_buf = offset_to_node(frontier_nhood.second, frontier_nhood.first);
            if (_merit_mem_pool != nullptr && _merit_mem_pool->active() &&
                (_merit_mem_runtime_admit || _merit_dyn_enabled))
            {
                this->merit_dyn_admit_node(frontier_nhood.first, node_disk_buf, stats, query_scratch,
                                           current_iteration);
                if (this->_merit_dyn_enabled)
                {
                    uint32_t parent = std::numeric_limits<uint32_t>::max();
                    const auto pit = query_scratch->profile_parent.find(frontier_nhood.first);
                    if (pit != query_scratch->profile_parent.end())
                        parent = pit->second;
                    this->merit_dyn_note_base_load(frontier_nhood.first, parent, node_disk_buf, query_scratch);
                }
            }
            uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
            uint64_t nnbrs = (uint64_t)(*node_buf);
            T *node_fp_coords = offset_to_node_coords(node_disk_buf);
            memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
            float cur_expanded_dist;
            if (!_use_disk_index_pq)
            {
                cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf);
                else
                    cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
            }
            full_retset.push_back(Neighbor(frontier_nhood.first, cur_expanded_dist));
            const uint32_t *node_nbrs = nullptr;
            merit_get_expand_neighbors(frontier_nhood.first, node_disk_buf, node_nbrs, nnbrs);
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            cpu_timer.reset();
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    profile_on_first_visit(id, frontier_nhood.first);
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !(point_has_label(id, filter_label)) &&
                        (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    if (stats != nullptr)
                        stats->n_cmps++;

                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }

            if (stats != nullptr)
                stats->cpu_us += (float)cpu_timer.elapsed();
        }
#else
            MERIT_SECTION(kSecExpand);
            for (uint32_t expand_id : expand_order)
            {
                const auto cached_it = cached_by_id.find(expand_id);
                if (cached_it != cached_by_id.end())
                {
                    auto &cached_nhood = cached_it->second;
                    T *node_fp_coords_copy = nullptr;
                    bool pool_hit = false;
                    if (_merit_mem_pool != nullptr && _merit_mem_pool->active())
                    {
                        MERIT_SECTION(kSecCachedLookup);
                        std::pair<uint32_t, uint32_t *> nh_dummy;
                        pool_hit = _merit_mem_pool->lookup(cached_nhood.first, node_fp_coords_copy, nh_dummy);
                    }
                    if (!pool_hit)
                    {
                        auto global_cache_iter = _coord_cache.find(cached_nhood.first);
                        if (global_cache_iter == _coord_cache.end())
                            continue;
                        node_fp_coords_copy = global_cache_iter->second;
                    }
                    float cur_expanded_dist;
                    if (!_use_disk_index_pq)
                    {
                        MERIT_SECTION(kSecExactDist);
                        cur_expanded_dist =
                            _dist_cmp->compare(aligned_query_T, node_fp_coords_copy, (uint32_t)_aligned_dim);
                    }
                    else
                    {
                        if (metric == diskann::Metric::INNER_PRODUCT)
                            cur_expanded_dist =
                                _disk_pq_table.inner_product(query_float, (uint8_t *)node_fp_coords_copy);
                        else
                            cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)node_fp_coords_copy);
                    }
                    full_retset.push_back(Neighbor((uint32_t)cached_nhood.first, cur_expanded_dist));

                    uint64_t nnbrs = cached_nhood.second.first;
                    uint32_t *node_nbrs = cached_nhood.second.second;

                    cpu_timer.reset();
                    compute_dists(node_nbrs, nnbrs, dist_scratch);
                    if (stats != nullptr)
                    {
                        stats->n_cmps += (uint32_t)nnbrs;
                        stats->cpu_us += (float)cpu_timer.elapsed();
                    }

                    MERIT_SECTION(kSecCachedNbrLoop);
                    cpu_timer.reset();
                    for (uint64_t m = 0; m < nnbrs; ++m)
                    {
                        uint32_t id = node_nbrs[m];
                        if (visited.insert(id).second)
                        {
                            profile_on_first_visit(id, cached_nhood.first);
                            if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                                continue;

                            if (use_filter && !(point_has_label(id, filter_label)) &&
                                (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                                continue;
                            cmps++;
                            float dist = dist_scratch[m];
                            Neighbor nn(id, dist);
                            retset.insert(nn);
                        }
                    }
                    if (stats != nullptr)
                        stats->cpu_us += (float)cpu_timer.elapsed();
                    continue;
                }

                const auto disk_it = disk_by_id.find(expand_id);
                if (disk_it == disk_by_id.end())
                    continue;

                char *node_disk_buf = offset_to_node(disk_it->second, expand_id);
                if (_merit_mem_pool != nullptr && _merit_mem_pool->active() &&
                    (_merit_mem_runtime_admit || _merit_dyn_enabled))
                {
                    {
                        MERIT_SECTION(kSecAdmit);
                        this->merit_dyn_admit_node(expand_id, node_disk_buf, stats, query_scratch, current_iteration);
                    }
                    if (this->_merit_dyn_enabled)
                    {
                        MERIT_SECTION(kSecNoteBase);
                        uint32_t parent = std::numeric_limits<uint32_t>::max();
                        const auto pit = query_scratch->profile_parent.find(expand_id);
                        if (pit != query_scratch->profile_parent.end())
                            parent = pit->second;
                        this->merit_dyn_note_base_load(expand_id, parent, node_disk_buf, query_scratch);
                    }
                }
                uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
                uint64_t nnbrs = (uint64_t)(*node_buf);
                T *node_fp_coords = offset_to_node_coords(node_disk_buf);
                memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
                float cur_expanded_dist;
                if (!_use_disk_index_pq)
                {
                    MERIT_SECTION(kSecExactDist);
                    cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)_aligned_dim);
                }
                else
                {
                    if (metric == diskann::Metric::INNER_PRODUCT)
                        cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *)data_buf);
                    else
                        cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *)data_buf);
                }
                full_retset.push_back(Neighbor(expand_id, cur_expanded_dist));
                const uint32_t *node_nbrs = nullptr;
                merit_get_expand_neighbors(expand_id, node_disk_buf, node_nbrs, nnbrs);
                cpu_timer.reset();
                compute_dists(node_nbrs, nnbrs, dist_scratch);
                if (stats != nullptr)
                {
                    stats->n_cmps += (uint32_t)nnbrs;
                    stats->cpu_us += (float)cpu_timer.elapsed();
                }

                cpu_timer.reset();
                for (uint64_t m = 0; m < nnbrs; ++m)
                {
                    uint32_t id = node_nbrs[m];
                    if (visited.insert(id).second)
                    {
                        profile_on_first_visit(id, expand_id);
                        if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                            continue;

                        if (use_filter && !(point_has_label(id, filter_label)) &&
                            (!_use_universal_label || !point_has_label(id, _universal_filter_label)))
                            continue;
                        cmps++;
                        float dist = dist_scratch[m];
                        if (stats != nullptr)
                            stats->n_cmps++;

                        Neighbor nn(id, dist);
                        retset.insert(nn);
                    }
                }

                if (stats != nullptr)
                    stats->cpu_us += (float)cpu_timer.elapsed();
            }
#endif

        hops++;
    }

    if (_merit_dyn_enabled)
    {
        {
            MERIT_SECTION(kSecQendPins);
            for (uint32_t sector : query_scratch->merit_dyn_pinned_sectors)
                merit_dyn_release_page_pin(sector);
            query_scratch->merit_dyn_pinned_sectors.clear();
        }
        MERIT_SECTION(kSecQendDyn);
        merit_dyn_on_query_end(stats, query_scratch);
    }
    if (_merit_mem_pool != nullptr && _merit_mem_pool->active())
    {
        MERIT_SECTION(kSecQendNcache);
        _merit_mem_pool->on_query_end(stats == nullptr ? 0 : stats->n_disk_reads,
                                      stats == nullptr ? 0.0 : stats->io_us);
    }
    uint64_t result_t0 = merit_section_on() ? merit_section_now() : 0;

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end());

    if (use_reorder_data)
    {
        if (!(this->_reorder_data_exists))
        {
            throw ANNException("Requested use of reordering data which does "
                               "not exist in index "
                               "file",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        std::vector<AlignedRead> vec_read_reqs;

        if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
            full_retset.erase(full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER, full_retset.end());

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            // MULTISECTORFIX
            vec_read_reqs.emplace_back(VECTOR_SECTOR_NO(((size_t)full_retset[i].id)) * defaults::SECTOR_LEN,
                                       defaults::SECTOR_LEN, sector_scratch + i * defaults::SECTOR_LEN);

            if (stats != nullptr)
            {
                stats->n_4k++;
                stats->n_ios++;
            }
        }

        io_timer.reset();
#ifdef USE_BING_INFRA
        reader->read(vec_read_reqs, ctx, true); // async reader windows.
#else
        reader->read(vec_read_reqs, ctx); // synchronous IO linux
#endif
        if (stats != nullptr)
        {
            stats->io_us += io_timer.elapsed();
        }

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            auto id = full_retset[i].id;
            // MULTISECTORFIX
            auto location = (sector_scratch + i * defaults::SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
            full_retset[i].distance = _dist_cmp->compare(aligned_query_T, (T *)location, (uint32_t)this->_data_dim);
        }

        std::sort(full_retset.begin(), full_retset.end());
    }

    // copy k_search values
    for (uint64_t i = 0; i < k_search; i++)
    {
        indices[i] = full_retset[i].id;
        auto key = (uint32_t)indices[i];
        if (_dummy_pts.find(key) != _dummy_pts.end())
        {
            indices[i] = _dummy_to_real_map[key];
        }

        if (distances != nullptr)
        {
            distances[i] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT)
            {
                // flip the sign to convert min to max
                distances[i] = (-distances[i]);
                // rescale to revert back to original norms (cancelling the
                // effect of base and query pre-processing)
                if (_max_base_norm != 0)
                    distances[i] *= (_max_base_norm * query_norm);
            }
        }
    }

#ifdef USE_BING_INFRA
    ctx.m_completeCount = 0;
#endif

    if (stats != nullptr)
    {
        stats->n_unique_sectors = (unsigned)query_scratch->read_sectors.size();
        stats->n_unique_merit_sectors = (unsigned)query_scratch->read_merit_disk_cache_sectors.size();
        stats->total_us = (float)query_timer.elapsed();
        if (result_t0 != 0)
        {
            auto &acc = merit_section_acc();
            acc.ns[kSecResult] += merit_section_now() - result_t0;
            acc.queries++;
        }
        if (_query_sector_cache_enabled)
        {
            const auto &seq = query_scratch->disk_read_sector_order;
            for (size_t i = 1; i < seq.size(); i++)
            {
                const uint64_t a = seq[i - 1];
                const uint64_t b = seq[i];
                const uint64_t d = (a >= b) ? (a - b) : (b - a);
                stats->sum_abs_sector_jump += d;
                stats->n_sector_jump_samples++;
                if (d <= 8)
                    stats->n_sector_jump_le8++;
                if (d > stats->max_sector_jump)
                    stats->max_sector_jump = d;
            }
        }
    }
}

// range search returns results of all neighbors within distance of range.
// indices and distances need to be pre-allocated of size l_search and the
// return value is the number of matching hits.
template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::range_search(const T *query1, const double range, const uint64_t min_l_search,
                                               const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                               std::vector<float> &distances, const uint64_t min_beam_width,
                                               QueryStats *stats)
{
    uint32_t res_count = 0;

    bool stop_flag = false;

    uint32_t l_search = (uint32_t)min_l_search; // starting size of the candidate list
    while (!stop_flag)
    {
        indices.resize(l_search);
        distances.resize(l_search);
        uint64_t cur_bw = min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
        cur_bw = (cur_bw > 100) ? 100 : cur_bw;
        for (auto &x : distances)
            x = std::numeric_limits<float>::max();
        this->cached_beam_search(query1, l_search, l_search, indices.data(), distances.data(), cur_bw, false, stats);
        for (uint32_t i = 0; i < l_search; i++)
        {
            if (distances[i] > (float)range)
            {
                res_count = i;
                break;
            }
            else if (i == l_search - 1)
                res_count = l_search;
        }
        if (res_count < (uint32_t)(l_search / 2.0))
            stop_flag = true;
        l_search = l_search * 2;
        if (l_search > max_l_search)
            stop_flag = true;
    }
    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::get_data_dim()
{
    return _data_dim;
}

template <typename T, typename LabelT> diskann::Metric PQFlashIndex<T, LabelT>::get_metric()
{
    return this->metric;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT> char *PQFlashIndex<T, LabelT>::getHeaderBytes()
{
    IOContext &ctx = reader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T, LabelT>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T, LabelT>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    reader->read(readReqs, ctx, false);

    return (char *)readReq.buf;
}
#endif

template <typename T, typename LabelT>
std::vector<std::uint8_t> PQFlashIndex<T, LabelT>::get_pq_vector(std::uint64_t vid)
{
    std::uint8_t *pqVec = &this->data[vid * this->_n_chunks];
    return std::vector<std::uint8_t>(pqVec, pqVec + this->_n_chunks);
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndex<T, LabelT>::get_num_points()
{
    return _num_points;
}

template <typename T, typename LabelT> std::uint64_t PQFlashIndex<T, LabelT>::get_max_degree()
{
    return _max_degree;
}

namespace
{
uint64_t read_host_mem_total_bytes()
{
    std::ifstream in("/proc/meminfo");
    if (!in.is_open())
        return 0;
    std::string key;
    uint64_t kb = 0;
    while (in >> key)
    {
        if (key == "MemTotal:")
        {
            in >> kb;
            break;
        }
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return kb * 1024ULL;
}
} // namespace

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::bytes_per_cached_node() const
{
    const uint64_t coord_bytes = _aligned_dim * sizeof(T);
    const uint64_t nhood_bytes = (_max_degree + 1) * sizeof(uint32_t);
    const uint64_t map_overhead = 64;
    return coord_bytes + nhood_bytes + map_overhead;
}

template <typename T, typename LabelT>
uint64_t PQFlashIndex<T, LabelT>::estimate_baseline_resident_bytes(uint32_t num_threads) const
{
    uint64_t bytes = 0;
    bytes += _num_points * _n_chunks;
    bytes += 256ULL * _data_dim * sizeof(float) * 4ULL;
    if (_use_disk_index_pq)
        bytes += 256ULL * _data_dim * sizeof(float) * 2ULL;
    bytes += _num_medoids * (_aligned_dim * sizeof(float) + sizeof(uint32_t));
    const uint64_t per_thread =
        defaults::MAX_N_SECTOR_READS * defaults::SECTOR_LEN + _aligned_dim * sizeof(T) +
        defaults::MAX_GRAPH_DEGREE * sizeof(float) + _aligned_dim * sizeof(float) * 2ULL + (1ULL << 20);
    bytes += static_cast<uint64_t>(num_threads) * per_thread;
    if (_pts_to_label_offsets != nullptr)
        bytes += (_num_points + 1) * sizeof(uint32_t);
    return bytes;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::plan_merit_memory_cache(double merit_memory_gb, double host_memory_gb, double reserve_gb,
                                                     uint32_t num_threads, uint64_t &out_max_nodes,
                                                     std::string &report) const
{
    out_max_nodes = 0;
    report.clear();
    if (merit_memory_gb <= 0.0)
    {
        report = "merit_memory_gb <= 0; MERIT memory cache disabled.";
        return 0;
    }

    const uint64_t host_bytes =
        (host_memory_gb > 0.0) ? static_cast<uint64_t>(host_memory_gb * (1024.0 * 1024.0 * 1024.0))
                               : read_host_mem_total_bytes();
    if (host_bytes == 0)
    {
        report = "Failed to determine host memory size (pass --merit_host_memory_gb).";
        return -1;
    }

    const uint64_t reserve_bytes =
        (reserve_gb > 0.0) ? static_cast<uint64_t>(reserve_gb * (1024.0 * 1024.0 * 1024.0)) : 0;
    const uint64_t baseline = estimate_baseline_resident_bytes(num_threads);
    const uint64_t per_node = bytes_per_cached_node();
    const uint64_t requested = static_cast<uint64_t>(merit_memory_gb * (1024.0 * 1024.0 * 1024.0));

    uint64_t residual = 0;
    if (host_bytes > baseline + reserve_bytes)
        residual = host_bytes - baseline - reserve_bytes;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << "MERIT memory budget:\n"
        << "  host_total          = " << (host_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  baseline_index      = " << (baseline / (1024.0 * 1024.0 * 1024.0))
        << " GB (PQ + pivots + thread scratch estimate)\n"
        << "  reserve             = " << (reserve_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  residual_for_cache  = " << (residual / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  requested_cache     = " << merit_memory_gb << " GB\n"
        << "  bytes_per_node      = " << per_node << " B"
        << " (aligned_dim=" << _aligned_dim << ", max_degree=" << _max_degree << ")";

    if (requested > residual)
    {
        oss << "\nERROR: requested MERIT memory cache (" << merit_memory_gb
            << " GB) exceeds residual budget (" << (residual / (1024.0 * 1024.0 * 1024.0))
            << " GB). Reduce --merit_memory_gb or --merit_memory_reserve_gb.";
        report = oss.str();
        return -1;
    }

    out_max_nodes = (per_node > 0) ? (requested / per_node) : 0;
    const uint64_t residual_cap = (per_node > 0) ? (residual / per_node) : 0;
    if (out_max_nodes > _num_points)
        out_max_nodes = _num_points;
    if (out_max_nodes > residual_cap)
        out_max_nodes = residual_cap;
    if (out_max_nodes == 0)
    {
        oss << "\nERROR: requested cache too small to hold even one node entry.";
        report = oss.str();
        return -1;
    }

    oss << "\n  max_nodes_in_residual = " << std::min<uint64_t>(residual_cap, _num_points)
        << "\n  selected_nodes      = " << out_max_nodes << " (Top-" << out_max_nodes
        << " by node_expand)\n"
        << "  approx_cache_bytes  = " << ((out_max_nodes * per_node) / (1024.0 * 1024.0 * 1024.0)) << " GB";
    report = oss.str();
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::build_merit_memory_node_list(const std::string &profile_prefix, uint64_t max_nodes,
                                                         std::vector<uint32_t> &node_list, uint64_t rank_skip) const
{
    node_list.clear();
    if (max_nodes == 0)
        return 0;

    std::vector<uint64_t> node_expand;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> edges;
    if (HotnessProfiler::load(profile_prefix, node_expand, edges) != 0)
        return -1;
    if (node_expand.size() != _num_points)
    {
        diskann::cerr << "MERIT profile node_expand size " << node_expand.size() << " != num_points " << _num_points
                      << std::endl;
        return -1;
    }

    std::vector<uint32_t> order(_num_points);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (node_expand[a] != node_expand[b])
            return node_expand[a] > node_expand[b];
        return a < b;
    });

    if (rank_skip >= _num_points)
        return 0;

    uint64_t skipped = 0;
    for (uint32_t id : order)
    {
        if (id >= node_expand.size() || node_expand[id] == 0)
            continue;
        if (skipped < rank_skip)
        {
            skipped++;
            continue;
        }
        node_list.push_back(id);
        if (node_list.size() >= max_nodes)
            break;
    }
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::build_merit_disk_node_list(const std::string &profile_prefix, uint64_t max_nodes,
                                                        uint64_t memory_tier_exclude_count, uint32_t k_hops,
                                                        const std::string &layout_in,
                                                        std::vector<uint32_t> &node_list,
                                                        std::vector<SeedPageGroup> *seed_page_groups)
{
    node_list.clear();
    if (seed_page_groups != nullptr)
        seed_page_groups->clear();
    if (max_nodes == 0)
        return 0;

    std::string layout = normalize_disk_cache_layout(layout_in);
    if (layout.empty())
        layout = (k_hops == 0) ? "flat" : "node";

    if (layout == "bfs")
    {
        std::vector<uint32_t> bfs_nodes;
        const uint64_t pool_nodes = _merit_mem_pool != nullptr ? _merit_mem_pool->size() : 0;
        const uint64_t exclude = std::max<uint64_t>(memory_tier_exclude_count, pool_nodes);
        const uint64_t request = std::min<uint64_t>(_num_points, max_nodes + std::min<uint64_t>(exclude, _num_points));
        cache_bfs_levels(request, bfs_nodes, false, 1.0);
        node_list.reserve(static_cast<size_t>(max_nodes));
        for (uint32_t id : bfs_nodes)
        {
            if (_merit_mem_pool != nullptr && _merit_mem_pool->contains(id))
                continue;
            node_list.push_back(id);
            if (node_list.size() >= max_nodes)
                break;
        }
        if (node_list.size() != max_nodes)
        {
            diskann::cerr << "MERIT disk-cache BFS prefill selected only " << node_list.size() << " / " << max_nodes
                          << " nodes after excluding the N-cache." << std::endl;
            return -1;
        }
        diskann::cout << "MERIT disk-cache node list: BFS prefill selected " << node_list.size()
                      << " nodes after excluding "
                      << (_merit_mem_pool != nullptr ? _merit_mem_pool->size() : 0) << " N-cache nodes." << std::endl;
        return 0;
    }

    std::vector<uint64_t> node_expand;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> edges;
    if (HotnessProfiler::load(profile_prefix, node_expand, edges) != 0)
        return -1;
    if (node_expand.size() != _num_points)
    {
        diskann::cerr << "MERIT profile node_expand size " << node_expand.size() << " != num_points " << _num_points
                      << std::endl;
        return -1;
    }

    std::string mem_index_path = _disk_index_file;
    const std::string disk_suffix = "_disk.index";
    if (mem_index_path.size() >= disk_suffix.size() &&
        mem_index_path.compare(mem_index_path.size() - disk_suffix.size(), disk_suffix.size(), disk_suffix) == 0)
    {
        mem_index_path.replace(mem_index_path.size() - disk_suffix.size(), disk_suffix.size(), "_mem.index");
    }
    else
    {
        diskann::cerr << "Cannot derive _mem.index from disk index path: " << _disk_index_file << std::endl;
        return -1;
    }

    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;

    std::unordered_set<uint32_t> exclude_ids;
    if (memory_tier_exclude_count > 0)
    {
        std::vector<uint32_t> mem_tier;
        if (build_merit_memory_node_list(profile_prefix, memory_tier_exclude_count, mem_tier, 0) != 0)
            return -1;
        exclude_ids.insert(mem_tier.begin(), mem_tier.end());
    }

    if (layout == "flat" ||
        (k_hops == 0 && layout != "edge" && layout != "edge_dir" && layout != "edge_star" && layout != "edge_u" &&
         layout != "edge_pair" && layout != "edge_clique" && layout != "edge_replica" &&
         layout != "edge_star_dup" && layout != "frontier_page" && layout != "frontier_dup" &&
         layout != "directed_beam" && !is_directed_beam_pct_layout(layout) &&
         !is_directed_child_only_layout(layout) && !is_directed_child_replica_layout(layout) &&
         !is_directed_seed_replica_layout(layout) &&
         !diskann::is_directed_seed_replica_benefit_layout(layout) &&
         !diskann::is_directed_seed_replica_budget_layout(layout) &&
         !diskann::is_directed_seed_core_unique_fill_layout(layout) &&
         layout != "directed_beam_dual" &&
         layout != "directed_beam_inseed" &&
         layout != "directed_beam_tight" && layout != "directed_beam_starfill" &&
         layout != "directed_beam_top4first" && layout != "node_top4first" && layout != "cooccur_star" &&
         layout != "dbeam_cooccur" &&
         layout != "frontier_topk" &&
         layout != "parent" && layout != "parent_star" &&
         layout != "dbeam_estar_split" && layout != "frontier"))
    {
        std::vector<uint32_t> order(_num_points);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (node_expand[a] != node_expand[b])
                return node_expand[a] > node_expand[b];
            return a < b;
        });
        node_list.reserve(static_cast<size_t>(max_nodes));
        for (uint32_t id : order)
        {
            if (exclude_ids.find(id) != exclude_ids.end())
                continue;
            if (id >= node_expand.size() || node_expand[id] == 0)
                continue;
            node_list.push_back(id);
            if (node_list.size() >= max_nodes)
                break;
        }
        if (node_list.empty())
        {
            diskann::cerr << "MERIT disk-cache: flat hot-node list empty (max_nodes=" << max_nodes << ")."
                          << std::endl;
            return -1;
        }
        diskann::cout << "MERIT disk-cache node list: flat Top-" << node_list.size()
                      << " by node_expand (k_hops=0, exclude memory-tier=" << exclude_ids.size() << ")."
                      << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    VamanaGraph graph;
    // SIFT100M _mem.index adjacency is too large for many machines. When
    // MERIT_DCACHE_PROFILE_GRAPH=1, build a lightweight undirected graph from
    // profile directed edges so directed_beam / pct / parent packing can run.
    const char *prof_graph_env = std::getenv("MERIT_DCACHE_PROFILE_GRAPH");
    const bool use_profile_graph =
        (prof_graph_env != nullptr && std::strcmp(prof_graph_env, "0") != 0);
    if (use_profile_graph)
    {
        graph.num_points = node_expand.size();
        graph.width = 0;
        graph.medoid = 0;
        graph.adjacency.assign(static_cast<size_t>(graph.num_points), {});
        for (const auto &e : edges)
        {
            const uint32_t u = std::get<0>(e);
            const uint32_t v = std::get<1>(e);
            const uint64_t c = std::get<2>(e);
            if (c == 0 || u >= graph.num_points || v >= graph.num_points || u == v)
                continue;
            graph.adjacency[u].push_back(v);
            graph.adjacency[v].push_back(u);
            if (graph.adjacency[u].size() > graph.width)
                graph.width = static_cast<uint32_t>(graph.adjacency[u].size());
            if (graph.adjacency[v].size() > graph.width)
                graph.width = static_cast<uint32_t>(graph.adjacency[v].size());
        }
        diskann::cout << "MERIT disk-cache: profile-edge graph num_points=" << graph.num_points
                      << " directed_edges=" << edges.size() << " (skip " << mem_index_path << ")" << std::endl;
    }
    else if (load_vamana_graph(mem_index_path, graph) != 0)
    {
        diskann::cerr << "MERIT disk-cache: failed to load Vamana graph from " << mem_index_path << std::endl;
        return -1;
    }

    if (layout == "edge" || layout == "jiang")
    {
        if (compute_edge_disk_cache_list(graph, node_expand, edges, nps, k_hops, max_nodes, exclude_ids, node_list) !=
            0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: edge-importance packing, k_hops=" << k_hops
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "edge_dir" || layout == "edge_star" || layout == "edge_u" || layout == "edge_pair" ||
        layout == "edge_clique" || layout == "dir_edge_star")
    {
        if (compute_edge_variant_disk_cache_list(graph, node_expand, edges, nps, k_hops, max_nodes, exclude_ids,
                                                 layout, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: edge-variant packing layout=" << layout << ", k_hops=" << k_hops
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "edge_replica")
    {
        if (compute_edge_replica_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids, node_list) !=
            0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: edge-replica packing, slots=" << node_list.size()
                      << " (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        return 0;
    }

    if (layout == "edge_star_dup")
    {
        if (compute_edge_star_dup_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids, node_list) !=
            0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: edge_star_dup packing, slots=" << node_list.size()
                      << " (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        return 0;
    }

    if (layout == "frontier")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_frontier_disk_cache_list(graph, node_expand, edges, profile_prefix, nps, beam_width, max_nodes,
                                             exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: frontier co-location packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "frontier_page")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_frontier_page_disk_cache_list(graph, node_expand, edges, profile_prefix, nps, beam_width,
                                                  max_nodes, exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: frontier_page packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "frontier_dup")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_frontier_dup_disk_cache_list(graph, node_expand, edges, profile_prefix, nps, beam_width, max_nodes,
                                                 exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: frontier_dup packing, slots=" << node_list.size()
                      << " (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        return 0;
    }

    if (layout == "parent")
    {
        if (compute_profile_parent_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids,
                                                   node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: profile-parent packing, selected " << node_list.size()
                      << " nodes (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "parent_star")
    {
        if (compute_parent_star_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: parent_star packing, selected " << node_list.size()
                      << " nodes (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "dbeam_estar_split")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 1;
        if (compute_dbeam_estar_split_disk_cache_list(graph, node_expand, edges, nps, beam_width, max_nodes,
                                                      exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: dbeam_estar_split packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "directed_beam")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_directed_beam_disk_cache_list(graph, node_expand, edges, nps, beam_width, max_nodes, exclude_ids,
                                                  node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: Layout E (directed_beam) packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (is_directed_beam_pct_layout(layout))
    {
        const double pct = directed_beam_pct_layout_threshold(layout);
        if (compute_directed_beam_pct_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids,
                                                        node_list, pct, layout.c_str(), seed_page_groups) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: Layout E (" << layout << ") packing, selected "
                      << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size() << ")."
                      << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (is_directed_child_only_layout(layout))
    {
        const double pct = directed_child_only_layout_threshold(layout);
        if (compute_directed_child_only_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids,
                                                        node_list, pct, layout.c_str(), seed_page_groups) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: child-only (" << layout << ") packing, selected "
                      << node_list.size() << " unique nodes (exclude memory-tier=" << exclude_ids.size() << ")."
                      << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (is_directed_child_replica_layout(layout))
    {
        const double pct = directed_child_replica_layout_threshold(layout);
        if (compute_directed_child_replica_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids,
                                                           node_list, pct, layout.c_str(), seed_page_groups) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: unlimited child-replica (" << layout << ") packing, slots="
                      << node_list.size() << " seed_groups="
                      << (seed_page_groups != nullptr ? seed_page_groups->size() : 0)
                      << " (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        return 0;
    }

    if (layout == "directed_seed_replica")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_directed_seed_replica_beam_disk_cache_list(graph, node_expand, edges, nps, beam_width, max_nodes,
                                                               exclude_ids, node_list, seed_page_groups) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: directed_seed_replica (Layout E replica), beam_width="
                      << beam_width << ", slots=" << node_list.size() << " seed_groups="
                      << (seed_page_groups != nullptr ? seed_page_groups->size() : 0)
                      << " (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        return 0;
    }

    if (is_directed_seed_replica_pct_layout(layout))
    {
        const double pct = directed_seed_replica_layout_threshold(layout);
        if (compute_directed_seed_replica_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids,
                                                            node_list, pct, layout.c_str(), seed_page_groups) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: seed_replica (" << layout << ") packing, slots="
                      << node_list.size() << " seed_groups="
                      << (seed_page_groups != nullptr ? seed_page_groups->size() : 0)
                      << " (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (diskann::is_directed_seed_replica_benefit_layout(layout))
    {
        const double pct = diskann::directed_seed_replica_benefit_layout_threshold(layout);
        if (compute_directed_seed_replica_benefit_disk_cache_list(graph, node_expand, edges, nps, max_nodes,
                                                                  exclude_ids, node_list, pct, layout.c_str(),
                                                                  seed_page_groups) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: seed_replica_benefit (" << layout << ") packing, slots="
                      << node_list.size() << " seed_groups="
                      << (seed_page_groups != nullptr ? seed_page_groups->size() : 0)
                      << " (exclude memory-tier=" << exclude_ids.size() << ", benefit-ranked seeds)." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (diskann::is_directed_seed_replica_budget_layout(layout))
    {
        const double pct = directed_seed_replica_layout_threshold("directed_seed_replica_pct100");
        if (compute_directed_seed_replica_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids,
                                                          node_list, pct, layout.c_str(), seed_page_groups) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: seed_replica_budget (" << layout << ") packing, slots="
                      << node_list.size() << " seed_groups="
                      << (seed_page_groups != nullptr ? seed_page_groups->size() : 0)
                      << " (strict ratio budget, out-heat ranked, no flat append)." << std::endl;
        return 0;
    }

    if (diskann::is_directed_seed_core_unique_fill_layout(layout))
    {
        const double core_frac = diskann::directed_seed_core_unique_fill_frac(layout);
        if (compute_directed_seed_core_unique_fill_disk_cache_list(graph, node_expand, edges, nps, max_nodes,
                                                                   exclude_ids, node_list, core_frac, layout.c_str(),
                                                                   seed_page_groups) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: seed_core_unique_fill (" << layout << ") slots="
                      << node_list.size() << " seed_groups="
                      << (seed_page_groups != nullptr ? seed_page_groups->size() : 0)
                      << " (core_frac=" << core_frac << ", unique expand fill)." << std::endl;
        return 0;
    }

    if (layout == "directed_beam_starfill")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_directed_beam_starfill_disk_cache_list(graph, node_expand, edges, nps, beam_width, max_nodes,
                                                           exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: directed_beam_starfill (legacy E), beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "directed_beam_top4first")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_directed_beam_top4first_disk_cache_list(graph, node_expand, edges, nps, beam_width, max_nodes,
                                                            exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: directed_beam_top4first, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "directed_star")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 1;
        if (compute_directed_star_disk_cache_list(graph, node_expand, edges, nps, beam_width, max_nodes, exclude_ids,
                                                  node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: directed_star packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "directed_beam_hybrid")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 1;
        if (compute_directed_beam_hybrid_disk_cache_list(graph, node_expand, edges, nps, beam_width, max_nodes,
                                                         exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: directed_beam_hybrid packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "directed_beam_inseed")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 1;
        if (compute_directed_beam_inseed_disk_cache_list(graph, node_expand, edges, nps, beam_width, max_nodes,
                                                         exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: directed_beam_inseed packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "directed_beam_tight")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_directed_beam_tight_disk_cache_list(graph, node_expand, edges, nps, beam_width, max_nodes,
                                                        exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: directed_beam_tight packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "cooccur_star")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_cooccur_star_disk_cache_list(graph, node_expand, edges, profile_prefix, nps, beam_width, max_nodes,
                                                 exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: cooccur_star packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "dbeam_cooccur")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_dbeam_cooccur_disk_cache_list(graph, node_expand, edges, profile_prefix, nps, beam_width, max_nodes,
                                                  exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: dbeam_cooccur packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "frontier_topk")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_frontier_topk_disk_cache_list(graph, node_expand, edges, profile_prefix, nps, beam_width, max_nodes,
                                                  exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: frontier_topk packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "directed_beam_dual")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 1;
        if (compute_directed_beam_dual_disk_cache_list(graph, node_expand, edges, nps, beam_width, max_nodes,
                                                       exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: directed_beam_dual packing, beam_width=" << beam_width
                      << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                      << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "node_top4first")
    {
        if (compute_hot_node_top4first_disk_cache_list(graph, node_expand, edges, nps, k_hops, max_nodes, exclude_ids,
                                                       node_list) != 0)
            return -1;
        diskann::cout << "MERIT disk-cache node list: node_top4first packing, k_hops=" << k_hops << ", selected "
                      << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (compute_hot_node_disk_cache_list(graph, node_expand, edges, nps, k_hops, max_nodes, exclude_ids, node_list) !=
        0)
        return -1;

    diskann::cout << "MERIT disk-cache node list: hot-node path packing (avg edge weight), k_hops=" << k_hops
                  << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                  << ")." << std::endl;
    append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
    return 0;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::merit_memory_cached_count() const
{
    if (_merit_mem_pool && _merit_mem_pool->active())
        return _merit_mem_pool->size();
    return _nhood_cache.size();
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::print_merit_memory_cache_stats() const
{
    if (!_merit_mem_pool || !_merit_mem_pool->active())
        return;
    const uint64_t ghost_actual_admissions = _merit_mem_pool->ghost_actual_admissions();
    const uint64_t ghost_post_hits = _merit_mem_pool->ghost_post_admission_hits();
    const uint64_t ghost_reused_admissions = _merit_mem_pool->ghost_reused_admissions();
    const uint64_t ghost_admission_evictions = _merit_mem_pool->ghost_admission_evictions();
    const uint64_t ghost_zero_hit_evictions = _merit_mem_pool->ghost_zero_hit_evictions();
    diskann::cout << "MERIT ncache admission: policy=" << _merit_mem_pool->admission_policy_name()
                  << " hop_threshold=" << _merit_mem_pool->admission_hop_threshold()
                  << " rejected=" << _merit_mem_pool->admission_rejections()
                  << " second_hits=" << _merit_mem_pool->admission_second_hits()
                  << " low_priority=" << _merit_mem_pool->low_priority_admissions()
                  << " evictions=" << _merit_mem_pool->eviction_count()
                  << " adaptive_queries=" << _merit_mem_pool->adaptive_query_count()
                  << " half_life_queries=" << _merit_mem_pool->adaptive_half_life_queries()
                  << " adaptive_reject="
                  << (_merit_mem_pool->adaptive_reject_enabled() ? "on" : "off")
                  << " fast_miss=" << (_merit_mem_pool->fast_miss_enabled() ? "on" : "off")
                  << " fast_miss_bytes=" << _merit_mem_pool->fast_miss_bitmap_bytes()
                  << " fast_miss_bypasses=" << _merit_mem_pool->fast_miss_bypasses()
                  << " clock=" << (_merit_mem_pool->clock_enabled() ? "on" : "off")
                  << " clock_second_chances=" << _merit_mem_pool->clock_second_chances()
                  << " ewma_hit_rate=" << _merit_mem_pool->adaptive_hit_rate()
                  << " ewma_ghost_reuse=" << _merit_mem_pool->adaptive_ghost_reuse_rate()
                  << " ewma_evictions_per_query=" << _merit_mem_pool->adaptive_evictions_per_query()
                  << " transitions=" << _merit_mem_pool->adaptive_transition_count() << std::endl;
    if (_merit_mem_pool->ghost_stats_enabled())
    {
        diskann::cout << "MERIT ncache ghost value: actual_admissions=" << ghost_actual_admissions
                      << " post_admission_hits=" << ghost_post_hits
                      << " reused_admissions=" << ghost_reused_admissions
                      << " admission_evictions=" << ghost_admission_evictions
                      << " zero_hit_evictions=" << ghost_zero_hit_evictions
                      << " hits_per_admission="
                      << (ghost_actual_admissions == 0
                              ? 0.0
                              : static_cast<double>(ghost_post_hits) /
                                    static_cast<double>(ghost_actual_admissions))
                      << " reused_pct="
                      << (ghost_actual_admissions == 0
                              ? 0.0
                              : 100.0 * static_cast<double>(ghost_reused_admissions) /
                                    static_cast<double>(ghost_actual_admissions))
                      << " zero_hit_eviction_pct="
                      << (ghost_admission_evictions == 0
                              ? 0.0
                              : 100.0 * static_cast<double>(ghost_zero_hit_evictions) /
                                    static_cast<double>(ghost_admission_evictions))
                      << " tracking_bytes=" << _merit_mem_pool->ghost_tracking_bitmap_bytes()
                      << std::endl;
    }
    if (_merit_mem_pool->shadow_enabled())
    {
        const auto shadow_stats = _merit_mem_pool->shadow_stats();
        const uint64_t real_admit_samples = _merit_mem_pool->shadow_real_admit_samples();
        const uint64_t real_admit_ns = _merit_mem_pool->shadow_real_admit_ns();
        const uint64_t query_disk_reads = _merit_mem_pool->shadow_query_disk_reads();
        const uint64_t query_io_ns = _merit_mem_pool->shadow_query_io_ns();
        diskann::cout << "MERIT ncache shadow cost: control="
                      << (_merit_mem_pool->shadow_control_enabled() ? "on" : "off")
                      << " sample_divisor="
                      << _merit_mem_pool->shadow_sample_divisor()
                      << " real_admit_samples=" << real_admit_samples
                      << " avg_real_admit_ns="
                      << (real_admit_samples == 0
                              ? 0.0
                              : static_cast<double>(real_admit_ns) /
                                    static_cast<double>(real_admit_samples))
                      << " disk_reads=" << query_disk_reads
                      << " avg_io_ns_per_read="
                      << (query_disk_reads == 0
                              ? 0.0
                              : static_cast<double>(query_io_ns) /
                                    static_cast<double>(query_disk_reads))
                      << std::endl;
        for (const auto &shadow : shadow_stats)
        {
            diskann::cout << "MERIT ncache shadow: policy=" << shadow.name
                          << " resident=" << shadow.resident
                          << " accesses=" << shadow.accesses
                          << " hits=" << shadow.hits
                          << " misses=" << shadow.misses
                          << " hit_rate="
                          << (shadow.accesses == 0
                                  ? 0.0
                                  : static_cast<double>(shadow.hits) /
                                        static_cast<double>(shadow.accesses))
                          << " admissions=" << shadow.admissions
                          << " rejected=" << shadow.rejections
                          << " second_hits=" << shadow.second_hits
                          << " evictions=" << shadow.evictions
                          << " clock_second_chances=" << shadow.clock_second_chances
                          << " sampled_capacity=" << shadow.sampled_capacity
                          << " dropped=" << _merit_mem_pool->shadow_dropped_samples()
                          << std::endl;
        }
    }
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::merit_mem_pool_contains(uint32_t node_id) const
{
    return _merit_mem_pool && _merit_mem_pool->active() && _merit_mem_pool->contains(node_id);
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::merit_dc_map_contains(uint32_t node_id) const
{
    return merit_disk_cache_lookup_hit(node_id, nullptr);
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::verify_merit_disk_cache_against_base(uint64_t max_reports)
{
    if (_merit_dc_map.empty() || !_merit_disk_reader)
    {
        diskann::cout << "MERIT verify: disk cache empty or reader missing." << std::endl;
        return 0;
    }

    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;
    _merit_disk_reader->register_thread();

    char *sidecar_buf = nullptr;
    alloc_aligned((void **)&sidecar_buf, defaults::SECTOR_LEN, defaults::SECTOR_LEN);

    uint64_t checked = 0;
    uint64_t mismatches = 0;
    uint64_t reports = 0;

    std::vector<uint32_t> one_id(1);
    std::vector<T *> coord_bufs(1);
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs(1);
    std::vector<T> base_coord(_aligned_dim);
    std::vector<uint32_t> base_nbr(_max_degree + 1);

    for (const auto &kv : _merit_dc_map)
    {
        const uint32_t node_id = kv.first;
        if (kv.second.empty())
            continue;
        const MeritDiskLoc &loc = kv.second[0];

        one_id[0] = node_id;
        coord_bufs[0] = base_coord.data();
        nbr_bufs[0] = {0, base_nbr.data()};
        const auto ok = read_nodes(one_id, coord_bufs, nbr_bufs);
        if (!ok[0])
        {
            diskann::cerr << "MERIT verify: base read failed for node " << node_id << std::endl;
            continue;
        }

        AlignedRead ar;
        ar.offset = static_cast<uint64_t>(loc.sector) * defaults::SECTOR_LEN;
        ar.len = defaults::SECTOR_LEN;
        ar.buf = sidecar_buf;
        std::vector<AlignedRead> sidecar_reqs;
        sidecar_reqs.push_back(ar);
        _merit_disk_reader->read(sidecar_reqs, ctx);

        char *slot = sidecar_buf + static_cast<uint64_t>(loc.slot) * _max_node_len;
        const T *dc_coords = offset_to_node_coords(slot);
        uint32_t *dc_nhood = offset_to_node_nhood(slot);
        const uint32_t dc_nnbrs = *dc_nhood;

        bool bad = false;
        std::string reason;
        if (nbr_bufs[0].first != dc_nnbrs)
        {
            bad = true;
            reason = "nnbrs base=" + std::to_string(nbr_bufs[0].first) + " sidecar=" + std::to_string(dc_nnbrs);
        }
        else if (memcmp(base_coord.data(), dc_coords, _disk_bytes_per_point) != 0)
        {
            bad = true;
            reason = "coords differ";
        }
        else if (memcmp(base_nbr.data() + 1, dc_nhood + 1, static_cast<size_t>(dc_nnbrs) * sizeof(uint32_t)) != 0)
        {
            bad = true;
            reason = "nbr list differs";
        }

        checked++;
        if (bad)
        {
            mismatches++;
            if (reports < max_reports)
            {
                diskann::cerr << "MERIT verify mismatch node=" << node_id << " sector=" << loc.sector
                              << " slot=" << loc.slot << " nsectors=" << loc.nsectors << " " << reason << std::endl;
                reports++;
            }
        }
    }

    aligned_free(sidecar_buf);
    diskann::cout << "MERIT verify: checked=" << checked << " mismatches=" << mismatches << std::endl;
    return mismatches > 0 ? -1 : 0;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::clear_merit_memory_cache()
{
    if (_merit_mem_pool)
        _merit_mem_pool->clear();
    _merit_mem_pool.reset();
    _coord_cache.clear();
    _nhood_cache.clear();
    if (_nhood_cache_buf != nullptr)
    {
        delete[] _nhood_cache_buf;
        _nhood_cache_buf = nullptr;
    }
    if (_coord_cache_buf != nullptr)
    {
        diskann::aligned_free(_coord_cache_buf);
        _coord_cache_buf = nullptr;
    }
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_merit_memory_pool(const std::string &profile_prefix,
                                                    std::vector<uint32_t> &node_list)
{
    clear_merit_memory_cache();
    if (node_list.empty())
        return 0;

    std::vector<uint64_t> node_expand;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> edges;
    if (!profile_prefix.empty() && HotnessProfiler::load(profile_prefix, node_expand, edges) != 0)
        return -1;

    _merit_mem_pool = std::make_unique<MeritMemoryPool<T>>();
    _merit_mem_pool->init(node_list.size(), _aligned_dim, _max_degree, _num_points);

    const size_t num_cached_nodes = node_list.size();
    const size_t BLOCK_SIZE = 8;
    const size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++)
    {
        const size_t start_idx = block * BLOCK_SIZE;
        const size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);

        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        nodes_to_read.reserve(end_idx - start_idx);
        coord_buffers.reserve(end_idx - start_idx);
        nbr_buffers.reserve(end_idx - start_idx);

        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            const uint32_t slot = static_cast<uint32_t>(node_idx);
            nodes_to_read.push_back(node_list[node_idx]);
            coord_buffers.push_back(_merit_mem_pool->coord_ptr(slot));
            nbr_buffers.emplace_back(0, _merit_mem_pool->nhood_ptr(slot));
        }

        const auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (!read_status[i])
            {
                diskann::cerr << "MERIT memory pool: failed to read node " << nodes_to_read[i] << std::endl;
                return -1;
            }
            const uint32_t slot = static_cast<uint32_t>(start_idx + i);
            _merit_mem_pool->nhood_ptr(slot)[0] = nbr_buffers[i].first;
        }
    }

    _merit_mem_pool->commit_initial_load(node_list, node_expand);
    diskann::cout << "MERIT memory pool loaded " << _merit_mem_pool->size() << " nodes"
                  << (profile_prefix.empty() ? " (BFS around medoid)" : " (profile Top-N)") << std::endl;
    return 0;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_merit_memory_runtime_admit(bool enable)
{
    _merit_mem_runtime_admit = enable;
}

namespace
{
inline uint64_t merit_env_u64(const char *name, uint64_t def)
{
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0')
        return def;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v)
        return def;
    return static_cast<uint64_t>(parsed);
}

} // namespace

template <typename T, typename LabelT>
size_t PQFlashIndex<T, LabelT>::merit_dyn_loc_shard_index(uint32_t node_id) const
{
    return (static_cast<uint64_t>(node_id) * 11400714819323198485ull) &
           (_merit_dyn_loc_shard_count - 1);
}

template <typename T, typename LabelT>
typename PQFlashIndex<T, LabelT>::MeritDynLocShard &
PQFlashIndex<T, LabelT>::merit_dyn_loc_shard(uint32_t node_id)
{
    return *_merit_dyn_loc_shards[merit_dyn_loc_shard_index(node_id)];
}

template <typename T, typename LabelT>
const typename PQFlashIndex<T, LabelT>::MeritDynLocShard &
PQFlashIndex<T, LabelT>::merit_dyn_loc_shard(uint32_t node_id) const
{
    return *_merit_dyn_loc_shards[merit_dyn_loc_shard_index(node_id)];
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_loc_clear_unlocked()
{
    for (size_t shard_index = 0; shard_index < _merit_dyn_loc_shard_count; ++shard_index)
    {
        MeritDynLocShard &shard = *_merit_dyn_loc_shards[shard_index];
        MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::DynamicUnique, true);
        shard.locations.clear();
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_loc_add_unlocked(uint32_t node_id, const MeritDiskLoc &loc)
{
    MeritDynLocShard &shard = merit_dyn_loc_shard(node_id);
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::DynamicUnique, true);
    auto &locations = shard.locations[node_id];
    for (const MeritDiskLoc &candidate : locations)
        if (candidate.sector == loc.sector && candidate.slot == loc.slot)
            return;
    locations.push_back(loc);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_loc_erase_sector_unlocked(uint32_t node_id, uint32_t sector)
{
    MeritDynLocShard &shard = merit_dyn_loc_shard(node_id);
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::DynamicUnique, true);
    auto it = shard.locations.find(node_id);
    if (it == shard.locations.end())
        return;
    auto &locations = it.value();
    locations.erase(std::remove_if(locations.begin(), locations.end(),
                                   [sector](const MeritDiskLoc &loc) { return loc.sector == sector; }),
                    locations.end());
    if (locations.empty())
        shard.locations.erase(it);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_seed_page_unlocked(uint32_t seed_id, uint32_t &page_idx) const
{
    const MeritDynLocShard &shard = merit_dyn_loc_shard(seed_id);
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::DynamicShared, false);
    const auto it = shard.locations.find(seed_id);
    if (it == shard.locations.end())
        return false;
    for (const MeritDiskLoc &loc : it.value())
    {
        if (loc.slot == 0 && merit_dyn_is_sector(loc.sector))
        {
            const uint32_t candidate = loc.sector - MERIT_DYN_SECTOR_BASE;
            if (candidate < _merit_dyn_page_secondary.size() && _merit_dyn_page_secondary[candidate])
                continue;
            if (merit_dyn_page_live(candidate))
            {
                page_idx = candidate;
                return true;
            }
        }
    }
    return false;
}

template <typename T, typename LabelT>
size_t PQFlashIndex<T, LabelT>::merit_dc_loc_count_unlocked(uint32_t node_id) const
{
    if (_merit_dyn_enabled)
    {
        const MeritDynLocShard &shard = merit_dyn_loc_shard(node_id);
        MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::DynamicShared, false);
        const auto it = shard.locations.find(node_id);
        return it == shard.locations.end() ? 0 : it.value().size();
    }
    const auto it = _merit_dc_map.find(node_id);
    return it == _merit_dc_map.end() ? 0 : it->second.size();
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dc_loc_at_unlocked(uint32_t node_id, size_t index, MeritDiskLoc &loc) const
{
    if (_merit_dyn_enabled)
    {
        const MeritDynLocShard &shard = merit_dyn_loc_shard(node_id);
        MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::DynamicShared, false);
        const auto it = shard.locations.find(node_id);
        if (it == shard.locations.end() || index >= it.value().size())
            return false;
        loc = it.value()[index];
        return true;
    }
    const auto it = _merit_dc_map.find(node_id);
    if (it == _merit_dc_map.end() || index >= it->second.size())
        return false;
    loc = it->second[index];
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dc_loc_contains_unlocked(uint32_t node_id, const MeritDiskLoc &loc) const
{
    bool found = false;
    merit_dc_for_each_loc_unlocked(node_id, [&](const MeritDiskLoc &candidate, size_t) {
        if (candidate.sector == loc.sector && candidate.slot == loc.slot)
            found = true;
    });
    return found;
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::merit_dc_loc_directory_empty_unlocked() const
{
    if (!_merit_dyn_enabled)
        return _merit_dc_map.empty();
    for (size_t shard_index = 0; shard_index < _merit_dyn_loc_shard_count; ++shard_index)
    {
        const MeritDynLocShard &shard = *_merit_dyn_loc_shards[shard_index];
        MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::DynamicShared, false);
        if (!shard.locations.empty())
            return false;
    }
    return true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_writer_loop()
{
    merit_pin_background_thread();
    while (true)
    {
        std::shared_ptr<MeritDynWriteJob> job;
        {
            std::unique_lock<std::mutex> lock(_merit_dyn_write_q_mu);
            _merit_dyn_write_q_cv.wait(lock, [this] {
                return _merit_dyn_writer_stop.load(std::memory_order_acquire) || !_merit_dyn_write_q.empty();
            });
            if (_merit_dyn_writer_stop.load(std::memory_order_acquire) && _merit_dyn_write_q.empty())
                return;
            job = std::move(_merit_dyn_write_q.front());
            _merit_dyn_write_q.pop_front();
        }
        _merit_dyn_write_q_cv.notify_all();
        if (job == nullptr)
            continue;

        const uint64_t io_started_ns = merit_lock_now_ns();
        if (job->enqueued_ns != 0)
            _merit_dyn_writer_queue_ns.fetch_add(io_started_ns - job->enqueued_ns, std::memory_order_relaxed);
        bool ok = false;
        if (job->op == MeritDynWriteOp::Sync)
        {
#ifndef _WINDOWS
            ok = _merit_dyn_write_fd >= 0 && ::fdatasync(_merit_dyn_write_fd) == 0;
#else
            std::lock_guard<std::mutex> writer_lock(_merit_dyn_writer_mu);
            _merit_dyn_write_stream.flush();
            ok = _merit_dyn_write_stream.good();
#endif
        }
        else
        {
#ifndef _WINDOWS
            if (_merit_dyn_write_fd >= 0 && job->data && job->nbytes > 0)
            {
                ssize_t written = -1;
                do
                {
                    written = ::pwrite(_merit_dyn_write_fd, job->data.get(), job->nbytes, job->offset);
                } while (written < 0 && errno == EINTR);
                ok = written == static_cast<ssize_t>(job->nbytes);
            }
#else
            if (job->data && job->nbytes > 0)
            {
                std::lock_guard<std::mutex> writer_lock(_merit_dyn_writer_mu);
                _merit_dyn_write_stream.clear();
                _merit_dyn_write_stream.seekp(static_cast<std::streamoff>(job->offset));
                _merit_dyn_write_stream.write(job->data.get(), static_cast<std::streamsize>(job->nbytes));
                _merit_dyn_write_stream.flush();
                ok = _merit_dyn_write_stream.good();
            }
#endif
        }
        _merit_dyn_writer_io_ns.fetch_add(merit_lock_now_ns() - io_started_ns, std::memory_order_relaxed);
        _merit_dyn_writer_completed.fetch_add(1, std::memory_order_relaxed);
        if (!ok)
            _merit_dyn_writer_failed.fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> done_lock(job->mu);
            job->ok = ok;
            job->done = true;
        }
        job->cv.notify_all();
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::start_merit_dyn_writer()
{
    if (_merit_dyn_writer_started)
        return;
    _merit_dyn_writer_stop.store(false, std::memory_order_release);
    _merit_dyn_writer_threads.clear();
    _merit_dyn_writer_threads.reserve(_merit_dyn_writer_thread_count);
    for (size_t thread = 0; thread < _merit_dyn_writer_thread_count; ++thread)
        _merit_dyn_writer_threads.emplace_back([this] { merit_dyn_writer_loop(); });
    _merit_dyn_writer_started = true;
    if (_merit_dyn_enabled && merit_env_u64("MERIT_REFRESH_THREAD", 0) != 0 && !_merit_refresh_thread_on)
    {
        _merit_refresh_thread_stop.store(false, std::memory_order_release);
        _merit_refresh_thread_on = true;
        _merit_refresh_thread = std::thread([this] { merit_refresh_thread_loop(); });
        diskann::cout << "MERIT refresh_thread=on" << std::endl;
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_refresh_thread_loop()
{
    merit_is_refresh_thread() = true;
    merit_pin_background_thread();
    while (!_merit_refresh_thread_stop.load(std::memory_order_acquire))
    {
        if (_merit_refresh_needed.load(std::memory_order_acquire) &&
            _merit_pending_pair_count.load(std::memory_order_acquire) < _merit_pending_pair_cap)
        {
            merit_dyn_request_refresh();
            std::this_thread::yield();
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::microseconds(5));
        }
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::stop_merit_dyn_writer()
{
    if (_merit_refresh_thread_on)
    {
        _merit_refresh_thread_stop.store(true, std::memory_order_release);
        if (_merit_refresh_thread.joinable())
            _merit_refresh_thread.join();
        _merit_refresh_thread_on = false;
    }
    if (!_merit_dyn_writer_started)
        return;
    {
        std::lock_guard<std::mutex> lock(_merit_dyn_write_q_mu);
        _merit_dyn_writer_stop.store(true, std::memory_order_release);
    }
    _merit_dyn_write_q_cv.notify_all();
    for (std::thread &thread : _merit_dyn_writer_threads)
        if (thread.joinable())
            thread.join();
    _merit_dyn_writer_threads.clear();
    _merit_dyn_writer_started = false;

    // Fail any leftover jobs (should be empty after drain; stop waits for empty).
    std::deque<std::shared_ptr<MeritDynWriteJob>> leftover;
    {
        std::lock_guard<std::mutex> lock(_merit_dyn_write_q_mu);
        leftover.swap(_merit_dyn_write_q);
    }
    for (auto &job : leftover)
    {
        if (job == nullptr)
            continue;
        std::lock_guard<std::mutex> done_lock(job->mu);
        job->ok = false;
        job->done = true;
        job->cv.notify_all();
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_writer_submit_and_wait(MeritDynWriteOp op, off_t offset, const void *data,
                                                               size_t nbytes)
{
    if (!_merit_dyn_writer_started || _merit_dyn_writer_stop.load(std::memory_order_acquire))
        return false;

    auto job = std::make_shared<MeritDynWriteJob>();
    job->op = op;
    job->offset = offset;
    job->nbytes = nbytes;
    job->enqueued_ns = merit_lock_now_ns();
    if (op == MeritDynWriteOp::Write)
    {
        if (data == nullptr || nbytes == 0)
            return false;
        void *aligned = nullptr;
#ifndef _WINDOWS
        if (::posix_memalign(&aligned, defaults::SECTOR_LEN, nbytes) != 0)
            return false;
#else
        aligned = std::malloc(nbytes);
        if (aligned == nullptr)
            return false;
#endif
        std::memcpy(aligned, data, nbytes);
        job->data.reset(static_cast<char *>(aligned));
    }

    {
        std::unique_lock<std::mutex> lock(_merit_dyn_write_q_mu);
        _merit_dyn_write_q_cv.wait(lock, [this] {
            return _merit_dyn_writer_stop.load(std::memory_order_acquire) ||
                   _merit_dyn_write_q.size() < _merit_dyn_write_q_cap;
        });
        if (_merit_dyn_writer_stop.load(std::memory_order_acquire))
            return false;
        _merit_dyn_write_q.push_back(job);
        _merit_dyn_writer_enqueued.fetch_add(1, std::memory_order_relaxed);
    }
    _merit_dyn_write_q_cv.notify_all();

    std::unique_lock<std::mutex> done_lock(job->mu);
    job->cv.wait(done_lock, [&job] { return job->done; });
    return job->ok;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::enable_merit_dynamic_3cache(bool enable, const std::string &flush_prefix)
{
    if (!enable)
    {
        stop_merit_dyn_patch_worker();
        stop_merit_dyn_commit_workers();
        stop_merit_dyn_writer();
        _merit_dyn_enabled = false;
        return;
    }
    stop_merit_dyn_patch_worker();
    stop_merit_dyn_commit_workers();
    stop_merit_dyn_writer();
    _merit_dyn_enabled = true;
    g_dyn_probe.reset();
    const bool import_static_prefill =
        merit_env_u64("MERIT_DYNAMIC_PREFILL_STATIC", 0) != 0 && !_merit_dc_path.empty() && !_merit_dc_map.empty();

    const uint64_t ncache_n =
        (_merit_mem_pool != nullptr && _merit_mem_pool->active()) ? _merit_mem_pool->capacity() : 0;
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    const uint64_t base_pages =
        (_num_points > 0) ? (1ULL + (_num_points + nps - 1ULL) / nps) : 0ULL;
    _merit_dyn_page_cap = merit_env_u64(
        "MERIT_DCACHE_CAP",
        base_pages > 0 ? std::max<uint64_t>((base_pages + 4ULL) / 5ULL, 1024) : 4096);
    const uint64_t requested_mcache_cap =
        merit_env_u64("MERIT_MCACHE_CAP", _merit_dyn_page_cap > 0 ? _merit_dyn_page_cap * 2ULL : 65536ULL);
    const uint64_t mcache_cap = std::max<uint64_t>(requested_mcache_cap, _merit_dyn_page_cap * 2ULL);
    const uint8_t sig_t = static_cast<uint8_t>(merit_env_u64("MERIT_SEED_T", MeritMetadataCache::kDefaultSigT));
    _merit_mcache.init(mcache_cap, sig_t, _num_points);
    _merit_deferred_ncache_page_write = merit_env_u64("MERIT_DEFERRED_NCACHE_PAGE_WRITE", 0) != 0;
    _merit_dcache_evict_ncache_on_commit =
        merit_env_u64("MERIT_DCACHE_EVICT_NCACHE_ON_COMMIT", 0) != 0;
    _merit_dcache_ncache_erases.store(0, std::memory_order_relaxed);
    _merit_dcache_no_seed_replace = merit_env_u64("MERIT_DCACHE_NO_SEED_REPLACE", 0) != 0;
    _merit_dcache_second_page = merit_env_u64("MERIT_DCACHE_SECOND_PAGE", 0) != 0;
    _merit_dcache_max_pages = 1;
    if (_merit_dcache_second_page)
    {
        uint64_t max_pages = merit_env_u64("MERIT_DCACHE_MAX_PAGES", 5);
        if (max_pages < 2)
            max_pages = 2;
        if (max_pages > 5)
            max_pages = 5;
        _merit_dcache_max_pages = static_cast<uint8_t>(max_pages);
    }
    _merit_dcache_pair_ios.store(0, std::memory_order_relaxed);
    _merit_span_adaptive = _merit_dcache_second_page && merit_env_u64("MERIT_DCACHE_SPAN_ADAPTIVE", 0) != 0;
    if (const char *v = std::getenv("MERIT_DCACHE_SPAN_MIN_HITS"))
        _merit_span_min_hits = std::strtof(v, nullptr);
    _merit_span_explore = static_cast<uint32_t>(merit_env_u64("MERIT_DCACHE_SPAN_EXPLORE", 16));
    _merit_span_full_reads.store(0, std::memory_order_relaxed);
    _merit_span_primary_only.store(0, std::memory_order_relaxed);
    _merit_dcache_freeze_after = merit_env_u64("MERIT_DCACHE_FREEZE_AFTER", 0);
    _merit_member_snapshot_at = merit_env_u64("MERIT_DCACHE_MEMBER_SNAPSHOT_AT", 0);
    _merit_member_snapshot_done.store(false, std::memory_order_relaxed);
    _merit_member_at_snapshot.clear();
    _merit_member_snapshot_slots.store(0, std::memory_order_relaxed);
    _merit_prefetch_from_early.store(0, std::memory_order_relaxed);
    _merit_prefetch_from_late.store(0, std::memory_order_relaxed);
    for (uint32_t h = 0; h < 128; ++h)
    {
        _merit_early_hop_hist[h].store(0, std::memory_order_relaxed);
        _merit_late_hop_hist[h].store(0, std::memory_order_relaxed);
    }
    _merit_dcache_query_patch = merit_env_u64("MERIT_DCACHE_QUERY_PATCH", 1) != 0;
    _merit_dcache_query_patch_async = merit_env_u64("MERIT_DCACHE_QUERY_PATCH_ASYNC", 1) != 0;
    _merit_dcache_query_patch_max = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(64, merit_env_u64("MERIT_DCACHE_QUERY_PATCH_MAX", 4))));
    _merit_patch_q_cap = static_cast<size_t>(
        std::max<uint64_t>(8, std::min<uint64_t>(4096, merit_env_u64("MERIT_DCACHE_QUERY_PATCH_Q", 64))));
    _merit_patch_batch_us = static_cast<uint32_t>(
        std::min<uint64_t>(10000, merit_env_u64("MERIT_DCACHE_QUERY_PATCH_BATCH_US", 100)));
    _merit_patch_worker_no_disk = merit_env_u64("MERIT_DCACHE_QUERY_PATCH_NO_DISK", 0) != 0;
    _merit_dcache_query_patch_trig.store(0, std::memory_order_relaxed);
    _merit_dcache_query_patch_ok.store(0, std::memory_order_relaxed);
    _merit_dcache_query_patch_drop.store(0, std::memory_order_relaxed);
    _merit_real_io_coaccess = merit_env_u64("MERIT_REAL_IO_COACCESS", 1) != 0;
    _merit_real_io_window = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(16, merit_env_u64("MERIT_REAL_IO_WINDOW", 4))));
    _merit_real_io_sample_queries = static_cast<uint32_t>(
        std::max<uint64_t>(1, merit_env_u64("MERIT_REAL_IO_SAMPLE_QUERIES", 64)));
    _merit_real_io_max_partners = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(128, merit_env_u64("MERIT_REAL_IO_MAX_PARTNERS", 64))));
    _merit_real_io_queries.store(0, std::memory_order_relaxed);
    _merit_real_io_sampled_queries.store(0, std::memory_order_relaxed);
    _merit_real_io_nodes.store(0, std::memory_order_relaxed);
    _merit_real_io_pairs.store(0, std::memory_order_relaxed);
    _merit_pending_buffer_cap_bytes =
        _merit_deferred_ncache_page_write
            ? merit_env_u64("MERIT_PENDING_BUFFER_BYTES", 4ULL * 1024ULL * 1024ULL)
            : 0;
    _merit_pending_buffer_payload_bytes = 0;
    _merit_pending_require_full_page =
        _merit_pending_buffer_cap_bytes > 0 &&
        merit_env_u64("MERIT_PENDING_REQUIRE_FULL_PAGE", 1) != 0;
    _merit_pending_buffer_ready_percent = static_cast<uint32_t>(
        std::min<uint64_t>(100, merit_env_u64("MERIT_PENDING_BUFFER_READY_PERCENT", 0)));
    _merit_pending_buffer_ready_min_members = static_cast<uint16_t>(
        std::min<uint64_t>(std::numeric_limits<uint16_t>::max(),
                           std::max<uint64_t>(1, merit_env_u64("MERIT_PENDING_BUFFER_READY_MIN_MEMBERS", 2))));
    _merit_pending_buffer_inserts.store(0, std::memory_order_relaxed);
    _merit_pending_buffer_ready_writes.store(0, std::memory_order_relaxed);
    _merit_pending_buffer_pressure_writes.store(0, std::memory_order_relaxed);
    _merit_pending_buffer_drops.store(0, std::memory_order_relaxed);
    _merit_pending_member_refreshes.store(0, std::memory_order_relaxed);
    _merit_pending_became_full.store(0, std::memory_order_relaxed);
    _merit_monitor_warmup_queries = static_cast<uint32_t>(
        std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                           merit_env_u64("MERIT_MONITOR_WARMUP_QUERIES", 0)));
    _merit_monitor_interval_queries = static_cast<uint32_t>(
        std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                           std::max<uint64_t>(1, merit_env_u64("MERIT_MONITOR_INTERVAL_QUERIES", 1000000))));
    _merit_dcache_net_gate_enabled =
        merit_env_u64("MERIT_DCACHE_NET_GATE", _merit_mcache.adaptive_update_enabled() ? 1 : 0) != 0;
    _merit_dcache_probe_period = static_cast<uint32_t>(
        std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                           std::max<uint64_t>(1, merit_env_u64("MERIT_DCACHE_PROBE_PERIOD", 100))));
    _merit_dcache_gate_active.store(true, std::memory_order_relaxed);
    _merit_dcache_query_issued.store(0, std::memory_order_relaxed);
    _merit_dcache_query_completed.store(0, std::memory_order_relaxed);
    _merit_dcache_enabled_queries.store(0, std::memory_order_relaxed);
    _merit_dcache_avoided_pages.store(0, std::memory_order_relaxed);
    _merit_dcache_physical_reads.store(0, std::memory_order_relaxed);
    _merit_dcache_served_nodes.store(0, std::memory_order_relaxed);
    _merit_dcache_page_writes.store(0, std::memory_order_relaxed);
    _merit_dcache_gate_transitions.store(0, std::memory_order_relaxed);
    // Each dedicated-page write costs SSD bandwidth that competes with query reads.
    if (const char *v = std::getenv("MERIT_DCACHE_NET_WRITE_WEIGHT"))
        _merit_dcache_net_write_weight = std::max(0.0, std::strtod(v, nullptr));
    else
        _merit_dcache_net_write_weight = 0.0;
    const auto merit_env_double = [](const char *name, double fallback, double minimum) {
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0')
            return fallback;
        char *end = nullptr;
        const double parsed = std::strtod(value, &end);
        return end == value || !std::isfinite(parsed) ? fallback : std::max(minimum, parsed);
    };
    _merit_dcache_base_read_us =
        merit_env_double("MERIT_DCACHE_BASE_READ_US", 40.0, 0.0);
    _merit_dcache_read_us =
        merit_env_double("MERIT_DCACHE_READ_US", 50.0, 0.0);
    _merit_dcache_write_us =
        merit_env_double("MERIT_DCACHE_WRITE_US", 56.0, 0.0);
    _merit_dcache_cpu_us =
        merit_env_double("MERIT_DCACHE_CPU_US", 20.0, 0.0);
    _merit_dcache_gate_off_us =
        merit_env_double("MERIT_DCACHE_GATE_OFF_US", 0.0, -1000000.0);
    _merit_dcache_gate_on_us =
        merit_env_double("MERIT_DCACHE_GATE_ON_US", 10.0, -1000000.0);
    {
        std::lock_guard<std::mutex> gate_lock(_merit_dcache_gate_mu);
        _merit_dcache_last_enabled_queries = 0;
        _merit_dcache_last_avoided_pages = 0;
        _merit_dcache_last_physical_reads = 0;
        _merit_dcache_last_served_nodes = 0;
        _merit_dcache_last_page_writes = 0;
        _merit_dcache_low_windows = 0;
        _merit_dcache_high_windows = 0;
        _merit_dcache_inactive_stages = 0;
    }
    const auto pending_knob = [](const char *name, uint32_t fallback) {
        return static_cast<uint32_t>(
            std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                               std::max<uint64_t>(1, merit_env_u64(name, fallback))));
    };
    _merit_pending_max_age_queries = pending_knob("MERIT_PENDING_MAX_AGE_QUERIES", 65536);
    _merit_pending_check_interval = pending_knob("MERIT_PENDING_CHECK_INTERVAL", 2048);
    _merit_pending_force_batch = pending_knob("MERIT_PENDING_FORCE_BATCH", 4);
    const uint64_t pending_pair_default = _merit_pending_buffer_cap_bytes > 0 ? 16384 : 4096;
    _merit_pending_pair_cap = static_cast<size_t>(
        std::min<uint64_t>(MERIT_PENDING_PAIR_CAP_MAX,
                           std::max<uint64_t>(1, merit_env_u64("MERIT_PENDING_PAIR_CAP", pending_pair_default))));
    const uint64_t requested_heap_shards =
        merit_env_u64("MERIT_HEAP_SHARDS", MERIT_DEFAULT_HEAP_SHARDS);
    _merit_heap_shard_count =
        requested_heap_shards >= 1 && requested_heap_shards <= MERIT_MAX_HEAP_SHARDS &&
                (requested_heap_shards & (requested_heap_shards - 1)) == 0
            ? static_cast<size_t>(requested_heap_shards)
            : MERIT_DEFAULT_HEAP_SHARDS;
    const size_t heap_slot_capacity = static_cast<size_t>(_merit_mcache.slot_capacity());
    for (auto &shard : _merit_heap_shards)
    {
        MeritTimedMutexGuard hlock(shard.mu, MeritLockKind::Heap);
        shard.max_heap.clear();
        shard.min_heap.clear();
    }
    _merit_heap_position.assign(heap_slot_capacity, MeritMetadataCache::kInvalid);
    _merit_heap_kind.assign(heap_slot_capacity, static_cast<uint8_t>(MeritHeapKind::None));
    _merit_node_state.reset(new std::atomic<uint8_t>[heap_slot_capacity]);
    _merit_node_state_capacity = heap_slot_capacity;
    for (size_t slot = 0; slot < heap_slot_capacity; ++slot)
        _merit_node_state[slot].store(static_cast<uint8_t>(MeritNodeState::NonSeed), std::memory_order_relaxed);
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
        _merit_ready_pairs.clear();
        merit_pending_member_clear();
        _merit_deletion_to_pending.clear();
        _merit_pending_buffer_buckets.clear();
        _merit_pending_buffer_buckets.resize(static_cast<size_t>(nps) + 1);
        _merit_reserved_free_pages = 0;
        _merit_pending_pair_count.store(0, std::memory_order_relaxed);
        _merit_deletion_pair_count.store(0, std::memory_order_relaxed);
        _merit_query_epoch.store(0, std::memory_order_relaxed);
        _merit_refresh_needed.store(false, std::memory_order_relaxed);
        _merit_refresh_running.store(false, std::memory_order_relaxed);
        _merit_score_unit.store(1.0f);
        _merit_decay_query_count.store(0, std::memory_order_relaxed);
        _merit_decay_access_count.store(0, std::memory_order_relaxed);
        _merit_decay_cycle_start.store(0, std::memory_order_relaxed);
        _merit_decay_half_life_accesses =
            static_cast<double>(std::max<uint64_t>(1, merit_env_u64("MERIT_DECAY_HALF_LIFE_ACCESSES", 2800000)));
        _merit_decay_cycles.store(0, std::memory_order_relaxed);
        _merit_decay_early_cycles.store(0, std::memory_order_relaxed);
        _merit_decay_started.store(false, std::memory_order_relaxed);
        _merit_pair_refresh = 0;
        _merit_heap_writes = 0;
        _merit_heap_deletes = 0;
        _merit_ncache_write_trig = 0;
    }

    if (_merit_mem_pool != nullptr && _merit_mem_pool->active() && _medoids != nullptr)
    {
        for (size_t i = 0; i < _num_medoids; i++)
            _merit_mem_pool->pin(_medoids[i]);
    }

    _merit_dyn_path = flush_prefix.empty() ? std::string() : (flush_prefix + "_merit_dc.dyn.data");
    if (_merit_dyn_path.empty())
        throw ANNException("Dynamic 3-cache requires a disk-cache file path.", -1, __FUNCSIG__, __FILE__, __LINE__);
    if (_merit_dyn_reader)
    {
        _merit_dyn_reader->close();
        _merit_dyn_reader.reset();
    }
    if (_merit_dyn_write_stream.is_open())
        _merit_dyn_write_stream.close();
#ifndef _WINDOWS
    if (_merit_dyn_write_fd >= 0)
    {
        ::close(_merit_dyn_write_fd);
        _merit_dyn_write_fd = -1;
    }
#endif
    const uint64_t requested_loc_shards =
        merit_env_u64("MERIT_DYN_LOC_SHARDS", 64);
    _merit_dyn_loc_shard_count =
        requested_loc_shards >= 1 && requested_loc_shards <= MERIT_DYN_LOC_MAX_SHARDS &&
                (requested_loc_shards & (requested_loc_shards - 1)) == 0
            ? static_cast<size_t>(requested_loc_shards)
            : 64;
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicUnique, true);
        _merit_dyn_page_members.clear();
        _merit_dyn_page_member_count.clear();
        _merit_dyn_members_per_page = static_cast<uint32_t>(nps);
        if (!import_static_prefill)
            _merit_dc_map.clear();
        merit_dyn_loc_clear_unlocked();
        const uint64_t page_multiplier =
            _merit_dcache_second_page ? static_cast<uint64_t>(_merit_dcache_max_pages) : 1ULL;
        // Keep seed_member_loc / canonical through static prefill import; they encode
        // the real seed-group page layout. Clear after import if not prefilling.
        if (!import_static_prefill)
        {
            _merit_dc_seed_member_loc.clear();
            _merit_dc_seed_canonical_loc.clear();
        }
        _merit_dyn_free.clear();
        _merit_dyn_committed_pages = 0;
        _merit_dyn_next_page = 0;
        _merit_dyn_physical_cap = _merit_dyn_page_cap * page_multiplier + std::max<uint64_t>(_max_nthreads, 1) + 1;
        if (_merit_dyn_physical_cap >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / _merit_dyn_members_per_page)
            throw ANNException("Dynamic d-cache member directory is too large.", -1, __FUNCSIG__, __FILE__,
                               __LINE__);
        _merit_dyn_page_members.assign(
            static_cast<size_t>(_merit_dyn_physical_cap) * _merit_dyn_members_per_page, MERIT_DYN_INVALID_PAGE);
        _merit_dyn_page_member_count.assign(static_cast<size_t>(_merit_dyn_physical_cap), 0);
        _merit_dyn_page_sibling.assign(static_cast<size_t>(_merit_dyn_physical_cap), MERIT_DYN_INVALID_PAGE);
        _merit_dyn_page_secondary.assign(static_cast<size_t>(_merit_dyn_physical_cap), 0);
        _merit_dyn_page_span_len.assign(static_cast<size_t>(_merit_dyn_physical_cap), 0);
        _merit_dyn_page_span_base.assign(static_cast<size_t>(_merit_dyn_physical_cap), MERIT_DYN_INVALID_PAGE);
        _merit_dyn_page_readers.reset(new std::atomic<uint32_t>[_merit_dyn_physical_cap]);
        _merit_dyn_page_retired.reset(new std::atomic<uint8_t>[_merit_dyn_physical_cap]);
        _merit_span_reads.reset(new std::atomic<uint32_t>[_merit_dyn_physical_cap]);
        _merit_span_hits.reset(new std::atomic<uint32_t>[_merit_dyn_physical_cap]);
        _merit_span_decisions.reset(new std::atomic<uint32_t>[_merit_dyn_physical_cap]);
        for (uint64_t i = 0; i < _merit_dyn_physical_cap; ++i)
        {
            _merit_dyn_page_readers[i].store(0, std::memory_order_relaxed);
            _merit_dyn_page_retired[i].store(0, std::memory_order_relaxed);
            _merit_span_reads[i].store(0, std::memory_order_relaxed);
            _merit_span_hits[i].store(0, std::memory_order_relaxed);
            _merit_span_decisions[i].store(0, std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> rlock(_merit_dyn_reclaim_mu);
            _merit_dyn_reclaim.clear();
        }
    }
    const uint64_t dyn_file_bytes = _merit_dyn_physical_cap * defaults::SECTOR_LEN;
#ifndef _WINDOWS
    _merit_dyn_write_fd =
        ::open(_merit_dyn_path.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_DIRECT | O_LARGEFILE, 0644);
    if (_merit_dyn_write_fd < 0 || ::ftruncate(_merit_dyn_write_fd, static_cast<off_t>(dyn_file_bytes)) != 0)
        throw ANNException("Failed to create dynamic disk-cache file.", -1, __FUNCSIG__, __FILE__, __LINE__);
#else
    _merit_dyn_write_stream.open(_merit_dyn_path,
                                 std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!_merit_dyn_write_stream.is_open())
        throw ANNException("Failed to create dynamic disk-cache file.", -1, __FUNCSIG__, __FILE__, __LINE__);
    _merit_dyn_write_stream.seekp(static_cast<std::streamoff>(dyn_file_bytes - 1));
    const char zero = 0;
    _merit_dyn_write_stream.write(&zero, 1);
    _merit_dyn_write_stream.flush();
    if (!_merit_dyn_write_stream.good())
        throw ANNException("Failed to preallocate dynamic disk-cache file.", -1, __FUNCSIG__, __FILE__, __LINE__);
#endif
    _merit_dyn_write_q_cap = static_cast<size_t>(
        std::max<uint64_t>(1, merit_env_u64("MERIT_DCACHE_WRITER_QUEUE", 64)));
    _merit_dyn_writer_thread_count = static_cast<size_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(16, merit_env_u64("MERIT_DCACHE_WRITER_THREADS", 1))));
    _merit_commit_thread_count = static_cast<size_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(16, merit_env_u64(
            "MERIT_DCACHE_COMMIT_THREADS", _merit_dyn_writer_thread_count))));
    _merit_commit_q_cap = static_cast<size_t>(
        std::max<uint64_t>(8, std::min<uint64_t>(65536, merit_env_u64("MERIT_DCACHE_COMMIT_QUEUE", 256))));
    _merit_dyn_writer_enqueued.store(0, std::memory_order_relaxed);
    _merit_dyn_writer_queue_ns.store(0, std::memory_order_relaxed);
    _merit_dyn_writer_io_ns.store(0, std::memory_order_relaxed);
    _merit_dyn_writer_completed.store(0, std::memory_order_relaxed);
    _merit_dyn_writer_failed.store(0, std::memory_order_relaxed);
    _merit_commit_enqueued.store(0, std::memory_order_relaxed);
    _merit_commit_coalesced.store(0, std::memory_order_relaxed);
    _merit_commit_dropped.store(0, std::memory_order_relaxed);
    _merit_commit_completed.store(0, std::memory_order_relaxed);
    _merit_commit_failed.store(0, std::memory_order_relaxed);
    _merit_commit_queue_ns.store(0, std::memory_order_relaxed);
    _merit_commit_work_ns.store(0, std::memory_order_relaxed);
    for (size_t source = 0; source < _merit_commit_source_enqueued.size(); ++source)
    {
        _merit_commit_source_enqueued[source].store(0, std::memory_order_relaxed);
        _merit_commit_source_completed[source].store(0, std::memory_order_relaxed);
    }
    start_merit_dyn_writer();
#ifndef _WINDOWS
    _merit_dyn_reader.reset(new LinuxAlignedFileReader());
#else
    _merit_dyn_reader.reset(new WindowsAlignedFileReader());
#endif
    _merit_dyn_reader->open(_merit_dyn_path);
#pragma omp parallel for num_threads((int)_max_nthreads)
    for (int64_t thread = 0; thread < static_cast<int64_t>(_max_nthreads); ++thread)
    {
#pragma omp critical
        { _merit_dyn_reader->register_thread(); }
    }
    start_merit_dyn_patch_worker();
    start_merit_dyn_commit_workers();
    if (import_static_prefill && merit_dyn_import_static_prefill() != 0)
        throw ANNException("Failed to import the static BFS d-cache into dynamic MERIT.", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    // After dyn import, per-member locs are authoritative. Seed-first canonical-extent
    // IO is for raw static layout; keeping it on with remapped sidecars corrupts reads
    // when a node also appears on another seed's page.
    if (import_static_prefill)
    {
        _merit_seed_first_lookup = false;
        diskann::cout << "MERIT static prefill: seed-first lookup forced off (use dyn member locs)"
                      << std::endl;
    }
    _merit_mem_runtime_admit = true;
    if (!import_static_prefill)
        _merit_seed_first_lookup = true;
    _merit_stash_cap = _merit_deferred_ncache_page_write
                           ? 0
                           : merit_env_u64("MERIT_STASH_CAP",
                                           ncache_n > 0 ? std::max<uint64_t>(ncache_n * 2ULL, 4096) : 16384);
    {
        std::lock_guard<std::mutex> slock(_merit_stash_mu);
        _merit_payload_stash.clear();
        _merit_stash_lru.clear();
        _merit_pending_buffer_payload_bytes = 0;
        _merit_parent_fetched_by_slot.assign(static_cast<size_t>(_merit_mcache.slot_capacity()),
                                             MeritRecentChildren{});
        _merit_parent_fetched_owner.assign(static_cast<size_t>(_merit_mcache.slot_capacity()),
                                           MeritMetadataCache::kInvalid);
    }
    diskann::cout << "MERIT dynamic 3-cache: m-cache cap=" << mcache_cap << " (2x seed cap) "
                  << "edge_storage=vector(unbounded) T=" << (int)sig_t
                  << " heap+delay-write nps=" << nps << " d-cache pages<=" << _merit_dyn_page_cap
                  << " (~20% of base " << base_pages << " pages)"
                  << " file=" << _merit_dyn_path << " physical_pages=" << _merit_dyn_physical_cap
                  << " score=float32 unit=1..16/100x1k-query"
                  << " heap_shards=" << _merit_heap_shard_count
                  << " dcache_net_gate=" << (_merit_dcache_net_gate_enabled ? "on" : "off")
                  << " dcache_probe=1/" << _merit_dcache_probe_period
                  << " deferred_ncache_page_write=" << (_merit_deferred_ncache_page_write ? "on" : "off")
                  << " dcache_writer=on queue=" << _merit_dyn_write_q_cap
                  << " dcache_ncache_exclusive="
                  << (_merit_dcache_evict_ncache_on_commit ? "on" : "off")
                  << " real_io_coaccess=" << (_merit_real_io_coaccess ? "on" : "off")
                  << " real_io_window=" << _merit_real_io_window
                  << " real_io_sample=1/" << _merit_real_io_sample_queries
                  << " pending_buffer_bytes=" << _merit_pending_buffer_cap_bytes
                  << " pending_buffer_ready=" << _merit_pending_buffer_ready_percent << "%"
                  << "/" << _merit_pending_buffer_ready_min_members << "-members"
                  << " pending_require_full_page=" << (_merit_pending_require_full_page ? "on" : "off")
                  << " pending_pair_cap=" << _merit_pending_pair_cap
                  << " dcache_no_seed_replace=" << (_merit_dcache_no_seed_replace ? "on" : "off")
                  << " dcache_second_page=" << (_merit_dcache_second_page ? "on" : "off")
                  << " dcache_max_pages=" << static_cast<unsigned>(_merit_dcache_max_pages)
                  << " dcache_freeze_after=" << _merit_dcache_freeze_after
                  << " dcache_query_patch=" << (_merit_dcache_query_patch ? "on" : "off")
                  << " patch_async=" << (_merit_dcache_query_patch_async ? "on" : "off")
                  << " patch_max=" << _merit_dcache_query_patch_max
                  << " patch_q=" << _merit_patch_q_cap
                  << " patch_batch_us=" << _merit_patch_batch_us
                  << " patch_no_disk=" << (_merit_patch_worker_no_disk ? "on" : "off")
                  << " patch_stash_cap=" << _merit_stash_cap
                  << std::endl;
}

template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::merit_dyn_import_static_prefill()
{
    if (_merit_dc_path.empty() || _merit_dc_map.empty() || _merit_dyn_write_fd < 0)
        return -1;

    // Prefer seed_member_loc for page_count: dense-fill / flat padding may extend
    // _merit_dc_map past the real seed-group pages and those extra pages have no seed.
    uint64_t page_count = 0;
    if (!_merit_dc_seed_member_loc.empty())
    {
        for (const auto &seed_members : _merit_dc_seed_member_loc)
            for (const auto &member : seed_members.second)
                page_count = std::max<uint64_t>(page_count, static_cast<uint64_t>(member.second.sector) + 1ULL);
    }
    else
    {
        for (const auto &entry : _merit_dc_map)
        {
            for (const MeritDiskLoc &loc : entry.second)
                page_count = std::max<uint64_t>(page_count, static_cast<uint64_t>(loc.sector) + 1ULL);
        }
    }
    const uint64_t file_pages = get_file_size(_merit_dc_path) / defaults::SECTOR_LEN;
    if (file_pages > 0 && page_count > file_pages)
    {
        diskann::cerr << "MERIT BFS prefill: map pages=" << page_count << " exceeds file pages=" << file_pages
                      << " (" << _merit_dc_path << ")." << std::endl;
        return -1;
    }
    if (page_count == 0 || page_count > _merit_dyn_page_cap)
    {
        diskann::cerr << "MERIT BFS prefill page count " << page_count << " exceeds dynamic capacity "
                      << _merit_dyn_page_cap << "." << std::endl;
        return -1;
    }

#ifndef _WINDOWS
    const int src_fd = ::open(_merit_dc_path.c_str(), O_RDONLY | O_DIRECT | O_LARGEFILE);
    if (src_fd < 0)
    {
        diskann::cerr << "MERIT BFS prefill could not open " << _merit_dc_path << "." << std::endl;
        return -1;
    }
    constexpr size_t copy_chunk = 4ULL * 1024ULL * 1024ULL;
    void *buffer = nullptr;
    if (::posix_memalign(&buffer, defaults::SECTOR_LEN, copy_chunk) != 0)
    {
        ::close(src_fd);
        return -1;
    }
    const uint64_t bytes_to_copy = page_count * defaults::SECTOR_LEN;
    bool copy_ok = true;
    for (uint64_t offset = 0; offset < bytes_to_copy;)
    {
        const size_t bytes = static_cast<size_t>(std::min<uint64_t>(copy_chunk, bytes_to_copy - offset));
        const ssize_t got = ::pread(src_fd, buffer, bytes, static_cast<off_t>(offset));
        if (got != static_cast<ssize_t>(bytes) ||
            !merit_dyn_writer_submit_and_wait(MeritDynWriteOp::Write, static_cast<off_t>(offset), buffer, bytes))
        {
            copy_ok = false;
            break;
        }
        offset += bytes;
    }
    ::free(buffer);
    ::close(src_fd);
    if (!copy_ok || !merit_dyn_writer_submit_and_wait(MeritDynWriteOp::Sync, 0, nullptr, 0))
    {
        diskann::cerr << "MERIT BFS prefill failed while copying " << bytes_to_copy << " bytes." << std::endl;
        return -1;
    }
#else
    diskann::cerr << "MERIT BFS dynamic prefill is not implemented on Windows." << std::endl;
    return -1;
#endif

    std::vector<uint32_t> seeds(static_cast<size_t>(page_count), MERIT_DYN_INVALID_PAGE);
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicUnique, true);
        // seed_member_loc / map store unique members only; padded seed replicas may be absent.
        // Reconstruct full pages: place known (sector,slot) members, then fill holes with the page seed.
        std::vector<std::vector<uint32_t>> page_slots(
            static_cast<size_t>(page_count),
            std::vector<uint32_t>(_merit_dyn_members_per_page, MERIT_DYN_INVALID_PAGE));
        auto place = [&](uint32_t node_id, const MeritDiskLoc &static_loc) {
            if (static_loc.sector >= page_count || static_loc.slot >= _merit_dyn_members_per_page)
                return;
            page_slots[static_loc.sector][static_loc.slot] = node_id;
        };
        if (!_merit_dc_seed_member_loc.empty())
        {
            // Page owner is the seed-group key, NOT "whoever sits in slot 0".
            // directed_seed_replica continuation pages are child-only in slots 0..n-1
            // and only pad with the group seed — treating slot0 as seed corrupts pads.
            for (const auto &seed_members : _merit_dc_seed_member_loc)
            {
                const uint32_t seed = seed_members.first;
                for (const auto &member : seed_members.second)
                {
                    place(member.first, member.second);
                    if (member.second.sector < page_count)
                        seeds[member.second.sector] = seed;
                }
            }
        }
        else
        {
            for (const auto &entry : _merit_dc_map)
                for (const MeritDiskLoc &static_loc : entry.second)
                {
                    place(entry.first, static_loc);
                    if (static_loc.slot == 0 && static_loc.sector < page_count)
                        seeds[static_loc.sector] = entry.first;
                }
        }
        for (const auto &kv : _merit_dc_seed_canonical_loc)
        {
            if (kv.second.sector < page_count && seeds[kv.second.sector] == MERIT_DYN_INVALID_PAGE)
                seeds[kv.second.sector] = kv.first;
        }
        for (uint64_t page = 0; page < page_count; ++page)
        {
            uint32_t seed = seeds[page];
            if (seed == MERIT_DYN_INVALID_PAGE)
            {
                for (uint32_t slot = 0; slot < _merit_dyn_members_per_page; ++slot)
                {
                    if (page_slots[page][slot] != MERIT_DYN_INVALID_PAGE)
                    {
                        seed = page_slots[page][slot];
                        break;
                    }
                }
            }
            if (seed == MERIT_DYN_INVALID_PAGE)
            {
                diskann::cerr << "MERIT BFS prefill page " << page << " has no seed." << std::endl;
                return -1;
            }
            seeds[page] = seed;
            if (page_slots[page][0] == MERIT_DYN_INVALID_PAGE)
                page_slots[page][0] = seed;
            for (uint32_t slot = 0; slot < _merit_dyn_members_per_page; ++slot)
            {
                if (page_slots[page][slot] == MERIT_DYN_INVALID_PAGE)
                    page_slots[page][slot] = seed;
                const uint32_t node_id = page_slots[page][slot];
                MeritDiskLoc dynamic_loc;
                dynamic_loc.sector = static_cast<uint32_t>(MERIT_DYN_SECTOR_BASE + page);
                dynamic_loc.slot = static_cast<uint16_t>(slot);
                dynamic_loc.nsectors = 1;
                merit_dyn_loc_add_unlocked(node_id, dynamic_loc);
                const size_t member_index =
                    static_cast<size_t>(page) * _merit_dyn_members_per_page + slot;
                _merit_dyn_page_members[member_index] = node_id;
            }
            _merit_dyn_page_member_count[page] = static_cast<uint16_t>(_merit_dyn_members_per_page);
        }
        _merit_dyn_committed_pages = page_count;
        _merit_dyn_next_page = page_count;
        _merit_dc_map.clear();

        // Remap static sidecar sectors → dynamic tagged sectors so any leftover
        // seed-member strict paths stay consistent with imported dyn pages.
        if (!_merit_dc_seed_member_loc.empty())
        {
            tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, MeritDiskLoc>> remapped_members;
            remapped_members.reserve(_merit_dc_seed_member_loc.size());
            for (const auto &seed_members : _merit_dc_seed_member_loc)
            {
                auto &out_members = remapped_members[seed_members.first];
                for (const auto &member : seed_members.second)
                {
                    MeritDiskLoc loc;
                    loc.sector = member.second.sector;
                    loc.slot = member.second.slot;
                    loc.nsectors = member.second.nsectors;
                    if (loc.sector < page_count)
                        loc.sector = static_cast<uint32_t>(MERIT_DYN_SECTOR_BASE + loc.sector);
                    out_members.emplace(member.first, loc);
                }
            }
            _merit_dc_seed_member_loc = std::move(remapped_members);

            tsl::robin_map<uint32_t, MeritDiskLoc> remapped_canonical;
            remapped_canonical.reserve(_merit_dc_seed_canonical_loc.size());
            for (const auto &kv : _merit_dc_seed_canonical_loc)
            {
                MeritDiskLoc loc;
                loc.sector = kv.second.sector;
                loc.slot = kv.second.slot;
                loc.nsectors = kv.second.nsectors;
                if (loc.sector < page_count)
                    loc.sector = static_cast<uint32_t>(MERIT_DYN_SECTOR_BASE + loc.sector);
                remapped_canonical.emplace(kv.first, loc);
            }
            _merit_dc_seed_canonical_loc = std::move(remapped_canonical);
        }
    }

    for (uint32_t seed : seeds)
    {
        const auto tr = _merit_mcache.on_expand(seed, _merit_score_unit);
        if (!tr.present || tr.evicted_id != MeritMetadataCache::kInvalid)
        {
            diskann::cerr << "MERIT BFS prefill could not register all page seeds in m-cache." << std::endl;
            return -1;
        }
        {
            MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
            merit_dyn_set_state_unlocked(seed, tr.slot_id, MeritNodeState::Seed);
        }
        MeritHeapShard &shard = merit_dyn_heap_shard(tr.slot_id);
        MeritTimedMutexGuard hlock(shard.mu, MeritLockKind::Heap);
        merit_dyn_heap_upsert_unlocked(tr.slot_id, tr.score);
    }

    if (_merit_disk_reader)
    {
        _merit_disk_reader->close();
        _merit_disk_reader.reset();
    }
    diskann::cout << "MERIT dynamic BFS prefill: " << page_count << "/" << _merit_dyn_page_cap
                  << " pages, " << (page_count * _merit_dyn_members_per_page)
                  << " nodes; all pages full before measurement." << std::endl;
    return 0;
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::merit_dynamic_3cache_enabled() const
{
    return _merit_dyn_enabled;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::merit_dynamic_flush_count() const
{
    return _merit_dyn_flush_count;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::merit_dynamic_page_count() const
{
    MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
    return _merit_dyn_committed_pages;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::merit_dynamic_page_capacity() const
{
    return _merit_dyn_page_cap;
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::merit_dynamic_disk_full() const
{
    return merit_dyn_disk_is_full();
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dynamic_begin_measurement()
{
    _merit_monitor_warmup_queries = _merit_query_epoch.load(std::memory_order_acquire);
    _merit_pending_buffer_inserts.store(0, std::memory_order_relaxed);
    _merit_pending_buffer_ready_writes.store(0, std::memory_order_relaxed);
    _merit_pending_buffer_pressure_writes.store(0, std::memory_order_relaxed);
    _merit_pending_buffer_drops.store(0, std::memory_order_relaxed);
    _merit_pending_member_refreshes.store(0, std::memory_order_relaxed);
    _merit_pending_became_full.store(0, std::memory_order_relaxed);
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
        _merit_heap_writes = 0;
        _merit_heap_deletes = 0;
        _merit_ncache_write_trig = 0;
    }
    // Default: when D-cache is already full, freeze M/pair/heap maintenance for
    // the timed run but keep D-page lookups. Override with
    // MERIT_FREEZE_MAINTENANCE=0 to keep updating, or =1 to force freeze.
    const uint64_t freeze_env = merit_env_u64("MERIT_FREEZE_MAINTENANCE", 2);
    const bool disk_full = merit_dyn_disk_is_full();
    _merit_freeze_maintenance = freeze_env == 1 || (freeze_env == 2 && disk_full);
    if (_merit_freeze_maintenance)
    {
        stop_merit_dyn_patch_worker();
        merit_dyn_wait_for_commits();
    }
    _merit_dyn_member_bits.clear();
    if (_merit_freeze_maintenance && _num_points > 0)
    {
        const uint64_t words = (_num_points + 63ULL) / 64ULL;
        _merit_dyn_member_bits.assign(static_cast<size_t>(words), 0ULL);
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
        for (size_t shard_index = 0; shard_index < _merit_dyn_loc_shard_count; ++shard_index)
        {
            const MeritDynLocShard &shard = *_merit_dyn_loc_shards[shard_index];
            MeritTimedSharedMutexGuard shard_lock(shard.mu, MeritLockKind::DynamicShared, false);
            for (const auto &kv : shard.locations)
            {
                const uint32_t node_id = kv.first;
                if (static_cast<uint64_t>(node_id) >= _num_points)
                    continue;
                bool live = false;
                for (const MeritDiskLoc &loc : kv.second)
                {
                    if (merit_dyn_is_sector(loc.sector))
                    {
                        const uint32_t idx = loc.sector - MERIT_DYN_SECTOR_BASE;
                        if (merit_dyn_page_live(idx))
                        {
                            live = true;
                            break;
                        }
                    }
                }
                if (!live)
                    continue;
                _merit_dyn_member_bits[static_cast<size_t>(node_id >> 6)] |=
                    (1ULL << (node_id & 63u));
            }
        }
    }
    diskann::cout << "MERIT measurement counters reset at query="
                  << _merit_monitor_warmup_queries
                  << " freeze_maintenance=" << (_merit_freeze_maintenance ? "on" : "off")
                  << " disk_full=" << (disk_full ? "yes" : "no")
                  << " member_filter_bits="
                  << (_merit_dyn_member_bits.size() * 64ULL) << std::endl;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::merit_dynamic_clean_seeds() const
{
    MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    uint64_t n = 0;
    for (uint16_t count : _merit_dyn_page_member_count)
        if (count >= nps)
            n++;
    return n;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::merit_dynamic_partial_seeds() const
{
    MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    uint64_t n = 0;
    for (uint16_t count : _merit_dyn_page_member_count)
        if (count > 0 && count < nps)
            n++;
    return n;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::print_merit_dynamic_3cache_stats() const
{
    if (!_merit_dyn_enabled)
        return;
    uint64_t dynamic_pages = 0;
    uint64_t dynamic_loc_entries = 0;
    uint64_t dynamic_locs = 0;
    uint64_t dynamic_loc_capacity = 0;
    uint64_t seed_dir_entries = 0;
    uint64_t seed_dir_members = 0;
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
        dynamic_pages = _merit_dyn_committed_pages;
        for (size_t shard_index = 0; shard_index < _merit_dyn_loc_shard_count; ++shard_index)
        {
            const MeritDynLocShard &shard = *_merit_dyn_loc_shards[shard_index];
            MeritTimedSharedMutexGuard shard_lock(shard.mu, MeritLockKind::DynamicShared, false);
            dynamic_loc_entries += shard.locations.size();
            for (const auto &kv : shard.locations)
            {
                dynamic_locs += kv.second.size();
                dynamic_loc_capacity += kv.second.capacity();
            }
        }
        seed_dir_entries = _merit_dyn_committed_pages;
        for (uint16_t count : _merit_dyn_page_member_count)
            seed_dir_members += count;
    }
    diskann::cout << "MERIT dynamic 3-cache stats: m-cache=" << _merit_mcache.size() << "/" << _merit_mcache.capacity()
                  << " mcache_edges=" << _merit_mcache.edge_count()
                  << " n-cache="
                  << ((_merit_mem_pool && _merit_mem_pool->active()) ? _merit_mem_pool->size() : 0) << "/"
                  << ((_merit_mem_pool && _merit_mem_pool->active()) ? _merit_mem_pool->capacity() : 0)
                  << " flushes=" << _merit_dyn_flush_count << " clean=" << merit_dynamic_clean_seeds()
                  << " partial=" << merit_dynamic_partial_seeds() << " dyn_pages=" << dynamic_pages
                  << std::endl;
    diskann::cout << "MERIT mcache update: adaptive="
                  << (_merit_mcache.adaptive_update_enabled() ? "on" : "off")
                  << " denominator=" << _merit_mcache.update_denominator()
                  << " maintenance_suspended=" << (_merit_mcache.maintenance_suspended() ? "on" : "off")
                  << " ewma_hit_rate=" << _merit_mcache.adaptive_hit_rate()
                  << " observed=" << _merit_mcache.observed_accesses()
                  << " updates=" << _merit_mcache.performed_updates()
                  << " skipped=" << _merit_mcache.skipped_updates()
                  << " transitions=" << _merit_mcache.adaptive_transition_count()
                  << " bitmap_bytes=" << _merit_mcache.membership_bitmap_bytes() << std::endl;
    const uint64_t dcache_enabled_queries =
        _merit_dcache_enabled_queries.load(std::memory_order_relaxed);
    const uint64_t dcache_avoided = _merit_dcache_avoided_pages.load(std::memory_order_relaxed);
    const uint64_t dcache_reads = _merit_dcache_physical_reads.load(std::memory_order_relaxed);
    diskann::cout << "MERIT dcache net: gate=" << (_merit_dcache_net_gate_enabled ? "on" : "off")
                  << " active="
                  << (_merit_dcache_gate_active.load(std::memory_order_relaxed) ? "on" : "off")
                  << " probe=1/" << _merit_dcache_probe_period
                  << " sampled_queries=" << dcache_enabled_queries
                  << " avoided_pages/query="
                  << (dcache_enabled_queries == 0
                          ? 0.0
                          : static_cast<double>(dcache_avoided) /
                                static_cast<double>(dcache_enabled_queries))
                  << " dcache_reads/query="
                  << (dcache_enabled_queries == 0
                          ? 0.0
                          : static_cast<double>(dcache_reads) /
                                static_cast<double>(dcache_enabled_queries))
                  << " net_pages/query="
                  << (dcache_enabled_queries == 0
                          ? 0.0
                          : (static_cast<double>(dcache_avoided) -
                             static_cast<double>(dcache_reads)) /
                                static_cast<double>(dcache_enabled_queries))
                  << " transitions="
                  << _merit_dcache_gate_transitions.load(std::memory_order_relaxed)
                  << " latency_model_us=" << _merit_dcache_base_read_us << "/"
                  << _merit_dcache_read_us << "/" << _merit_dcache_write_us << "/"
                  << _merit_dcache_cpu_us << std::endl;
    diskann::cout << "MERIT dcache_locations: shards=" << _merit_dyn_loc_shard_count
                  << " entries=" << dynamic_loc_entries << " locations=" << dynamic_locs
                  << " vector_capacity=" << dynamic_loc_capacity
                  << " loc_bytes=" << sizeof(MeritDiskLoc) << std::endl;
    diskann::cout << "MERIT seed_directory: entries=" << seed_dir_entries << " members=" << seed_dir_members
                  << " member_arena_slots=" << _merit_dyn_page_members.size()
                  << " member_arena_bytes=" << _merit_dyn_page_members.capacity() * sizeof(uint32_t)
                  << " count_bytes=" << _merit_dyn_page_member_count.capacity() * sizeof(uint16_t) << std::endl;
    {
        uint64_t paired = 0, first_sum = 0, final_sum = 0, grew = 0, same = 0, shrunk = 0;
        uint64_t first_hist[11] = {};
        uint64_t final_hist[11] = {};
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
        for (size_t page = 0; page < _merit_dyn_page_member_count.size(); ++page)
        {
            const uint16_t count = _merit_dyn_page_member_count[page];
            if (count == 0)
                continue;
            if (page < _merit_dyn_page_secondary.size() && _merit_dyn_page_secondary[page])
                continue;
            const size_t begin = page * _merit_dyn_members_per_page;
            if (begin >= _merit_dyn_page_members.size())
                continue;
            const uint32_t seed = _merit_dyn_page_members[begin];
            auto it = _merit_seed_first_fill.find(seed);
            if (it == _merit_seed_first_fill.end())
                continue;
            const uint16_t first = it->second;
            if (first == 0 || first > 10 || count > 10)
                continue;
            ++paired;
            first_sum += first;
            final_sum += count;
            if (count > first)
                ++grew;
            else if (count < first)
                ++shrunk;
            else
                ++same;
            ++first_hist[first];
            ++final_hist[count];
        }
        const double first_avg = paired == 0 ? 0.0 : static_cast<double>(first_sum) / static_cast<double>(paired);
        const double final_avg = paired == 0 ? 0.0 : static_cast<double>(final_sum) / static_cast<double>(paired);
        diskann::cout << "MERIT dcache_fill: resident_seeds=" << paired
                      << " first_members=" << first_avg << " first_neighbors=" << (first_avg - 1.0)
                      << " first_fill=" << (first_avg / 10.0)
                      << " final_members=" << final_avg << " final_neighbors=" << (final_avg - 1.0)
                      << " final_fill=" << (final_avg / 10.0)
                      << " grew=" << grew << " same=" << same << " shrunk=" << shrunk << std::endl;
        diskann::cout << "MERIT dcache_fill_hist first";
        for (int n = 1; n <= 10; ++n)
            diskann::cout << " " << n << ":" << first_hist[n];
        diskann::cout << " final";
        for (int n = 1; n <= 10; ++n)
            diskann::cout << " " << n << ":" << final_hist[n];
        diskann::cout << std::endl;
        if (_merit_dcache_second_page)
        {
            uint64_t extra_pages = 0, extra_neighbors = 0;
            uint64_t span_hist[6] = {};
            for (size_t page = 0; page < _merit_dyn_page_member_count.size(); ++page)
            {
                if (!merit_dyn_page_live(static_cast<uint32_t>(page)))
                    continue;
                const bool secondary =
                    page < _merit_dyn_page_secondary.size() && _merit_dyn_page_secondary[page];
                if (secondary)
                {
                    ++extra_pages;
                    const uint16_t count = _merit_dyn_page_member_count[page];
                    if (count > 1)
                        extra_neighbors += static_cast<uint64_t>(count - 1);
                    continue;
                }
                uint32_t n = 1;
                if (page < _merit_dyn_page_span_len.size() && _merit_dyn_page_span_len[page] > 1)
                    n = _merit_dyn_page_span_len[page];
                if (n > 5)
                    n = 5;
                ++span_hist[n];
            }
            diskann::cout << "MERIT dcache_pages: max=" << static_cast<uint32_t>(_merit_dcache_max_pages)
                          << " p1=" << span_hist[1] << " p2=" << span_hist[2] << " p3=" << span_hist[3]
                          << " p4=" << span_hist[4] << " p5=" << span_hist[5]
                          << " extra_pages=" << extra_pages << " extra_neighbors=" << extra_neighbors
                          << " multi_ios=" << _merit_dcache_pair_ios.load(std::memory_order_relaxed)
                          << " span_adaptive=" << (_merit_span_adaptive ? "on" : "off")
                          << " span_min_hits=" << _merit_span_min_hits << " span_explore=" << _merit_span_explore
                          << " span_full_reads=" << _merit_span_full_reads.load(std::memory_order_relaxed)
                          << " span_primary_only=" << _merit_span_primary_only.load(std::memory_order_relaxed)
                          << std::endl;
        }
    }
    if (_merit_member_snapshot_done.load(std::memory_order_acquire))
    {
        uint64_t neighbor_slots = 0, early_still = 0, late_slots = 0;
        {
            MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
            for (size_t page = 0; page < _merit_dyn_page_member_count.size(); ++page)
            {
                const uint16_t count = _merit_dyn_page_member_count[page];
                if (count <= 1)
                    continue;
                const size_t begin = page * _merit_dyn_members_per_page;
                if (begin + count > _merit_dyn_page_members.size())
                    continue;
                for (size_t mi = 1; mi < count; ++mi)
                {
                    const uint32_t mid = _merit_dyn_page_members[begin + mi];
                    if (mid == MERIT_DYN_INVALID_PAGE)
                        continue;
                    ++neighbor_slots;
                    if (_merit_member_at_snapshot.find(mid) != _merit_member_at_snapshot.end())
                        ++early_still;
                    else
                        ++late_slots;
                }
            }
        }
        const uint64_t early_hits = _merit_prefetch_from_early.load(std::memory_order_relaxed);
        const uint64_t late_hits = _merit_prefetch_from_late.load(std::memory_order_relaxed);
        auto hop_stats = [this](const std::atomic<uint64_t> *hist, uint64_t n, double &mean, uint32_t &p50,
                                uint32_t &p90) {
            uint64_t sum = 0;
            for (uint32_t h = 0; h < 128; ++h)
                sum += hist[h].load(std::memory_order_relaxed) * static_cast<uint64_t>(h);
            mean = n == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(n);
            auto pct = [&](double q) {
                if (n == 0)
                    return 0u;
                uint64_t need = static_cast<uint64_t>(std::ceil(q * static_cast<double>(n)));
                if (need < 1)
                    need = 1;
                uint64_t seen = 0;
                for (uint32_t h = 0; h < 128; ++h)
                {
                    seen += hist[h].load(std::memory_order_relaxed);
                    if (seen >= need)
                        return h;
                }
                return 127u;
            };
            p50 = pct(0.50);
            p90 = pct(0.90);
        };
        double early_mean = 0, late_mean = 0;
        uint32_t early_p50 = 0, early_p90 = 0, late_p50 = 0, late_p90 = 0;
        hop_stats(_merit_early_hop_hist, early_hits, early_mean, early_p50, early_p90);
        hop_stats(_merit_late_hop_hist, late_hits, late_mean, late_p50, late_p90);
        diskann::cout << "MERIT prefetch_member_age: snapshot_query=" << _merit_member_snapshot_at
                      << " early_neighbor_ids=" << _merit_member_at_snapshot.size()
                      << " early_neighbor_slots=" << _merit_member_snapshot_slots.load(std::memory_order_relaxed)
                      << " final_neighbor_slots=" << neighbor_slots
                      << " early_still=" << early_still << " late_slots=" << late_slots
                      << " prefetch_early=" << early_hits << " prefetch_late=" << late_hits
                      << " early_hop_mean=" << early_mean << " early_hop_p50=" << early_p50
                      << " early_hop_p90=" << early_p90 << " late_hop_mean=" << late_mean
                      << " late_hop_p50=" << late_p50 << " late_hop_p90=" << late_p90 << std::endl;
    }
    const uint64_t ncache_payload_bytes =
        (_merit_mem_pool && _merit_mem_pool->active()) ? _merit_mem_pool->payload_capacity_bytes() : 0;
    uint64_t stash_payload_bytes = 0;
    uint64_t pending_buffer_payload_bytes = 0;
    uint64_t parent_fetched_bytes = 0;
    uint64_t stash_entries = 0;
    uint64_t parent_entries = 0;
    {
        std::lock_guard<std::mutex> lock(_merit_stash_mu);
        stash_entries = _merit_payload_stash.size();
        for (const auto &kv : _merit_payload_stash)
            stash_payload_bytes +=
                kv.second.coords.capacity() * sizeof(T) + kv.second.nbrs.capacity() * sizeof(uint32_t);
        pending_buffer_payload_bytes = _merit_pending_buffer_payload_bytes;
        for (const auto &kids : _merit_parent_fetched_by_slot)
            if (!kids.empty())
                ++parent_entries;
        parent_fetched_bytes = _merit_parent_fetched_by_slot.capacity() * sizeof(MeritRecentChildren) +
                               _merit_parent_fetched_owner.capacity() * sizeof(uint32_t);
    }
    diskann::cout << "MERIT ncache_memory: payload_capacity_bytes=" << ncache_payload_bytes
                  << " lock=" << ((_merit_mem_pool != nullptr && _merit_mem_pool->spin_lock_enabled()) ? "spin" : "mutex")
                  << " stash_entries=" << stash_entries << "/" << _merit_stash_cap
                  << " stash_payload_capacity_bytes=" << stash_payload_bytes
                  << " parent_fetched_entries=" << parent_entries
                  << " parent_fetched_capacity_bytes=" << parent_fetched_bytes << std::endl;
    diskann::cout << "MERIT pending_buffer: payload_bytes=" << pending_buffer_payload_bytes << "/"
                  << _merit_pending_buffer_cap_bytes
                  << " entries=" << stash_entries
                  << " inserts=" << _merit_pending_buffer_inserts.load(std::memory_order_relaxed)
                  << " ready_writes="
                  << _merit_pending_buffer_ready_writes.load(std::memory_order_relaxed)
                  << " pressure_writes="
                  << _merit_pending_buffer_pressure_writes.load(std::memory_order_relaxed)
                  << " drops=" << _merit_pending_buffer_drops.load(std::memory_order_relaxed)
                  << " member_refreshes="
                  << _merit_pending_member_refreshes.load(std::memory_order_relaxed)
                  << " became_full=" << _merit_pending_became_full.load(std::memory_order_relaxed)
                  << " dcache_ncache_erases="
                  << _merit_dcache_ncache_erases.load(std::memory_order_relaxed)
                  << " query_patch_trig=" << _merit_dcache_query_patch_trig.load(std::memory_order_relaxed)
                  << " query_patch_ok=" << _merit_dcache_query_patch_ok.load(std::memory_order_relaxed)
                  << " query_patch_drop=" << _merit_dcache_query_patch_drop.load(std::memory_order_relaxed)
                  << std::endl;
    const uint64_t writer_done = _merit_dyn_writer_completed.load(std::memory_order_relaxed);
    const uint64_t commit_done = _merit_commit_completed.load(std::memory_order_relaxed);
    const uint64_t commit_failed = _merit_commit_failed.load(std::memory_order_relaxed);
    const uint64_t commit_attempted = commit_done + commit_failed;
    diskann::cout << "MERIT async_commit: commit_threads=" << _merit_commit_thread_count
                  << " writer_threads=" << _merit_dyn_writer_thread_count
                  << " enqueued=" << _merit_commit_enqueued.load(std::memory_order_relaxed)
                  << " coalesced=" << _merit_commit_coalesced.load(std::memory_order_relaxed)
                  << " dropped=" << _merit_commit_dropped.load(std::memory_order_relaxed)
                  << " completed=" << commit_done << " failed=" << commit_failed
                  << " queue_us="
                  << (commit_attempted == 0
                          ? 0.0
                          : static_cast<double>(_merit_commit_queue_ns.load(std::memory_order_relaxed)) /
                                1000.0 / static_cast<double>(commit_attempted))
                  << " work_us="
                  << (commit_attempted == 0
                          ? 0.0
                          : static_cast<double>(_merit_commit_work_ns.load(std::memory_order_relaxed)) /
                                1000.0 / static_cast<double>(commit_attempted))
                  << " source_enqueued=" << _merit_commit_source_enqueued[0].load(std::memory_order_relaxed)
                  << "/" << _merit_commit_source_enqueued[1].load(std::memory_order_relaxed)
                  << "/" << _merit_commit_source_enqueued[2].load(std::memory_order_relaxed)
                  << " source_completed=" << _merit_commit_source_completed[0].load(std::memory_order_relaxed)
                  << "/" << _merit_commit_source_completed[1].load(std::memory_order_relaxed)
                  << "/" << _merit_commit_source_completed[2].load(std::memory_order_relaxed)
                  << " writer_enqueued=" << _merit_dyn_writer_enqueued.load(std::memory_order_relaxed)
                  << " writer_failed=" << _merit_dyn_writer_failed.load(std::memory_order_relaxed)
                  << " writer_queue_us="
                  << (writer_done == 0
                          ? 0.0
                          : static_cast<double>(_merit_dyn_writer_queue_ns.load(std::memory_order_relaxed)) /
                                1000.0 / static_cast<double>(writer_done))
                  << " writer_io_us="
                  << (writer_done == 0
                          ? 0.0
                          : static_cast<double>(_merit_dyn_writer_io_ns.load(std::memory_order_relaxed)) /
                                1000.0 / static_cast<double>(writer_done))
                  << std::endl;
    diskann::cout << "MERIT real_io_coaccess: enabled=" << (_merit_real_io_coaccess ? "yes" : "no")
                  << " queries=" << _merit_real_io_queries.load(std::memory_order_relaxed)
                  << " sampled_queries=" << _merit_real_io_sampled_queries.load(std::memory_order_relaxed)
                  << " nodes=" << _merit_real_io_nodes.load(std::memory_order_relaxed)
                  << " directed_pairs=" << _merit_real_io_pairs.load(std::memory_order_relaxed)
                  << " window=" << _merit_real_io_window
                  << " max_partners=" << _merit_real_io_max_partners << std::endl;
    diskann::cout << "MERIT dyn probe: ncache_hit=" << g_dyn_probe.ncache_hit
                  << " ncache_miss=" << g_dyn_probe.ncache_miss << " resolve=" << g_dyn_probe.resolve_call
                  << " parent_hit=" << g_dyn_probe.parent_hit << " parent_absent=" << g_dyn_probe.parent_absent
                  << " parent_no_page=" << g_dyn_probe.parent_no_page
                  << " parent_not_member=" << g_dyn_probe.parent_not_member << " self_hit=" << g_dyn_probe.self_hit
                  << " map_hit=" << g_dyn_probe.map_hit << " resolve_miss=" << g_dyn_probe.resolve_miss
                  << " miss_but_in_map=" << g_dyn_probe.miss_but_in_map << std::endl;
    uint32_t max_id = std::numeric_limits<uint32_t>::max();
    uint32_t min_id = std::numeric_limits<uint32_t>::max();
    float max_c = 0.0f, min_c = 0.0f;
    uint32_t ins = std::numeric_limits<uint32_t>::max();
    uint32_t del = std::numeric_limits<uint32_t>::max();
    uint64_t pair_n = 0;
    uint64_t writes = 0, deletes = 0, trig = 0, refresh = 0, max_n = 0, min_n = 0, heap_storage_bytes = 0;
    uint64_t decay_queries = 0, decay_accesses = 0, decay_cycles = 0, decay_early_cycles = 0;
    bool decay_started = false;
    float score_unit = 1.0f;
    bool have_max = false;
    bool have_min = false;
    MeritHeapEntry best_max;
    MeritHeapEntry best_min;
    for (auto &shard : _merit_heap_shards)
    {
        MeritTimedMutexGuard hlock(shard.mu, MeritLockKind::Heap);
        max_n += shard.max_heap.size();
        min_n += shard.min_heap.size();
        heap_storage_bytes +=
            (shard.max_heap.capacity() + shard.min_heap.capacity()) * sizeof(MeritHeapEntry);
        if (!shard.max_heap.empty() &&
            (!have_max || merit_dyn_heap_better_unlocked(shard.max_heap.front(), best_max, MeritHeapKind::Max)))
        {
            best_max = shard.max_heap.front();
            have_max = true;
        }
        if (!shard.min_heap.empty() &&
            (!have_min || merit_dyn_heap_better_unlocked(shard.min_heap.front(), best_min, MeritHeapKind::Min)))
        {
            best_min = shard.min_heap.front();
            have_min = true;
        }
    }
    heap_storage_bytes += _merit_heap_position.capacity() * sizeof(uint32_t) +
                          _merit_heap_kind.capacity() * sizeof(uint8_t) +
                          _merit_node_state_capacity * sizeof(std::atomic<uint8_t>);
    if (have_max)
    {
        max_c = best_max.score;
        max_id = _merit_mcache.node_at(best_max.slot_id);
    }
    if (have_min)
    {
        min_c = best_min.score;
        min_id = _merit_mcache.node_at(best_min.slot_id);
    }
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
        pair_n = _merit_ready_pairs.size();
        if (!_merit_ready_pairs.empty())
        {
            ins = _merit_ready_pairs.begin()->second.insertion_id;
            del = _merit_ready_pairs.begin()->second.deletion_id;
        }
        writes = _merit_heap_writes;
        deletes = _merit_heap_deletes;
        trig = _merit_ncache_write_trig;
        refresh = _merit_pair_refresh;
        decay_queries = _merit_decay_query_count.load(std::memory_order_relaxed);
        decay_accesses = _merit_decay_access_count.load(std::memory_order_relaxed);
        decay_cycles = _merit_decay_cycles.load(std::memory_order_relaxed);
        decay_early_cycles = _merit_decay_early_cycles.load(std::memory_order_relaxed);
        decay_started = _merit_decay_started.load(std::memory_order_relaxed);
        score_unit = _merit_score_unit.load();
    }
    diskann::cout << "MERIT heap: max_top=" << max_id << ":" << max_c << " min_top=" << min_id << ":" << min_c
                  << " max_n=" << max_n << " min_n=" << min_n << " pending=" << pair_n
                  << " ins=" << ins << " del=" << del << " writes=" << writes << " deletes=" << deletes
                  << " trig=" << trig << " pair_refresh=" << refresh << " storage_bytes=" << heap_storage_bytes
                  << std::endl;
    diskann::cout << "MERIT score: unit=" << score_unit << "/" << MERIT_SCORE_UNIT_MAX
                  << " half_life_accesses=" << _merit_decay_half_life_accesses << " cycles=" << decay_cycles
                  << " early_cycles=" << decay_early_cycles << " decay_queries=" << decay_queries
                  << " decay_accesses=" << decay_accesses << " accesses_per_query="
                  << (decay_queries == 0 ? 0.0 : static_cast<double>(decay_accesses) / decay_queries)
                  << " decay=" << (decay_started ? "on" : "off")
                  << " edge_saturations=" << _merit_mcache.edge_saturations()
                  << " mcache_evictable=" << _merit_mcache.evictable_size()
                  << " probe pair_set=" << g_dyn_probe.pair_set
                  << " skip_on_disk=" << g_dyn_probe.pair_skip_max_on_disk
                  << " skip_count=" << g_dyn_probe.pair_skip_count << " write_trig=" << g_dyn_probe.write_trig
                  << " timeout_trig=" << g_dyn_probe.timeout_trig << " timeout_ok=" << g_dyn_probe.timeout_ok
                  << " write_ok=" << g_dyn_probe.write_ok << " delete_ok=" << g_dyn_probe.delete_ok << std::endl;
    static constexpr const char *lock_names[] = {"score", "heap", "pair", "mcache", "dyn_shared",
                                                 "dyn_unique", "commit_io"};
    {
        auto &p = merit_hop_io_probe();
        diskann::cout << "MERIT hop_io:";
        static constexpr const char *kinds[] = {"single_only", "with_multi"};
        for (int k = 0; k < 2; ++k)
        {
            const uint64_t b = p.batches[k].load(std::memory_order_relaxed);
            const double avg_us = b == 0 ? 0.0 : p.batch_ns[k].load() / 1000.0 / b;
            const double avg_reads = b == 0 ? 0.0 : static_cast<double>(p.reads[k].load()) / b;
            const double avg_kb = b == 0 ? 0.0 : p.bytes[k].load() / 1024.0 / b;
            diskann::cout << " " << kinds[k] << "_batches=" << b << " " << kinds[k] << "_us=" << avg_us << " "
                          << kinds[k] << "_reads=" << avg_reads << " " << kinds[k] << "_kb=" << avg_kb;
        }
        diskann::cout << " multi_reads=" << p.multi_reads.load() << " multi_kb="
                      << (p.multi_reads.load() == 0 ? 0.0 : p.multi_bytes.load() / 1024.0 / p.multi_reads.load())
                      << std::endl;
    }
    merit_io_latency_report();
    if (merit_section_on())
    {
        MeritSectionAcc total;
        {
            std::lock_guard<std::mutex> lock(merit_section_mu());
            for (auto *a : merit_section_registry())
            {
                for (int i = 0; i < kSecCount; ++i)
                    total.ns[i] += a->ns[i];
                total.queries += a->queries;
            }
        }
        diskann::cout << "MERIT section_us_per_query: queries=" << total.queries;
        for (int i = 0; i < kSecCount; ++i)
            diskann::cout << " " << kMeritSectionNames[i] << "="
                          << (total.queries == 0 ? 0.0 : total.ns[i] / 1000.0 / total.queries);
        diskann::cout << std::endl;
    }
    diskann::cout << "MERIT lock sample=1/1024:";
    auto &lock_metrics = merit_lock_metrics();
    for (size_t i = 0; i < static_cast<size_t>(MeritLockKind::Count); ++i)
    {
        const uint64_t samples = lock_metrics[i].samples.load(std::memory_order_relaxed);
        const double wait_us =
            samples == 0 ? 0.0 : static_cast<double>(lock_metrics[i].wait_ns.load(std::memory_order_relaxed)) /
                                         static_cast<double>(samples) / 1000.0;
        const double hold_us =
            samples == 0 ? 0.0 : static_cast<double>(lock_metrics[i].hold_ns.load(std::memory_order_relaxed)) /
                                         static_cast<double>(samples) / 1000.0;
        diskann::cout << " " << lock_names[i] << "=" << wait_us << "/" << hold_us << "us(" << samples << ")";
    }
    diskann::cout << std::endl;
}

template <typename T, typename LabelT> uint32_t PQFlashIndex<T, LabelT>::merit_dyn_alloc_page_unlocked()
{
    merit_dyn_drain_reclaim_unlocked();
    if (!_merit_dyn_free.empty())
    {
        const uint32_t idx = _merit_dyn_free.back();
        _merit_dyn_free.pop_back();
        return idx;
    }
    if (_merit_dyn_next_page < _merit_dyn_physical_cap)
        return _merit_dyn_next_page++;
    return MERIT_DYN_INVALID_PAGE;
}

template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::merit_dyn_alloc_contiguous_unlocked(uint32_t count)
{
    if (count == 0)
        return MERIT_DYN_INVALID_PAGE;
    if (count == 1)
        return merit_dyn_alloc_page_unlocked();
    merit_dyn_drain_reclaim_unlocked();
    if (_merit_dyn_next_page + count <= _merit_dyn_physical_cap)
    {
        const uint32_t base = _merit_dyn_next_page;
        _merit_dyn_next_page = static_cast<uint32_t>(_merit_dyn_next_page + count);
        return base;
    }
    std::sort(_merit_dyn_free.begin(), _merit_dyn_free.end());
    for (size_t i = 0; i + 1 < _merit_dyn_free.size(); ++i)
    {
        bool run = true;
        for (uint32_t k = 1; k < count; ++k)
        {
            if (i + k >= _merit_dyn_free.size() || _merit_dyn_free[i + k] != _merit_dyn_free[i] + k)
            {
                run = false;
                break;
            }
        }
        if (!run)
            continue;
        const uint32_t base = _merit_dyn_free[i];
        _merit_dyn_free.erase(_merit_dyn_free.begin() + static_cast<std::ptrdiff_t>(i),
                              _merit_dyn_free.begin() + static_cast<std::ptrdiff_t>(i + count));
        return base;
    }
    return MERIT_DYN_INVALID_PAGE;
}

template <typename T, typename LabelT>
std::vector<uint32_t> PQFlashIndex<T, LabelT>::merit_dyn_now_page_ids(uint32_t seed_id,
                                                                     const MeritMetadataCache::Snapshot &snap,
                                                                     bool ncache_only) const
{
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    std::vector<uint32_t> now;
    now.reserve(static_cast<size_t>(nps));
    tsl::robin_set<uint32_t> used;
    auto try_add = [&](uint32_t id) {
        if (now.size() >= nps || id == seed_id)
            return;
        if (!used.insert(id).second)
            return;
        bool has_payload = false;
        if (ncache_only)
        {
            has_payload = _merit_mem_pool != nullptr && _merit_mem_pool->active() &&
                          _merit_mem_pool->contains(id);
            if (!has_payload && _merit_pending_buffer_cap_bytes > 0)
            {
                std::lock_guard<std::mutex> slock(_merit_stash_mu);
                has_payload = _merit_payload_stash.find(id) != _merit_payload_stash.end();
            }
        }
        else
        {
            has_payload = merit_dyn_has_payload(id);
        }
        if (!has_payload)
            return;
        now.push_back(id);
    };

    now.push_back(seed_id);
    used.insert(seed_id);

    std::vector<uint32_t> old_members;
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
        uint32_t page_idx = MERIT_DYN_INVALID_PAGE;
        if (merit_dyn_seed_page_unlocked(seed_id, page_idx))
        {
            const size_t count = _merit_dyn_page_member_count[page_idx];
            const size_t begin = static_cast<size_t>(page_idx) * _merit_dyn_members_per_page;
            old_members.assign(_merit_dyn_page_members.begin() + begin,
                               _merit_dyn_page_members.begin() + begin + count);
        }
    }
    std::vector<uint32_t> fetched;
    const uint32_t fetched_slot = _merit_mcache.slot_of(seed_id);
    if (fetched_slot != MeritMetadataCache::kInvalid)
    {
        const uint32_t slot_id = fetched_slot;
        std::lock_guard<std::mutex> slock(merit_parent_fetched_mu(slot_id));
        if (slot_id < _merit_parent_fetched_by_slot.size() && _merit_parent_fetched_owner[slot_id] == seed_id)
        {
            const auto &kids = _merit_parent_fetched_by_slot[slot_id];
            fetched.assign(kids.begin(), kids.end());
        }
    }

    // Directed W is the primary co-packing signal. Recently fetched children
    // and old members are only fallback fillers when the page still has room.
    for (const auto &sc : snap.significant)
        try_add(sc.first);

    for (auto it = fetched.rbegin(); it != fetched.rend(); ++it)
        try_add(*it);

    if (!_merit_real_io_coaccess)
    {
        for (uint32_t id : old_members)
        {
            if (_merit_mem_pool != nullptr && _merit_mem_pool->contains(id))
                continue;
            try_add(id);
        }
        for (uint32_t id : old_members)
            try_add(id);
    }
    return now;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_hotter_mismatch(const uint32_t *page_members, size_t page_member_count,
                                                        const std::vector<uint32_t> &now_ids) const
{
    if (page_member_count == 0)
        return true;
    for (uint32_t id : now_ids)
    {
        if (std::find(page_members, page_members + page_member_count, id) == page_members + page_member_count)
            return true;
    }
    return false;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_retire_secondary_unlocked(uint32_t page_idx)
{
    if (page_idx >= _merit_dyn_page_member_count.size())
        return;
    const uint16_t member_count = _merit_dyn_page_member_count[page_idx];
    const size_t member_begin = static_cast<size_t>(page_idx) * _merit_dyn_members_per_page;
    const uint32_t old_sector = static_cast<uint32_t>(MERIT_DYN_SECTOR_BASE + page_idx);
    for (uint16_t i = 0; i < member_count; ++i)
        merit_dyn_loc_erase_sector_unlocked(_merit_dyn_page_members[member_begin + i], old_sector);
    _merit_dyn_page_member_count[page_idx] = 0;
    if (page_idx < _merit_dyn_page_sibling.size())
        _merit_dyn_page_sibling[page_idx] = MERIT_DYN_INVALID_PAGE;
    if (page_idx < _merit_dyn_page_secondary.size())
        _merit_dyn_page_secondary[page_idx] = 0;
    if (page_idx < _merit_dyn_page_span_len.size())
        _merit_dyn_page_span_len[page_idx] = 0;
    if (page_idx < _merit_dyn_page_span_base.size())
        _merit_dyn_page_span_base[page_idx] = MERIT_DYN_INVALID_PAGE;
    merit_dyn_retire_page_unlocked(page_idx);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_invalidate_seed_page(uint32_t seed_id)
{
    uint32_t page_idx = MERIT_DYN_INVALID_PAGE;
    if (!merit_dyn_seed_page_unlocked(seed_id, page_idx))
        return;
    uint8_t span = 1;
    if (page_idx < _merit_dyn_page_span_len.size() && _merit_dyn_page_span_len[page_idx] > 1)
        span = _merit_dyn_page_span_len[page_idx];
    const uint16_t member_count = _merit_dyn_page_member_count[page_idx];
    const size_t member_begin = static_cast<size_t>(page_idx) * _merit_dyn_members_per_page;
    const uint32_t old_sector = static_cast<uint32_t>(MERIT_DYN_SECTOR_BASE + page_idx);

    for (uint16_t i = 0; i < member_count; ++i)
        merit_dyn_loc_erase_sector_unlocked(_merit_dyn_page_members[member_begin + i], old_sector);
    if (merit_dyn_page_live(page_idx))
    {
        if (_merit_dyn_committed_pages > 0)
            _merit_dyn_committed_pages--;
    }
    _merit_dyn_page_member_count[page_idx] = 0;
    if (page_idx < _merit_dyn_page_sibling.size())
        _merit_dyn_page_sibling[page_idx] = MERIT_DYN_INVALID_PAGE;
    if (page_idx < _merit_dyn_page_secondary.size())
        _merit_dyn_page_secondary[page_idx] = 0;
    if (page_idx < _merit_dyn_page_span_len.size())
        _merit_dyn_page_span_len[page_idx] = 0;
    if (page_idx < _merit_dyn_page_span_base.size())
        _merit_dyn_page_span_base[page_idx] = MERIT_DYN_INVALID_PAGE;
    merit_dyn_retire_page_unlocked(page_idx);
    for (uint8_t extra = 1; extra < span; ++extra)
    {
        const uint32_t extra_page = page_idx + extra;
        if (extra_page < _merit_dyn_page_member_count.size())
            merit_dyn_retire_secondary_unlocked(extra_page);
    }
    if (!merit_dyn_seed_page_unlocked(seed_id, page_idx))
        _merit_seed_first_fill.erase(seed_id);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_release_page_pin(uint32_t sector)
{
    if (!merit_dyn_is_sector(sector) || _merit_dyn_page_readers == nullptr)
        return;
    const uint32_t page_idx = sector - MERIT_DYN_SECTOR_BASE;
    if (page_idx >= _merit_dyn_physical_cap)
        return;
    // seq_cst pairs with merit_dyn_retire_page_unlocked: whichever side observes
    // readers==0 && retired==1 last claims the page via CAS, exactly once.
    if (_merit_dyn_page_readers[page_idx].fetch_sub(1, std::memory_order_seq_cst) != 1)
        return;
    if (_merit_dyn_page_retired[page_idx].load(std::memory_order_seq_cst) == 0)
        return;
    uint8_t expected = 1;
    if (!_merit_dyn_page_retired[page_idx].compare_exchange_strong(expected, 0, std::memory_order_seq_cst))
        return;
    std::lock_guard<std::mutex> rlock(_merit_dyn_reclaim_mu);
    _merit_dyn_reclaim.push_back(page_idx);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_retire_page_unlocked(uint32_t page_idx)
{
    _merit_dyn_page_retired[page_idx].store(1, std::memory_order_seq_cst);
    if (_merit_dyn_page_readers[page_idx].load(std::memory_order_seq_cst) != 0)
        return;
    uint8_t expected = 1;
    if (_merit_dyn_page_retired[page_idx].compare_exchange_strong(expected, 0, std::memory_order_seq_cst))
        _merit_dyn_free.push_back(page_idx);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_span_should_read_full(uint32_t base, uint32_t span) const
{
    if (!_merit_span_adaptive || span <= 1)
        return true;
    const uint32_t decision = _merit_span_decisions[base].fetch_add(1, std::memory_order_relaxed) + 1;
    const uint32_t reads = _merit_span_reads[base].load(std::memory_order_relaxed);
    bool full = reads < 4 || (_merit_span_explore > 0 && decision % _merit_span_explore == 0);
    if (!full)
    {
        const double hits = _merit_span_hits[base].load(std::memory_order_relaxed);
        full = hits >= static_cast<double>(_merit_span_min_hits) * reads * (span - 1);
    }
    if (full)
    {
        if (_merit_span_reads[base].fetch_add(1, std::memory_order_relaxed) + 1 >= 64)
        {
            _merit_span_reads[base].store(32, std::memory_order_relaxed);
            _merit_span_hits[base].store(_merit_span_hits[base].load(std::memory_order_relaxed) / 2,
                                         std::memory_order_relaxed);
        }
        _merit_span_full_reads.fetch_add(1, std::memory_order_relaxed);
    }
    else
        _merit_span_primary_only.fetch_add(1, std::memory_order_relaxed);
    return full;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_span_carry_stats(uint32_t from_page, uint32_t to_page)
{
    if (!_merit_span_reads || to_page >= _merit_dyn_physical_cap)
        return;
    uint32_t reads = 0, hits = 0, decisions = 0;
    if (from_page < _merit_dyn_physical_cap)
    {
        reads = _merit_span_reads[from_page].load(std::memory_order_relaxed);
        hits = _merit_span_hits[from_page].load(std::memory_order_relaxed);
        decisions = _merit_span_decisions[from_page].load(std::memory_order_relaxed);
    }
    _merit_span_reads[to_page].store(reads, std::memory_order_relaxed);
    _merit_span_hits[to_page].store(hits, std::memory_order_relaxed);
    _merit_span_decisions[to_page].store(decisions, std::memory_order_relaxed);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_drain_reclaim_unlocked()
{
    std::lock_guard<std::mutex> rlock(_merit_dyn_reclaim_mu);
    if (_merit_dyn_reclaim.empty())
        return;
    _merit_dyn_free.insert(_merit_dyn_free.end(), _merit_dyn_reclaim.begin(), _merit_dyn_reclaim.end());
    _merit_dyn_reclaim.clear();
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_maybe_mark_seed(uint32_t node_id)
{
    (void)node_id;
    if (!_merit_dyn_enabled)
        return;
    merit_mark_refresh_needed();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_note_touch(const MeritMetadataCache::TouchResult &tr,
                                                   SSDQueryScratch<T> *query_scratch)
{
    if (!_merit_dyn_enabled || !tr.present)
        return;
    if (tr.evicted_id != MeritMetadataCache::kInvalid)
        merit_dyn_on_mcache_evict(tr.evicted_id, tr.evicted_slot);
    if (_merit_mcache.node_at(tr.slot_id) != tr.node_id || query_scratch == nullptr)
        return;
    query_scratch->mark_merit_heap_dirty(tr.slot_id);
    merit_mark_refresh_needed();
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_ncache_evictable(uint32_t node_id) const
{
    if (_merit_mem_pool == nullptr || !_merit_mem_pool->active())
        return false;
    return _merit_mem_pool->contains_evictable(node_id);
}

template <typename T, typename LabelT>
typename PQFlashIndex<T, LabelT>::MeritNodeState PQFlashIndex<T, LabelT>::merit_dyn_state_unlocked(uint32_t slot_id) const
{
    return slot_id < _merit_node_state_capacity
               ? static_cast<MeritNodeState>(_merit_node_state[slot_id].load(std::memory_order_acquire))
               : MeritNodeState::NonSeed;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_set_state_unlocked(uint32_t node_id, uint32_t slot_id, MeritNodeState st)
{
    if (slot_id < _merit_node_state_capacity)
        _merit_node_state[slot_id].store(static_cast<uint8_t>(st), std::memory_order_release);
    _merit_mcache.set_evictable(node_id, st == MeritNodeState::NonSeed);
}

template <typename T, typename LabelT>
size_t PQFlashIndex<T, LabelT>::merit_dyn_heap_shard_index(uint32_t slot_id) const
{
    return static_cast<size_t>(slot_id) & (_merit_heap_shard_count - 1);
}

template <typename T, typename LabelT>
typename PQFlashIndex<T, LabelT>::MeritHeapShard &PQFlashIndex<T, LabelT>::merit_dyn_heap_shard(uint32_t slot_id)
{
    return _merit_heap_shards[merit_dyn_heap_shard_index(slot_id)];
}

template <typename T, typename LabelT>
const typename PQFlashIndex<T, LabelT>::MeritHeapShard &PQFlashIndex<T, LabelT>::merit_dyn_heap_shard(
    uint32_t slot_id) const
{
    return _merit_heap_shards[merit_dyn_heap_shard_index(slot_id)];
}

template <typename T, typename LabelT>
std::vector<typename PQFlashIndex<T, LabelT>::MeritHeapEntry> &PQFlashIndex<T, LabelT>::merit_dyn_heap_for(
    MeritHeapShard &shard, MeritHeapKind kind)
{
    return kind == MeritHeapKind::Max ? shard.max_heap : shard.min_heap;
}

template <typename T, typename LabelT>
const std::vector<typename PQFlashIndex<T, LabelT>::MeritHeapEntry> &PQFlashIndex<T, LabelT>::merit_dyn_heap_for(
    const MeritHeapShard &shard, MeritHeapKind kind) const
{
    return kind == MeritHeapKind::Max ? shard.max_heap : shard.min_heap;
}

template <typename T, typename LabelT>
typename PQFlashIndex<T, LabelT>::MeritHeapKind PQFlashIndex<T, LabelT>::merit_dyn_heap_kind_unlocked(
    uint32_t slot_id) const
{
    return slot_id < _merit_heap_kind.size()
               ? static_cast<MeritHeapKind>(_merit_heap_kind[slot_id])
               : MeritHeapKind::None;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_set_heap_kind_unlocked(uint32_t slot_id, MeritHeapKind kind)
{
    if (slot_id < _merit_heap_kind.size())
        _merit_heap_kind[slot_id] = static_cast<uint8_t>(kind);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_heap_better_unlocked(const MeritHeapEntry &a, const MeritHeapEntry &b,
                                                             MeritHeapKind kind) const
{
    if (a.score != b.score)
        return kind == MeritHeapKind::Max ? a.score > b.score : a.score < b.score;
    return kind == MeritHeapKind::Max ? a.slot_id > b.slot_id : a.slot_id < b.slot_id;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_heap_swap_unlocked(std::vector<MeritHeapEntry> &heap, size_t a, size_t b)
{
    if (a == b)
        return;
    std::swap(heap[a], heap[b]);
    _merit_heap_position[heap[a].slot_id] = static_cast<uint32_t>(a);
    _merit_heap_position[heap[b].slot_id] = static_cast<uint32_t>(b);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_heap_sift_up_unlocked(std::vector<MeritHeapEntry> &heap, size_t index,
                                                              MeritHeapKind kind)
{
    while (index > 0)
    {
        const size_t parent = (index - 1) / 2;
        if (!merit_dyn_heap_better_unlocked(heap[index], heap[parent], kind))
            break;
        merit_dyn_heap_swap_unlocked(heap, index, parent);
        index = parent;
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_heap_sift_down_unlocked(std::vector<MeritHeapEntry> &heap, size_t index,
                                                                MeritHeapKind kind)
{
    while (true)
    {
        const size_t left = index * 2 + 1;
        if (left >= heap.size())
            return;
        const size_t right = left + 1;
        size_t best = left;
        if (right < heap.size() && merit_dyn_heap_better_unlocked(heap[right], heap[left], kind))
            best = right;
        if (!merit_dyn_heap_better_unlocked(heap[best], heap[index], kind))
            return;
        merit_dyn_heap_swap_unlocked(heap, index, best);
        index = best;
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_heap_insert_unlocked(const MeritHeapEntry &entry, MeritHeapKind kind)
{
    if (entry.slot_id >= _merit_heap_position.size() || kind == MeritHeapKind::None)
        return;
    auto &heap = merit_dyn_heap_for(merit_dyn_heap_shard(entry.slot_id), kind);
    merit_dyn_set_heap_kind_unlocked(entry.slot_id, kind);
    _merit_heap_position[entry.slot_id] = static_cast<uint32_t>(heap.size());
    heap.push_back(entry);
    merit_dyn_heap_sift_up_unlocked(heap, heap.size() - 1, kind);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_heap_erase_unlocked(uint32_t slot_id)
{
    if (slot_id >= _merit_heap_position.size())
        return;
    const MeritHeapKind kind = merit_dyn_heap_kind_unlocked(slot_id);
    if (kind == MeritHeapKind::None)
        return;
    auto &heap = merit_dyn_heap_for(merit_dyn_heap_shard(slot_id), kind);
    const size_t index = _merit_heap_position[slot_id];
    if (index >= heap.size() || heap[index].slot_id != slot_id)
    {
        merit_dyn_set_heap_kind_unlocked(slot_id, MeritHeapKind::None);
        _merit_heap_position[slot_id] = MeritMetadataCache::kInvalid;
        return;
    }
    const MeritHeapEntry replacement = heap.back();
    heap.pop_back();
    merit_dyn_set_heap_kind_unlocked(slot_id, MeritHeapKind::None);
    _merit_heap_position[slot_id] = MeritMetadataCache::kInvalid;
    if (index >= heap.size())
        return;
    heap[index] = replacement;
    _merit_heap_position[replacement.slot_id] = static_cast<uint32_t>(index);
    const size_t parent = index > 0 ? (index - 1) / 2 : 0;
    if (index > 0 &&
        merit_dyn_heap_better_unlocked(heap[index], heap[parent], kind))
        merit_dyn_heap_sift_up_unlocked(heap, index, kind);
    else
        merit_dyn_heap_sift_down_unlocked(heap, index, kind);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_heap_apply_unlocked(uint32_t slot_id, float score, MeritHeapKind desired)
{
    if (slot_id >= _merit_heap_position.size())
        return;
    if (merit_dyn_heap_kind_unlocked(slot_id) == desired && desired != MeritHeapKind::None)
    {
        auto &heap = merit_dyn_heap_for(merit_dyn_heap_shard(slot_id), desired);
        const size_t index = _merit_heap_position[slot_id];
        if (index < heap.size() && heap[index].slot_id == slot_id)
        {
            heap[index].score = score;
            const size_t parent = index > 0 ? (index - 1) / 2 : 0;
            if (index > 0 &&
                merit_dyn_heap_better_unlocked(heap[index], heap[parent], desired))
                merit_dyn_heap_sift_up_unlocked(heap, index, desired);
            else
                merit_dyn_heap_sift_down_unlocked(heap, index, desired);
            return;
        }
    }
    merit_dyn_heap_erase_unlocked(slot_id);
    if (desired != MeritHeapKind::None)
        merit_dyn_heap_insert_unlocked(MeritHeapEntry{score, slot_id}, desired);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_heap_upsert_unlocked(uint32_t slot_id, float score)
{
    if (slot_id >= _merit_heap_position.size())
        return;
    const uint32_t node_id = _merit_mcache.node_at(slot_id);
    if (node_id == MeritMetadataCache::kInvalid)
    {
        merit_dyn_heap_erase_unlocked(slot_id);
        return;
    }
    const auto st = merit_dyn_state_unlocked(slot_id);
    MeritHeapKind desired = MeritHeapKind::None;
    if (st == MeritNodeState::Seed)
        desired = MeritHeapKind::Min;
    else if (st == MeritNodeState::NonSeed && merit_dyn_ncache_evictable(node_id))
        desired = MeritHeapKind::Max;
    merit_dyn_heap_apply_unlocked(slot_id, score, desired);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_clean_heap_unlocked(MeritHeapShard &shard, MeritHeapKind kind)
{
    auto &heap = merit_dyn_heap_for(shard, kind);
    while (!heap.empty())
    {
        const MeritHeapEntry entry = heap.front();
        const uint32_t node_id = _merit_mcache.node_at(entry.slot_id);
        if (kind == MeritHeapKind::Max)
        {
            if (merit_dyn_state_unlocked(entry.slot_id) == MeritNodeState::NonSeed &&
                node_id != MeritMetadataCache::kInvalid && merit_dyn_ncache_evictable(node_id))
                break;
        }
        else if (merit_dyn_state_unlocked(entry.slot_id) == MeritNodeState::Seed &&
                 node_id != MeritMetadataCache::kInvalid)
        {
            break;
        }
        merit_dyn_heap_erase_unlocked(entry.slot_id);
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_best_heap_entry(MeritHeapKind kind, MeritHeapEntry &entry,
                                                        size_t &shard_index)
{
    bool found = false;
    for (size_t index = 0; index < _merit_heap_shard_count; ++index)
    {
        MeritHeapShard &shard = _merit_heap_shards[index];
        MeritTimedMutexGuard hlock(shard.mu, MeritLockKind::Heap);
        merit_dyn_clean_heap_unlocked(shard, kind);
        const auto &heap = merit_dyn_heap_for(shard, kind);
        if (heap.empty())
            continue;
        if (!found || merit_dyn_heap_better_unlocked(heap.front(), entry, kind))
        {
            entry = heap.front();
            shard_index = index;
            found = true;
        }
    }
    return found;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_flush_dirty_heap(SSDQueryScratch<T> *query_scratch)
{
    if (query_scratch == nullptr || query_scratch->merit_heap_dirty_slots.empty())
        return;

    const size_t dirty_count = query_scratch->merit_heap_dirty_slots.size();
    query_scratch->merit_heap_dirty_nodes.resize(dirty_count);
    query_scratch->merit_heap_dirty_scores.resize(dirty_count);
    query_scratch->merit_heap_dirty_eligible.resize(dirty_count);
    for (size_t index = 0; index < dirty_count; ++index)
    {
        const uint32_t slot_id = query_scratch->merit_heap_dirty_slots[index];
        uint32_t node_id = MeritMetadataCache::kInvalid;
        float score = 0.0f;
        if (!_merit_mcache.node_score_at(slot_id, node_id, score))
        {
            query_scratch->merit_heap_dirty_nodes[index] = MeritMetadataCache::kInvalid;
            continue;
        }
        query_scratch->merit_heap_dirty_nodes[index] = node_id;
        query_scratch->merit_heap_dirty_scores[index] = score;
        query_scratch->merit_heap_dirty_eligible[index] =
            static_cast<uint8_t>(merit_dyn_ncache_evictable(node_id));
    }
    thread_local std::vector<uint32_t> shard_begin;
    thread_local std::vector<uint32_t> by_shard;
    shard_begin.assign(_merit_heap_shard_count + 1, 0);
    for (size_t index = 0; index < dirty_count; ++index)
        ++shard_begin[merit_dyn_heap_shard_index(query_scratch->merit_heap_dirty_slots[index]) + 1];
    for (size_t s = 0; s < _merit_heap_shard_count; ++s)
        shard_begin[s + 1] += shard_begin[s];
    by_shard.resize(dirty_count);
    {
        thread_local std::vector<uint32_t> cursor;
        cursor.assign(shard_begin.begin(), shard_begin.end() - 1);
        for (size_t index = 0; index < dirty_count; ++index)
            by_shard[cursor[merit_dyn_heap_shard_index(query_scratch->merit_heap_dirty_slots[index])]++] =
                static_cast<uint32_t>(index);
    }
    for (size_t shard_index = 0; shard_index < _merit_heap_shard_count; ++shard_index)
    {
        if (shard_begin[shard_index] == shard_begin[shard_index + 1])
            continue;
        MeritTimedMutexGuard hlock(_merit_heap_shards[shard_index].mu, MeritLockKind::Heap);
        for (uint32_t bi = shard_begin[shard_index]; bi < shard_begin[shard_index + 1]; ++bi)
        {
            const size_t index = by_shard[bi];
            const uint32_t slot_id = query_scratch->merit_heap_dirty_slots[index];
            const uint32_t node_id = query_scratch->merit_heap_dirty_nodes[index];
            if (node_id == MeritMetadataCache::kInvalid)
                continue;
            if (_merit_mcache.node_at(slot_id) != node_id)
                continue;
            MeritHeapKind desired = MeritHeapKind::None;
            const MeritNodeState state = merit_dyn_state_unlocked(slot_id);
            if (state == MeritNodeState::Seed)
                desired = MeritHeapKind::Min;
            else if (state == MeritNodeState::NonSeed && query_scratch->merit_heap_dirty_eligible[index] != 0)
                desired = MeritHeapKind::Max;
            merit_dyn_heap_apply_unlocked(slot_id, query_scratch->merit_heap_dirty_scores[index], desired);
        }
    }

    std::vector<uint32_t> cancellation_slots;
    if (_merit_deletion_pair_count.load(std::memory_order_acquire) != 0)
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
        for (uint32_t slot_id : query_scratch->merit_heap_dirty_slots)
        {
            const auto pending_it = _merit_deletion_to_pending.find(slot_id);
            if (pending_it == _merit_deletion_to_pending.end())
                continue;
            const uint32_t insertion_slot = pending_it->second;
            if (_merit_mcache.score_at(slot_id) >= _merit_mcache.score_at(insertion_slot))
                cancellation_slots.push_back(insertion_slot);
        }
    }
    if (!cancellation_slots.empty())
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
        for (uint32_t insertion_slot : cancellation_slots)
        {
            const auto pair_it = _merit_ready_pairs.find(insertion_slot);
            if (pair_it == _merit_ready_pairs.end())
                continue;
            const uint32_t deletion_slot = pair_it->second.deletion_slot;
            if (deletion_slot != MeritMetadataCache::kInvalid &&
                _merit_mcache.score_at(deletion_slot) >= _merit_mcache.score_at(insertion_slot))
            {
                merit_dyn_clear_ready_pair_unlocked(insertion_slot);
            }
        }
    }
    query_scratch->clear_merit_heap_dirty();
    query_scratch->merit_heap_dirty_nodes.clear();
    query_scratch->merit_heap_dirty_scores.clear();
    query_scratch->merit_heap_dirty_eligible.clear();
}

template <typename T, typename LabelT>
typename PQFlashIndex<T, LabelT>::MeritPendingMemberShard &
PQFlashIndex<T, LabelT>::merit_pending_member_shard(uint32_t node_id)
{
    return *_merit_pending_member_shards[node_id & (MERIT_PENDING_MEMBER_SHARDS - 1)];
}

template <typename T, typename LabelT>
const typename PQFlashIndex<T, LabelT>::MeritPendingMemberShard &
PQFlashIndex<T, LabelT>::merit_pending_member_shard(uint32_t node_id) const
{
    return *_merit_pending_member_shards[node_id & (MERIT_PENDING_MEMBER_SHARDS - 1)];
}

template <typename T, typename LabelT>
std::vector<uint32_t> PQFlashIndex<T, LabelT>::merit_pending_member_slots(uint32_t node_id) const
{
    const MeritPendingMemberShard &shard = merit_pending_member_shard(node_id);
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Pair, false);
    const auto it = shard.slots.find(node_id);
    return it == shard.slots.end() ? std::vector<uint32_t>{} : it.value();
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_pending_member_contains(uint32_t node_id) const
{
    const MeritPendingMemberShard &shard = merit_pending_member_shard(node_id);
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Pair, false);
    return shard.slots.find(node_id) != shard.slots.end();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_pending_member_add(uint32_t node_id, uint32_t insertion_slot)
{
    MeritPendingMemberShard &shard = merit_pending_member_shard(node_id);
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Pair, true);
    auto &slots = shard.slots[node_id];
    if (std::find(slots.begin(), slots.end(), insertion_slot) == slots.end())
        slots.push_back(insertion_slot);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_pending_member_remove(uint32_t node_id, uint32_t insertion_slot)
{
    MeritPendingMemberShard &shard = merit_pending_member_shard(node_id);
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Pair, true);
    auto it = shard.slots.find(node_id);
    if (it == shard.slots.end())
        return;
    auto &slots = it.value();
    slots.erase(std::remove(slots.begin(), slots.end(), insertion_slot), slots.end());
    if (slots.empty())
        shard.slots.erase(it);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_pending_member_clear()
{
    for (auto &shard_ptr : _merit_pending_member_shards)
    {
        MeritPendingMemberShard &shard = *shard_ptr;
        MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Pair, true);
        shard.slots.clear();
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_clear_ready_pair_unlocked(uint32_t insertion_slot, bool restore_states)
{
    auto pair_it = _merit_ready_pairs.find(insertion_slot);
    if (pair_it == _merit_ready_pairs.end())
        return;
    if (restore_states && pair_it->second.committing)
        return;

    MeritReadyPair pair = std::move(pair_it->second);
    if (pair.retained_members < _merit_pending_buffer_buckets.size())
        _merit_pending_buffer_buckets[pair.retained_members].erase(insertion_slot);
    _merit_ready_pairs.erase(pair_it);
    _merit_pending_pair_count.fetch_sub(1, std::memory_order_release);
    const uint32_t ins = pair.insertion_id;
    const uint32_t del = pair.deletion_id;
    const uint32_t ins_slot = pair.insertion_slot;
    const uint32_t del_slot = pair.deletion_slot;
    if (del_slot == MeritMetadataCache::kInvalid && _merit_reserved_free_pages > 0)
        --_merit_reserved_free_pages;
    else if (del_slot != MeritMetadataCache::kInvalid)
    {
        _merit_deletion_to_pending.erase(del_slot);
        _merit_deletion_pair_count.fetch_sub(1, std::memory_order_release);
    }
    for (uint32_t member : pair.snap.member_ids)
        merit_pending_member_remove(member, ins_slot);

    if (!restore_states)
        return;
    if (ins != std::numeric_limits<uint32_t>::max() &&
        merit_dyn_state_unlocked(ins_slot) == MeritNodeState::ReadyToInsertion)
        merit_dyn_set_state_unlocked(ins, ins_slot, MeritNodeState::NonSeed);
    if (del != std::numeric_limits<uint32_t>::max() &&
        merit_dyn_state_unlocked(del_slot) == MeritNodeState::ReadyToDeletion)
        merit_dyn_set_state_unlocked(del, del_slot, MeritNodeState::Seed);
    if (ins_slot != MeritMetadataCache::kInvalid)
    {
        MeritTimedMutexGuard hlock(merit_dyn_heap_shard(ins_slot).mu, MeritLockKind::Heap);
        merit_dyn_heap_upsert_unlocked(ins_slot, _merit_mcache.score_at(ins_slot));
    }
    if (del_slot != MeritMetadataCache::kInvalid)
    {
        MeritTimedMutexGuard hlock(merit_dyn_heap_shard(del_slot).mu, MeritLockKind::Heap);
        merit_dyn_heap_upsert_unlocked(del_slot, _merit_mcache.score_at(del_slot));
    }
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::merit_dcache_updates_frozen() const
{
    return _merit_dcache_freeze_after > 0 &&
           _merit_query_epoch.load(std::memory_order_acquire) >= _merit_dcache_freeze_after;
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::merit_dyn_disk_is_full() const
{
    MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
    return _merit_dyn_committed_pages >= _merit_dyn_page_cap;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_on_mcache_evict(uint32_t node_id, uint32_t slot_id)
{
    if (node_id == MeritMetadataCache::kInvalid)
        return;
    if (slot_id != MeritMetadataCache::kInvalid)
    {
        std::lock_guard<std::mutex> slock(merit_parent_fetched_mu(slot_id));
        if (slot_id < _merit_parent_fetched_by_slot.size() &&
            _merit_parent_fetched_owner[slot_id] == node_id)
        {
            _merit_parent_fetched_by_slot[slot_id].clear();
            _merit_parent_fetched_owner[slot_id] = MeritMetadataCache::kInvalid;
        }
    }
    merit_mark_refresh_needed();
    MeritNodeState st = MeritNodeState::NonSeed;
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
        st = merit_dyn_state_unlocked(slot_id);
        if (st == MeritNodeState::NonSeed)
            return;
    }
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
        st = merit_dyn_state_unlocked(slot_id);
        if (st == MeritNodeState::ReadyToInsertion)
            merit_dyn_clear_ready_pair_unlocked(slot_id);
        else if (st == MeritNodeState::ReadyToDeletion)
        {
            const auto pending_it = _merit_deletion_to_pending.find(slot_id);
            if (pending_it != _merit_deletion_to_pending.end())
                merit_dyn_clear_ready_pair_unlocked(pending_it->second);
        }
        merit_dyn_set_state_unlocked(node_id, slot_id, MeritNodeState::NonSeed);
    }
    {
        MeritTimedMutexGuard hlock(merit_dyn_heap_shard(slot_id).mu, MeritLockKind::Heap);
        merit_dyn_heap_erase_unlocked(slot_id);
    }
    if (st != MeritNodeState::Seed && st != MeritNodeState::ReadyToDeletion)
        return;
    if (merit_dcache_updates_frozen() || (_merit_dcache_no_seed_replace && merit_dyn_disk_is_full()))
        return;
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicUnique, true);
        merit_dyn_invalidate_seed_page(node_id);
    }
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
        _merit_heap_deletes++;
    }
    g_dyn_probe.delete_ok++;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_build_pending_flush(uint32_t seed_id,
                                                           const MeritMetadataCache::Snapshot &snap,
                                                           MeritPendingFlush &pf) const
{
    pf = MeritPendingFlush{};
    pf.seed_id = seed_id;
    pf.snap = snap;
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    const std::vector<uint32_t> now =
        merit_dyn_now_page_ids(seed_id, snap, _merit_deferred_ncache_page_write);
    std::vector<T> coords;
    std::vector<uint32_t> nbrs;
    for (uint32_t mid : now)
    {
        if (_merit_deferred_ncache_page_write)
        {
            pf.member_ids.push_back(mid);
            continue;
        }
        if (!merit_dyn_copy_member_payload(mid, coords, nbrs))
        {
            if (mid == seed_id)
                return false;
            continue;
        }
        pf.member_ids.push_back(mid);
        pf.member_coords.push_back(std::move(coords));
        pf.member_nbrs.push_back(std::move(nbrs));
    }
    if (pf.member_ids.empty() || pf.member_ids.front() != seed_id)
        return false;
    // A seed-only snapshot can wait forever because the hottest candidate may
    // never leave n-cache. Require one co-packed member when a page has room;
    // this is a flush-safety rule, not a heat threshold for seed admission.
    return nps <= 1 || pf.member_ids.size() >= 2;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_refresh_pending_members(uint32_t insertion_slot)
{
    uint32_t seed_id = MeritMetadataCache::kInvalid;
    std::vector<uint32_t> old_members;
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
        const auto pair_it = _merit_ready_pairs.find(insertion_slot);
        if (pair_it == _merit_ready_pairs.end() || pair_it->second.committing)
            return false;
        seed_id = pair_it->second.insertion_id;
        old_members = pair_it->second.snap.member_ids;
    }

    MeritMetadataCache::Snapshot snap;
    if (!_merit_mcache.snapshot(seed_id, snap))
        return false;
    std::vector<uint32_t> refreshed =
        merit_dyn_now_page_ids(seed_id, snap, _merit_deferred_ncache_page_write);
    const size_t page_capacity = static_cast<size_t>(
        (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
    for (uint32_t member_id : old_members)
    {
        if (refreshed.size() >= page_capacity)
            break;
        if (std::find(refreshed.begin(), refreshed.end(), member_id) != refreshed.end())
            continue;
        bool available = _merit_mem_pool != nullptr && _merit_mem_pool->contains(member_id);
        if (!available)
        {
            std::lock_guard<std::mutex> slock(_merit_stash_mu);
            available = _merit_payload_stash.find(member_id) != _merit_payload_stash.end();
        }
        if (available)
            refreshed.push_back(member_id);
    }
    if (refreshed.empty() || refreshed.front() != seed_id)
        return false;

    bool became_full = false;
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
        auto pair_it = _merit_ready_pairs.find(insertion_slot);
        if (pair_it == _merit_ready_pairs.end() || pair_it.value().committing ||
            pair_it.value().insertion_id != seed_id)
            return false;
        MeritReadyPair &pair = pair_it.value();
        const size_t previous_count = pair.snap.member_ids.size();
        if (pair.snap.member_ids == refreshed)
            return refreshed.size() >= page_capacity;

        if (pair.retained_members < _merit_pending_buffer_buckets.size())
            _merit_pending_buffer_buckets[pair.retained_members].erase(insertion_slot);
        for (uint32_t member_id : pair.snap.member_ids)
            merit_pending_member_remove(member_id, insertion_slot);

        pair.snap.snap = std::move(snap);
        pair.snap.member_ids = refreshed;
        pair.retained_members = 0;
        {
            std::lock_guard<std::mutex> slock(_merit_stash_mu);
            for (uint32_t member_id : pair.snap.member_ids)
                if (_merit_payload_stash.find(member_id) != _merit_payload_stash.end())
                    ++pair.retained_members;
        }
        for (uint32_t member_id : pair.snap.member_ids)
            merit_pending_member_add(member_id, insertion_slot);
        if (pair.retained_members < _merit_pending_buffer_buckets.size())
            _merit_pending_buffer_buckets[pair.retained_members].insert(insertion_slot);
        became_full = previous_count < page_capacity && pair.snap.member_ids.size() >= page_capacity;
    }

    _merit_pending_member_refreshes.fetch_add(1, std::memory_order_relaxed);
    if (became_full)
        _merit_pending_became_full.fetch_add(1, std::memory_order_relaxed);
    merit_dyn_release_unreferenced_buffer_members(old_members);
    return refreshed.size() >= page_capacity;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_maybe_refresh_pair()
{
    if (!_merit_dyn_enabled || merit_dcache_updates_frozen())
        return;

    bool full = false;
    uint32_t cand_ins = std::numeric_limits<uint32_t>::max();
    uint32_t cand_del = std::numeric_limits<uint32_t>::max();
    uint32_t cand_ins_slot = MeritMetadataCache::kInvalid;
    uint32_t cand_del_slot = MeritMetadataCache::kInvalid;
    size_t cand_ins_shard = 0;
    size_t cand_del_shard = 0;
    MeritHeapEntry max_entry;
    MeritHeapEntry min_entry;
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
        MeritTimedSharedMutexGuard dlock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
        if (_merit_ready_pairs.size() >= _merit_pending_pair_cap)
            return;
        const uint64_t committed_pages = _merit_dyn_committed_pages;
        full = committed_pages >= _merit_dyn_page_cap;
    }
    if (!merit_dyn_best_heap_entry(MeritHeapKind::Max, max_entry, cand_ins_shard))
        return;
    cand_ins_slot = max_entry.slot_id;
    cand_ins = _merit_mcache.node_at(cand_ins_slot);
    if (cand_ins == MeritMetadataCache::kInvalid)
        return;
    const auto ins_st = merit_dyn_state_unlocked(cand_ins_slot);
    if (ins_st == MeritNodeState::Seed || ins_st == MeritNodeState::ReadyToDeletion)
    {
        g_dyn_probe.pair_skip_max_on_disk++;
        return;
    }
    if (full)
    {
        if (_merit_dcache_no_seed_replace)
        {
            g_dyn_probe.pair_skip_count++;
            return;
        }
        if (!merit_dyn_best_heap_entry(MeritHeapKind::Min, min_entry, cand_del_shard))
            return;
        cand_del_slot = min_entry.slot_id;
        cand_del = _merit_mcache.node_at(cand_del_slot);
        if (cand_del == MeritMetadataCache::kInvalid)
            return;
        if (max_entry.score <= min_entry.score || cand_ins == cand_del)
        {
            g_dyn_probe.pair_skip_count++;
            return;
        }
    }

    MeritMetadataCache::Snapshot snap;
    if (!_merit_mcache.snapshot(cand_ins, snap))
        return;
    MeritPendingFlush pf;
    if (!merit_dyn_build_pending_flush(cand_ins, snap, pf))
    {
        MeritHeapShard &shard = _merit_heap_shards[cand_ins_shard];
        MeritTimedMutexGuard hlock(shard.mu, MeritLockKind::Heap);
        merit_dyn_clean_heap_unlocked(shard, MeritHeapKind::Max);
        if (!shard.max_heap.empty() && shard.max_heap.front().slot_id == cand_ins_slot &&
            _merit_mcache.node_at(cand_ins_slot) == cand_ins)
            merit_dyn_heap_erase_unlocked(cand_ins_slot);
        return;
    }

    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
        MeritTimedSharedMutexGuard dlock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
        if (_merit_ready_pairs.size() >= _merit_pending_pair_cap)
            return;
        const uint64_t committed_pages = _merit_dyn_committed_pages;
        const bool now_full = committed_pages >= _merit_dyn_page_cap;
        if (now_full != full)
            return;
        float current_max_score = 0.0f;
        {
            MeritHeapShard &shard = _merit_heap_shards[cand_ins_shard];
            MeritTimedMutexGuard hlock(shard.mu, MeritLockKind::Heap);
            merit_dyn_clean_heap_unlocked(shard, MeritHeapKind::Max);
            if (shard.max_heap.empty() || shard.max_heap.front().slot_id != cand_ins_slot ||
                _merit_mcache.node_at(cand_ins_slot) != cand_ins)
                return;
            current_max_score = shard.max_heap.front().score;
        }
        if (full)
        {
            MeritHeapShard &shard = _merit_heap_shards[cand_del_shard];
            MeritTimedMutexGuard hlock(shard.mu, MeritLockKind::Heap);
            merit_dyn_clean_heap_unlocked(shard, MeritHeapKind::Min);
            if (shard.min_heap.empty() || shard.min_heap.front().slot_id != cand_del_slot ||
                _merit_mcache.node_at(cand_del_slot) != cand_del ||
                current_max_score <= shard.min_heap.front().score)
                return;
        }
        MeritReadyPair pair;
        pair.insertion_id = cand_ins;
        pair.deletion_id = cand_del;
        pair.insertion_slot = cand_ins_slot;
        pair.deletion_slot = cand_del_slot;
        pair.created_epoch = _merit_query_epoch.load(std::memory_order_relaxed);
        pair.snap = std::move(pf);
        if (_merit_pending_buffer_cap_bytes > 0)
        {
            std::lock_guard<std::mutex> slock(_merit_stash_mu);
            for (uint32_t mid : pair.snap.member_ids)
                if (_merit_payload_stash.find(mid) != _merit_payload_stash.end())
                    ++pair.retained_members;
        }
        for (uint32_t mid : pair.snap.member_ids)
            merit_pending_member_add(mid, cand_ins_slot);
        if (cand_del_slot == MeritMetadataCache::kInvalid)
            ++_merit_reserved_free_pages;
        else
        {
            _merit_deletion_to_pending[cand_del_slot] = cand_ins_slot;
            _merit_deletion_pair_count.fetch_add(1, std::memory_order_release);
        }
        _merit_ready_pairs[cand_ins_slot] = std::move(pair);
        const uint16_t retained_members = _merit_ready_pairs[cand_ins_slot].retained_members;
        if (retained_members < _merit_pending_buffer_buckets.size())
            _merit_pending_buffer_buckets[retained_members].insert(cand_ins_slot);
        _merit_pending_pair_count.fetch_add(1, std::memory_order_release);
        merit_dyn_set_state_unlocked(cand_ins, cand_ins_slot, MeritNodeState::ReadyToInsertion);
        {
            MeritTimedMutexGuard hlock(merit_dyn_heap_shard(cand_ins_slot).mu, MeritLockKind::Heap);
            merit_dyn_heap_upsert_unlocked(cand_ins_slot, _merit_mcache.score_at(cand_ins_slot));
        }
        if (cand_del != std::numeric_limits<uint32_t>::max())
        {
            merit_dyn_set_state_unlocked(cand_del, cand_del_slot, MeritNodeState::ReadyToDeletion);
            MeritTimedMutexGuard hlock(merit_dyn_heap_shard(cand_del_slot).mu, MeritLockKind::Heap);
            merit_dyn_heap_upsert_unlocked(cand_del_slot, _merit_mcache.score_at(cand_del_slot));
        }
        _merit_pair_refresh++;
        g_dyn_probe.pair_set++;
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_request_refresh()
{
    if (!_merit_dyn_enabled)
        return;
    merit_mark_refresh_needed();
    if (_merit_refresh_thread_on && !merit_is_refresh_thread())
        return;
    if (_merit_pending_pair_count.load(std::memory_order_acquire) >= _merit_pending_pair_cap)
        return;
    if (_merit_refresh_running.load(std::memory_order_relaxed))
        return;

    bool expected = false;
    if (!_merit_refresh_running.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        return;

    for (unsigned pass = 0; pass < 4; ++pass)
    {
        _merit_refresh_needed.store(false, std::memory_order_release);
        if (_merit_pending_pair_count.load(std::memory_order_acquire) < _merit_pending_pair_cap)
            merit_dyn_maybe_refresh_pair();
        if (!_merit_refresh_needed.load(std::memory_order_acquire))
            break;
    }
    _merit_refresh_running.store(false, std::memory_order_release);

    if (_merit_refresh_needed.load(std::memory_order_acquire) &&
        _merit_pending_pair_count.load(std::memory_order_acquire) < _merit_pending_pair_cap)
    {
        expected = false;
        if (_merit_refresh_running.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        {
            _merit_refresh_needed.store(false, std::memory_order_release);
            merit_dyn_maybe_refresh_pair();
            _merit_refresh_running.store(false, std::memory_order_release);
        }
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_enqueue_ready_pair(uint32_t insertion_slot, MeritCommitSource source)
{
    if (!_merit_commit_started || _merit_commit_stop.load(std::memory_order_acquire) ||
        insertion_slot == MeritMetadataCache::kInvalid || _merit_freeze_maintenance ||
        merit_dcache_updates_frozen())
        return false;

    {
        std::lock_guard<std::mutex> lock(_merit_commit_q_mu);
        if (_merit_commit_pending_slots.find(insertion_slot) != _merit_commit_pending_slots.end())
        {
            _merit_commit_coalesced.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        if (_merit_commit_q.size() >= _merit_commit_q_cap)
        {
            _merit_commit_dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        MeritPairCommitJob job;
        job.insertion_slot = insertion_slot;
        job.source = source;
        job.enqueued_ns = merit_lock_now_ns();
        _merit_commit_q.push_back(job);
        _merit_commit_pending_slots.insert(insertion_slot);
        _merit_commit_enqueued.fetch_add(1, std::memory_order_relaxed);
        _merit_commit_source_enqueued[static_cast<size_t>(source)].fetch_add(1, std::memory_order_relaxed);
    }
    _merit_commit_q_cv.notify_one();
    return true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_commit_loop()
{
    merit_pin_background_thread();
    if (_merit_dyn_reader != nullptr)
        _merit_dyn_reader->register_thread();
    while (true)
    {
        MeritPairCommitJob job;
        {
            std::unique_lock<std::mutex> lock(_merit_commit_q_mu);
            _merit_commit_q_cv.wait(lock, [this] {
                return _merit_commit_stop.load(std::memory_order_acquire) || !_merit_commit_q.empty();
            });
            if (_merit_commit_stop.load(std::memory_order_acquire) && _merit_commit_q.empty())
                return;
            job = _merit_commit_q.front();
            _merit_commit_q.pop_front();
        }

        const uint64_t work_started_ns = merit_lock_now_ns();
        if (job.enqueued_ns != 0)
            _merit_commit_queue_ns.fetch_add(work_started_ns - job.enqueued_ns, std::memory_order_relaxed);
        const bool committed = merit_dyn_commit_ready_pair(job.insertion_slot);
        _merit_commit_work_ns.fetch_add(merit_lock_now_ns() - work_started_ns, std::memory_order_relaxed);
        if (committed)
        {
            _merit_commit_completed.fetch_add(1, std::memory_order_relaxed);
            _merit_commit_source_completed[static_cast<size_t>(job.source)].fetch_add(1,
                                                                                      std::memory_order_relaxed);
        }
        else
            _merit_commit_failed.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(_merit_commit_q_mu);
            _merit_commit_pending_slots.erase(job.insertion_slot);
        }
        _merit_commit_q_cv.notify_all();
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::start_merit_dyn_commit_workers()
{
    if (_merit_commit_started)
        return;
    {
        std::lock_guard<std::mutex> lock(_merit_commit_q_mu);
        _merit_commit_q.clear();
        _merit_commit_pending_slots.clear();
        _merit_commit_stop.store(false, std::memory_order_release);
    }
    _merit_commit_threads.clear();
    _merit_commit_threads.reserve(_merit_commit_thread_count);
    for (size_t thread = 0; thread < _merit_commit_thread_count; ++thread)
        _merit_commit_threads.emplace_back([this] { merit_dyn_commit_loop(); });
    _merit_commit_started = true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::stop_merit_dyn_commit_workers()
{
    if (!_merit_commit_started)
        return;
    {
        std::lock_guard<std::mutex> lock(_merit_commit_q_mu);
        _merit_commit_stop.store(true, std::memory_order_release);
    }
    _merit_commit_q_cv.notify_all();
    for (std::thread &thread : _merit_commit_threads)
        if (thread.joinable())
            thread.join();
    _merit_commit_threads.clear();
    {
        std::lock_guard<std::mutex> lock(_merit_commit_q_mu);
        _merit_commit_q.clear();
        _merit_commit_pending_slots.clear();
    }
    _merit_commit_started = false;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_wait_for_commits()
{
    if (!_merit_commit_started)
        return;
    std::unique_lock<std::mutex> lock(_merit_commit_q_mu);
    _merit_commit_q_cv.wait(lock, [this] {
        return _merit_commit_pending_slots.empty() ||
               _merit_commit_stop.load(std::memory_order_acquire);
    });
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::merit_dyn_commit_ready_pair(uint32_t insertion_slot)
{
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
        auto pair_it = _merit_ready_pairs.find(insertion_slot);
        if (pair_it == _merit_ready_pairs.end() || pair_it->second.committing)
            return false;
        const MeritReadyPair &candidate = pair_it->second;
        const size_t page_capacity = static_cast<size_t>(
            (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
        if (_merit_pending_require_full_page &&
            candidate.snap.member_ids.size() < page_capacity)
            return false;
        const float insertion_score = _merit_mcache.score_at(candidate.insertion_slot);
        if (_merit_mcache.node_at(candidate.insertion_slot) != candidate.insertion_id ||
            (candidate.deletion_id != std::numeric_limits<uint32_t>::max() &&
             _merit_mcache.node_at(candidate.deletion_slot) != candidate.deletion_id))
        {
            merit_dyn_clear_ready_pair_unlocked(insertion_slot);
            return false;
        }
        if (candidate.deletion_id != std::numeric_limits<uint32_t>::max() &&
            insertion_score <= _merit_mcache.score_at(candidate.deletion_slot))
        {
            merit_dyn_clear_ready_pair_unlocked(insertion_slot);
            return false;
        }
        if (pair_it.value().retained_members < _merit_pending_buffer_buckets.size())
            _merit_pending_buffer_buckets[pair_it.value().retained_members].erase(insertion_slot);
        pair_it.value().committing = true;
    }

    MeritReadyPair pair;
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
        const auto pair_it = _merit_ready_pairs.find(insertion_slot);
        if (pair_it == _merit_ready_pairs.end() || !pair_it->second.committing)
            return false;
        pair = pair_it->second;
    }
    MeritPendingFlush pf = pair.snap;
    if (_merit_deferred_ncache_page_write)
    {
        pf.member_coords.clear();
        pf.member_nbrs.clear();
        pf.member_coords.reserve(pf.member_ids.size());
        pf.member_nbrs.reserve(pf.member_ids.size());
        for (uint32_t member_id : pf.member_ids)
        {
            std::vector<T> coords;
            std::vector<uint32_t> nbrs;
            if (!merit_dyn_copy_member_payload(member_id, coords, nbrs))
            {
                pf.member_coords.clear();
                pf.member_nbrs.clear();
                break;
            }
            pf.member_coords.push_back(std::move(coords));
            pf.member_nbrs.push_back(std::move(nbrs));
        }
        if (pf.member_coords.size() != pf.member_ids.size())
        {
            MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
            auto pair_it = _merit_ready_pairs.find(insertion_slot);
            if (pair_it != _merit_ready_pairs.end())
            {
                pair_it.value().committing = false;
                merit_dyn_clear_ready_pair_unlocked(insertion_slot);
            }
            return false;
        }
    }
    const bool wrote = merit_dyn_commit_one(pf, pair.deletion_id);

    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
        auto pair_it = _merit_ready_pairs.find(insertion_slot);
        if (pair_it == _merit_ready_pairs.end())
            return false;
        if (!wrote)
        {
            pair_it.value().committing = false;
            merit_dyn_clear_ready_pair_unlocked(insertion_slot);
            return false;
        }

        const uint32_t ins = pair.insertion_id;
        const uint32_t del = pair.deletion_id;
        const uint32_t ins_slot = pair.insertion_slot;
        const uint32_t del_slot = pair.deletion_slot;
        const float insertion_score = _merit_mcache.score_at(ins_slot);
        merit_dyn_clear_ready_pair_unlocked(ins_slot, false);
        if (del == std::numeric_limits<uint32_t>::max() && merit_dyn_disk_is_full())
        {
            std::vector<uint32_t> surplus_free_pairs;
            surplus_free_pairs.reserve(_merit_ready_pairs.size());
            for (const auto &kv : _merit_ready_pairs)
                if (kv.second.deletion_slot == MeritMetadataCache::kInvalid)
                    surplus_free_pairs.push_back(kv.first);
            for (uint32_t slot_id : surplus_free_pairs)
                merit_dyn_clear_ready_pair_unlocked(slot_id);
        }
        if (del != std::numeric_limits<uint32_t>::max())
        {
            merit_dyn_set_state_unlocked(del, del_slot, MeritNodeState::NonSeed);
            {
                MeritTimedMutexGuard hlock(merit_dyn_heap_shard(del_slot).mu, MeritLockKind::Heap);
                merit_dyn_heap_upsert_unlocked(del_slot, _merit_mcache.score_at(del_slot));
            }
            _merit_heap_deletes++;
            g_dyn_probe.delete_ok++;
        }
        merit_dyn_set_state_unlocked(ins, ins_slot, MeritNodeState::Seed);
        {
            MeritTimedMutexGuard hlock(merit_dyn_heap_shard(ins_slot).mu, MeritLockKind::Heap);
            merit_dyn_heap_upsert_unlocked(ins_slot, insertion_score);
        }
        _merit_heap_writes++;
        g_dyn_probe.write_ok++;
    }
    merit_dyn_release_unreferenced_buffer_members(pair.snap.member_ids);
    return true;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_commit_expired_pairs(uint32_t query_epoch, QueryStats *stats)
{
    if (!_merit_dyn_enabled || !_merit_deferred_ncache_page_write)
        return;

    std::vector<uint32_t> expired;
    expired.reserve(_merit_pending_force_batch);
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
        for (const auto &entry : _merit_ready_pairs)
        {
            const MeritReadyPair &pair = entry.second;
            if (!pair.committing && pair.created_epoch <= query_epoch &&
                static_cast<uint32_t>(query_epoch - pair.created_epoch) >= _merit_pending_max_age_queries)
            {
                expired.push_back(entry.first);
                if (expired.size() >= _merit_pending_force_batch)
                    break;
            }
        }
    }

    g_dyn_probe.timeout_trig.fetch_add(expired.size(), std::memory_order_relaxed);
    for (uint32_t insertion_slot : expired)
    {
        if (_merit_pending_require_full_page)
            merit_dyn_refresh_pending_members(insertion_slot);
        if (!merit_dyn_enqueue_ready_pair(insertion_slot, MeritCommitSource::Expired))
        {
            if (_merit_pending_require_full_page)
            {
                MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
                const auto pair_it = _merit_ready_pairs.find(insertion_slot);
                if (pair_it != _merit_ready_pairs.end() && !pair_it->second.committing &&
                    pair_it->second.created_epoch <= query_epoch &&
                    static_cast<uint32_t>(query_epoch - pair_it->second.created_epoch) >=
                        _merit_pending_max_age_queries)
                    merit_dyn_clear_ready_pair_unlocked(insertion_slot);
            }
            continue;
        }
        g_dyn_probe.timeout_ok.fetch_add(1, std::memory_order_relaxed);
        if (stats != nullptr)
            stats->n_merit_dyn_flushes++;
    }
}

template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::merit_dyn_admit_node(uint32_t node_id, const char *node_disk_buf, QueryStats *stats,
                                                      SSDQueryScratch<T> *query_scratch, uint32_t search_hop)
{
    if (_merit_mem_pool == nullptr || !_merit_mem_pool->active() ||
        !(_merit_mem_runtime_admit || _merit_dyn_enabled))
        return MeritMemoryPool<T>::INVALID_NODE;

    bool admitted_after_rejection = false;
    {
        MERIT_SECTION(kSecAdmitShould);
        if (!_merit_mem_pool->should_admit(node_id, search_hop, admitted_after_rejection))
            return MeritMemoryPool<T>::INVALID_NODE;
    }

    const bool dynamic_maintenance =
        _merit_dyn_enabled && query_scratch != nullptr &&
        query_scratch->merit_dcache_maintenance_enabled;
    if (dynamic_maintenance && !_merit_mem_pool->contains(node_id))
    {
        MERIT_SECTION(kSecAdmitEvictHook);
        uint32_t victim;
        {
            MERIT_SECTION(kSecEvictPeek);
            victim = _merit_mem_pool->peek_lru_victim(node_id);
        }
        if (victim != MeritMemoryPool<T>::INVALID_NODE)
            merit_dyn_on_ncache_evict(victim, stats);
    }
    uint32_t evicted;
    {
        MERIT_SECTION(kSecAdmitTry);
        evicted = _merit_mem_pool->try_admit(node_id, node_disk_buf, _disk_bytes_per_point, _max_node_len,
                                             search_hop, admitted_after_rejection);
    }
    MERIT_SECTION(kSecAdmitTail);
    if (stats != nullptr && evicted != MeritMemoryPool<T>::INVALID_NODE)
        stats->n_merit_mem_evictions++;
    if (dynamic_maintenance && _merit_mem_pool->contains(node_id) && query_scratch != nullptr)
    {
        const uint32_t slot_id = _merit_mcache.slot_of(node_id);
        if (slot_id != MeritMetadataCache::kInvalid)
            query_scratch->mark_merit_heap_dirty(slot_id);
    }
    return evicted;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_stash_evict_unlocked()
{
    size_t pinned_rotations = 0;
    while (_merit_payload_stash.size() > _merit_stash_cap && !_merit_stash_lru.empty())
    {
        const uint32_t victim = _merit_stash_lru.back();
        if (_merit_patch_pin_count.find(victim) != _merit_patch_pin_count.end())
        {
            if (++pinned_rotations > _merit_payload_stash.size())
                break;
            _merit_stash_lru.splice(_merit_stash_lru.begin(), _merit_stash_lru, std::prev(_merit_stash_lru.end()));
            continue;
        }
        _merit_stash_lru.pop_back();
        _merit_payload_stash.erase(victim);
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_note_base_load(uint32_t node_id, uint32_t parent, const char *node_disk_buf,
                                                       SSDQueryScratch<T> *query_scratch)
{
    if (!_merit_dyn_enabled || query_scratch == nullptr ||
        !query_scratch->merit_dcache_maintenance_enabled || node_disk_buf == nullptr)
        return;

    if (_merit_real_io_coaccess)
        query_scratch->merit_storage_node_order.push_back(node_id);

    const T *coord_src = offset_to_node_coords(const_cast<char *>(node_disk_buf));
    const uint32_t *nhood = offset_to_node_nhood(const_cast<char *>(node_disk_buf));
    const uint32_t nnbrs = nhood[0];
    if (nnbrs > _max_degree)
        return;

    uint64_t note_t0 = merit_section_on() ? merit_section_now() : 0;
    if (!_merit_deferred_ncache_page_write)
    {
        std::lock_guard<std::mutex> slock(_merit_stash_mu);
        {
            const size_t ncoords = _disk_bytes_per_point / sizeof(T);
            std::vector<T> coords(coord_src, coord_src + ncoords);
            std::vector<uint32_t> nbrs(nhood + 1, nhood + 1 + nnbrs);
            auto it = _merit_payload_stash.find(node_id);
            if (it != _merit_payload_stash.end())
            {
                it.value().coords = std::move(coords);
                it.value().nbrs = std::move(nbrs);
                _merit_stash_lru.splice(_merit_stash_lru.begin(), _merit_stash_lru, it.value().lru_it);
                it.value().lru_it = _merit_stash_lru.begin();
            }
            else
            {
                _merit_stash_lru.push_front(node_id);
                MeritPayloadStashEntry ent;
                ent.coords = std::move(coords);
                ent.nbrs = std::move(nbrs);
                ent.lru_it = _merit_stash_lru.begin();
                _merit_payload_stash[node_id] = std::move(ent);
                merit_dyn_stash_evict_unlocked();
            }
        }
    }

    const uint32_t parent_slot = parent != std::numeric_limits<uint32_t>::max()
                                     ? _merit_mcache.slot_of(parent)
                                     : MeritMetadataCache::kInvalid;
    if (parent_slot != MeritMetadataCache::kInvalid && parent_slot < _merit_parent_fetched_by_slot.size())
    {
        std::lock_guard<std::mutex> slock(merit_parent_fetched_mu(parent_slot));
        if (_merit_parent_fetched_owner[parent_slot] != parent)
        {
            _merit_parent_fetched_by_slot[parent_slot].clear();
            _merit_parent_fetched_owner[parent_slot] = parent;
        }
        _merit_parent_fetched_by_slot[parent_slot].touch(node_id);
    }

    if (parent_slot != MeritMetadataCache::kInvalid && query_scratch != nullptr)
        query_scratch->mark_merit_heap_dirty(parent_slot);
    if (parent != std::numeric_limits<uint32_t>::max())
        merit_dyn_maybe_mark_seed(parent);
    if (note_t0 != 0)
        merit_section_acc().ns[kSecNoteStash] += merit_section_now() - note_t0;

    MERIT_SECTION(kSecNotePatch);
    if (_merit_dcache_query_patch && parent != std::numeric_limits<uint32_t>::max() &&
        merit_dyn_note_patch_candidate(node_id, parent, query_scratch))
    {
        const size_t ncoords = _disk_bytes_per_point / sizeof(T);
        std::vector<T> coords(coord_src, coord_src + ncoords);
        std::vector<uint32_t> nbrs(nhood + 1, nhood + 1 + nnbrs);
        std::lock_guard<std::mutex> slock(_merit_stash_mu);
        auto it = _merit_payload_stash.find(node_id);
        if (it != _merit_payload_stash.end())
        {
            it.value().coords = std::move(coords);
            it.value().nbrs = std::move(nbrs);
            _merit_stash_lru.splice(_merit_stash_lru.begin(), _merit_stash_lru, it.value().lru_it);
            it.value().lru_it = _merit_stash_lru.begin();
        }
        else
        {
            _merit_stash_lru.push_front(node_id);
            MeritPayloadStashEntry ent;
            ent.coords = std::move(coords);
            ent.nbrs = std::move(nbrs);
            ent.lru_it = _merit_stash_lru.begin();
            _merit_payload_stash[node_id] = std::move(ent);
        }
        merit_dyn_stash_evict_unlocked();
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_has_payload(uint32_t node_id) const
{
    if (_merit_mem_pool != nullptr && _merit_mem_pool->active() && _merit_mem_pool->contains(node_id))
        return true;
    {
        std::lock_guard<std::mutex> slock(_merit_stash_mu);
        if (_merit_payload_stash.find(node_id) != _merit_payload_stash.end())
            return true;
    }
    MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
    bool found = false;
    merit_dc_for_each_loc_unlocked(node_id, [&](const MeritDiskLoc &loc, size_t) {
        if (found)
            return;
        if (!merit_dyn_is_sector(loc.sector))
            return;
        const uint32_t idx = loc.sector - MERIT_DYN_SECTOR_BASE;
        if (merit_dyn_page_live(idx))
            found = true;
    });
    return found;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_extract_overlay_payload(uint32_t node_id, std::vector<T> &coords,
                                                               std::vector<uint32_t> &nbrs) const
{
    MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
    bool found = false;
    merit_dc_for_each_loc_unlocked(node_id, [&](const MeritDiskLoc &loc, size_t) {
        if (found)
            return;
        if (!merit_dyn_is_sector(loc.sector))
            return;
        const uint32_t idx = loc.sector - MERIT_DYN_SECTOR_BASE;
        if (!merit_dyn_page_live(idx) || _merit_dyn_reader == nullptr)
            return;
        char *page = nullptr;
        alloc_aligned(reinterpret_cast<void **>(&page), defaults::SECTOR_LEN, defaults::SECTOR_LEN);
        std::vector<AlignedRead> reads;
        reads.emplace_back(static_cast<uint64_t>(idx) * defaults::SECTOR_LEN, defaults::SECTOR_LEN, page);
        _merit_dyn_reader->read(reads, _merit_dyn_reader->get_ctx());
        const char *slot = page + static_cast<uint64_t>(loc.slot) * _max_node_len;
        const T *coord_src = reinterpret_cast<const T *>(slot);
        const uint32_t *nhood = reinterpret_cast<const uint32_t *>(slot + _disk_bytes_per_point);
        const uint32_t nnbrs = nhood[0];
        if (nnbrs > _max_degree)
        {
            aligned_free(page);
            return;
        }
        const size_t ncoords = _disk_bytes_per_point / sizeof(T);
        coords.assign(coord_src, coord_src + ncoords);
        nbrs.assign(nhood + 1, nhood + 1 + nnbrs);
        aligned_free(page);
        found = true;
    });
    return found;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_copy_member_payload(uint32_t node_id, std::vector<T> &coords,
                                                            std::vector<uint32_t> &nbrs, bool allow_disk) const
{
    if (_merit_mem_pool != nullptr && _merit_mem_pool->copy_payload(node_id, coords, nbrs))
        return true;
    {
        std::lock_guard<std::mutex> slock(_merit_stash_mu);
        const auto it = _merit_payload_stash.find(node_id);
        if (it != _merit_payload_stash.end())
        {
            coords = it.value().coords;
            nbrs = it.value().nbrs;
            return true;
        }
    }
    if (!allow_disk)
        return false;
    return merit_dyn_extract_overlay_payload(node_id, coords, nbrs);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_buffer_evicted_member(uint32_t node_id, bool &inserted)
{
    inserted = false;
    if (_merit_pending_buffer_cap_bytes == 0 || _merit_mem_pool == nullptr)
        return false;

    std::vector<T> coords;
    std::vector<uint32_t> nbrs;
    if (!_merit_mem_pool->copy_payload(node_id, coords, nbrs))
        return false;

    std::lock_guard<std::mutex> slock(_merit_stash_mu);
    auto existing = _merit_payload_stash.find(node_id);
    if (existing != _merit_payload_stash.end())
    {
        _merit_stash_lru.splice(_merit_stash_lru.begin(), _merit_stash_lru, existing.value().lru_it);
        existing.value().lru_it = _merit_stash_lru.begin();
        return _merit_pending_buffer_payload_bytes > _merit_pending_buffer_cap_bytes;
    }

    _merit_stash_lru.push_front(node_id);
    MeritPayloadStashEntry entry;
    entry.coords = std::move(coords);
    entry.nbrs = std::move(nbrs);
    entry.lru_it = _merit_stash_lru.begin();
    _merit_pending_buffer_payload_bytes +=
        entry.coords.size() * sizeof(T) + entry.nbrs.size() * sizeof(uint32_t);
    _merit_payload_stash[node_id] = std::move(entry);
    _merit_pending_buffer_inserts.fetch_add(1, std::memory_order_relaxed);
    inserted = true;
    return _merit_pending_buffer_payload_bytes > _merit_pending_buffer_cap_bytes;
}

template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::merit_dyn_best_buffered_pair(bool require_ready) const
{
    MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
    for (size_t retained = _merit_pending_buffer_buckets.size(); retained > 1; --retained)
    {
        const auto &bucket = _merit_pending_buffer_buckets[retained - 1];
        for (uint32_t insertion_slot : bucket)
        {
            const auto pair_it = _merit_ready_pairs.find(insertion_slot);
            if (pair_it != _merit_ready_pairs.end() && !pair_it->second.committing &&
                pair_it->second.retained_members == retained - 1)
            {
                const size_t page_capacity = static_cast<size_t>(
                    (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
                if (_merit_pending_require_full_page &&
                    pair_it->second.snap.member_ids.size() < page_capacity)
                    continue;
                if (require_ready &&
                    (pair_it->second.retained_members < _merit_pending_buffer_ready_min_members ||
                     static_cast<uint64_t>(pair_it->second.retained_members) * 100 <
                         static_cast<uint64_t>(pair_it->second.snap.member_ids.size()) *
                             _merit_pending_buffer_ready_percent))
                    continue;
                return insertion_slot;
            }
        }
    }
    return MeritMetadataCache::kInvalid;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_release_unreferenced_buffer_members(
    const std::vector<uint32_t> &member_ids)
{
    if (_merit_pending_buffer_cap_bytes == 0)
        return;

    MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
    std::lock_guard<std::mutex> slock(_merit_stash_mu);
    for (uint32_t member_id : member_ids)
    {
        if (merit_pending_member_contains(member_id))
            continue;
        auto stash_it = _merit_payload_stash.find(member_id);
        if (stash_it == _merit_payload_stash.end())
            continue;
        const uint64_t payload_bytes =
            stash_it.value().coords.size() * sizeof(T) +
            stash_it.value().nbrs.size() * sizeof(uint32_t);
        _merit_stash_lru.erase(stash_it.value().lru_it);
        _merit_payload_stash.erase(stash_it);
        _merit_pending_buffer_payload_bytes =
            payload_bytes >= _merit_pending_buffer_payload_bytes
                ? 0
                : _merit_pending_buffer_payload_bytes - payload_bytes;
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_trim_pending_buffer()
{
    if (_merit_pending_buffer_cap_bytes == 0)
        return;

    for (;;)
    {
        uint32_t victim = MeritMetadataCache::kInvalid;
        {
            std::lock_guard<std::mutex> slock(_merit_stash_mu);
            if (_merit_pending_buffer_payload_bytes <= _merit_pending_buffer_cap_bytes ||
                _merit_stash_lru.empty())
                return;
            victim = _merit_stash_lru.back();
        }

        bool cleared_pair = false;
        {
            MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
            const std::vector<uint32_t> dependent_slots = merit_pending_member_slots(victim);
            for (uint32_t insertion_slot : dependent_slots)
            {
                const auto pair_it = _merit_ready_pairs.find(insertion_slot);
                if (pair_it != _merit_ready_pairs.end() && pair_it->second.committing)
                    return;
            }
            for (uint32_t insertion_slot : dependent_slots)
            {
                if (_merit_ready_pairs.find(insertion_slot) == _merit_ready_pairs.end())
                    continue;
                merit_dyn_clear_ready_pair_unlocked(insertion_slot);
                cleared_pair = true;
            }

            std::lock_guard<std::mutex> slock(_merit_stash_mu);
            auto stash_it = _merit_payload_stash.find(victim);
            if (stash_it == _merit_payload_stash.end())
                continue;
            const uint64_t payload_bytes =
                stash_it.value().coords.size() * sizeof(T) +
                stash_it.value().nbrs.size() * sizeof(uint32_t);
            _merit_stash_lru.erase(stash_it.value().lru_it);
            _merit_payload_stash.erase(stash_it);
            _merit_pending_buffer_payload_bytes =
                payload_bytes >= _merit_pending_buffer_payload_bytes
                    ? 0
                    : _merit_pending_buffer_payload_bytes - payload_bytes;
            _merit_pending_buffer_drops.fetch_add(1, std::memory_order_relaxed);
        }
        if (cleared_pair)
            merit_dyn_request_refresh();
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_on_ncache_evict(uint32_t node_id, QueryStats *stats)
{
    if (!_merit_dyn_enabled || node_id == MeritMemoryPool<T>::INVALID_NODE)
        return;

    std::vector<uint32_t> pending_slots;
    std::vector<uint32_t> refresh_slots;
    {
        MERIT_SECTION(kSecEvictPair);
        pending_slots = merit_pending_member_slots(node_id);
        if (!pending_slots.empty())
        {
            if (_merit_pending_require_full_page)
            {
                MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
                const size_t page_capacity = static_cast<size_t>(
                    (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
                for (uint32_t insertion_slot : pending_slots)
                {
                    const auto pair_it = _merit_ready_pairs.find(insertion_slot);
                    if (pair_it != _merit_ready_pairs.end() &&
                        pair_it->second.snap.member_ids.size() < page_capacity)
                        refresh_slots.push_back(insertion_slot);
                }
            }
        }
    }
    for (uint32_t insertion_slot : refresh_slots)
        merit_dyn_refresh_pending_members(insertion_slot);
    if (!refresh_slots.empty())
    {
        pending_slots = merit_pending_member_slots(node_id);
    }
    if (!pending_slots.empty())
    {
        g_dyn_probe.write_trig += pending_slots.size();
        {
            MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
            _merit_ncache_write_trig += pending_slots.size();
        }
        if (_merit_pending_buffer_cap_bytes > 0)
        {
            bool inserted = false;
            const bool pressure = merit_dyn_buffer_evicted_member(node_id, inserted);
            if (inserted)
            {
                MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, true);
                const std::vector<uint32_t> member_slots = merit_pending_member_slots(node_id);
                if (!member_slots.empty())
                {
                    for (uint32_t insertion_slot : member_slots)
                    {
                        auto pair_it = _merit_ready_pairs.find(insertion_slot);
                        if (pair_it != _merit_ready_pairs.end() && !pair_it.value().committing &&
                            pair_it.value().retained_members < pair_it.value().snap.member_ids.size())
                        {
                            const uint16_t old_retained = pair_it.value().retained_members;
                            if (old_retained < _merit_pending_buffer_buckets.size())
                                _merit_pending_buffer_buckets[old_retained].erase(insertion_slot);
                            ++pair_it.value().retained_members;
                            if (pair_it.value().retained_members < _merit_pending_buffer_buckets.size())
                                _merit_pending_buffer_buckets[pair_it.value().retained_members].insert(
                                    insertion_slot);
                        }
                    }
                }
            }
            uint32_t best_slot = MeritMetadataCache::kInvalid;
            bool ready_trigger = false;
            if (_merit_pending_buffer_ready_percent > 0)
            {
                best_slot = merit_dyn_best_buffered_pair(true);
                ready_trigger = best_slot != MeritMetadataCache::kInvalid;
            }
            if (pressure && best_slot == MeritMetadataCache::kInvalid)
                best_slot = merit_dyn_best_buffered_pair(false);
            if (best_slot != MeritMetadataCache::kInvalid &&
                merit_dyn_enqueue_ready_pair(best_slot, MeritCommitSource::Buffered))
            {
                if (ready_trigger)
                    _merit_pending_buffer_ready_writes.fetch_add(1, std::memory_order_relaxed);
                else
                    _merit_pending_buffer_pressure_writes.fetch_add(1, std::memory_order_relaxed);
                if (stats != nullptr)
                    stats->n_merit_dyn_flushes++;
            }
            if (pressure)
                merit_dyn_trim_pending_buffer();
        }
        else
        {
            for (uint32_t insertion_slot : pending_slots)
            {
                if (merit_dyn_enqueue_ready_pair(insertion_slot, MeritCommitSource::Evict) && stats != nullptr)
                    stats->n_merit_dyn_flushes++;
            }
        }
        {
            const uint32_t slot_id = _merit_mcache.slot_of(node_id);
            if (slot_id != MeritMetadataCache::kInvalid)
            {
                MeritTimedMutexGuard hlock(merit_dyn_heap_shard(slot_id).mu, MeritLockKind::Heap);
                if (merit_dyn_state_unlocked(slot_id) == MeritNodeState::NonSeed)
                    merit_dyn_heap_erase_unlocked(slot_id);
            }
        }
        merit_dyn_request_refresh();
        return;
    }

    {
        MERIT_SECTION(kSecEvictHeap);
        const uint32_t slot_id = _merit_mcache.slot_of(node_id);
        if (slot_id != MeritMetadataCache::kInvalid)
        {
            MeritTimedMutexGuard hlock(merit_dyn_heap_shard(slot_id).mu, MeritLockKind::Heap);
            if (merit_dyn_state_unlocked(slot_id) == MeritNodeState::NonSeed)
                merit_dyn_heap_erase_unlocked(slot_id);
        }
    }
    MERIT_SECTION(kSecEvictRefresh);
    merit_dyn_request_refresh();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_prepare_query(SSDQueryScratch<T> *query_scratch)
{
    if (query_scratch == nullptr)
        return;
    if (!_merit_dcache_net_gate_enabled)
    {
        // Keep D lookups available even when maintenance is frozen/suspended.
        query_scratch->merit_dcache_lookup_enabled = true;
        query_scratch->merit_dcache_maintenance_enabled =
            !_merit_freeze_maintenance && !_merit_mcache.maintenance_suspended();
        return;
    }

    const uint64_t ticket = _merit_dcache_query_issued.fetch_add(1, std::memory_order_relaxed);
    const bool active = _merit_dcache_gate_active.load(std::memory_order_acquire);
    const bool probe = !active && ticket % _merit_dcache_probe_period == 0;
    query_scratch->merit_dcache_probe = probe;
    query_scratch->merit_dcache_lookup_enabled = active || probe;
    query_scratch->merit_dcache_maintenance_enabled =
        !_merit_freeze_maintenance &&
        (probe || (active && !_merit_mcache.maintenance_suspended()));
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_update_dcache_gate(QueryStats *stats,
                                                           SSDQueryScratch<T> *query_scratch)
{
    if (query_scratch == nullptr)
        return;

    uint32_t avoided_pages = 0;
    for (uint64_t sector : query_scratch->merit_avoided_base_sectors)
    {
        if (query_scratch->read_sectors.find(sector) == query_scratch->read_sectors.end())
            ++avoided_pages;
    }
    if (stats != nullptr)
    {
        stats->n_merit_base_pages_avoided = avoided_pages;
        stats->n_merit_dcache_probe = query_scratch->merit_dcache_probe ? 1u : 0u;
    }
    if (!_merit_dcache_net_gate_enabled)
        return;

    if (query_scratch->merit_dcache_lookup_enabled)
    {
        _merit_dcache_enabled_queries.fetch_add(1, std::memory_order_relaxed);
        _merit_dcache_avoided_pages.fetch_add(avoided_pages, std::memory_order_relaxed);
        _merit_dcache_physical_reads.fetch_add(query_scratch->merit_dyn_physical_reads,
                                                std::memory_order_relaxed);
        _merit_dcache_served_nodes.fetch_add(query_scratch->merit_dyn_served_nodes,
                                              std::memory_order_relaxed);
    }

    constexpr uint64_t stage_queries = 1000;
    const uint64_t completed =
        _merit_dcache_query_completed.fetch_add(1, std::memory_order_relaxed) + 1;
    if (completed % stage_queries != 0)
        return;

    std::lock_guard<std::mutex> gate_lock(_merit_dcache_gate_mu);
    bool active = _merit_dcache_gate_active.load(std::memory_order_relaxed);
    if (!active && ++_merit_dcache_inactive_stages < 10)
        return;
    _merit_dcache_inactive_stages = 0;

    const uint64_t enabled_total = _merit_dcache_enabled_queries.load(std::memory_order_relaxed);
    const uint64_t avoided_total = _merit_dcache_avoided_pages.load(std::memory_order_relaxed);
    const uint64_t reads_total = _merit_dcache_physical_reads.load(std::memory_order_relaxed);
    const uint64_t served_total = _merit_dcache_served_nodes.load(std::memory_order_relaxed);
    const uint64_t enabled_delta = enabled_total - _merit_dcache_last_enabled_queries;
    const uint64_t avoided_delta = avoided_total - _merit_dcache_last_avoided_pages;
    const uint64_t reads_delta = reads_total - _merit_dcache_last_physical_reads;
    const uint64_t served_delta = served_total - _merit_dcache_last_served_nodes;
    _merit_dcache_last_enabled_queries = enabled_total;
    _merit_dcache_last_avoided_pages = avoided_total;
    _merit_dcache_last_physical_reads = reads_total;
    _merit_dcache_last_served_nodes = served_total;
    const uint64_t writes_total = _merit_dcache_page_writes.load(std::memory_order_relaxed);
    const uint64_t writes_delta = writes_total - _merit_dcache_last_page_writes;
    _merit_dcache_last_page_writes = writes_total;
    if (enabled_delta == 0)
        return;

    const double net_per_query =
        (static_cast<double>(avoided_delta) - static_cast<double>(reads_delta) -
         _merit_dcache_net_write_weight * static_cast<double>(writes_delta)) /
        static_cast<double>(enabled_delta);
    const double writes_per_query =
        static_cast<double>(writes_delta) / static_cast<double>(enabled_delta);
    const double avoided_per_query =
        static_cast<double>(avoided_delta) / static_cast<double>(enabled_delta);
    const double reads_per_query =
        static_cast<double>(reads_delta) / static_cast<double>(enabled_delta);
    const double served_per_query =
        static_cast<double>(served_delta) / static_cast<double>(enabled_delta);
    const double write_multiplier =
        _merit_dcache_net_write_weight > 0.0 ? _merit_dcache_net_write_weight : 1.0;
    const double net_us_per_query =
        avoided_per_query * _merit_dcache_base_read_us -
        reads_per_query * _merit_dcache_read_us -
        writes_per_query * _merit_dcache_write_us * write_multiplier -
        _merit_dcache_cpu_us;

    bool transition = false;
    if (active)
    {
        _merit_dcache_low_windows =
            net_us_per_query < _merit_dcache_gate_off_us ? _merit_dcache_low_windows + 1 : 0;
        if (_merit_dcache_low_windows >= 5)
        {
            active = false;
            transition = true;
        }
    }
    else
    {
        const bool profitable = net_us_per_query >= _merit_dcache_gate_on_us;
        _merit_dcache_high_windows = profitable ? _merit_dcache_high_windows + 1 : 0;
        if (_merit_dcache_high_windows >= 2)
        {
            active = true;
            transition = true;
        }
    }
    if (!transition)
        return;

    _merit_dcache_gate_active.store(active, std::memory_order_release);
    _merit_dcache_gate_transitions.fetch_add(1, std::memory_order_relaxed);
    _merit_dcache_low_windows = 0;
    _merit_dcache_high_windows = 0;
    _merit_dcache_inactive_stages = 0;
    std::cout << "MERIT dcache net gate: queries=" << completed
              << " active=" << (active ? "off->on" : "on->off")
              << " sampled_queries=" << enabled_delta
              << " avoided_pages/query=" << avoided_per_query
              << " dcache_reads/query=" << reads_per_query
              << " page_writes/query=" << writes_per_query
              << " net_pages/query=" << net_per_query
              << " net_us/query=" << net_us_per_query
              << " model_us=" << _merit_dcache_base_read_us << "/"
              << _merit_dcache_read_us << "/" << _merit_dcache_write_us << "/"
              << _merit_dcache_cpu_us
              << " served_nodes/query=" << served_per_query << std::endl;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_log_monitor(uint32_t query_epoch) const
{
    if (_merit_monitor_warmup_queries == 0 || query_epoch < _merit_monitor_warmup_queries ||
        (query_epoch - _merit_monitor_warmup_queries) % _merit_monitor_interval_queries != 0)
        return;

    uint64_t pages = 0;
    uint64_t members = 0;
    uint64_t full_pages = 0;
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
        pages = _merit_dyn_committed_pages;
        const uint16_t page_capacity = static_cast<uint16_t>(
            (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
        for (uint16_t count : _merit_dyn_page_member_count)
        {
            if (count == 0)
                continue;
            members += count;
            if (count >= page_capacity)
                ++full_pages;
        }
    }
    uint64_t pending = 0;
    uint64_t pending_full = 0;
    uint64_t writes = 0;
    {
        MeritTimedSharedMutexGuard plock(_merit_pair_mu, MeritLockKind::Pair, false);
        pending = _merit_ready_pairs.size();
        const size_t page_capacity = static_cast<size_t>(
            (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
        for (const auto &entry : _merit_ready_pairs)
            if (entry.second.snap.member_ids.size() >= page_capacity)
                ++pending_full;
        writes = _merit_heap_writes;
    }
    uint64_t buffer_bytes = 0;
    uint64_t buffer_entries = 0;
    {
        std::lock_guard<std::mutex> slock(_merit_stash_mu);
        buffer_bytes = _merit_pending_buffer_payload_bytes;
        buffer_entries = _merit_payload_stash.size();
    }
    diskann::cout << "MERIT page monitor: query=" << query_epoch
                  << " measured=" << query_epoch - _merit_monitor_warmup_queries
                  << " pages=" << pages
                  << " members=" << members
                  << " avg_members=" << (pages == 0 ? 0.0 : static_cast<double>(members) / pages)
                  << " full_pages=" << full_pages
                  << " pending=" << pending
                  << " pending_full=" << pending_full
                  << " buffer_entries=" << buffer_entries
                  << " buffer_bytes=" << buffer_bytes
                  << " member_refreshes="
                  << _merit_pending_member_refreshes.load(std::memory_order_relaxed)
                  << " became_full=" << _merit_pending_became_full.load(std::memory_order_relaxed)
                  << " writes=" << writes << std::endl;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_record_real_io_coaccess(SSDQueryScratch<T> *query_scratch)
{
    if (!_merit_real_io_coaccess || query_scratch == nullptr ||
        !query_scratch->merit_dcache_maintenance_enabled)
        return;

    const uint64_t query_number = _merit_real_io_queries.fetch_add(1, std::memory_order_relaxed);
    const std::vector<uint32_t> &nodes = query_scratch->merit_storage_node_order;
    if (nodes.size() < 2 || query_number % _merit_real_io_sample_queries != 0)
        return;

    _merit_real_io_sampled_queries.fetch_add(1, std::memory_order_relaxed);
    _merit_real_io_nodes.fetch_add(nodes.size(), std::memory_order_relaxed);
    uint64_t recorded_pairs = 0;
    bool touched = false;
    const float unit = _merit_score_unit.load(std::memory_order_relaxed);
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        const size_t end = std::min(nodes.size(), i + static_cast<size_t>(_merit_real_io_window) + 1);
        for (size_t j = i + 1; j < end; ++j)
        {
            if (nodes[i] == nodes[j])
                continue;
            const auto forward =
                _merit_mcache.on_real_io_edge(nodes[i], nodes[j], _merit_real_io_max_partners, unit);
            if (forward.present)
            {
                merit_dyn_note_touch(forward, query_scratch);
                touched = true;
            }
            const auto reverse =
                _merit_mcache.on_real_io_edge(nodes[j], nodes[i], _merit_real_io_max_partners, unit);
            if (reverse.present)
            {
                merit_dyn_note_touch(reverse, query_scratch);
                touched = true;
            }
            recorded_pairs += 2;
        }
    }
    _merit_real_io_pairs.fetch_add(recorded_pairs, std::memory_order_relaxed);
    if (touched)
        merit_mark_refresh_needed();
}

template <typename T, typename LabelT>
std::vector<uint32_t> PQFlashIndex<T, LabelT>::merit_dyn_assemble_patch_members(
    uint32_t seed_id, const std::vector<uint32_t> &new_ids, const std::vector<uint32_t> &prior_members,
    const std::vector<std::vector<uint32_t>> &prior_extra, std::vector<std::vector<uint32_t>> *extra_out) const
{
    if (extra_out != nullptr)
        extra_out->clear();
    const size_t page_capacity = static_cast<size_t>((_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
    if (!_merit_dcache_second_page || _merit_dcache_max_pages <= 1)
    {
        std::vector<uint32_t> now;
        now.reserve(page_capacity);
        tsl::robin_set<uint32_t> used;
        auto try_add = [&](uint32_t id) {
            if (now.size() >= page_capacity || !used.insert(id).second)
                return;
            now.push_back(id);
        };
        try_add(seed_id);
        for (uint32_t id : new_ids)
            try_add(id);
        for (uint32_t id : prior_members)
            try_add(id);
        if (now.size() < page_capacity)
        {
            MeritMetadataCache::Snapshot snap;
            if (_merit_mcache.snapshot(seed_id, snap))
            {
                const std::vector<uint32_t> filled =
                    merit_dyn_now_page_ids(seed_id, snap, _merit_deferred_ncache_page_write);
                for (uint32_t id : filled)
                    try_add(id);
            }
        }
        return now;
    }

    const uint32_t max_pages = _merit_dcache_max_pages;
    std::vector<std::vector<uint32_t>> pages(1);
    tsl::robin_set<uint32_t> seen;
    auto add_to = [&](std::vector<uint32_t> &dst, uint32_t id) {
        if (dst.size() >= page_capacity || id == MERIT_DYN_INVALID_PAGE)
            return false;
        if (id == seed_id)
        {
            if (!dst.empty())
                return false;
            dst.push_back(id);
            return true;
        }
        if (!seen.insert(id).second)
            return false;
        dst.push_back(id);
        return true;
    };
    add_to(pages[0], seed_id);
    for (uint32_t id : prior_members)
    {
        if (id != seed_id)
            add_to(pages[0], id);
    }
    std::vector<uint32_t> rest;
    for (uint32_t id : new_ids)
    {
        if (id == seed_id)
            continue;
        if (!add_to(pages[0], id) && seen.find(id) == seen.end())
            rest.push_back(id);
    }
    if (pages[0].size() < page_capacity)
    {
        MeritMetadataCache::Snapshot snap;
        if (_merit_mcache.snapshot(seed_id, snap))
        {
            const std::vector<uint32_t> filled =
                merit_dyn_now_page_ids(seed_id, snap, _merit_deferred_ncache_page_write);
            for (uint32_t id : filled)
                add_to(pages[0], id);
        }
    }
    size_t prior_i = 0;
    while (pages.size() < max_pages)
    {
        const std::vector<uint32_t> *old_page =
            prior_i < prior_extra.size() ? &prior_extra[prior_i] : nullptr;
        bool need = !rest.empty();
        if (!need && old_page != nullptr)
        {
            for (uint32_t id : *old_page)
            {
                if (id != seed_id && seen.find(id) == seen.end())
                {
                    need = true;
                    break;
                }
            }
        }
        if (!need)
            break;
        pages.emplace_back();
        add_to(pages.back(), seed_id);
        if (old_page != nullptr)
        {
            for (uint32_t id : *old_page)
            {
                if (id != seed_id)
                    add_to(pages.back(), id);
            }
        }
        std::vector<uint32_t> still;
        for (uint32_t id : rest)
        {
            if (!add_to(pages.back(), id) && seen.find(id) == seen.end())
                still.push_back(id);
        }
        rest.swap(still);
        ++prior_i;
        if (pages.back().size() < 2)
        {
            pages.pop_back();
            break;
        }
    }
    if (extra_out != nullptr)
    {
        for (size_t i = 1; i < pages.size(); ++i)
        {
            if (pages[i].size() >= 2)
                extra_out->push_back(std::move(pages[i]));
        }
    }
    return pages[0];
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_patch_pin(const std::vector<uint32_t> &member_ids)
{
    std::lock_guard<std::mutex> slock(_merit_stash_mu);
    for (uint32_t id : member_ids)
        _merit_patch_pin_count[id] += 1;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_patch_unpin(const std::vector<uint32_t> &member_ids)
{
    std::lock_guard<std::mutex> slock(_merit_stash_mu);
    for (uint32_t id : member_ids)
    {
        auto it = _merit_patch_pin_count.find(id);
        if (it == _merit_patch_pin_count.end())
            continue;
        if (it.value() <= 1)
            _merit_patch_pin_count.erase(it);
        else
            it.value() -= 1;
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::merit_dyn_patch_loop()
{
    merit_pin_background_thread();
    if (_merit_dyn_reader != nullptr)
        _merit_dyn_reader->register_thread();
    while (true)
    {
        std::shared_ptr<MeritPatchJob> job;
        {
            std::unique_lock<std::mutex> lock(_merit_patch_q_mu);
            _merit_patch_q_cv.wait(lock, [this] {
                return _merit_patch_stop.load(std::memory_order_acquire) || !_merit_patch_q.empty();
            });
            if (_merit_patch_stop.load(std::memory_order_acquire) && _merit_patch_q.empty())
                return;
            if (_merit_patch_batch_us > 0 && !_merit_patch_stop.load(std::memory_order_acquire))
            {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(_merit_patch_batch_us));
                lock.lock();
                if (_merit_patch_stop.load(std::memory_order_acquire) && _merit_patch_q.empty())
                    return;
                if (_merit_patch_q.empty())
                    continue;
            }
            job = std::move(_merit_patch_q.front());
            _merit_patch_q.pop_front();
        }
        _merit_patch_q_cv.notify_all();
        if (job == nullptr)
            continue;
        if (_merit_freeze_maintenance || merit_dcache_updates_frozen())
        {
            merit_dyn_patch_unpin(job->new_ids);
            merit_dyn_patch_unpin(job->prior_members);
            for (const auto &page : job->prior_extra)
                merit_dyn_patch_unpin(page);
            continue;
        }

        std::vector<std::vector<uint32_t>> extra_pages;
        std::vector<uint32_t> now = merit_dyn_assemble_patch_members(
            job->seed_id, job->new_ids, job->prior_members, job->prior_extra, &extra_pages);
        std::vector<uint32_t> extra_pins;
        {
            tsl::robin_set<uint32_t> known;
            known.insert(job->seed_id);
            for (uint32_t id : job->new_ids)
                known.insert(id);
            for (uint32_t id : job->prior_members)
                known.insert(id);
            for (const auto &page : job->prior_extra)
                for (uint32_t id : page)
                    known.insert(id);
            for (uint32_t id : now)
            {
                if (known.find(id) == known.end())
                    extra_pins.push_back(id);
            }
            for (const auto &page : extra_pages)
                for (uint32_t id : page)
                    if (known.find(id) == known.end())
                        extra_pins.push_back(id);
            if (!extra_pins.empty())
                merit_dyn_patch_pin(extra_pins);
        }
        std::vector<MeritPendingFlush> flushes(1 + extra_pages.size());
        flushes[0].seed_id = job->seed_id;
        flushes[0].member_ids = std::move(now);
        for (size_t i = 0; i < extra_pages.size(); ++i)
        {
            flushes[i + 1].seed_id = job->seed_id;
            flushes[i + 1].member_ids = std::move(extra_pages[i]);
        }
        const bool allow_disk = !_merit_patch_worker_no_disk;
        const bool use_span = flushes.size() > 1;
        bool ok = !flushes[0].member_ids.empty() && flushes[0].member_ids.front() == job->seed_id &&
                  flushes[0].member_ids.size() >= 2;
        if (ok && !use_span && !job->prior_members.empty() && flushes[0].member_ids == job->prior_members)
            ok = false;
        if (ok && use_span && flushes[0].member_ids == job->prior_members)
        {
            bool same = flushes.size() == job->prior_extra.size() + 1;
            for (size_t i = 1; same && i < flushes.size(); ++i)
                same = flushes[i].member_ids == job->prior_extra[i - 1];
            if (same)
                ok = false;
        }
        auto copy_payloads = [&](MeritPendingFlush &dst) {
            dst.member_coords.reserve(dst.member_ids.size());
            dst.member_nbrs.reserve(dst.member_ids.size());
            for (uint32_t mid : dst.member_ids)
            {
                std::vector<T> coords;
                std::vector<uint32_t> nbrs;
                if (!merit_dyn_copy_member_payload(mid, coords, nbrs, allow_disk))
                    return false;
                dst.member_coords.push_back(std::move(coords));
                dst.member_nbrs.push_back(std::move(nbrs));
            }
            return true;
        };
        if (ok)
        {
            for (MeritPendingFlush &dst : flushes)
            {
                if (!copy_payloads(dst))
                {
                    ok = false;
                    break;
                }
            }
        }
        const bool committed = ok && (use_span ? merit_dyn_commit_span(flushes, job->replacement_seed)
                                               : merit_dyn_commit_one(flushes[0], job->replacement_seed));
        if (committed)
        {
            _merit_dcache_query_patch_ok.fetch_add(1, std::memory_order_relaxed);
            g_dyn_probe.write_ok.fetch_add(1, std::memory_order_relaxed);
        }
        merit_dyn_patch_unpin(job->new_ids);
        merit_dyn_patch_unpin(job->prior_members);
        for (const auto &page : job->prior_extra)
            merit_dyn_patch_unpin(page);
        if (!extra_pins.empty())
            merit_dyn_patch_unpin(extra_pins);
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::start_merit_dyn_patch_worker()
{
    if (_merit_patch_started || !_merit_dcache_query_patch)
        return;
    _merit_patch_stop.store(false, std::memory_order_release);
    _merit_patch_thread = std::thread([this] { merit_dyn_patch_loop(); });
    _merit_patch_started = true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::stop_merit_dyn_patch_worker()
{
    if (!_merit_patch_started)
        return;
    {
        std::lock_guard<std::mutex> lock(_merit_patch_q_mu);
        _merit_patch_stop.store(true, std::memory_order_release);
    }
    _merit_patch_q_cv.notify_all();
    if (_merit_patch_thread.joinable())
        _merit_patch_thread.join();
    _merit_patch_started = false;
    std::deque<std::shared_ptr<MeritPatchJob>> leftover;
    {
        std::lock_guard<std::mutex> lock(_merit_patch_q_mu);
        leftover.swap(_merit_patch_q);
    }
    for (auto &job : leftover)
    {
        if (job != nullptr)
        {
            merit_dyn_patch_unpin(job->new_ids);
            merit_dyn_patch_unpin(job->prior_members);
            for (const auto &page : job->prior_extra)
                merit_dyn_patch_unpin(page);
        }
    }
    if (!leftover.empty())
        _merit_dcache_query_patch_drop.fetch_add(leftover.size(), std::memory_order_relaxed);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_enqueue_patch(uint32_t seed_id, std::vector<uint32_t> new_ids,
                                                      std::vector<uint32_t> prior_members,
                                                      std::vector<std::vector<uint32_t>> prior_extra,
                                                      uint32_t replacement_seed)
{
    if (!_merit_patch_started || _merit_patch_stop.load(std::memory_order_acquire) ||
        _merit_freeze_maintenance || new_ids.empty())
        return false;
    auto job = std::make_shared<MeritPatchJob>();
    job->seed_id = seed_id;
    job->new_ids = std::move(new_ids);
    job->prior_members = std::move(prior_members);
    job->prior_extra = std::move(prior_extra);
    job->replacement_seed = replacement_seed;
    merit_dyn_patch_pin(job->new_ids);
    merit_dyn_patch_pin(job->prior_members);
    for (const auto &page : job->prior_extra)
        merit_dyn_patch_pin(page);
    {
        std::unique_lock<std::mutex> lock(_merit_patch_q_mu);
        for (auto it = _merit_patch_q.begin(); it != _merit_patch_q.end(); ++it)
        {
            if (*it != nullptr && (*it)->seed_id == job->seed_id)
            {
                auto old = *it;
                *it = job;
                lock.unlock();
                if (old != nullptr)
                {
                    merit_dyn_patch_unpin(old->new_ids);
                    merit_dyn_patch_unpin(old->prior_members);
                    for (const auto &page : old->prior_extra)
                        merit_dyn_patch_unpin(page);
                }
                _merit_patch_q_cv.notify_all();
                return true;
            }
        }
        if (_merit_patch_q.size() >= _merit_patch_q_cap)
        {
            lock.unlock();
            merit_dyn_patch_unpin(job->new_ids);
            merit_dyn_patch_unpin(job->prior_members);
            for (const auto &page : job->prior_extra)
                merit_dyn_patch_unpin(page);
            _merit_dcache_query_patch_drop.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        _merit_patch_q.push_back(job);
    }
    _merit_patch_q_cv.notify_all();
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_note_patch_candidate(uint32_t node_id, uint32_t parent,
                                                             SSDQueryScratch<T> *query_scratch)
{
    if (!_merit_dcache_query_patch || query_scratch == nullptr || parent == std::numeric_limits<uint32_t>::max())
        return false;
    const auto seed_it = query_scratch->merit_dcache_member_to_seed.find(parent);
    if (seed_it == query_scratch->merit_dcache_member_to_seed.end())
        return false;
    const uint32_t seed_id = seed_it->second;
    if (query_scratch->merit_dcache_member_to_seed.find(node_id) != query_scratch->merit_dcache_member_to_seed.end())
        return false;
    const auto mem_it = query_scratch->merit_dcache_seed_members.find(seed_id);
    if (mem_it != query_scratch->merit_dcache_seed_members.end())
    {
        for (uint32_t mid : mem_it->second)
            if (mid == node_id)
                return false;
    }
    const auto extra_it = query_scratch->merit_dcache_seed_extra.find(seed_id);
    if (extra_it != query_scratch->merit_dcache_seed_extra.end())
    {
        for (const auto &page : extra_it->second)
            for (uint32_t mid : page)
                if (mid == node_id)
                    return false;
    }
    auto &bucket = query_scratch->merit_dcache_patch_new[seed_id];
    if (std::find(bucket.begin(), bucket.end(), node_id) != bucket.end())
        return false;
    bucket.push_back(node_id);
    return true;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_apply_query_patches(QueryStats *stats, SSDQueryScratch<T> *query_scratch)
{
    if (!_merit_dcache_query_patch || query_scratch == nullptr)
        return;
    uint32_t applied = 0;
    tsl::robin_set<uint32_t> enqueued_ids;
    for (const auto &kv : query_scratch->merit_dcache_patch_new)
    {
        if (applied >= _merit_dcache_query_patch_max)
            break;
        const uint32_t seed_id = kv.first;
        const auto &new_ids = kv.second;
        if (new_ids.empty())
            continue;
        const auto mem_it = query_scratch->merit_dcache_seed_members.find(seed_id);
        if (mem_it == query_scratch->merit_dcache_seed_members.end() || mem_it->second.empty())
            continue;
        std::vector<std::vector<uint32_t>> prior_extra;
        const auto extra_it = query_scratch->merit_dcache_seed_extra.find(seed_id);
        if (extra_it != query_scratch->merit_dcache_seed_extra.end())
            prior_extra = extra_it->second;
        _merit_dcache_query_patch_trig.fetch_add(1, std::memory_order_relaxed);
        if (_merit_patch_started)
        {
            if (merit_dyn_enqueue_patch(seed_id, new_ids, mem_it->second, prior_extra, seed_id))
            {
                enqueued_ids.insert(new_ids.begin(), new_ids.end());
                ++applied;
                if (stats != nullptr)
                    stats->n_merit_dyn_flushes++;
            }
            continue;
        }
        std::vector<std::vector<uint32_t>> extra_pages;
        std::vector<uint32_t> now =
            merit_dyn_assemble_patch_members(seed_id, new_ids, mem_it->second, prior_extra, &extra_pages);
        if (now.size() < 2 || now.front() != seed_id)
            continue;
        if (extra_pages.empty() && now == mem_it->second)
            continue;
        std::vector<MeritPendingFlush> flushes(1 + extra_pages.size());
        flushes[0].seed_id = seed_id;
        flushes[0].member_ids = now;
        bool ok = true;
        auto copy_one = [&](MeritPendingFlush &dst) {
            for (uint32_t mid : dst.member_ids)
            {
                std::vector<T> coords;
                std::vector<uint32_t> nbrs;
                if (!merit_dyn_copy_member_payload(mid, coords, nbrs))
                    return false;
                dst.member_coords.push_back(std::move(coords));
                dst.member_nbrs.push_back(std::move(nbrs));
            }
            return true;
        };
        for (size_t i = 0; i < extra_pages.size(); ++i)
        {
            flushes[i + 1].seed_id = seed_id;
            flushes[i + 1].member_ids = extra_pages[i];
        }
        for (MeritPendingFlush &dst : flushes)
        {
            if (!copy_one(dst))
            {
                ok = false;
                break;
            }
        }
        const bool committed = ok && (flushes.size() > 1 ? merit_dyn_commit_span(flushes, seed_id)
                                                          : merit_dyn_commit_one(flushes[0], seed_id));
        if (!committed)
            continue;
        _merit_dcache_query_patch_ok.fetch_add(1, std::memory_order_relaxed);
        g_dyn_probe.write_ok.fetch_add(1, std::memory_order_relaxed);
        ++applied;
        if (stats != nullptr)
            stats->n_merit_dyn_flushes++;
    }
    std::vector<uint32_t> release;
    for (const auto &kv : query_scratch->merit_dcache_patch_new)
        for (uint32_t id : kv.second)
            if (enqueued_ids.find(id) == enqueued_ids.end())
                release.push_back(id);
    if (!release.empty())
        merit_dyn_patch_unpin(release);
    query_scratch->merit_dcache_patch_new.clear();
    query_scratch->merit_dcache_seed_members.clear();
    query_scratch->merit_dcache_seed_extra.clear();
    query_scratch->merit_dcache_member_to_seed.clear();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_dyn_on_query_end(QueryStats *stats, SSDQueryScratch<T> *query_scratch)
{
    if (_merit_dcache_query_patch && !merit_dcache_updates_frozen())
    {
        MERIT_SECTION(kSecQendPatch);
        merit_dyn_apply_query_patches(stats, query_scratch);
    }
    {
        MERIT_SECTION(kSecQendRealIo);
        merit_dyn_record_real_io_coaccess(query_scratch);
    }
    _merit_mcache.on_query_end();
    {
        MERIT_SECTION(kSecQendHeap);
        if (query_scratch != nullptr && query_scratch->merit_dcache_maintenance_enabled)
            merit_dyn_flush_dirty_heap(query_scratch);
        else if (query_scratch != nullptr)
            query_scratch->clear_merit_heap_dirty();
    }
    MERIT_SECTION(kSecQendGate);
    merit_dyn_update_dcache_gate(stats, query_scratch);
    if (query_scratch == nullptr || !query_scratch->merit_dcache_maintenance_enabled)
        return;
    const uint32_t query_epoch = _merit_query_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (query_epoch % 100000 == 0)
    {
        uint64_t snap_pages = 0, snap_members = 0, extra_pages = 0, extra_neighbors = 0;
        uint64_t span_hist[6] = {};
        {
            MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
            for (size_t page = 0; page < _merit_dyn_page_member_count.size(); ++page)
            {
                const uint16_t count = _merit_dyn_page_member_count[page];
                if (count == 0)
                    continue;
                if (page < _merit_dyn_page_secondary.size() && _merit_dyn_page_secondary[page])
                {
                    ++extra_pages;
                    if (count > 1)
                        extra_neighbors += static_cast<uint64_t>(count - 1);
                    continue;
                }
                ++snap_pages;
                snap_members += count;
                uint32_t n = 1;
                if (page < _merit_dyn_page_span_len.size() && _merit_dyn_page_span_len[page] > 1)
                    n = _merit_dyn_page_span_len[page];
                if (n > 5)
                    n = 5;
                ++span_hist[n];
            }
        }
        const double snap_avg =
            snap_pages == 0 ? 0.0 : static_cast<double>(snap_members) / static_cast<double>(snap_pages);
        diskann::cout << "MERIT dcache_fill_snap query=" << query_epoch << " pages=" << snap_pages
                      << " avg_members=" << snap_avg << " avg_neighbors=" << (snap_avg - 1.0)
                      << " extra_pages=" << extra_pages << " extra_neighbors=" << extra_neighbors
                      << " p1=" << span_hist[1] << " p2=" << span_hist[2] << " p3=" << span_hist[3]
                      << " p4=" << span_hist[4] << " p5=" << span_hist[5] << std::endl;
    }
    if (_merit_member_snapshot_at > 0 && query_epoch == _merit_member_snapshot_at &&
        !_merit_member_snapshot_done.load(std::memory_order_relaxed))
    {
        std::vector<uint32_t> ids;
        {
            MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
            for (size_t page = 0; page < _merit_dyn_page_member_count.size(); ++page)
            {
                const uint16_t count = _merit_dyn_page_member_count[page];
                if (count <= 1)
                    continue;
                const size_t begin = page * _merit_dyn_members_per_page;
                if (begin + count > _merit_dyn_page_members.size())
                    continue;
                for (size_t mi = 1; mi < count; ++mi)
                {
                    const uint32_t mid = _merit_dyn_page_members[begin + mi];
                    if (mid != MERIT_DYN_INVALID_PAGE)
                        ids.push_back(mid);
                }
            }
        }
        _merit_member_snapshot_slots.store(ids.size(), std::memory_order_relaxed);
        for (uint32_t id : ids)
            _merit_member_at_snapshot.insert(id);
        _merit_member_snapshot_done.store(true, std::memory_order_release);
        diskann::cout << "MERIT member_snapshot query=" << query_epoch << " neighbor_slots=" << ids.size()
                      << " neighbor_ids=" << _merit_member_at_snapshot.size() << std::endl;
    }
    if (query_epoch % _merit_pending_check_interval == 0)
        merit_dyn_commit_expired_pairs(query_epoch, stats);
    merit_dyn_log_monitor(query_epoch);
    uint64_t page_count = 0;
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
        page_count = _merit_dyn_committed_pages;
    }
    if (stats != nullptr)
        stats->n_merit_dyn_pages = static_cast<unsigned>(
            std::min<uint64_t>(page_count, static_cast<uint64_t>(std::numeric_limits<unsigned>::max())));
    merit_dyn_request_refresh();
    const bool disk_full = page_count >= _merit_dyn_page_cap;
    if (!_merit_decay_started.load(std::memory_order_acquire))
    {
        if (!disk_full)
            return;
        bool expected = false;
        _merit_decay_started.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
    }

    _merit_decay_query_count.fetch_add(1, std::memory_order_relaxed);
    const uint64_t accesses = query_scratch->merit_expand_count;
    const uint64_t now = _merit_decay_access_count.fetch_add(accesses, std::memory_order_acq_rel) + accesses;
    const uint64_t elapsed = now - std::min(now, _merit_decay_cycle_start.load(std::memory_order_acquire));
    const float unit = static_cast<float>(std::exp2(static_cast<double>(elapsed) / _merit_decay_half_life_accesses));
    const bool saturated = _merit_mcache.edge_saturation_pending();
    if (unit < MERIT_SCORE_UNIT_MAX && !(saturated && unit >= 2.0f))
    {
        _merit_score_unit.store(unit, std::memory_order_release);
        return;
    }

    std::unique_lock<std::mutex> decay_lock(_merit_decay_mu, std::try_to_lock);
    if (!decay_lock.owns_lock())
        return;
    const uint64_t cycle_start = _merit_decay_cycle_start.load(std::memory_order_acquire);
    if (cycle_start > now)
        return;
    const float divisor = std::min(
        MERIT_SCORE_UNIT_MAX,
        static_cast<float>(std::exp2(static_cast<double>(now - cycle_start) / _merit_decay_half_life_accesses)));
    if (divisor < 2.0f)
        return;
    const float factor = 1.0f / divisor;
    std::vector<std::unique_ptr<MeritTimedMutexGuard>> heap_locks;
    heap_locks.reserve(_merit_heap_shard_count);
    for (size_t shard_index = 0; shard_index < _merit_heap_shard_count; ++shard_index)
        heap_locks.emplace_back(
            std::make_unique<MeritTimedMutexGuard>(_merit_heap_shards[shard_index].mu, MeritLockKind::Heap));
    _merit_mcache.scale_scores(factor, _merit_score_unit, 1.0f);
    for (size_t shard_index = 0; shard_index < _merit_heap_shard_count; ++shard_index)
    {
        MeritHeapShard &shard = _merit_heap_shards[shard_index];
        for (MeritHeapEntry &entry : shard.max_heap)
            entry.score *= factor;
        for (MeritHeapEntry &entry : shard.min_heap)
            entry.score *= factor;
    }
    _merit_decay_cycle_start.store(now, std::memory_order_release);
    _merit_decay_cycles.fetch_add(1, std::memory_order_relaxed);
    if (divisor < MERIT_SCORE_UNIT_MAX)
        _merit_decay_early_cycles.fetch_add(1, std::memory_order_relaxed);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_commit_span(std::vector<MeritPendingFlush> &pages,
                                                    uint32_t replacement_seed)
{
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    if (pages.size() < 2 || pages.size() > _merit_dcache_max_pages)
        return false;
    const uint32_t seed_id = pages[0].seed_id;
    for (MeritPendingFlush &pf : pages)
    {
        if (pf.member_ids.size() < 2 || pf.member_ids.front() != seed_id)
            return false;
        if (pf.member_ids.size() > nps)
        {
            pf.member_ids.resize(nps);
            pf.member_coords.resize(std::min(pf.member_coords.size(), nps));
            pf.member_nbrs.resize(std::min(pf.member_nbrs.size(), nps));
        }
    }
    const uint8_t span = static_cast<uint8_t>(pages.size());
    auto fill_buf = [&](const MeritPendingFlush &pf, std::vector<char> &buf) -> bool {
        if (pf.member_coords.size() < pf.member_ids.size() || pf.member_nbrs.size() < pf.member_ids.size())
            return false;
        buf.assign(defaults::SECTOR_LEN, 0);
        for (size_t mi = 0; mi < pf.member_ids.size(); ++mi)
        {
            const auto &coords = pf.member_coords[mi];
            const auto &nbrs = pf.member_nbrs[mi];
            const uint32_t nnbrs = static_cast<uint32_t>(nbrs.size());
            if (_disk_bytes_per_point > coords.size() * sizeof(T) || nnbrs > _max_degree)
                return false;
            char *slot_dst = buf.data() + static_cast<uint64_t>(mi) * _max_node_len;
            memcpy(slot_dst, coords.data(), _disk_bytes_per_point);
            uint32_t *nhood = reinterpret_cast<uint32_t *>(slot_dst + _disk_bytes_per_point);
            nhood[0] = nnbrs;
            if (nnbrs > 0)
                memcpy(nhood + 1, nbrs.data(), nnbrs * sizeof(uint32_t));
            if (mi == 0)
                write_merit_dc_seed_tail(slot_dst, _max_node_len, _disk_bytes_per_point, nnbrs, span);
        }
        return true;
    };
    std::vector<std::vector<char>> bufs(pages.size());
    for (size_t i = 0; i < pages.size(); ++i)
        if (!fill_buf(pages[i], bufs[i]))
            return false;

    uint32_t base = MERIT_DYN_INVALID_PAGE;
    uint32_t expected_victim_page = MERIT_DYN_INVALID_PAGE;
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicUnique, true);
        if (replacement_seed != MERIT_DYN_INVALID_PAGE)
        {
            if (!merit_dyn_seed_page_unlocked(replacement_seed, expected_victim_page))
                return false;
        }
        else if (_merit_dyn_committed_pages >= _merit_dyn_page_cap)
            return false;
        base = merit_dyn_alloc_contiguous_unlocked(span);
    }
    if (base == MERIT_DYN_INVALID_PAGE || static_cast<uint64_t>(base) + span > _merit_dyn_physical_cap)
        return false;
    auto write_page = [&](uint32_t page_idx, const std::vector<char> &buf) {
#ifndef _WINDOWS
        char *aligned_page = nullptr;
        alloc_aligned(reinterpret_cast<void **>(&aligned_page), defaults::SECTOR_LEN, defaults::SECTOR_LEN);
        memcpy(aligned_page, buf.data(), defaults::SECTOR_LEN);
        const bool write_ok = merit_dyn_writer_submit_and_wait(
            MeritDynWriteOp::Write, static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN),
            aligned_page, defaults::SECTOR_LEN);
        aligned_free(aligned_page);
        return write_ok;
#else
        return merit_dyn_writer_submit_and_wait(
            MeritDynWriteOp::Write, static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN), buf.data(),
            defaults::SECTOR_LEN);
#endif
    };
    bool write_ok = true;
    {
        MeritTimedRegion commit_io(MeritLockKind::CommitIo);
        for (uint32_t i = 0; i < span; ++i)
            write_ok = write_ok && write_page(base + i, bufs[i]);
    }
    auto free_span = [&]() {
        for (uint32_t i = 0; i < span; ++i)
            _merit_dyn_free.push_back(base + i);
    };
    if (!write_ok)
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicUnique, true);
        free_span();
        return false;
    }
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicUnique, true);
        bool valid = true;
        if (replacement_seed == MERIT_DYN_INVALID_PAGE)
            valid = _merit_dyn_committed_pages < _merit_dyn_page_cap;
        else
        {
            uint32_t victim_page = MERIT_DYN_INVALID_PAGE;
            valid = merit_dyn_seed_page_unlocked(replacement_seed, victim_page) && victim_page == expected_victim_page;
        }
        if (!valid)
        {
            free_span();
            return false;
        }
        uint16_t kept_first = 0;
        bool keep_first = false;
        if (replacement_seed == seed_id)
        {
            auto first_it = _merit_seed_first_fill.find(seed_id);
            if (first_it != _merit_seed_first_fill.end())
            {
                kept_first = first_it->second;
                keep_first = true;
            }
        }
        merit_span_carry_stats(replacement_seed == seed_id ? expected_victim_page : MERIT_DYN_INVALID_PAGE, base);
        if (replacement_seed != MERIT_DYN_INVALID_PAGE)
            merit_dyn_invalidate_seed_page(replacement_seed);
        if (keep_first)
            _merit_seed_first_fill[seed_id] = kept_first;
        else if (_merit_seed_first_fill.find(seed_id) == _merit_seed_first_fill.end())
            _merit_seed_first_fill[seed_id] = static_cast<uint16_t>(pages[0].member_ids.size());

        for (uint32_t i = 0; i < span; ++i)
        {
            const uint32_t page_idx = base + i;
            const MeritPendingFlush &pf = pages[i];
            if (page_idx < _merit_dyn_page_secondary.size())
                _merit_dyn_page_secondary[page_idx] = i == 0 ? 0 : 1;
            if (page_idx < _merit_dyn_page_span_len.size())
                _merit_dyn_page_span_len[page_idx] = span;
            if (page_idx < _merit_dyn_page_span_base.size())
                _merit_dyn_page_span_base[page_idx] = base;
            if (page_idx < _merit_dyn_page_sibling.size())
                _merit_dyn_page_sibling[page_idx] = MERIT_DYN_INVALID_PAGE;
            const size_t member_begin = static_cast<size_t>(page_idx) * _merit_dyn_members_per_page;
            std::copy(pf.member_ids.begin(), pf.member_ids.end(), _merit_dyn_page_members.begin() + member_begin);
            _merit_dyn_page_member_count[page_idx] = static_cast<uint16_t>(pf.member_ids.size());
            // Publish locations last. Acquiring a location shard after this
            // release also publishes the immutable page metadata above.
            const uint32_t sector = MERIT_DYN_SECTOR_BASE + page_idx;
            for (size_t mi = 0; mi < pf.member_ids.size(); ++mi)
            {
                MeritDiskLoc loc;
                loc.sector = sector;
                loc.slot = static_cast<uint16_t>(mi);
                loc.nsectors = 1;
                merit_dyn_loc_add_unlocked(pf.member_ids[mi], loc);
            }
        }
        _merit_dyn_committed_pages++;
        _merit_dyn_flush_count++;
        _merit_dcache_page_writes.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_commit_one(MeritPendingFlush &pf, uint32_t replacement_seed)
{
    if (pf.member_ids.empty())
        return false;
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    if (pf.member_ids.size() > nps)
    {
        pf.member_ids.resize(nps);
        pf.member_coords.resize(nps);
        pf.member_nbrs.resize(nps);
    }

    std::vector<char> buf(defaults::SECTOR_LEN, 0);
    for (size_t mi = 0; mi < pf.member_ids.size(); mi++)
    {
        const uint16_t slot = static_cast<uint16_t>(mi);
        char *slot_dst = buf.data() + static_cast<uint64_t>(slot) * _max_node_len;
        const auto &coords = pf.member_coords[mi];
        const auto &nbrs = pf.member_nbrs[mi];
        const uint32_t nnbrs = static_cast<uint32_t>(nbrs.size());
        if (_disk_bytes_per_point > coords.size() * sizeof(T) || nnbrs > _max_degree)
            return false;
        memcpy(slot_dst, coords.data(), _disk_bytes_per_point);
        uint32_t *nhood = reinterpret_cast<uint32_t *>(slot_dst + _disk_bytes_per_point);
        nhood[0] = nnbrs;
        if (nnbrs > 0)
            memcpy(nhood + 1, nbrs.data(), nnbrs * sizeof(uint32_t));
        if (slot == 0)
            write_merit_dc_seed_tail(slot_dst, _max_node_len, _disk_bytes_per_point, nnbrs, 1);
    }

    uint32_t page_idx = MERIT_DYN_INVALID_PAGE;
    uint32_t expected_victim_page = MERIT_DYN_INVALID_PAGE;
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicUnique, true);
        uint32_t existing_page = MERIT_DYN_INVALID_PAGE;
        if (merit_dyn_seed_page_unlocked(pf.seed_id, existing_page))
        {
            const size_t existing_count = _merit_dyn_page_member_count[existing_page];
            const uint32_t *existing_members =
                _merit_dyn_page_members.data() +
                static_cast<size_t>(existing_page) * _merit_dyn_members_per_page;
            if (existing_count > 0 &&
                !merit_dyn_hotter_mismatch(existing_members, existing_count, pf.member_ids))
                return false;
        }
        if (replacement_seed == MERIT_DYN_INVALID_PAGE && _merit_dyn_committed_pages >= _merit_dyn_page_cap)
            return false;
        if (replacement_seed != MERIT_DYN_INVALID_PAGE)
        {
            if (!merit_dyn_seed_page_unlocked(replacement_seed, expected_victim_page))
                return false;
        }
        page_idx = merit_dyn_alloc_page_unlocked();
    }
    if (page_idx == MERIT_DYN_INVALID_PAGE)
        return false;
    bool write_ok = false;
    {
        MeritTimedRegion commit_io(MeritLockKind::CommitIo);
#ifndef _WINDOWS
        char *aligned_page = nullptr;
        alloc_aligned(reinterpret_cast<void **>(&aligned_page), defaults::SECTOR_LEN, defaults::SECTOR_LEN);
        memcpy(aligned_page, buf.data(), defaults::SECTOR_LEN);
        write_ok = merit_dyn_writer_submit_and_wait(
            MeritDynWriteOp::Write,
            static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN), aligned_page,
            defaults::SECTOR_LEN);
        aligned_free(aligned_page);
#else
        write_ok = merit_dyn_writer_submit_and_wait(
            MeritDynWriteOp::Write,
            static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN), buf.data(),
            defaults::SECTOR_LEN);
#endif
    }
    if (!write_ok)
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicUnique, true);
        _merit_dyn_free.push_back(page_idx);
        return false;
    }
    {
        MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicUnique, true);
        bool valid = true;
        uint32_t existing_page = MERIT_DYN_INVALID_PAGE;
        if (merit_dyn_seed_page_unlocked(pf.seed_id, existing_page))
        {
            const size_t existing_count = _merit_dyn_page_member_count[existing_page];
            const uint32_t *existing_members =
                _merit_dyn_page_members.data() +
                static_cast<size_t>(existing_page) * _merit_dyn_members_per_page;
            if (existing_count > 0 &&
                !merit_dyn_hotter_mismatch(existing_members, existing_count, pf.member_ids))
                valid = false;
        }
        if (replacement_seed == MERIT_DYN_INVALID_PAGE)
            valid = valid && _merit_dyn_committed_pages < _merit_dyn_page_cap;
        else
        {
            uint32_t victim_page = MERIT_DYN_INVALID_PAGE;
            valid = valid && merit_dyn_seed_page_unlocked(replacement_seed, victim_page) &&
                    victim_page == expected_victim_page;
        }
        if (!valid)
        {
            _merit_dyn_free.push_back(page_idx);
            return false;
        }

        // Copy-on-write: publish the new physical page before retiring the old one.
        uint16_t kept_first = 0;
        bool keep_first = false;
        if (replacement_seed == pf.seed_id)
        {
            auto first_it = _merit_seed_first_fill.find(pf.seed_id);
            if (first_it != _merit_seed_first_fill.end())
            {
                kept_first = first_it->second;
                keep_first = true;
            }
        }
        merit_span_carry_stats(replacement_seed == pf.seed_id ? expected_victim_page : MERIT_DYN_INVALID_PAGE,
                               page_idx);
        if (replacement_seed != MERIT_DYN_INVALID_PAGE)
            merit_dyn_invalidate_seed_page(replacement_seed);
        if (keep_first)
            _merit_seed_first_fill[pf.seed_id] = kept_first;
        else if (_merit_seed_first_fill.find(pf.seed_id) == _merit_seed_first_fill.end())
            _merit_seed_first_fill[pf.seed_id] = static_cast<uint16_t>(pf.member_ids.size());

        const size_t member_begin = static_cast<size_t>(page_idx) * _merit_dyn_members_per_page;
        std::copy(pf.member_ids.begin(), pf.member_ids.end(), _merit_dyn_page_members.begin() + member_begin);
        _merit_dyn_page_member_count[page_idx] = static_cast<uint16_t>(pf.member_ids.size());
        // Publish locations only after the page table is complete.
        const uint32_t sector = MERIT_DYN_SECTOR_BASE + page_idx;
        for (size_t mi = 0; mi < pf.member_ids.size(); mi++)
        {
            MeritDiskLoc loc;
            loc.sector = sector;
            loc.slot = static_cast<uint16_t>(mi);
            loc.nsectors = 1;
            const uint32_t mid = pf.member_ids[mi];
            merit_dyn_loc_add_unlocked(mid, loc);
        }
        _merit_dyn_committed_pages++;
        _merit_dyn_flush_count++;
        _merit_dcache_page_writes.fetch_add(1, std::memory_order_relaxed);
    }
    if (_merit_dcache_evict_ncache_on_commit && _merit_mem_pool != nullptr && _merit_mem_pool->active())
    {
        uint64_t erased = 0;
        for (uint32_t member_id : pf.member_ids)
            erased += _merit_mem_pool->erase(member_id) ? 1ULL : 0ULL;
        _merit_dcache_ncache_erases.fetch_add(erased, std::memory_order_relaxed);
    }
    return true;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::reload_merit_memory_cache(const std::string &profile_prefix, double merit_memory_gb,
                                                       double host_memory_gb, double reserve_gb, uint32_t num_threads,
                                                       uint64_t &evicted_nodes, std::string &report)
{
    evicted_nodes = 0;
    report.clear();
    const uint64_t before = merit_memory_cached_count();

    uint64_t max_nodes = 0;
    if (plan_merit_memory_cache(merit_memory_gb, host_memory_gb, reserve_gb, num_threads, max_nodes, report) != 0)
        return -1;

    clear_merit_memory_cache();
    std::vector<uint32_t> node_list;
    if (build_merit_memory_node_list(profile_prefix, max_nodes, node_list, 0) != 0)
        return -1;
    if (!node_list.empty())
    {
        if (load_merit_memory_pool(profile_prefix, node_list) != 0)
            return -1;
    }

    const uint64_t after = merit_memory_cached_count();
    if (before > after)
        evicted_nodes = before - after;

    std::ostringstream oss;
    oss << "MERIT memory reload: cached " << after << " nodes";
    if (evicted_nodes > 0)
        oss << ", evicted " << evicted_nodes << " from DRAM cache";
    oss << ".";
    report = oss.str();
    diskann::cout << report << std::endl;
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::plan_merit_disk_cache(double base_ratio, uint64_t &out_max_nodes,
                                                   std::string &report, bool allow_replica_slots) const
{
    out_max_nodes = 0;
    report.clear();
    if (base_ratio <= 0.0)
    {
        report = "merit_disk_cache_ratio <= 0; MERIT disk cache disabled.";
        return 0;
    }
    const double max_ratio = allow_replica_slots ? 10.0 : 1.0;
    if (base_ratio > max_ratio)
    {
        report = "ERROR: merit_disk_cache_ratio exceeds allowed maximum.";
        return -1;
    }
    if (_disk_index_file.empty() || _max_node_len == 0)
    {
        report = "ERROR: disk index not loaded; cannot size MERIT disk cache.";
        return -1;
    }

    const uint64_t base_bytes =
        (_base_disk_index_bytes > 0) ? _base_disk_index_bytes : get_file_size(_disk_index_file);
    const uint64_t budget = static_cast<uint64_t>(base_ratio * static_cast<double>(base_bytes));
    uint64_t nodes = 0;
    if (_nnodes_per_sector > 0)
    {
        const uint64_t sectors = budget / defaults::SECTOR_LEN;
        nodes = sectors * _nnodes_per_sector;
    }
    else
    {
        const uint64_t secs_per_node = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
        const uint64_t bytes_per_node = secs_per_node * defaults::SECTOR_LEN;
        nodes = (bytes_per_node > 0) ? (budget / bytes_per_node) : 0;
    }
    if (!allow_replica_slots && nodes > _num_points)
        nodes = _num_points;
    if (nodes == 0)
    {
        report = "ERROR: MERIT disk-cache budget too small for one node.";
        return -1;
    }
    out_max_nodes = nodes;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(4);
    oss << "MERIT disk-cache plan:\n"
        << "  base_index_file     = " << _disk_index_file << "\n"
        << "  base_size           = " << (base_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  ratio               = " << base_ratio << "\n"
        << "  budget              = " << (budget / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  max_node_len        = " << _max_node_len << " B, nnodes_per_sector=" << _nnodes_per_sector << "\n"
        << "  selected_nodes      = " << out_max_nodes
        << " (disk tier; k_hops=0 flat by node_expand, else k-hop disk cache; exclude memory-tier ids)";
    report = oss.str();
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::reload_merit_disk_cache(const std::string &profile_prefix, double base_ratio,
                                                     const std::string &output_prefix, uint64_t rank_skip,
                                                     uint64_t &evicted_nodes, std::string &report,
                                                     bool unified_single_file, uint32_t k_hops,
                                                     const std::string &layout)
{
    evicted_nodes = 0;
    report.clear();
    const uint64_t before = _merit_dc_map.size();

    uint64_t max_nodes = 0;
    if (plan_merit_disk_cache(base_ratio, max_nodes, report) != 0)
        return -1;

    if (build_and_load_merit_disk_cache(profile_prefix, max_nodes, output_prefix, rank_skip, unified_single_file,
                                        k_hops, layout) != 0)
        return -1;

    const uint64_t after = _merit_dc_map.size();
    if (before > after)
        evicted_nodes = before - after;

    std::ostringstream oss;
    oss << "MERIT disk-cache reload: " << after << " nodes in map";
    if (rank_skip > 0)
        oss << " (rank_skip=" << rank_skip << " for memory tier)";
    if (evicted_nodes > 0)
        oss << ", evicted " << evicted_nodes << " from disk-cache directory";
    oss << ".";
    report = oss.str();
    diskann::cout << report << std::endl;
    return 0;
}

namespace
{
inline int copy_file_binary(const std::string &src, const std::string &dst)
{
    std::ifstream in(src, std::ios::binary);
    if (!in.is_open())
        return -1;
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
        return -1;
    out << in.rdbuf();
    return out.good() ? 0 : -1;
}
} // namespace

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_merit_disk_cache_from_prefix(const std::string &output_prefix)
{
    if (_merit_disk_reader)
    {
        _merit_disk_reader->close();
        _merit_disk_reader.reset();
    }
    _merit_unified_disk = false;
    _merit_region_byte_offset = 0;
    _merit_dc_map.clear();
    _merit_dc_seed_page_nbrs.clear();
    _merit_dc_seed_member_loc.clear();
    _merit_dc_seed_canonical_loc.clear();
    _merit_seed_only_layout = false;
    _merit_seed_only_expand = false;
    _merit_seed_first_lookup = false;
    _merit_child_only_layout = false;
    _merit_dc_num_nodes = 0;
    _merit_dc_path.clear();

    const std::string data_path = output_prefix + "_merit_dc.data";
    const std::string nodes_path = output_prefix + "_merit_dc.nodes";
    if (!file_exists(data_path) || !file_exists(nodes_path))
    {
        diskann::cerr << "MERIT disk-cache reuse: missing " << data_path << " or " << nodes_path << std::endl;
        return -1;
    }

    uint32_t *nodes = nullptr;
    size_t npts = 0, nd = 0;
    diskann::load_bin<uint32_t>(nodes_path, nodes, npts, nd);
    std::vector<uint32_t> node_list(nodes, nodes + npts);
    delete[] nodes;

    if (node_list.empty() || _nnodes_per_sector == 0)
    {
        diskann::cerr << "MERIT disk-cache reuse: empty node list or nnodes_per_sector=0." << std::endl;
        return -1;
    }

    const uint64_t nps = _nnodes_per_sector;
    uint64_t slot_in_sector = 0;
    uint32_t cur_sector = 0;
    _merit_dc_map.reserve(node_list.size());
    for (uint32_t id : node_list)
    {
        MeritDiskLoc loc;
        loc.sector = cur_sector;
        loc.slot = static_cast<uint16_t>(slot_in_sector);
        loc.nsectors = 1;
        _merit_dc_map[id].push_back(loc);
        slot_in_sector++;
        if (slot_in_sector == nps)
        {
            slot_in_sector = 0;
            cur_sector++;
        }
    }

    const std::string seed_pages_path = output_prefix + "_merit_dc.seed_pages";
    const std::string seed_nbrs_path = output_prefix + "_merit_dc.seed_nbrs";
    const std::string seed_member_locs_path = output_prefix + "_merit_dc.seed_member_locs";
    std::vector<std::pair<uint32_t, uint8_t>> seed_pages_sidecar;
    if (load_seed_pages_sidecar(seed_pages_path, seed_pages_sidecar) != 0)
        return -1;
    if (load_seed_nbrs_sidecar(seed_nbrs_path, _merit_dc_seed_page_nbrs) == 0 && !_merit_dc_seed_page_nbrs.empty())
    {
        _merit_seed_only_expand =
            (std::getenv("MERIT_SEED_ONLY_EXPAND") != nullptr &&
             std::strcmp(std::getenv("MERIT_SEED_ONLY_EXPAND"), "0") != 0);
        if (std::getenv("MERIT_SEED_ONLY_LOOKUP") != nullptr &&
            std::strcmp(std::getenv("MERIT_SEED_ONLY_LOOKUP"), "0") != 0)
        {
            _merit_seed_only_layout = true;
            std::unordered_set<uint32_t> seed_ids;
            seed_ids.reserve(_merit_dc_seed_page_nbrs.size());
            for (const auto &kv : _merit_dc_seed_page_nbrs)
                seed_ids.insert(kv.first);
            tsl::robin_map<uint32_t, std::vector<MeritDiskLoc>> seed_only_map;
            seed_only_map.reserve(seed_ids.size());
            for (const auto &kv : _merit_dc_map)
            {
                if (seed_ids.count(kv.first) > 0)
                    seed_only_map[kv.first] = kv.second;
            }
            _merit_dc_map = std::move(seed_only_map);
        }
    }
    if (load_seed_member_locs_sidecar(seed_member_locs_path, _merit_dc_seed_member_loc, _merit_dc_seed_canonical_loc) !=
        0)
    {
        diskann::cerr << "MERIT disk-cache reuse: failed to load " << seed_member_locs_path << std::endl;
        return -1;
    }
    for (const auto &entry : seed_pages_sidecar)
    {
        const uint32_t seed_id = entry.first;
        const uint8_t total_pages = entry.second;
        const auto member_it = _merit_dc_seed_member_loc.find(seed_id);
        if (member_it == _merit_dc_seed_member_loc.end() || member_it->second.empty())
            continue;

        uint32_t first_sector = std::numeric_limits<uint32_t>::max();
        for (const auto &member : member_it->second)
            first_sector = std::min(first_sector, member.second.sector);
        if (first_sector != std::numeric_limits<uint32_t>::max())
        {
            MeritDiskLoc canonical;
            canonical.sector = first_sector;
            canonical.slot = 0;
            canonical.nsectors = total_pages;
            _merit_dc_seed_canonical_loc[seed_id] = canonical;
        }
    }
    for (const auto &seed_members : _merit_dc_seed_member_loc)
    {
        if (seed_members.second.find(seed_members.first) == seed_members.second.end())
        {
            _merit_child_only_layout = true;
            break;
        }
    }
    const bool record_driven_seed_access =
        (std::getenv("MERIT_RECORD_DRIVEN_SEED_ACCESS") != nullptr &&
         std::strcmp(std::getenv("MERIT_RECORD_DRIVEN_SEED_ACCESS"), "0") != 0);
    if (_merit_child_only_layout || record_driven_seed_access)
    {
        _merit_dc_map.clear();
        for (const auto &seed_members : _merit_dc_seed_member_loc)
        {
            for (const auto &member : seed_members.second)
                _merit_dc_map[member.first].push_back(member.second);
        }
    }
    update_merit_seed_first_lookup_flag();

    _merit_dc_path = data_path;
    _merit_dc_num_nodes = node_list.size();

#ifndef _WINDOWS
    _merit_disk_reader.reset(new LinuxAlignedFileReader());
#else
    _merit_disk_reader.reset(new WindowsAlignedFileReader());
#endif
    _merit_disk_reader->open(data_path);
#pragma omp parallel for num_threads((int)_max_nthreads)
    for (int64_t thread = 0; thread < (int64_t)_max_nthreads; thread++)
    {
#pragma omp critical
        {
            _merit_disk_reader->register_thread();
        }
    }

    diskann::cout << "MERIT disk-cache loaded (reuse): " << _merit_dc_num_nodes << " nodes -> " << data_path
                  << " (map size=" << _merit_dc_map.size()
                  << ", seed-first lookup=" << (_merit_seed_first_lookup ? "on" : "off") << ")" << std::endl;
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::build_and_load_merit_disk_cache(const std::string &profile_prefix, uint64_t max_nodes,
                                                             const std::string &output_prefix, uint64_t rank_skip,
                                                             bool unified_single_file, uint32_t k_hops,
                                                             const std::string &layout)
{
    if (_merit_disk_reader)
    {
        _merit_disk_reader->close();
        _merit_disk_reader.reset();
    }
    _merit_unified_disk = false;
    _merit_region_byte_offset = 0;
    _merit_dc_map.clear();
    _merit_dc_seed_page_nbrs.clear();
    _merit_dc_seed_member_loc.clear();
    _merit_dc_seed_canonical_loc.clear();
    _merit_seed_only_layout = false;
    _merit_seed_only_expand = false;
    _merit_seed_first_lookup = false;
    _merit_child_only_layout = false;
    _merit_dc_num_nodes = 0;
    _merit_dc_path.clear();

    if (max_nodes == 0)
        return 0;

    std::vector<uint32_t> node_list;
    std::vector<SeedPageGroup> seed_groups;
    const std::string layout_norm = normalize_disk_cache_layout(layout);
    _merit_child_only_layout =
        is_directed_child_only_layout(layout_norm) || is_directed_child_replica_layout(layout_norm);
    _merit_seed_only_layout = (std::getenv("MERIT_SEED_ONLY_LOOKUP") != nullptr &&
                               std::strcmp(std::getenv("MERIT_SEED_ONLY_LOOKUP"), "0") != 0);
    _merit_seed_only_expand =
        (std::getenv("MERIT_SEED_ONLY_EXPAND") != nullptr &&
         std::strcmp(std::getenv("MERIT_SEED_ONLY_EXPAND"), "0") != 0);
    std::vector<SeedPageGroup> *seed_groups_ptr =
        (is_directed_beam_pct_layout(layout_norm) || is_directed_child_only_layout(layout_norm) ||
         is_directed_child_replica_layout(layout_norm) || is_directed_seed_replica_layout(layout_norm) ||
         diskann::is_directed_seed_replica_benefit_layout(layout_norm) ||
         diskann::is_directed_seed_replica_budget_layout(layout_norm) ||
         diskann::is_directed_seed_core_unique_fill_layout(layout_norm))
            ? &seed_groups
            : nullptr;
    if (build_merit_disk_node_list(profile_prefix, max_nodes, rank_skip, k_hops, layout, node_list, seed_groups_ptr) !=
        0)
        return -1;

    const std::string data_path =
        unified_single_file ? (output_prefix + "_disk_merit_unified.index") : (output_prefix + "_merit_dc.data");
    const std::string nodes_path = output_prefix + "_merit_dc.nodes";
    const std::string seed_pages_path = output_prefix + "_merit_dc.seed_pages";
    const std::string seed_nbrs_path = output_prefix + "_merit_dc.seed_nbrs";
    const std::string seed_member_locs_path = output_prefix + "_merit_dc.seed_member_locs";

    // Persist node order for debugging / reload.
    diskann::save_bin<uint32_t>(nodes_path, node_list.data(), node_list.size(), 1);

    if (unified_single_file)
    {
        if (copy_file_binary(_disk_index_file, data_path) != 0)
        {
            diskann::cerr << "Failed to copy base index into unified MERIT file: " << data_path << std::endl;
            return -1;
        }
        uint64_t sz = get_file_size(data_path);
        const uint64_t aligned_sz = ROUND_UP(sz, defaults::SECTOR_LEN);
        if (aligned_sz > sz)
        {
            std::ofstream pad(data_path, std::ios::binary | std::ios::app);
            std::vector<char> zeros(static_cast<size_t>(aligned_sz - sz), 0);
            pad.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
            pad.close();
        }
        _merit_region_byte_offset = aligned_sz;
    }

    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    const uint64_t secs_per_node =
        (_nnodes_per_sector > 0) ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    // Space capacity is max_nodes slot-equivalents from the ratio budget (sectors * nps),
    // not "unique nodes". Seed-group layouts may use partial pages; cap file size to this
    // many sectors and pad slots with replicas so occupied bytes stay within ratio.
    const uint64_t space_budget_sectors =
        (_nnodes_per_sector > 0) ? (max_nodes / nps) : (max_nodes * secs_per_node);
    if (space_budget_sectors > 0)
    {
        diskann::cout << "MERIT disk-cache space budget: sectors<=" << space_budget_sectors << " ("
                      << (space_budget_sectors * defaults::SECTOR_LEN) / (1024.0 * 1024.0 * 1024.0)
                      << " GiB), slot_cap=" << (space_budget_sectors * nps)
                      << " (pad partial pages with seed replicas; stop groups when full)." << std::endl;
    }

    char *sector_buf = nullptr;
    alloc_aligned((void **)&sector_buf, defaults::SECTOR_LEN, defaults::SECTOR_LEN);

    std::ofstream out(data_path, std::ios::binary | (unified_single_file ? std::ios::app : std::ios::trunc));
    if (!out.is_open())
    {
        aligned_free(sector_buf);
        diskann::cerr << "Failed to open MERIT disk-cache for write: " << data_path << std::endl;
        return -1;
    }

    // Scratch for reading base nodes one block at a time.
    const size_t BLOCK = 64;
    std::vector<T *> coord_ptrs;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_ptrs;
    std::vector<uint32_t> nbr_storage(BLOCK * (_max_degree + 1));
    std::vector<T> coord_storage(BLOCK * _aligned_dim);

    auto flush_sector = [&](char *buf) {
        out.write(buf, defaults::SECTOR_LEN);
        memset(buf, 0, defaults::SECTOR_LEN);
    };

    auto write_node_into_slot = [&](uint32_t node_id, char *slot_dst, const T *coords, uint32_t nnbrs,
                                    const uint32_t *nbrs, uint32_t sector_id, uint16_t slot_id, uint16_t nsectors,
                                    bool write_seed_tail, uint8_t seed_total_pages, bool index_in_map) {
        memcpy(slot_dst, coords, _disk_bytes_per_point);
        uint32_t *nhood = reinterpret_cast<uint32_t *>(slot_dst + _disk_bytes_per_point);
        nhood[0] = nnbrs;
        memcpy(nhood + 1, nbrs, nnbrs * sizeof(uint32_t));
        if (write_seed_tail)
            write_merit_dc_seed_tail(slot_dst, _max_node_len, _disk_bytes_per_point, nnbrs, seed_total_pages);

        if (index_in_map)
        {
            MeritDiskLoc loc;
            loc.sector = sector_id;
            loc.slot = slot_id;
            loc.nsectors = nsectors;
            _merit_dc_map[node_id].push_back(loc);
        }
    };

    memset(sector_buf, 0, defaults::SECTOR_LEN);
    uint64_t slot_in_sector = 0;
    uint32_t cur_sector = 0;
    size_t flat_write_begin = 0;

    if (!seed_groups.empty() && _nnodes_per_sector > 0)
    {
        std::vector<std::pair<uint32_t, uint8_t>> seed_pages_sidecar;
        seed_pages_sidecar.reserve(seed_groups.size());

        const std::vector<std::pair<uint32_t, uint32_t>> copack_pairs = load_merit_copack_pairs();
        std::unordered_map<uint32_t, uint32_t> copack_partner;
        for (const auto &pr : copack_pairs)
        {
            copack_partner[pr.first] = pr.second;
            copack_partner[pr.second] = pr.first;
        }
        std::unordered_map<uint32_t, size_t> seed_to_group_idx;
        for (size_t gi = 0; gi < seed_groups.size(); ++gi)
            seed_to_group_idx[seed_groups[gi].seed] = gi;
        std::unordered_set<uint32_t> copack_written_seeds;

        auto pad_page_slots = [&](std::vector<uint32_t> page_nodes, uint32_t pad_id) {
            if (page_nodes.empty() || nps == 0)
                return page_nodes;
            if (pad_id == UINT32_MAX)
                pad_id = page_nodes.front();
            while (page_nodes.size() < nps)
                page_nodes.push_back(pad_id);
            if (page_nodes.size() > nps)
                page_nodes.resize(static_cast<size_t>(nps));
            return page_nodes;
        };

        auto group_sector_cost = [&](const SeedPageGroup &group) -> uint64_t {
            uint64_t cost = group.pages.size();
            for (const auto &page : group.cold_pages)
            {
                if (!page.empty())
                    ++cost;
            }
            return cost;
        };

        size_t group_nodes_emitted = 0;
        uint64_t groups_written = 0;
        uint64_t groups_skipped_budget = 0;

        auto write_group_page = [&](const SeedPageGroup &group, size_t pi) -> int {
            if (space_budget_sectors > 0 && cur_sector >= space_budget_sectors)
                return 0;
            const auto &raw_page = group.pages[pi];
            if (raw_page.empty())
                return 0;
            std::vector<uint32_t> page_nodes = pad_page_slots(raw_page, group.seed);

            const uint32_t page_sector = cur_sector;
            if (pi == 0)
            {
                MeritDiskLoc canonical;
                canonical.sector = page_sector;
                canonical.slot = 0;
                canonical.nsectors = group.total_pages;
                _merit_dc_seed_canonical_loc[group.seed] = canonical;
            }
            for (size_t si = 0; si < page_nodes.size(); ++si)
            {
                const uint32_t member_id = page_nodes[si];
                MeritDiskLoc member_loc;
                member_loc.sector = page_sector;
                member_loc.slot = static_cast<uint16_t>(si);
                member_loc.nsectors = (pi == 0 && member_id == group.seed) ? group.total_pages : 1;
                _merit_dc_seed_member_loc[group.seed][member_id] = member_loc;
                if (member_id == group.seed && pi == 0)
                    _merit_dc_seed_canonical_loc[group.seed] = member_loc;
            }

            memset(sector_buf, 0, defaults::SECTOR_LEN);
            for (size_t begin = 0; begin < page_nodes.size(); begin += BLOCK)
            {
                const size_t end = std::min(page_nodes.size(), begin + BLOCK);
                const size_t bn = end - begin;
                std::vector<uint32_t> batch(page_nodes.begin() + begin, page_nodes.begin() + end);
                coord_ptrs.resize(bn);
                nbr_ptrs.resize(bn);
                for (size_t i = 0; i < bn; i++)
                {
                    coord_ptrs[i] = coord_storage.data() + i * _aligned_dim;
                    nbr_ptrs[i] = {0, nbr_storage.data() + i * (_max_degree + 1)};
                }
                auto ok = read_nodes(batch, coord_ptrs, nbr_ptrs);
                for (size_t i = 0; i < bn; i++)
                {
                    if (!ok[i])
                    {
                        aligned_free(sector_buf);
                        out.close();
                        diskann::cerr << "Failed to read base node " << batch[i]
                                      << " for MERIT disk-cache (pct80 group)." << std::endl;
                        return -1;
                    }
                    const uint32_t node_id = batch[i];
                    const uint16_t slot_id = static_cast<uint16_t>(begin + i);
                    const bool is_seed = (pi == 0 && node_id == group.seed);
                    const bool index_in_map = !_merit_seed_only_layout || is_seed;
                    write_node_into_slot(node_id, sector_buf + static_cast<uint64_t>(slot_id) * _max_node_len,
                                         coord_ptrs[i], nbr_ptrs[i].first, nbr_ptrs[i].second, cur_sector, slot_id,
                                         is_seed ? group.total_pages : 1, is_seed, group.total_pages, index_in_map);
                }
            }
            flush_sector(sector_buf);
            cur_sector++;
            group_nodes_emitted += page_nodes.size();
            return 0;
        };

        auto write_group_cold_pages = [&](const SeedPageGroup &group) -> int {
            for (size_t pi = 0; pi < group.cold_pages.size(); ++pi)
            {
                if (space_budget_sectors > 0 && cur_sector >= space_budget_sectors)
                    return 0;
                const auto &raw_page = group.cold_pages[pi];
                if (raw_page.empty())
                    continue;
                std::vector<uint32_t> page_nodes = pad_page_slots(raw_page, group.seed);

                const uint32_t page_sector = cur_sector;
                for (size_t si = 0; si < page_nodes.size(); ++si)
                {
                    const uint32_t member_id = page_nodes[si];
                    MeritDiskLoc member_loc;
                    member_loc.sector = page_sector;
                    member_loc.slot = static_cast<uint16_t>(si);
                    member_loc.nsectors = 1;
                    _merit_dc_seed_member_loc[group.seed][member_id] = member_loc;
                }

                memset(sector_buf, 0, defaults::SECTOR_LEN);
                for (size_t begin = 0; begin < page_nodes.size(); begin += BLOCK)
                {
                    const size_t end = std::min(page_nodes.size(), begin + BLOCK);
                    const size_t bn = end - begin;
                    std::vector<uint32_t> batch(page_nodes.begin() + begin, page_nodes.begin() + end);
                    coord_ptrs.resize(bn);
                    nbr_ptrs.resize(bn);
                    for (size_t i = 0; i < bn; i++)
                    {
                        coord_ptrs[i] = coord_storage.data() + i * _aligned_dim;
                        nbr_ptrs[i] = {0, nbr_storage.data() + i * (_max_degree + 1)};
                    }
                    auto ok = read_nodes(batch, coord_ptrs, nbr_ptrs);
                    for (size_t i = 0; i < bn; i++)
                    {
                        if (!ok[i])
                        {
                            aligned_free(sector_buf);
                            out.close();
                            diskann::cerr << "Failed to read base node " << batch[i]
                                          << " for MERIT disk-cache (pct cold tail)." << std::endl;
                            return -1;
                        }
                        const uint32_t node_id = batch[i];
                        const uint16_t slot_id = static_cast<uint16_t>(begin + i);
                        write_node_into_slot(node_id, sector_buf + static_cast<uint64_t>(slot_id) * _max_node_len,
                                             coord_ptrs[i], nbr_ptrs[i].first, nbr_ptrs[i].second, cur_sector,
                                             slot_id, 1, false, group.total_pages, true);
                    }
                }
                flush_sector(sector_buf);
                cur_sector++;
                group_nodes_emitted += page_nodes.size();
            }
            return 0;
        };

        auto finalize_group_sidecar = [&](const SeedPageGroup &group) {
            seed_pages_sidecar.emplace_back(group.seed, group.total_pages);
            std::vector<uint32_t> page_nbrs;
            for (const auto &page : group.pages)
            {
                for (uint32_t nid : page)
                {
                    if (nid != group.seed)
                        page_nbrs.push_back(nid);
                }
            }
            _merit_dc_seed_page_nbrs[group.seed] = std::move(page_nbrs);
        };

        auto apply_copack_canonical = [&](uint32_t seed_a, uint32_t seed_b, uint32_t extent_base) {
            MeritDiskLoc canon;
            canon.sector = extent_base;
            canon.slot = 0;
            canon.nsectors = 2;
            _merit_dc_seed_canonical_loc[seed_a] = canon;
            _merit_dc_seed_canonical_loc[seed_b] = canon;
            auto fix_seed = [&](uint32_t seed_id) {
                const auto outer = _merit_dc_seed_member_loc.find(seed_id);
                if (outer == _merit_dc_seed_member_loc.end())
                    return;
                const auto inner = outer->second.find(seed_id);
                if (inner == outer->second.end())
                    return;
                MeritDiskLoc updated = inner->second;
                updated.nsectors = 2;
                _merit_dc_seed_member_loc[seed_id][seed_id] = updated;
            };
            fix_seed(seed_a);
            fix_seed(seed_b);
        };

        for (size_t gi = 0; gi < seed_groups.size(); ++gi)
        {
            const SeedPageGroup &group = seed_groups[gi];
            if (copack_written_seeds.count(group.seed) > 0)
                continue;

            uint32_t partner_seed = 0;
            const auto partner_it = copack_partner.find(group.seed);
            if (partner_it != copack_partner.end())
                partner_seed = partner_it->second;

            size_t partner_gi = std::numeric_limits<size_t>::max();
            const auto pgi_it = seed_to_group_idx.find(partner_seed);
            if (pgi_it != seed_to_group_idx.end())
                partner_gi = pgi_it->second;

            const bool do_copack = !copack_pairs.empty() && partner_seed != 0 &&
                                   partner_gi != std::numeric_limits<size_t>::max() &&
                                   copack_written_seeds.count(partner_seed) == 0 && !group.pages.empty() &&
                                   !seed_groups[partner_gi].pages.empty() && group.total_pages == 1 &&
                                   seed_groups[partner_gi].total_pages == 1;

            if (do_copack)
            {
                const SeedPageGroup &partner_group = seed_groups[partner_gi];
                const uint64_t need = group_sector_cost(group) + group_sector_cost(partner_group);
                if (space_budget_sectors > 0 && cur_sector + need > space_budget_sectors)
                {
                    ++groups_skipped_budget;
                    break;
                }
                const uint32_t extent_base = cur_sector;
                if (write_group_page(group, 0) != 0)
                    return -1;
                if (write_group_page(partner_group, 0) != 0)
                    return -1;
                apply_copack_canonical(group.seed, partner_seed, extent_base);

                for (size_t pi = 1; pi < group.pages.size(); ++pi)
                {
                    if (write_group_page(group, pi) != 0)
                        return -1;
                }
                for (size_t pi = 1; pi < partner_group.pages.size(); ++pi)
                {
                    if (write_group_page(partner_group, pi) != 0)
                        return -1;
                }
                if (write_group_cold_pages(group) != 0)
                    return -1;
                if (write_group_cold_pages(partner_group) != 0)
                    return -1;

                finalize_group_sidecar(group);
                finalize_group_sidecar(partner_group);
                copack_written_seeds.insert(group.seed);
                copack_written_seeds.insert(partner_seed);
                groups_written += 2;
                continue;
            }

            const uint64_t need = group_sector_cost(group);
            if (space_budget_sectors > 0 && cur_sector + need > space_budget_sectors)
            {
                ++groups_skipped_budget;
                // Keep scanning for a later smaller group that still fits.
                continue;
            }

            for (size_t pi = 0; pi < group.pages.size(); ++pi)
            {
                if (write_group_page(group, pi) != 0)
                    return -1;
            }
            if (write_group_cold_pages(group) != 0)
                return -1;
            finalize_group_sidecar(group);
            copack_written_seeds.insert(group.seed);
            ++groups_written;
        }

        diskann::cout << "MERIT disk-cache seed-group write: groups_written=" << groups_written
                      << " groups_skipped_for_space_budget=" << groups_skipped_budget
                      << " sectors=" << cur_sector << "/" << space_budget_sectors
                      << " padded_slots_emitted=" << group_nodes_emitted << std::endl;

        save_seed_pages_sidecar(seed_pages_path, seed_pages_sidecar);
        if (!_merit_dc_seed_page_nbrs.empty())
            save_seed_nbrs_sidecar(seed_nbrs_path, _merit_dc_seed_page_nbrs);
        if (!_merit_dc_seed_member_loc.empty())
            save_seed_member_locs_sidecar(seed_member_locs_path, _merit_dc_seed_member_loc);
        update_merit_seed_first_lookup_flag();
        // Seed-group write may skip groups under the space budget; node_list is no longer a
        // simple written-prefix. Dense-fill any remaining budget sectors from node_list
        // (duplicates allowed) instead of trusting a flat prefix offset.
        flat_write_begin = node_list.size();
        slot_in_sector = 0;
    }

    auto write_flat_node = [&](uint32_t node_id, const T *coords, uint32_t nnbrs, const uint32_t *nbrs) -> int {
        if (_nnodes_per_sector > 0)
        {
            if (space_budget_sectors > 0 && cur_sector >= space_budget_sectors && slot_in_sector == 0)
                return 1; // budget full
            char *dst = sector_buf + slot_in_sector * _max_node_len;
            write_node_into_slot(node_id, dst, coords, nnbrs, nbrs, cur_sector,
                                 static_cast<uint16_t>(slot_in_sector), 1, false, 1, true);
            slot_in_sector++;
            if (slot_in_sector == nps)
            {
                flush_sector(sector_buf);
                slot_in_sector = 0;
                cur_sector++;
            }
            return 0;
        }
        if (space_budget_sectors > 0 && cur_sector >= space_budget_sectors)
            return 1;
        std::vector<char> node_pack(secs_per_node * defaults::SECTOR_LEN, 0);
        memcpy(node_pack.data(), coords, _disk_bytes_per_point);
        uint32_t *nhood = reinterpret_cast<uint32_t *>(node_pack.data() + _disk_bytes_per_point);
        nhood[0] = nnbrs;
        memcpy(nhood + 1, nbrs, nnbrs * sizeof(uint32_t));
        MeritDiskLoc loc;
        loc.sector = cur_sector;
        loc.slot = 0;
        loc.nsectors = static_cast<uint16_t>(secs_per_node);
        _merit_dc_map[node_id].push_back(loc);
        out.write(node_pack.data(), static_cast<std::streamsize>(node_pack.size()));
        cur_sector += static_cast<uint32_t>(secs_per_node);
        return 0;
    };

    for (size_t begin = flat_write_begin; begin < node_list.size(); begin += BLOCK)
    {
        if (space_budget_sectors > 0 && cur_sector >= space_budget_sectors && slot_in_sector == 0)
            break;
        const size_t end = std::min(node_list.size(), begin + BLOCK);
        const size_t bn = end - begin;
        std::vector<uint32_t> batch(node_list.begin() + begin, node_list.begin() + end);
        coord_ptrs.resize(bn);
        nbr_ptrs.resize(bn);
        for (size_t i = 0; i < bn; i++)
        {
            coord_ptrs[i] = coord_storage.data() + i * _aligned_dim;
            nbr_ptrs[i] = {0, nbr_storage.data() + i * (_max_degree + 1)};
        }
        auto ok = read_nodes(batch, coord_ptrs, nbr_ptrs);
        for (size_t i = 0; i < bn; i++)
        {
            if (!ok[i])
            {
                aligned_free(sector_buf);
                out.close();
                diskann::cerr << "Failed to read base node " << batch[i] << " for MERIT disk-cache." << std::endl;
                return -1;
            }
            const int wr = write_flat_node(batch[i], coord_ptrs[i], nbr_ptrs[i].first, nbr_ptrs[i].second);
            if (wr != 0)
                break;
        }
    }

    // Fill remaining space-budget sectors by cycling node_list (explicit replicas OK).
    // Skip when seed-group pages already consumed the real working set; padding the
    // full ratio budget creates seedless pages that break static prefill.
    const bool skip_dense_budget_fill =
        (!_merit_dc_seed_member_loc.empty() && flat_write_begin >= node_list.size() && cur_sector > 0);
    if (skip_dense_budget_fill)
    {
        diskann::cout << "MERIT disk-cache: skip dense budget fill after seed-groups "
                      << "(sectors_written=" << cur_sector << ", budget=" << space_budget_sectors << ")."
                      << std::endl;
    }
    if (!skip_dense_budget_fill && _nnodes_per_sector > 0 && space_budget_sectors > 0 && !node_list.empty())
    {
        size_t fill_idx = 0;
        uint64_t replica_slots = 0;
        while (cur_sector < space_budget_sectors || slot_in_sector != 0)
        {
            if (cur_sector >= space_budget_sectors && slot_in_sector == 0)
                break;
            const size_t batch_n = std::min<size_t>(BLOCK, nps - slot_in_sector);
            std::vector<uint32_t> batch(batch_n);
            for (size_t i = 0; i < batch_n; ++i)
            {
                batch[i] = node_list[fill_idx % node_list.size()];
                ++fill_idx;
            }
            coord_ptrs.resize(batch_n);
            nbr_ptrs.resize(batch_n);
            for (size_t i = 0; i < batch_n; i++)
            {
                coord_ptrs[i] = coord_storage.data() + i * _aligned_dim;
                nbr_ptrs[i] = {0, nbr_storage.data() + i * (_max_degree + 1)};
            }
            auto ok = read_nodes(batch, coord_ptrs, nbr_ptrs);
            for (size_t i = 0; i < batch_n; i++)
            {
                if (!ok[i])
                {
                    aligned_free(sector_buf);
                    out.close();
                    diskann::cerr << "Failed to read base node " << batch[i]
                                  << " while padding MERIT disk-cache to space budget." << std::endl;
                    return -1;
                }
                write_flat_node(batch[i], coord_ptrs[i], nbr_ptrs[i].first, nbr_ptrs[i].second);
                ++replica_slots;
            }
        }
        if (replica_slots > 0)
        {
            diskann::cout << "MERIT disk-cache space fill: wrote " << replica_slots
                          << " extra slot(s) (replicas allowed) to reach sectors=" << cur_sector << "/"
                          << space_budget_sectors << std::endl;
        }
    }

    if (slot_in_sector > 0 && _nnodes_per_sector > 0)
    {
        flush_sector(sector_buf);
        cur_sector++;
        slot_in_sector = 0;
    }

    out.close();
    aligned_free(sector_buf);

    diskann::cout << "MERIT disk-cache packed file sectors=" << cur_sector
                  << " bytes=" << (static_cast<uint64_t>(cur_sector) * defaults::SECTOR_LEN)
                  << " space_budget_sectors=" << space_budget_sectors << std::endl;

    _merit_dc_path = data_path;
    _merit_dc_num_nodes = node_list.size();

    if (unified_single_file)
    {
        _merit_unified_disk = true;
        reader->close();
        reader->open(data_path);
        _disk_index_file = data_path;
        diskann::cout << "MERIT unified disk-cache ready: " << _merit_dc_num_nodes << " nodes appended at offset "
                      << (_merit_region_byte_offset / (1024.0 * 1024.0)) << " MiB -> " << data_path
                      << " (map size=" << _merit_dc_map.size() << ", seed-first lookup="
                      << (_merit_seed_first_lookup ? "on" : "off") << ", single-fd merged IO)" << std::endl;
        return 0;
    }

#ifndef _WINDOWS
    _merit_disk_reader.reset(new LinuxAlignedFileReader());
#else
    _merit_disk_reader.reset(new WindowsAlignedFileReader());
#endif
    _merit_disk_reader->open(data_path);
#pragma omp parallel for num_threads((int)_max_nthreads)
    for (int64_t thread = 0; thread < (int64_t)_max_nthreads; thread++)
    {
#pragma omp critical
        {
            _merit_disk_reader->register_thread();
        }
    }

    diskann::cout << "MERIT disk-cache ready: " << _merit_dc_num_nodes << " nodes -> " << data_path
                  << " (map size=" << _merit_dc_map.size() << ", seed-first lookup="
                  << (_merit_seed_first_lookup ? "on" : "off");
    if (_merit_seed_only_layout || _merit_seed_only_expand)
        diskann::cout << " (seed-only lookup=" << (_merit_seed_only_layout ? "on" : "off")
                      << ", page-local expand=" << (_merit_seed_only_expand ? "on" : "off") << ")";
    diskann::cout << ")" << std::endl;
    return 0;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::update_merit_seed_first_lookup_flag()
{
    const bool record_driven_seed_access =
        (std::getenv("MERIT_RECORD_DRIVEN_SEED_ACCESS") != nullptr &&
         std::strcmp(std::getenv("MERIT_RECORD_DRIVEN_SEED_ACCESS"), "0") != 0);
    const char *env = std::getenv("MERIT_SEED_FIRST_LOOKUP");
    if (env != nullptr)
    {
        _merit_seed_first_lookup = (std::strcmp(env, "0") != 0);
    }
    else
    {
        _merit_seed_first_lookup = !_merit_dc_seed_member_loc.empty();
    }
    if (record_driven_seed_access)
    {
        _merit_seed_first_lookup = true;
        diskann::cout << "MERIT record-driven self-seed access: on" << std::endl;
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_loc_in_query_cache(const SSDQueryScratch<T> *query_scratch,
                                                        uint32_t base_sector,
                                                        uint16_t nsectors) const
{
    if (!_query_sector_cache_enabled || query_scratch == nullptr || nsectors == 0)
        return false;
    for (uint16_t si = 0; si < nsectors; ++si)
    {
        if (query_scratch->merit_sector_cache.find(base_sector + static_cast<uint32_t>(si)) ==
            query_scratch->merit_sector_cache.end())
            return false;
    }
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_resolve_disk_cache_loc(uint32_t node_id,
                                                           const SSDQueryScratch<T> *query_scratch,
                                                           MeritDiskLoc &out_loc) const
{
    // The dynamic location directory is independently sharded. Publishing or
    // retiring a page takes the affected location shard locks, so lookup does
    // not need the global page-allocation mutex.
    if (_merit_dyn_enabled)
    {
        g_dyn_probe.resolve_call++;
        auto loc_from_dir = [&](uint32_t seed, uint32_t member) -> bool {
            MeritTimedSharedMutexGuard page_lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
            uint32_t page_idx = MERIT_DYN_INVALID_PAGE;
            if (!merit_dyn_seed_page_unlocked(seed, page_idx))
                return false;
            const size_t count = _merit_dyn_page_member_count[page_idx];
            const size_t begin = static_cast<size_t>(page_idx) * _merit_dyn_members_per_page;
            for (size_t k = 0; k < count; k++)
            {
                if (_merit_dyn_page_members[begin + k] != member)
                    continue;
                out_loc.sector = MERIT_DYN_SECTOR_BASE + page_idx;
                out_loc.slot = static_cast<uint16_t>(k);
                out_loc.nsectors = 1;
                return true;
            }
            return false;
        };
        // Default: loc-map, then self-seed page. Parent page only if
        // MERIT_RESOLVE_PARENT=1 (parent-first hurt the small dynamic d-cache).
        {
            bool found = false;
            merit_dc_for_each_loc_unlocked(node_id, [&](const MeritDiskLoc &loc, size_t) {
                if (found)
                    return;
                if (!merit_dyn_is_sector(loc.sector))
                    return;
                const uint32_t idx = loc.sector - MERIT_DYN_SECTOR_BASE;
                if (merit_dyn_page_live(idx))
                {
                    out_loc = loc;
                    g_dyn_probe.map_hit++;
                    found = true;
                }
            });
            if (found)
                return true;
        }
        if (loc_from_dir(node_id, node_id))
        {
            g_dyn_probe.self_hit++;
            return true;
        }
        static const bool resolve_parent =
            merit_env_u64("MERIT_RESOLVE_PARENT", 0) != 0;
        if (resolve_parent && query_scratch != nullptr)
        {
            const auto parent_it = query_scratch->profile_parent.find(node_id);
            if (parent_it == query_scratch->profile_parent.end())
                g_dyn_probe.parent_absent++;
            else if (loc_from_dir(parent_it->second, node_id))
            {
                g_dyn_probe.parent_hit++;
                return true;
            }
            else
            {
                uint32_t parent_page = MERIT_DYN_INVALID_PAGE;
                if (!merit_dyn_seed_page_unlocked(parent_it->second, parent_page))
                    g_dyn_probe.parent_no_page++;
                else
                    g_dyn_probe.parent_not_member++;
            }
        }
        else if (query_scratch == nullptr ||
                 query_scratch->profile_parent.find(node_id) == query_scratch->profile_parent.end())
            g_dyn_probe.parent_absent++;
        g_dyn_probe.resolve_miss++;
        return false;
    }
    if (_merit_dc_seed_member_loc.empty())
        return false;

    const bool record_driven_seed_access =
        (std::getenv("MERIT_RECORD_DRIVEN_SEED_ACCESS") != nullptr &&
         std::strcmp(std::getenv("MERIT_RECORD_DRIVEN_SEED_ACCESS"), "0") != 0);
    if (record_driven_seed_access)
    {
        if (query_scratch != nullptr)
        {
            const auto map_it = _merit_dc_map.find(node_id);
            if (map_it != _merit_dc_map.end())
            {
                for (const MeritDiskLoc &loc : map_it->second)
                {
                    if (merit_loc_in_query_cache(query_scratch, loc.sector, loc.nsectors))
                    {
                        out_loc = loc;
                        return true;
                    }
                }
            }
        }
        const auto seed_it = _merit_dc_seed_member_loc.find(node_id);
        const bool own_extent_contains_record =
            seed_it != _merit_dc_seed_member_loc.end() && seed_it->second.find(node_id) != seed_it->second.end();
        const auto canonical_it = _merit_dc_seed_canonical_loc.find(node_id);
        if (own_extent_contains_record && canonical_it != _merit_dc_seed_canonical_loc.end())
        {
            out_loc = canonical_it->second;
            return true;
        }
        const auto map_it = _merit_dc_map.find(node_id);
        if (map_it != _merit_dc_map.end() && !map_it->second.empty())
        {
            out_loc = map_it->second.front();
            return true;
        }
        return false;
    }

    const bool cache_replica_fallback =
        (std::getenv("MERIT_CACHE_REPLICA_FALLBACK") != nullptr &&
         std::strcmp(std::getenv("MERIT_CACHE_REPLICA_FALLBACK"), "0") != 0);
    if (cache_replica_fallback && query_scratch != nullptr)
    {
        const auto map_it = _merit_dc_map.find(node_id);
        if (map_it != _merit_dc_map.end())
        {
            for (const MeritDiskLoc &loc : map_it->second)
            {
                if (merit_loc_in_query_cache(query_scratch, loc.sector, loc.nsectors))
                {
                    out_loc = loc;
                    return true;
                }
            }
        }
    }

    // Strict parent-seed lookup: every non-medoid node may only use the page
    // belonging to the graph parent that first discovered it.
    if (query_scratch != nullptr)
    {
        const auto parent_it = query_scratch->profile_parent.find(node_id);
        if (parent_it != query_scratch->profile_parent.end())
        {
            const uint32_t parent_seed = parent_it->second;
            const auto member_it = _merit_dc_seed_member_loc.find(parent_seed);
            if (member_it != _merit_dc_seed_member_loc.end())
            {
                const auto loc_it = member_it->second.find(node_id);
                if (loc_it != member_it->second.end())
                {
                    out_loc = loc_it->second;
                    return true;
                }
            }
            if (_merit_child_only_layout)
            {
                const auto global_it = _merit_dc_map.find(node_id);
                if (global_it != _merit_dc_map.end() && !global_it->second.empty())
                {
                    out_loc = global_it->second.front();
                    return true;
                }
            }
            return false;
        }
    }

    // Bootstrap exception: the medoid has no graph parent, so start from its
    // own seed page. Non-medoid nodes never fall back to self-seed.
    const auto seed_it = _merit_dc_seed_member_loc.find(node_id);
    if (seed_it != _merit_dc_seed_member_loc.end())
    {
        const auto self_it = seed_it->second.find(node_id);
        if (self_it != seed_it->second.end())
        {
            out_loc = self_it->second;
            return true;
        }
    }

    return false;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_dyn_query_local_dcache_hit(uint32_t node_id,
                                                               SSDQueryScratch<T> *query_scratch) const
{
    if (query_scratch == nullptr)
        return false;
    const auto it = query_scratch->merit_dcache_member_packed_loc.find(node_id);
    if (it == query_scratch->merit_dcache_member_packed_loc.end())
        return false;
    query_scratch->merit_dcache_loc_cache[node_id] = it->second;
    if (_merit_span_adaptive)
    {
        MeritTimedSharedMutexGuard page_lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
        const uint32_t sector = static_cast<uint32_t>(it->second >> 32);
        if (merit_dyn_is_sector(sector))
        {
            const uint32_t page = sector - MERIT_DYN_SECTOR_BASE;
            if (page < _merit_dyn_page_secondary.size() && _merit_dyn_page_secondary[page])
            {
                const uint32_t sb = _merit_dyn_page_span_base[page];
                if (sb < _merit_dyn_physical_cap &&
                    query_scratch->merit_dyn_full_span.find(sb) != query_scratch->merit_dyn_full_span.end())
                    _merit_span_hits[sb].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    return true;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_pack_dyn_page_into_query_prefetch(uint32_t sector,
                                                                       SSDQueryScratch<T> *query_scratch) const
{
    if (query_scratch == nullptr || !merit_dyn_is_sector(sector) || _merit_dyn_members_per_page == 0)
        return;
    MeritTimedSharedMutexGuard page_lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
    const uint32_t page_idx = sector - MERIT_DYN_SECTOR_BASE;
    if (page_idx >= _merit_dyn_physical_cap || page_idx >= _merit_dyn_page_member_count.size())
        return;
    uint32_t base = page_idx;
    uint32_t span = 1;
    if (_merit_dcache_second_page && page_idx < _merit_dyn_page_span_len.size())
    {
        const uint8_t sl = _merit_dyn_page_span_len[page_idx];
        const uint32_t sb = page_idx < _merit_dyn_page_span_base.size() ? _merit_dyn_page_span_base[page_idx]
                                                                        : MERIT_DYN_INVALID_PAGE;
        if (sl > 1 && sb != MERIT_DYN_INVALID_PAGE &&
            static_cast<uint64_t>(sb) + sl <= _merit_dyn_physical_cap)
        {
            base = sb;
            span = sl;
        }
    }
    bool full = true;
    if (span > 1)
    {
        const bool known = query_scratch->merit_dyn_full_span.find(base) != query_scratch->merit_dyn_full_span.end();
        full = known || merit_span_should_read_full(base, span);
        if (full)
            query_scratch->merit_dyn_full_span.insert(base);
    }
    auto collect_page = [&](uint32_t page, std::vector<uint32_t> &members) {
        if (!merit_dyn_page_live(page))
            return;
        const size_t begin = static_cast<size_t>(page) * _merit_dyn_members_per_page;
        const size_t count = _merit_dyn_page_member_count[page];
        if (count == 0 || begin + count > _merit_dyn_page_members.size())
            return;
        members.assign(_merit_dyn_page_members.begin() + begin, _merit_dyn_page_members.begin() + begin + count);
    };
    auto pack_page = [&](uint32_t page, std::vector<uint32_t> &members) {
        if (!merit_dyn_page_live(page))
            return;
        const size_t begin = static_cast<size_t>(page) * _merit_dyn_members_per_page;
        const size_t count = _merit_dyn_page_member_count[page];
        if (count == 0 || begin + count > _merit_dyn_page_members.size())
            return;
        const uint32_t page_sector = MERIT_DYN_SECTOR_BASE + page;
        members.reserve(count);
        for (size_t mi = 0; mi < count; ++mi)
        {
            const uint32_t mid = _merit_dyn_page_members[begin + mi];
            if (mid == MERIT_DYN_INVALID_PAGE)
                continue;
            const uint64_t packed = (static_cast<uint64_t>(page_sector) << 32) | (static_cast<uint64_t>(1) << 16) |
                                    static_cast<uint64_t>(mi);
            query_scratch->merit_dcache_member_packed_loc[mid] = packed;
            members.push_back(mid);
        }
    };
    std::vector<std::vector<uint32_t>> packed(span);
    for (uint32_t i = 0; i < span; ++i)
    {
        if (full || base + i == page_idx)
            pack_page(base + i, packed[i]);
        else
            collect_page(base + i, packed[i]);
    }
    if (!_merit_dcache_query_patch || packed[0].empty())
        return;
    const uint32_t seed_id = packed[0].front();
    if (seed_id == MERIT_DYN_INVALID_PAGE)
        return;
    if (query_scratch->merit_dcache_seed_members.find(seed_id) != query_scratch->merit_dcache_seed_members.end())
        return;
    for (uint32_t i = 0; i < span; ++i)
    {
        if (!full && base + i != page_idx)
            continue;
        for (uint32_t mid : packed[i])
            query_scratch->merit_dcache_member_to_seed[mid] = seed_id;
    }
    query_scratch->merit_dcache_seed_members[seed_id] = std::move(packed[0]);
    std::vector<std::vector<uint32_t>> extra;
    for (uint32_t i = 1; i < span; ++i)
    {
        if (packed[i].size() >= 2)
            extra.push_back(std::move(packed[i]));
    }
    if (!extra.empty())
        query_scratch->merit_dcache_seed_extra[seed_id] = std::move(extra);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::merit_disk_cache_lookup_hit(uint32_t node_id,
                                                          SSDQueryScratch<T> *query_scratch) const
{
    if (_merit_dyn_enabled)
    {
        const bool lookup_enabled =
            query_scratch != nullptr
                ? query_scratch->merit_dcache_lookup_enabled
                : (_merit_dcache_net_gate_enabled
                       ? _merit_dcache_gate_active.load(std::memory_order_acquire)
                       : !_merit_mcache.maintenance_suspended());
        if (!lookup_enabled)
            return false;
        if (merit_dyn_query_local_dcache_hit(node_id, query_scratch))
            return true;
    }
    if (_merit_dyn_enabled || _merit_seed_first_lookup)
    {
        if (!_merit_dyn_member_bits.empty() && static_cast<uint64_t>(node_id) < _num_points)
        {
            const uint64_t word =
                _merit_dyn_member_bits[static_cast<size_t>(node_id >> 6)];
            if ((word & (1ULL << (node_id & 63u))) == 0)
                return false;
        }
        MeritDiskLoc loc;
        if (!merit_resolve_disk_cache_loc(node_id, query_scratch, loc))
            return false;
        if (query_scratch != nullptr)
        {
            const uint64_t packed =
                (static_cast<uint64_t>(loc.sector) << 32) |
                (static_cast<uint64_t>(loc.nsectors) << 16) |
                static_cast<uint64_t>(loc.slot);
            query_scratch->merit_dcache_loc_cache[node_id] = packed;
        }
        return true;
    }

    MeritTimedSharedMutexGuard lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
    const auto it = _merit_dc_map.find(node_id);
    return it != _merit_dc_map.end() && !it->second.empty();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::merit_get_expand_neighbors(uint32_t expand_id, char *node_disk_buf,
                                                         const uint32_t *&out_nbrs, uint64_t &out_nnbrs) const
{
    if (_merit_seed_only_expand)
    {
        const auto it = _merit_dc_seed_page_nbrs.find(expand_id);
        if (it != _merit_dc_seed_page_nbrs.end() && !it->second.empty())
        {
            out_nbrs = it->second.data();
            out_nnbrs = it->second.size();
            return;
        }
    }
    uint32_t *node_buf = reinterpret_cast<uint32_t *>(node_disk_buf + _disk_bytes_per_point);
    out_nnbrs = static_cast<uint64_t>(*node_buf);
    out_nbrs = node_buf + 1;
}

struct MeritSectorSpan
{
    uint32_t base = 0;
    uint16_t nsectors = 1;
};

static std::vector<MeritSectorSpan> coalesce_sectors_to_spans(std::vector<uint32_t> sectors)
{
    if (sectors.empty())
        return {};
    std::sort(sectors.begin(), sectors.end());
    sectors.erase(std::unique(sectors.begin(), sectors.end()), sectors.end());
    std::vector<MeritSectorSpan> spans;
    uint32_t span_base = sectors[0];
    uint32_t span_end = sectors[0] + 1;
    for (size_t i = 1; i < sectors.size(); ++i)
    {
        if (sectors[i] == span_end)
            span_end++;
        else
        {
            spans.push_back({span_base, static_cast<uint16_t>(span_end - span_base)});
            span_base = sectors[i];
            span_end = sectors[i] + 1;
        }
    }
    spans.push_back({span_base, static_cast<uint16_t>(span_end - span_base)});
    return spans;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::prepare_merit_disk_cache_io(
    const std::vector<uint32_t> &merit_ids, SSDQueryScratch<T> *query_scratch, char *sector_scratch,
    uint64_t &sector_scratch_idx, size_t num_sectors_per_node, std::vector<MeritReadPending> &pending,
    std::vector<AlignedRead> &merit_io, std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout_groups,
    QueryStats *stats, uint32_t &num_ios)
{
    pending.clear();
    merit_io.clear();
    disk_fanout_groups.clear();
    if (merit_ids.empty())
        return;

    const bool disable_multiread = (std::getenv("MERIT_DISABLE_MULTIREAD") != nullptr);
    const bool record_driven_seed_access =
        (std::getenv("MERIT_RECORD_DRIVEN_SEED_ACCESS") != nullptr &&
         std::strcmp(std::getenv("MERIT_RECORD_DRIVEN_SEED_ACCESS"), "0") != 0);
    const bool adaptive_parent_or_self =
        record_driven_seed_access && std::getenv("MERIT_ADAPTIVE_PARENT_OR_SELF") != nullptr &&
        std::strcmp(std::getenv("MERIT_ADAPTIVE_PARENT_OR_SELF"), "0") != 0;

    const size_t read_len = num_sectors_per_node * defaults::SECTOR_LEN;

    // Batch replica selection: greedy set-cover on sectors to minimize distinct reads.
    struct PendingChoice
    {
        uint32_t id = 0;
        size_t loc_idx = 0;
        MeritDiskLoc loc;
        uint32_t io_base_sector = 0;
        uint16_t io_nsectors = 1;
        bool dynamic_page_pinned = false;
    };
    std::vector<PendingChoice> choices;
    choices.reserve(merit_ids.size());
    std::unordered_set<uint32_t> chosen_ids;
    std::vector<uint32_t> unresolved;
    unresolved.reserve(merit_ids.size());

    if (_merit_seed_first_lookup)
    {
        const bool cache_replica_fallback =
            (std::getenv("MERIT_CACHE_REPLICA_FALLBACK") != nullptr &&
             std::strcmp(std::getenv("MERIT_CACHE_REPLICA_FALLBACK"), "0") != 0);
        const bool adaptive_extent =
            (std::getenv("MERIT_ADAPTIVE_EXTENT") != nullptr &&
             std::strcmp(std::getenv("MERIT_ADAPTIVE_EXTENT"), "0") != 0);
        const bool parent_single_page =
            (std::getenv("MERIT_PARENT_SINGLE_PAGE") != nullptr &&
             std::strcmp(std::getenv("MERIT_PARENT_SINGLE_PAGE"), "0") != 0);

        struct SeedFirstChoiceInput
        {
            uint32_t id = 0;
            uint32_t seed_id = 0;
            MeritDiskLoc loc{};
            bool replica_cache_only = false;
        };
        std::vector<SeedFirstChoiceInput> sf_inputs;
        sf_inputs.reserve(merit_ids.size());

        for (uint32_t id : merit_ids)
        {
            MeritDiskLoc loc;
            if (!merit_resolve_disk_cache_loc(id, query_scratch, loc) || !chosen_ids.insert(id).second)
                continue;

            uint32_t seed_id = id;
            if (!record_driven_seed_access && query_scratch != nullptr)
            {
                const auto parent_it = query_scratch->profile_parent.find(id);
                if (parent_it != query_scratch->profile_parent.end())
                    seed_id = parent_it->second;
            }

            MeritDiskLoc strict_loc;
            bool has_strict_loc = false;
            if (record_driven_seed_access)
            {
                const auto canonical_it = _merit_dc_seed_canonical_loc.find(id);
                if (canonical_it != _merit_dc_seed_canonical_loc.end())
                {
                    strict_loc = canonical_it->second;
                    has_strict_loc = true;
                }
            }
            else if (query_scratch != nullptr)
            {
                const auto parent_it = query_scratch->profile_parent.find(id);
                if (parent_it != query_scratch->profile_parent.end())
                {
                    const auto member_it = _merit_dc_seed_member_loc.find(parent_it->second);
                    if (member_it != _merit_dc_seed_member_loc.end())
                    {
                        const auto strict_it = member_it->second.find(id);
                        if (strict_it != member_it->second.end())
                        {
                            strict_loc = strict_it->second;
                            has_strict_loc = true;
                        }
                    }
                }
                else
                {
                    const auto seed_it = _merit_dc_seed_member_loc.find(id);
                    if (seed_it != _merit_dc_seed_member_loc.end())
                    {
                        const auto self_it = seed_it->second.find(id);
                        if (self_it != seed_it->second.end())
                        {
                            strict_loc = self_it->second;
                            has_strict_loc = true;
                        }
                    }
                }
            }

            const bool record_already_available =
                record_driven_seed_access && query_scratch != nullptr &&
                merit_loc_in_query_cache(query_scratch, loc.sector, loc.nsectors);
            const bool shared_child_only_loc = _merit_child_only_layout && !has_strict_loc;
            const bool selected_non_strict_loc =
                has_strict_loc &&
                (loc.sector != strict_loc.sector || loc.slot != strict_loc.slot ||
                 loc.nsectors != strict_loc.nsectors);
            const bool replica_cache_only =
                record_already_available || shared_child_only_loc ||
                ((cache_replica_fallback || _merit_child_only_layout) && selected_non_strict_loc);

            sf_inputs.push_back({id, seed_id, loc, replica_cache_only});
        }

        if (adaptive_parent_or_self && query_scratch != nullptr)
        {
            auto is_leaf_like_node = [&](uint32_t node_id) -> bool {
                const auto seed_it = _merit_dc_seed_member_loc.find(node_id);
                if (seed_it == _merit_dc_seed_member_loc.end())
                    return true;
                for (const auto &member : seed_it->second)
                {
                    if (member.first != node_id)
                        return false;
                }
                return true;
            };

            std::unordered_map<uint32_t, std::vector<size_t>> inputs_by_parent;
            for (size_t input_idx = 0; input_idx < sf_inputs.size(); ++input_idx)
            {
                const SeedFirstChoiceInput &input = sf_inputs[input_idx];
                if (input.replica_cache_only)
                    continue;
                const auto parent_it = query_scratch->profile_parent.find(input.id);
                if (parent_it == query_scratch->profile_parent.end())
                    continue;
                const auto members_it = _merit_dc_seed_member_loc.find(parent_it->second);
                if (members_it == _merit_dc_seed_member_loc.end() ||
                    members_it->second.find(input.id) == members_it->second.end())
                    continue;
                inputs_by_parent[parent_it->second].push_back(input_idx);
            }

            for (const auto &parent_group : inputs_by_parent)
            {
                const uint32_t parent_seed = parent_group.first;
                const std::vector<size_t> &input_indices = parent_group.second;

                const auto canonical_it = _merit_dc_seed_canonical_loc.find(parent_seed);
                const auto members_it = _merit_dc_seed_member_loc.find(parent_seed);
                if (canonical_it == _merit_dc_seed_canonical_loc.end() ||
                    members_it == _merit_dc_seed_member_loc.end())
                    continue;

                std::vector<size_t> leaf_indices;
                std::vector<size_t> internal_indices;
                leaf_indices.reserve(input_indices.size());
                internal_indices.reserve(input_indices.size());
                for (size_t input_idx : input_indices)
                {
                    if (is_leaf_like_node(sf_inputs[input_idx].id))
                        leaf_indices.push_back(input_idx);
                    else
                        internal_indices.push_back(input_idx);
                }

                auto assign_parent_extent = [&](const std::vector<size_t> &indices) {
                    for (size_t input_idx : indices)
                    {
                        SeedFirstChoiceInput &input = sf_inputs[input_idx];
                        const auto member_it = members_it->second.find(input.id);
                        if (member_it == members_it->second.end())
                            continue;
                        input.seed_id = parent_seed;
                        input.loc = member_it->second;
                        input.replica_cache_only = false;
                    }
                };

                if (!leaf_indices.empty())
                    assign_parent_extent(leaf_indices);

                if (internal_indices.size() < 2)
                    continue;

                std::unordered_set<uint64_t> uncached_self_reads;
                for (size_t input_idx : internal_indices)
                {
                    const MeritDiskLoc &self_loc = sf_inputs[input_idx].loc;
                    if (!merit_loc_in_query_cache(query_scratch, self_loc.sector, self_loc.nsectors))
                    {
                        const uint64_t read_key = (static_cast<uint64_t>(self_loc.sector) << 16) |
                                                  static_cast<uint64_t>(self_loc.nsectors);
                        uncached_self_reads.insert(read_key);
                    }
                }
                const size_t parent_reads =
                    merit_loc_in_query_cache(query_scratch, canonical_it->second.sector,
                                             canonical_it->second.nsectors)
                        ? 0
                        : 1;
                if (parent_reads >= uncached_self_reads.size())
                    continue;
                assign_parent_extent(internal_indices);
            }
        }

        std::unordered_map<uint32_t, std::vector<MeritSectorSpan>> seed_spans;
        if (adaptive_extent && !parent_single_page)
        {
            std::unordered_map<uint32_t, std::vector<uint32_t>> seed_sectors;
            for (const auto &in : sf_inputs)
            {
                if (!in.replica_cache_only)
                    seed_sectors[in.seed_id].push_back(in.loc.sector);
            }
            for (auto &kv : seed_sectors)
                seed_spans[kv.first] = coalesce_sectors_to_spans(std::move(kv.second));
        }

        for (const auto &in : sf_inputs)
        {
            const auto canonical_it = _merit_dc_seed_canonical_loc.find(in.seed_id);
            uint32_t io_base = in.loc.sector;
            uint16_t io_nsectors = in.loc.nsectors;

            if (!in.replica_cache_only && !parent_single_page)
            {
                if (adaptive_extent)
                {
                    const auto spans_it = seed_spans.find(in.seed_id);
                    if (spans_it != seed_spans.end() && !spans_it->second.empty())
                    {
                        if (spans_it->second.size() == 1)
                        {
                            io_base = spans_it->second[0].base;
                            io_nsectors = spans_it->second[0].nsectors;
                        }
                        else if (canonical_it != _merit_dc_seed_canonical_loc.end())
                        {
                            // Disjoint needed pages: one full-extent read beats many IOs.
                            io_base = canonical_it->second.sector;
                            io_nsectors = canonical_it->second.nsectors;
                        }
                    }
                }
                else if (canonical_it != _merit_dc_seed_canonical_loc.end())
                {
                    io_base = canonical_it->second.sector;
                    io_nsectors = canonical_it->second.nsectors;
                }
            }

            // Member loc must lie inside the planned IO window. Otherwise finalize
            // remaps unpack_base to loc.sector while the buffer still holds the
            // canonical extent → wrong page bytes (seen as recall collapse).
            if (in.loc.sector < io_base ||
                in.loc.sector >= io_base + static_cast<uint32_t>(io_nsectors))
            {
                io_base = in.loc.sector;
                io_nsectors = 1;
            }

            choices.push_back({in.id, 0, in.loc, io_base, io_nsectors});
        }
    }
    else
    {
        // Reuse locs discovered during beam lookup to avoid a second map walk.
        if (query_scratch != nullptr && !query_scratch->merit_dcache_loc_cache.empty())
        {
            for (uint32_t id : merit_ids)
            {
                const auto cit = query_scratch->merit_dcache_loc_cache.find(id);
                if (cit == query_scratch->merit_dcache_loc_cache.end() || !chosen_ids.insert(id).second)
                    continue;
                MeritDiskLoc loc;
                loc.sector = static_cast<uint32_t>(cit->second >> 32);
                loc.nsectors = static_cast<uint16_t>((cit->second >> 16) & 0xffffu);
                loc.slot = static_cast<uint16_t>(cit->second & 0xffffu);
                if (!merit_dyn_is_sector(loc.sector))
                    continue;
                choices.push_back({id, 0, loc, loc.sector, loc.nsectors});
            }
            for (uint32_t id : merit_ids)
            {
                if (chosen_ids.find(id) == chosen_ids.end())
                    unresolved.push_back(id);
            }
        }
        else
            unresolved.assign(merit_ids.begin(), merit_ids.end());
    }

    std::unordered_set<uint32_t> need_ids(unresolved.begin(), unresolved.end());

    // Under freeze, page/loc tables are immutable: skip shared lock when every
    // merit id already has a cached loc (common after lookup).
    struct OptionalDynSharedLock
    {
        MeritTimedSharedMutexGuard *guard = nullptr;
        alignas(MeritTimedSharedMutexGuard) unsigned char storage[sizeof(MeritTimedSharedMutexGuard)];
        OptionalDynSharedLock(std::shared_mutex &mu, bool take)
        {
            if (take)
                guard = new (storage) MeritTimedSharedMutexGuard(mu, MeritLockKind::DynamicShared, false);
        }
        ~OptionalDynSharedLock()
        {
            if (guard)
                guard->~MeritTimedSharedMutexGuard();
        }
        void unlock()
        {
            if (guard)
            {
                guard->~MeritTimedSharedMutexGuard();
                guard = nullptr;
            }
        }
    } loc_lock(_merit_dyn_mu, !_merit_freeze_maintenance || !need_ids.empty());
    while (!need_ids.empty())
    {
        std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, size_t>>> sector_hits;
        for (uint32_t id : need_ids)
        {
            merit_dc_for_each_loc_unlocked(id, [&](const MeritDiskLoc &loc, size_t li) {
                sector_hits[loc.sector].emplace_back(id, li);
            });
        }
        if (sector_hits.empty())
            break;

        uint32_t best_sector = 0;
        size_t best_cover = 0;
        int best_cache_bonus = -1;
        for (const auto &kv : sector_hits)
        {
            const int cache_bonus =
                (_query_sector_cache_enabled &&
                 query_scratch->merit_sector_cache.find(kv.first) != query_scratch->merit_sector_cache.end())
                    ? 1
                    : 0;
            if (kv.second.size() > best_cover ||
                (kv.second.size() == best_cover && cache_bonus > best_cache_bonus))
            {
                best_cover = kv.second.size();
                best_cache_bonus = cache_bonus;
                best_sector = kv.first;
            }
        }

        const auto &hit = sector_hits[best_sector];
        const size_t choices_before = choices.size();
        for (const auto &id_loc : hit)
        {
            if (need_ids.count(id_loc.first) == 0)
                continue;
            MeritDiskLoc selected_loc;
            if (!merit_dc_loc_at_unlocked(id_loc.first, id_loc.second, selected_loc))
                continue;
            PendingChoice pc;
            pc.id = id_loc.first;
            pc.loc_idx = id_loc.second;
            pc.loc = selected_loc;
            pc.io_base_sector = pc.loc.sector;
            pc.io_nsectors = pc.loc.nsectors;
            if (!chosen_ids.insert(pc.id).second)
                continue;
            choices.push_back(pc);
            need_ids.erase(id_loc.first);
        }

        for (size_t ci = choices_before; ci < choices.size(); ++ci)
        {
            const PendingChoice &anchor = choices[ci];
            if (disable_multiread || anchor.loc.nsectors <= 1)
                continue;
            const uint32_t span_base = anchor.loc.sector;
            const uint32_t span_end = span_base + anchor.loc.nsectors;
            for (auto id_it = need_ids.begin(); id_it != need_ids.end();)
            {
                const uint32_t id = *id_it;
                const size_t loc_count = merit_dc_loc_count_unlocked(id);
                if (loc_count == 0)
                {
                    ++id_it;
                    continue;
                }
                size_t cover_li = 0;
                bool covered = false;
                MeritDiskLoc cover_loc;
                merit_dc_for_each_loc_unlocked(id, [&](const MeritDiskLoc &loc, size_t li) {
                    if (!covered && loc.sector >= span_base && loc.sector < span_end)
                    {
                        cover_li = li;
                        cover_loc = loc;
                        covered = true;
                    }
                });
                if (!covered)
                {
                    ++id_it;
                    continue;
                }
                PendingChoice pc;
                pc.id = id;
                pc.loc_idx = cover_li;
                pc.loc = cover_loc;
                pc.io_base_sector = span_base;
                pc.io_nsectors = anchor.loc.nsectors;
                if (!chosen_ids.insert(pc.id).second)
                {
                    id_it = need_ids.erase(id_it);
                    continue;
                }
                choices.push_back(pc);
                if (stats != nullptr)
                    stats->n_merit_setcover_grouped++;
                id_it = need_ids.erase(id_it);
            }
        }
    }
    choices.erase(std::remove_if(choices.begin(), choices.end(),
                                 [&](PendingChoice &pc) {
                                     if (!merit_dyn_is_sector(pc.loc.sector))
                                         return false;
                                     const uint32_t page_idx = pc.loc.sector - MERIT_DYN_SECTOR_BASE;
                                     if (page_idx >= _merit_dyn_physical_cap ||
                                         !merit_dyn_page_live(page_idx))
                                         return true;
                                     if (!merit_dc_loc_contains_unlocked(pc.id, pc.loc))
                                         return true;
                                     // Freeze keeps pages live for the whole run; skip pin/unpin.
                                     if (!_merit_freeze_maintenance &&
                                         !merit_loc_in_query_cache(query_scratch, pc.loc.sector, 1))
                                     {
                                         _merit_dyn_page_readers[page_idx].fetch_add(1, std::memory_order_acq_rel);
                                         query_scratch->merit_dyn_pinned_sectors.push_back(pc.loc.sector);
                                         pc.dynamic_page_pinned = true;
                                     }
                                     return false;
                                 }),
                  choices.end());
    if (_merit_dcache_second_page)
    {
        for (PendingChoice &pc : choices)
        {
            if (!merit_dyn_is_sector(pc.loc.sector))
                continue;
            const uint32_t page_idx = pc.loc.sector - MERIT_DYN_SECTOR_BASE;
            if (page_idx >= _merit_dyn_page_span_len.size())
                continue;
            const uint8_t sl = _merit_dyn_page_span_len[page_idx];
            const uint32_t sb = page_idx < _merit_dyn_page_span_base.size() ? _merit_dyn_page_span_base[page_idx]
                                                                            : MERIT_DYN_INVALID_PAGE;
            if (sl <= 1 || sb == MERIT_DYN_INVALID_PAGE ||
                static_cast<uint64_t>(sb) + sl > _merit_dyn_physical_cap)
                continue;
            if (_merit_span_adaptive &&
                query_scratch->merit_dyn_full_span.find(sb) == query_scratch->merit_dyn_full_span.end())
                continue;
            pc.io_base_sector = MERIT_DYN_SECTOR_BASE + sb;
            pc.io_nsectors = sl;
            pc.loc.nsectors = sl;
            for (uint32_t extra = 0; extra < sl; ++extra)
            {
                const uint32_t ep = sb + extra;
                if (ep == page_idx || !merit_dyn_page_live(ep))
                    continue;
                const uint32_t es = MERIT_DYN_SECTOR_BASE + ep;
                if (!_merit_freeze_maintenance && !merit_loc_in_query_cache(query_scratch, es, 1))
                {
                    _merit_dyn_page_readers[ep].fetch_add(1, std::memory_order_acq_rel);
                    query_scratch->merit_dyn_pinned_sectors.push_back(es);
                }
            }
        }
    }
    loc_lock.unlock();

    struct MultireadSpan
    {
        uint32_t base = 0;
        uint16_t nsectors = 1;
    };
    std::vector<MultireadSpan> multiread_spans;
    multiread_spans.reserve(choices.size());
    if (!disable_multiread)
    {
        for (const PendingChoice &pc : choices)
        {
            if (pc.io_nsectors > 1)
                multiread_spans.push_back({pc.io_base_sector, pc.io_nsectors});
        }
    }

    auto multiread_root_for_sector = [&](uint32_t sector) -> uint32_t {
        for (const MultireadSpan &sp : multiread_spans)
        {
            if (sector >= sp.base && sector < sp.base + sp.nsectors)
                return sp.base;
        }
        return sector;
    };

    pending.reserve(choices.size());
    for (const PendingChoice &pc : choices)
    {
        const MeritDiskLoc loc = pc.loc;
        const bool dynamic_page = merit_dyn_is_sector(loc.sector);
        const bool pair_read = dynamic_page && pc.io_nsectors > 1;
        const uint32_t io_base =
            pair_read ? pc.io_base_sector
                      : (dynamic_page || disable_multiread ? loc.sector
                                                           : multiread_root_for_sector(pc.io_base_sector));
        const uint16_t io_nsectors =
            pair_read ? pc.io_nsectors
                      : (dynamic_page || disable_multiread ? uint16_t{1} : pc.io_nsectors);
        query_scratch->read_merit_disk_cache_sectors.insert(loc.sector);
        if (pair_read)
            query_scratch->read_merit_disk_cache_sectors.insert(io_base);
        pending.push_back({pc.id, nullptr, loc, io_base, io_nsectors, pc.dynamic_page_pinned});
        const size_t pidx = pending.size() - 1;
        if (stats != nullptr && merit_dyn_is_sector(loc.sector))
            stats->n_merit_dyn_hits++;

        bool served = false;
        if (!served && _query_sector_cache_enabled)
        {
            const uint32_t cache_base = pair_read ? io_base : pc.io_base_sector;
            const uint16_t cache_nsectors =
                pair_read ? io_nsectors : (disable_multiread ? uint16_t{1} : pc.io_nsectors);
            if (merit_loc_in_query_cache(query_scratch, cache_base, cache_nsectors))
            {
                char *read_buf = sector_scratch + sector_scratch_idx * defaults::SECTOR_LEN;
                sector_scratch_idx += cache_nsectors;
                pending[pidx].sec_buf = read_buf;
                pending[pidx].io_base_sector = cache_base;
                for (uint16_t si = 0; si < cache_nsectors; ++si)
                {
                    const auto cit = query_scratch->merit_sector_cache.find(cache_base + si);
                    memcpy(read_buf + static_cast<size_t>(si) * defaults::SECTOR_LEN, cit->second.data(),
                           defaults::SECTOR_LEN);
                }
                if (stats != nullptr)
                    stats->n_sector_cache_hits++;
                served = true;
            }
        }
        if (!served)
        {
            const uint32_t group_sec = io_base;
            if (group_sec != loc.sector && stats != nullptr)
                stats->n_merit_io_avoided++;
            disk_fanout_groups[group_sec].push_back(pidx);
            if (stats != nullptr)
            {
                stats->n_merit_dc_hits++;
                stats->n_4k++;
                stats->n_ios++;
            }
            num_ios++;
        }
    }

    for (auto &kv : disk_fanout_groups)
    {
        const uint32_t disk_cache_sec = kv.first;
        std::vector<size_t> &indices = kv.second;
        if (indices.empty())
            continue;

        size_t root_idx = indices[0];
        uint16_t group_nsectors = 1;
        if (disable_multiread)
            group_nsectors = 1;
        else
        {
            for (size_t idx : indices)
                group_nsectors = std::max(group_nsectors, pending[idx].loc.nsectors);
            for (const MultireadSpan &sp : multiread_spans)
            {
                if (sp.base == disk_cache_sec)
                {
                    group_nsectors = std::max(group_nsectors, sp.nsectors);
                    break;
                }
            }
        }
        if (!disable_multiread)
        {
            for (size_t idx : indices)
            {
                if (pending[idx].loc.sector == disk_cache_sec && pending[idx].loc.nsectors == group_nsectors)
                {
                    root_idx = idx;
                    break;
                }
            }
        }
        if (root_idx != indices[0])
            std::swap(indices[0], root_idx);

        const uint64_t buf_sectors =
            disable_multiread ? static_cast<uint64_t>(num_sectors_per_node)
                              : std::max(static_cast<uint64_t>(group_nsectors),
                                         static_cast<uint64_t>(num_sectors_per_node));
        char *read_buf = sector_scratch + sector_scratch_idx * defaults::SECTOR_LEN;
        sector_scratch_idx += buf_sectors;
        pending[indices[0]].sec_buf = read_buf;
        for (size_t idx : indices)
        {
            pending[idx].io_base_sector =
                disable_multiread ? pending[idx].loc.sector : disk_cache_sec;
            pending[idx].io_nsectors = group_nsectors;
        }
        const uint64_t byte_len = static_cast<uint64_t>(group_nsectors) * defaults::SECTOR_LEN;
        const uint64_t off = merit_dyn_is_sector(disk_cache_sec)
                                 ? static_cast<uint64_t>(disk_cache_sec - MERIT_DYN_SECTOR_BASE) *
                                       defaults::SECTOR_LEN
                                 : (_merit_unified_disk ? _merit_region_byte_offset : 0) +
                                       static_cast<uint64_t>(disk_cache_sec) * defaults::SECTOR_LEN;
        merit_io.emplace_back(off, byte_len, read_buf);
        if (merit_dyn_is_sector(disk_cache_sec))
        {
            query_scratch->merit_dyn_physical_reads += group_nsectors;
            if (group_nsectors > 1)
                _merit_dcache_pair_ios.fetch_add(1, std::memory_order_relaxed);
        }
        if (stats != nullptr)
        {
            stats->n_disk_reads++;
            if (merit_dyn_is_sector(disk_cache_sec))
                stats->n_merit_dyn_disk_reads++;
            if (group_nsectors > 1)
                stats->n_merit_multiread_ios++;
            if (_record_hop_frontier && !stats->hop_frontier_trace.empty())
                stats->hop_frontier_trace.back().physical_reads.push_back(
                    {true, disk_cache_sec, group_nsectors});
        }
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::complete_merit_disk_cache_io(
    SSDQueryScratch<T> *query_scratch, std::vector<MeritReadPending> &pending,
    const std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout_groups)
{
    for (const auto &kv : disk_fanout_groups)
    {
        const std::vector<size_t> &indices = kv.second;
        if (indices.empty())
            continue;
        const char *primary = pending[indices[0]].sec_buf;
        if (_query_sector_cache_enabled && pending[indices[0]].io_nsectors == 1)
            memcpy(query_scratch->merit_sector_cache[kv.first].data(), primary, defaults::SECTOR_LEN);
        else if (_query_sector_cache_enabled && pending[indices[0]].io_nsectors > 1)
        {
            for (uint16_t si = 0; si < pending[indices[0]].io_nsectors; ++si)
                memcpy(query_scratch->merit_sector_cache[kv.first + static_cast<uint32_t>(si)].data(),
                       primary + static_cast<size_t>(si) * defaults::SECTOR_LEN, defaults::SECTOR_LEN);
        }
        for (size_t idx : indices)
            pending[idx].sec_buf = const_cast<char *>(primary);
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::finalize_merit_pending_nodes(const std::vector<MeritReadPending> &pending,
                                                           const std::vector<uint32_t> &merit_order,
                                                           SSDQueryScratch<T> *query_scratch, char *sector_scratch,
                                                           uint64_t &sector_scratch_idx,
                                                           std::vector<std::pair<uint32_t, char *>> &frontier_nhoods,
                                                           QueryStats *stats)
{
    std::unordered_map<uint32_t, char *> ready;
    ready.reserve(pending.size());
    for (const auto &mp : pending)
    {
        char *sec_buf = mp.sec_buf;
        uint32_t io_base = mp.io_base_sector;
        if (sec_buf == nullptr && _query_sector_cache_enabled)
        {
            const auto cit = query_scratch->merit_sector_cache.find(mp.loc.sector);
            if (cit != query_scratch->merit_sector_cache.end())
            {
                sec_buf = copy_to_sector_scratch(sector_scratch, sector_scratch_idx, cit->second.data(),
                                                 defaults::SECTOR_LEN);
                io_base = mp.loc.sector;
            }
        }
        if (sec_buf == nullptr)
            continue;

        char *out_buf = sec_buf;
        if (_nnodes_per_sector > 0)
        {
            const uint64_t base_slot = mp.id % _nnodes_per_sector;
            uint32_t unpack_base = io_base;
            if (mp.loc.sector < io_base || mp.loc.sector >= io_base + static_cast<uint32_t>(mp.io_nsectors))
                unpack_base = mp.loc.sector;
            const uint64_t sector_off =
                (mp.loc.sector >= unpack_base)
                    ? static_cast<uint64_t>(mp.loc.sector - unpack_base) * defaults::SECTOR_LEN
                    : 0;
            // Merit pages use loc.slot (dense page layout); normalize to id % nps for offset_to_node.
            out_buf = sector_scratch + sector_scratch_idx * defaults::SECTOR_LEN;
            sector_scratch_idx++;
            memset(out_buf, 0, defaults::SECTOR_LEN);
            const char *packed = sec_buf + sector_off + static_cast<uint64_t>(mp.loc.slot) * _max_node_len;
            memcpy(out_buf + base_slot * _max_node_len, packed, _max_node_len);
        }
        ready[mp.id] = out_buf;
        if (merit_dyn_is_sector(mp.loc.sector))
        {
            query_scratch->merit_avoided_base_sectors.insert(get_node_sector(mp.id));
            query_scratch->merit_dyn_served_nodes++;
            if (_merit_real_io_coaccess)
                query_scratch->merit_storage_node_order.push_back(mp.id);
            // Prefetch buffer: register all members on this served page for same-query hits.
            if (merit_dyn_is_sector(mp.loc.sector) && _merit_dyn_members_per_page > 0)
            {
                MeritTimedSharedMutexGuard page_lock(_merit_dyn_mu, MeritLockKind::DynamicShared, false);
                const uint32_t page_idx = mp.loc.sector - MERIT_DYN_SECTOR_BASE;
                if (page_idx < _merit_dyn_physical_cap)
                {
                    const size_t begin = static_cast<size_t>(page_idx) * _merit_dyn_members_per_page;
                    const size_t count = _merit_dyn_page_member_count[page_idx];
                    for (size_t mi = 0; mi < count; ++mi)
                    {
                        const uint32_t mid = _merit_dyn_page_members[begin + mi];
                        if (mid == MERIT_DYN_INVALID_PAGE)
                            continue;
                        const uint64_t packed = (static_cast<uint64_t>(mp.loc.sector) << 32) |
                                               (static_cast<uint64_t>(1) << 16) |
                                               static_cast<uint64_t>(mi);
                        query_scratch->merit_dcache_member_packed_loc[mid] = packed;
                    }
                }
            }
        }
    }

    for (uint32_t id : merit_order)
    {
        const auto it = ready.find(id);
        if (it == ready.end())
        {
            if (stats != nullptr)
                stats->n_merit_finalize_skipped++;
            continue;
        }
        frontier_nhoods.emplace_back(id, it->second);
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_access_profile(bool enable)
{
    _hotness_profiler.set_enabled(enable);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::reset_access_profile()
{
    _hotness_profiler.reset();
    _hotness_profiler.set_enabled(true);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_query_sector_cache(bool enable)
{
    _query_sector_cache_enabled = enable;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_base_frontier_recording(bool enable)
{
    _record_base_frontier = enable;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_hop_frontier_recording(bool enable)
{
    _record_hop_frontier = enable;
}

template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::save_access_profile(const std::string &output_prefix) const
{
    return _hotness_profiler.save(output_prefix);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::print_access_profile_cdf() const
{
    _hotness_profiler.print_cdf_summary();
}

// instantiations
template class PQFlashIndex<uint8_t>;
template class PQFlashIndex<int8_t>;
template class PQFlashIndex<float>;
template class PQFlashIndex<uint8_t, uint16_t>;
template class PQFlashIndex<int8_t, uint16_t>;
template class PQFlashIndex<float, uint16_t>;

} // namespace diskann
