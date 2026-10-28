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
#include <iomanip>
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
#include "neighborcache_lock_metrics.h"

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
struct NeighborCacheDemandProbe
{
    bool attempted = false;
    bool hit = false;
};
thread_local NeighborCacheDemandProbe *g_neighborcache_demand_probe = nullptr;
struct NeighborCacheDemandProbeScope
{
    NeighborCacheDemandProbe *previous;
    explicit NeighborCacheDemandProbeScope(NeighborCacheDemandProbe &probe)
        : previous(g_neighborcache_demand_probe)
    {
        g_neighborcache_demand_probe = &probe;
    }
    ~NeighborCacheDemandProbeScope() { g_neighborcache_demand_probe = previous; }
};
inline bool neighborcache_demand_probe_hit()
{
    if (g_neighborcache_demand_probe != nullptr)
        g_neighborcache_demand_probe->hit = true;
    return true;
}
inline size_t neighborcache_thread_shard()
{
    static std::atomic<size_t> next{0};
    thread_local size_t id = next.fetch_add(1, std::memory_order_relaxed);
    return id;
}
class NeighborCacheShardedCounter
{
  public:
    static constexpr size_t kShards = 64;
    NeighborCacheShardedCounter &operator++()
    {
        fetch_add(1);
        return *this;
    }
    NeighborCacheShardedCounter &operator++(int)
    {
        fetch_add(1);
        return *this;
    }
    NeighborCacheShardedCounter &operator+=(uint64_t v)
    {
        fetch_add(v);
        return *this;
    }
    void fetch_add(uint64_t v, std::memory_order = std::memory_order_relaxed)
    {
        _slots[neighborcache_thread_shard() & (kShards - 1)].v.fetch_add(v, std::memory_order_relaxed);
    }
    NeighborCacheShardedCounter &operator=(uint64_t v)
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
// NEIGHBORCACHE_MISS_DIAG=1: classify static D-cache lookup misses (each one becomes a base read).
static const bool g_neighborcache_miss_diag = [] {
    const char *e = std::getenv("NEIGHBORCACHE_MISS_DIAG");
    return e != nullptr && std::strcmp(e, "0") != 0;
}();
static std::atomic<uint64_t> g_neighborcache_miss_not_in_dcache{0};
static std::atomic<uint64_t> g_neighborcache_miss_no_parent{0};
static std::atomic<uint64_t> g_neighborcache_miss_parent_ncache{0};
static std::atomic<uint64_t> g_neighborcache_miss_parent_not_seed{0};
static std::atomic<uint64_t> g_neighborcache_miss_parent_lacks{0};
struct NeighborCacheDynProbe
{
    NeighborCacheShardedCounter ncache_hit;
    NeighborCacheShardedCounter ncache_miss;
    NeighborCacheShardedCounter resolve_call;
    NeighborCacheShardedCounter parent_hit;
    NeighborCacheShardedCounter parent_absent;
    NeighborCacheShardedCounter parent_no_page;
    NeighborCacheShardedCounter parent_not_member;
    NeighborCacheShardedCounter self_hit;
    NeighborCacheShardedCounter map_hit;
    NeighborCacheShardedCounter resolve_miss;
    NeighborCacheShardedCounter miss_but_in_map;
    NeighborCacheShardedCounter qualify_ok;
    NeighborCacheShardedCounter qualify_no_ncache;
    NeighborCacheShardedCounter qualify_no_meta;
    NeighborCacheShardedCounter qualify_heat;
    NeighborCacheShardedCounter qualify_smin;
    NeighborCacheShardedCounter qualify_half;
    NeighborCacheShardedCounter pair_set;
    NeighborCacheShardedCounter pair_skip_max_on_disk;
    NeighborCacheShardedCounter pair_skip_count;
    NeighborCacheShardedCounter write_trig;
    NeighborCacheShardedCounter timeout_trig;
    NeighborCacheShardedCounter timeout_ok;
    NeighborCacheShardedCounter write_ok;
    NeighborCacheShardedCounter delete_ok;

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
NeighborCacheDynProbe g_dyn_probe;

// Per successful seed-page write: how many of that seed's neighbors are in the
// N-cache at commit time. Does not change which pages are written.
constexpr int kNeighborCacheSeedNbrHist = 129;
std::atomic<uint64_t> g_neighborcache_seed_writes{0};
std::mutex g_neighborcache_seed_id_mu;
std::vector<uint32_t> g_neighborcache_seed_ids;

static bool neighborcache_log_seed_ids()
{
    static const int on = []() {
        const char *p = std::getenv("NEIGHBORCACHE_SEED_WRITE_IDS");
        return (p != nullptr && p[0] != '\0') ? 1 : 0;
    }();
    return on == 1;
}

void neighborcache_seed_write_note_id(uint32_t seed_id)
{
    if (!neighborcache_log_seed_ids())
        return;
    std::lock_guard<std::mutex> lock(g_neighborcache_seed_id_mu);
    g_neighborcache_seed_ids.push_back(seed_id);
}
std::atomic<uint64_t> g_neighborcache_seed_graph_sum{0};
std::atomic<uint64_t> g_neighborcache_seed_followed_sum{0};
std::atomic<uint64_t> g_neighborcache_seed_sig_sum{0};
std::atomic<uint64_t> g_neighborcache_seed_packed_sum{0};
std::atomic<uint64_t> g_neighborcache_seed_graph_hist[kNeighborCacheSeedNbrHist];
std::atomic<uint64_t> g_neighborcache_seed_followed_hist[kNeighborCacheSeedNbrHist];

void neighborcache_seed_write_reset()
{
    g_neighborcache_seed_writes.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(g_neighborcache_seed_id_mu);
        g_neighborcache_seed_ids.clear();
    }
    g_neighborcache_seed_graph_sum.store(0, std::memory_order_relaxed);
    g_neighborcache_seed_followed_sum.store(0, std::memory_order_relaxed);
    g_neighborcache_seed_sig_sum.store(0, std::memory_order_relaxed);
    g_neighborcache_seed_packed_sum.store(0, std::memory_order_relaxed);
    for (int i = 0; i < kNeighborCacheSeedNbrHist; ++i)
    {
        g_neighborcache_seed_graph_hist[i].store(0, std::memory_order_relaxed);
        g_neighborcache_seed_followed_hist[i].store(0, std::memory_order_relaxed);
    }
}

void neighborcache_seed_write_tally(uint32_t graph_in, uint32_t followed_in, uint32_t sig, uint32_t packed)
{
    auto bump = [](std::atomic<uint64_t> *hist, uint32_t v) {
        if (v >= static_cast<uint32_t>(kNeighborCacheSeedNbrHist))
            v = static_cast<uint32_t>(kNeighborCacheSeedNbrHist - 1);
        hist[v].fetch_add(1, std::memory_order_relaxed);
    };
    bump(g_neighborcache_seed_graph_hist, graph_in);
    bump(g_neighborcache_seed_followed_hist, followed_in);
    g_neighborcache_seed_graph_sum.fetch_add(graph_in, std::memory_order_relaxed);
    g_neighborcache_seed_followed_sum.fetch_add(followed_in, std::memory_order_relaxed);
    g_neighborcache_seed_sig_sum.fetch_add(sig, std::memory_order_relaxed);
    g_neighborcache_seed_packed_sum.fetch_add(packed, std::memory_order_relaxed);
    g_neighborcache_seed_writes.fetch_add(1, std::memory_order_relaxed);
}

double neighborcache_seed_hist_median(const std::atomic<uint64_t> *hist, uint64_t total)
{
    if (total == 0)
        return 0.0;
    const uint64_t left = (total - 1) / 2;
    const uint64_t right = total / 2;
    uint64_t acc = 0;
    int lo = 0;
    int hi = 0;
    bool got_lo = false;
    for (int i = 0; i < kNeighborCacheSeedNbrHist; ++i)
    {
        const uint64_t c = hist[i].load(std::memory_order_relaxed);
        if (!got_lo && acc + c > left)
        {
            lo = i;
            got_lo = true;
        }
        if (acc + c > right)
        {
            hi = i;
            return 0.5 * static_cast<double>(lo + hi);
        }
        acc += c;
    }
    return static_cast<double>(lo);
}

template <typename Pool>
bool neighborcache_seed_write_count(const Pool *pool, uint32_t seed_id, const std::vector<uint32_t> &member_ids,
                            const std::vector<uint32_t> &graph_nbrs,
                            const std::vector<std::pair<uint32_t, uint8_t>> &significant, uint32_t &graph_in,
                            uint32_t &followed_in, uint32_t &sig, uint32_t &packed)
{
    graph_in = 0;
    followed_in = 0;
    sig = 0;
    packed = 0;
    if (member_ids.empty() || member_ids.front() != seed_id || graph_nbrs.empty())
        return false;
    sig = static_cast<uint32_t>(significant.size());
    packed = static_cast<uint32_t>(member_ids.size() - 1);
    if (pool != nullptr && pool->active())
    {
        for (uint32_t id : graph_nbrs)
            if (id != seed_id && pool->contains(id))
                ++graph_in;
        for (const auto &sc : significant)
            if (sc.first != seed_id && pool->contains(sc.first))
                ++followed_in;
    }
    return true;
}

// NEIGHBORCACHE_BG_CPUS="10-18" pins each background thread to one core in order.
inline void neighborcache_pin_background_thread()
{
#ifndef _WINDOWS
    const char *spec = std::getenv("NEIGHBORCACHE_BG_CPUS");
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
    static std::atomic<int> next{0};
    const int cpu = cpus[static_cast<size_t>(next.fetch_add(1, std::memory_order_relaxed)) % cpus.size()];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0)
        diskann::cout << "NEIGHBORCACHE pin background cpu=" << cpu << std::endl;
#endif
}

inline bool &neighborcache_is_refresh_thread()
{
    thread_local bool is_refresh = false;
    return is_refresh;
}

inline bool neighborcache_debug_verify_enabled()
{
    static const bool on = std::getenv("NEIGHBORCACHE_DEBUG_VERIFY") != nullptr;
    return on;
}

inline uint64_t neighborcache_disk_append_cap_for_full_coverage(const std::vector<uint64_t> &node_expand,
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

inline void write_neighborcache_dc_seed_tail(char *slot_dst, uint64_t max_node_len, uint64_t disk_bytes_per_point,
                                     uint32_t nnbrs, uint8_t total_pages)
{
    const uint64_t used = disk_bytes_per_point + static_cast<uint64_t>(nnbrs + 1) * sizeof(uint32_t);
    if (used + 4 > max_node_len)
        return;
    slot_dst[used] = static_cast<char>(diskann::NEIGHBORCACHE_DC_SEED_MAGIC);
    slot_dst[used + 1] = static_cast<char>(total_pages & diskann::NEIGHBORCACHE_DC_FLAG_TOTAL_PAGES_MASK);
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
inline void neighborcache_fallback_unserved_to_base(const std::vector<uint32_t> &neighborcache_ids,
                                            const std::vector<PendingT> &pending,
                                            const std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout,
                                            std::vector<uint32_t> &frontier)
{
    tsl::robin_set<uint32_t> served;
    served.reserve(pending.size());
    for (const auto &p : pending)
    {
        if (p.sec_buf != nullptr || p.node_ptr != nullptr)
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
    for (uint32_t id : neighborcache_ids)
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

template <typename NeighborCacheDiskLocT>
inline void save_seed_member_locs_sidecar(
    const std::string &path,
    const tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, NeighborCacheDiskLocT>> &seed_member_loc)
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
            const NeighborCacheDiskLocT &loc = member.second;
            out.write(reinterpret_cast<const char *>(&seed_id), sizeof(seed_id));
            out.write(reinterpret_cast<const char *>(&member_id), sizeof(member_id));
            out.write(reinterpret_cast<const char *>(&loc.sector), sizeof(loc.sector));
            out.write(reinterpret_cast<const char *>(&loc.slot), sizeof(loc.slot));
            out.write(reinterpret_cast<const char *>(&loc.nsectors), sizeof(loc.nsectors));
        }
    }
}

template <typename NeighborCacheDiskLocT>
inline int load_seed_member_locs_sidecar(
    const std::string &path, tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, NeighborCacheDiskLocT>> &seed_member_loc,
    tsl::robin_map<uint32_t, NeighborCacheDiskLocT> &seed_canonical_loc)
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
        NeighborCacheDiskLocT loc;
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

static std::vector<std::pair<uint32_t, uint32_t>> load_neighborcache_copack_pairs()
{
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    const char *path = std::getenv("NEIGHBORCACHE_COPACK_PAIRS_FILE");
    if (path == nullptr || path[0] == '\0')
        return pairs;
    std::ifstream in(path);
    if (!in.is_open())
    {
        diskann::cerr << "NEIGHBORCACHE copack: failed to open " << path << std::endl;
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
    diskann::cout << "NEIGHBORCACHE copack: loaded " << pairs.size() << " seed pairs from " << path << std::endl;
    return pairs;
}

struct NeighborCacheHopIoProbe
{
    std::atomic<uint64_t> batches[2]{};
    std::atomic<uint64_t> batch_ns[2]{};
    std::atomic<uint64_t> reads[2]{};
    std::atomic<uint64_t> bytes[2]{};
    std::atomic<uint64_t> multi_reads{0};
    std::atomic<uint64_t> multi_bytes{0};
};
inline NeighborCacheHopIoProbe &neighborcache_hop_io_probe()
{
    static NeighborCacheHopIoProbe probe;
    return probe;
}

inline void neighborcache_record_hop_io(const std::vector<AlignedRead> &neighborcache_reqs, const std::vector<AlignedRead> &base_reqs,
                                float us)
{
    uint64_t bytes = 0, multi = 0, multi_bytes = 0;
    for (const auto &r : neighborcache_reqs)
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
    auto &p = neighborcache_hop_io_probe();
    const int k = multi > 0 ? 1 : 0;
    p.batches[k].fetch_add(1, std::memory_order_relaxed);
    p.batch_ns[k].fetch_add(static_cast<uint64_t>(us * 1000.0f), std::memory_order_relaxed);
    p.reads[k].fetch_add(neighborcache_reqs.size() + base_reqs.size(), std::memory_order_relaxed);
    p.bytes[k].fetch_add(bytes, std::memory_order_relaxed);
    p.multi_reads.fetch_add(multi, std::memory_order_relaxed);
    p.multi_bytes.fetch_add(multi_bytes, std::memory_order_relaxed);
}

inline float issue_neighborcache_and_base_disk_reads(std::shared_ptr<AlignedFileReader> &base_reader,
                                            std::shared_ptr<AlignedFileReader> &neighborcache_reader, IOContext &base_ctx,
                                            std::vector<AlignedRead> &neighborcache_reqs, std::vector<AlignedRead> &base_reqs,
                                            diskann::Timer &io_timer, diskann::QueryStats *stats)
{
    if (neighborcache_reqs.empty() && base_reqs.empty())
        return 0.f;

    io_timer.reset();
#ifndef _WINDOWS
    if (!neighborcache_reqs.empty() && !base_reqs.empty() && neighborcache_reader != nullptr)
    {
        auto *base_linux = dynamic_cast<LinuxAlignedFileReader *>(base_reader.get());
        auto *neighborcache_linux = dynamic_cast<LinuxAlignedFileReader *>(neighborcache_reader.get());
        if (base_linux != nullptr && neighborcache_linux != nullptr)
        {
            std::vector<FdAlignedRead> multi;
            multi.reserve(neighborcache_reqs.size() + base_reqs.size());
            const int neighborcache_fd = neighborcache_linux->get_file_desc();
            const int base_fd = base_linux->get_file_desc();
            for (auto &r : neighborcache_reqs)
                multi.emplace_back(neighborcache_fd, r.offset, r.len, r.buf);
            for (auto &r : base_reqs)
                multi.emplace_back(base_fd, r.offset, r.len, r.buf);
            base_reader->read_multi(multi, base_ctx);
            const float us = (float)io_timer.elapsed();
            if (stats != nullptr)
                stats->io_us += us;
            neighborcache_record_hop_io(neighborcache_reqs, base_reqs, us);
            return us;
        }
    }
#endif
    if (!neighborcache_reqs.empty() && neighborcache_reader != nullptr)
        neighborcache_reader->read(neighborcache_reqs, neighborcache_reader->get_ctx());
    if (!base_reqs.empty())
        base_reader->read(base_reqs, base_ctx);
    const float us = (float)io_timer.elapsed();
    if (stats != nullptr)
        stats->io_us += us;
    neighborcache_record_hop_io(neighborcache_reqs, base_reqs, us);
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
    for (auto &shard : _neighborcache_dyn_loc_shards)
        shard = std::make_unique<NeighborCacheDynLocShard>();
    for (auto &shard : _neighborcache_pending_member_shards)
        shard = std::make_unique<NeighborCachePendingMemberShard>();
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
    stop_neighborcache_dyn_patch_worker();
    stop_neighborcache_dyn_commit_workers();
    stop_neighborcache_dyn_writer();
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

    if (_neighborcache_disk_reader)
    {
        _neighborcache_disk_reader->close();
        _neighborcache_disk_reader.reset();
    }
    if (_neighborcache_dyn_reader)
    {
        _neighborcache_dyn_reader->close();
        _neighborcache_dyn_reader.reset();
    }
    if (_neighborcache_dyn_write_stream.is_open())
        _neighborcache_dyn_write_stream.close();
#ifndef _WINDOWS
    if (_neighborcache_dyn_write_fd >= 0)
    {
        ::close(_neighborcache_dyn_write_fd);
        _neighborcache_dyn_write_fd = -1;
    }
#endif
#ifndef _WINDOWS
    if (_neighborcache_dyn_write_fd >= 0)
    {
        ::close(_neighborcache_dyn_write_fd);
        _neighborcache_dyn_write_fd = -1;
    }
#endif
    _neighborcache_dc_map.clear();

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
    NeighborCacheSharedPageCache::ReadSection neighborcache_shared_read_section(_neighborcache_shared_cache);

    // reset query scratch
    query_scratch->reset();
    if (_neighborcache_dyn_enabled)
        neighborcache_dyn_prepare_query(query_scratch);
    if (_query_sector_cache_enabled)
    {
        query_scratch->sector_cache.clear();
        query_scratch->neighborcache_sector_cache.clear();
        query_scratch->neighborcache_fetched_sectors.clear();
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
        query_scratch->neighborcache_expand_count++;
        if (this->_neighborcache_dyn_enabled && query_scratch->neighborcache_dcache_maintenance_enabled &&
            !this->_neighborcache_coread_pack)
        {
            const bool force_update =
                this->_neighborcache_dcache_net_gate_enabled && query_scratch->neighborcache_dcache_probe;
            const bool suppress_update =
                this->_neighborcache_dcache_net_gate_enabled && !query_scratch->neighborcache_dcache_lookup_enabled;
            const auto decision =
                this->_neighborcache_mcache.plan_expand_update(node_id, force_update, suppress_update);
            if (stats != nullptr && this->_neighborcache_mcache.adaptive_update_enabled())
            {
                if (decision.observed_hit)
                    stats->n_neighborcache_mcache_hits++;
                else
                    stats->n_neighborcache_mcache_misses++;
            }
            if (decision.update)
            {
                NeighborCacheMetadataCache::TouchResult tr;
                {
                    tr = this->_neighborcache_mcache.on_expand(node_id, this->_neighborcache_score_unit);
                }
                if (stats != nullptr && !this->_neighborcache_mcache.adaptive_update_enabled() && tr.present)
                {
                    if (tr.inserted)
                        stats->n_neighborcache_mcache_misses++;
                    else
                        stats->n_neighborcache_mcache_hits++;
                }
                this->neighborcache_dyn_note_touch(tr, query_scratch);
            }
        }
        if (!this->_hotness_profiler.enabled())
            return;
        this->_hotness_profiler.on_node_expand(node_id);
        auto parent_it = query_scratch->parent_of(node_id);
        if (parent_it.has_value())
            this->_hotness_profiler.on_directed_edge(parent_it.value(), node_id);
    };
    auto profile_on_first_visit = [this](uint32_t id, uint32_t) {
        this->_hotness_profiler.on_node_visit(id);
    };
    auto profile_on_acquired = [&](uint32_t id) {
        const bool track_parent =
            (this->_neighborcache_dyn_enabled && query_scratch->neighborcache_dcache_maintenance_enabled) ||
            this->_hotness_profiler.enabled();
        const auto source = track_parent ? query_scratch->parent_of(id) : std::nullopt;
        if (source) {
            const uint32_t parent = *source;
            query_scratch->profile_parent.insert({id, parent});
            if (this->_neighborcache_dyn_enabled && query_scratch->neighborcache_dcache_maintenance_enabled &&
                !this->_neighborcache_real_io_coaccess)
            {
                bool touched = false;
                {
                    const auto tr = this->_neighborcache_mcache.on_edge(
                        parent, id, this->_neighborcache_score_unit.load(std::memory_order_relaxed));
                    touched = tr.present;
                    if (touched)
                        this->neighborcache_dyn_note_touch(tr, query_scratch);
                }
                if (touched)
                    this->neighborcache_mark_refresh_needed();
            }
        }
        profile_on_expand(id);
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
        (_neighborcache_dyn_enabled || !_neighborcache_dc_map.empty()) && (std::getenv("NEIGHBORCACHE_SAME_SEED_BEAM") != nullptr &&
                                   std::strcmp(std::getenv("NEIGHBORCACHE_SAME_SEED_BEAM"), "0") != 0);
    const bool seed_batch_expand =
        (_neighborcache_dyn_enabled || !_neighborcache_dc_map.empty()) && !same_seed_beam &&
        (std::getenv("NEIGHBORCACHE_SEED_BATCH_EXPAND") != nullptr &&
         std::strcmp(std::getenv("NEIGHBORCACHE_SEED_BATCH_EXPAND"), "0") != 0);
    size_t same_seed_beam_pool = std::max<uint64_t>(beam_width, 4 * beam_width);
    size_t seed_batch_expand_max = 16;
    float seed_batch_expand_dist_eps = 0.0f;
    if (same_seed_beam)
    {
        const char *pool_env = std::getenv("NEIGHBORCACHE_SAME_SEED_BEAM_POOL");
        if (pool_env != nullptr)
            same_seed_beam_pool = static_cast<size_t>(std::strtoul(pool_env, nullptr, 10));
    }
    if (seed_batch_expand)
    {
        const char *max_env = std::getenv("NEIGHBORCACHE_SEED_BATCH_EXPAND_MAX");
        if (max_env != nullptr)
            seed_batch_expand_max = static_cast<size_t>(std::strtoul(max_env, nullptr, 10));
        const char *eps_env = std::getenv("NEIGHBORCACHE_SEED_BATCH_EXPAND_DIST_EPS");
        if (eps_env != nullptr)
            seed_batch_expand_dist_eps = std::strtof(eps_env, nullptr);
    }
    const bool record_driven_seed_access =
        (std::getenv("NEIGHBORCACHE_RECORD_DRIVEN_SEED_ACCESS") != nullptr &&
         std::strcmp(std::getenv("NEIGHBORCACHE_RECORD_DRIVEN_SEED_ACCESS"), "0") != 0);
    auto neighborcache_io_seed = [query_scratch, record_driven_seed_access](uint32_t node_id) -> uint32_t {
        if (record_driven_seed_access)
            return node_id;
        const auto parent_it = query_scratch->parent_of(node_id);
        if (parent_it.has_value())
            return parent_it.value();
        return node_id;
    };

    // cleared every iteration
        const size_t hop_reserve = seed_batch_expand ? l_search : (2 * beam_width);
        std::vector<uint32_t> frontier;
        frontier.reserve(hop_reserve);
        std::vector<uint32_t> neighborcache_frontier;
        neighborcache_frontier.reserve(hop_reserve);
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
            neighborcache_frontier.clear();
            frontier_nhoods.clear();
            frontier_read_reqs.clear();
            cached_nhoods.clear();
            expand_order.clear();
            if ((_neighborcache_seed_page_lru || _neighborcache_seed_prefetch) && query_scratch != nullptr)
            {
                query_scratch->neighborcache_seed_page_coords.clear();
                query_scratch->neighborcache_seed_page_nbrs.clear();
                query_scratch->neighborcache_seed_page_coord_ptr.clear();
            }
            sector_scratch_idx = 0;
            float beam_max_dist = 0.0f;
            // find new beam
            query_scratch->neighborcache_beam_parents.clear();
            auto process_beam_neighbor = [&](const Neighbor &nbr) {
                query_scratch->neighborcache_beam_parents.emplace_back(nbr.id, nbr.nc_parent);
                beam_max_dist = std::max(beam_max_dist, nbr.distance);
                expand_order.push_back(nbr.id);
                if (stats != nullptr)
                    stats->n_beam_lookups++;
                if (stats != nullptr && query_scratch != nullptr && !_neighborcache_dc_seed_member_loc.empty())
                {
                    const auto parent_it = query_scratch->parent_of(nbr.id);
                    if (parent_it.has_value())
                    {
                        const auto seed_it = _neighborcache_dc_seed_member_loc.find(parent_it.value());
                        if (seed_it != _neighborcache_dc_seed_member_loc.end())
                        {
                            stats->n_dparent_nbrs++;
                            if (seed_it->second.find(nbr.id) != seed_it->second.end())
                                stats->n_dparent_nbr_on_page++;
                        }
                    }
                }
                NeighborCacheDemandProbe demand_probe;
                bool mem_hit = false;
                if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active())
                {
                    _neighborcache_mem_pool->shadow_access(nbr.id, current_iteration);
                    T *coord_ptr = nullptr;
                    std::pair<uint32_t, uint32_t *> nh_pair;
                    if (_neighborcache_mem_pool->lookup(nbr.id, coord_ptr, nh_pair, true))
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
                if (!mem_hit && (_neighborcache_seed_page_lru || _neighborcache_seed_prefetch) && query_scratch != nullptr)
                {
                    std::vector<T> sp_coords;
                    std::vector<uint32_t> sp_nbrs;
                    bool sp_hit;
                    {
                        NeighborCacheDemandProbeScope probe_scope(demand_probe);
                        sp_hit = neighborcache_seed_page_lookup(nbr.id, sp_coords, sp_nbrs);
                    }
                    if (sp_hit && !sp_coords.empty())
                    {
                        query_scratch->neighborcache_seed_page_coords.push_back(std::move(sp_coords));
                        query_scratch->neighborcache_seed_page_nbrs.push_back(std::move(sp_nbrs));
                        auto &stored_nbrs = query_scratch->neighborcache_seed_page_nbrs.back();
                        std::pair<uint32_t, uint32_t *> nh_pair(
                            static_cast<uint32_t>(stored_nbrs.size()),
                            stored_nbrs.empty() ? nullptr : stored_nbrs.data());
                        cached_nhoods.push_back(std::make_pair(nbr.id, nh_pair));
                        query_scratch->neighborcache_seed_page_coord_ptr[nbr.id] =
                            query_scratch->neighborcache_seed_page_coords.back().data();
                        mem_hit = true;
                    }
                }
                if (!mem_hit)
                {
                    g_dyn_probe.ncache_miss++;
                    // Prefetch buffer = same-query members packed when a seed page was served.
                    bool pref_hit = false;
                    {
                        pref_hit = neighborcache_dyn_query_local_dcache_hit(nbr.id, query_scratch);
                    }
                    bool dc_hit = false;
                    if (!pref_hit)
                    {
                        NeighborCacheDemandProbeScope probe_scope(demand_probe);
                        dc_hit = neighborcache_disk_cache_lookup_hit(nbr.id, query_scratch);
                    }
                    if (pref_hit)
                    {
                        if (stats != nullptr)
                            stats->n_neighborcache_prefetch_hits++;
                        if (_neighborcache_member_snapshot_done.load(std::memory_order_acquire))
                        {
                            const uint32_t hop = current_iteration < 128 ? current_iteration : 127;
                            if (_neighborcache_member_at_snapshot.find(nbr.id) != _neighborcache_member_at_snapshot.end())
                            {
                                _neighborcache_prefetch_from_early.fetch_add(1, std::memory_order_relaxed);
                                _neighborcache_early_hop_hist[hop].fetch_add(1, std::memory_order_relaxed);
                            }
                            else
                            {
                                _neighborcache_prefetch_from_late.fetch_add(1, std::memory_order_relaxed);
                                _neighborcache_late_hop_hist[hop].fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                        neighborcache_frontier.push_back(nbr.id);
                    }
                    else if (dc_hit)
                    {
                        // First touch of this page in the query → seed hit; immediately expose
                        // siblings in the query-local prefetch map so same-hop co-members are
                        // counted as prefetch (IO still coalesced later).
                        if (query_scratch != nullptr)
                        {
                            const auto lit = query_scratch->neighborcache_dcache_loc_cache.find(nbr.id);
                            if (lit != query_scratch->neighborcache_dcache_loc_cache.end())
                            {
                                const uint32_t sector = static_cast<uint32_t>(lit->second >> 32);
                                neighborcache_pack_dyn_page_into_query_prefetch(sector, query_scratch);
                            }
                        }
                        if (stats != nullptr)
                            stats->n_neighborcache_dcache_seed_hits++;
                        neighborcache_frontier.push_back(nbr.id);
                    }
                    else
                    {
                        frontier.push_back(nbr.id);
                        if (_record_base_frontier && stats != nullptr)
                            stats->base_frontier_nodes.push_back(nbr.id);
                    }
                }
                if (stats != nullptr && demand_probe.attempted)
                {
                    stats->n_neighborcache_dcache_demand_probes++;
                    if (demand_probe.hit)
                        stats->n_neighborcache_dcache_demand_hits++;
                }
                if (this->_count_visited_nodes)
                {
                    reinterpret_cast<std::atomic<uint32_t> &>(this->_node_visit_counter[nbr.id].second).fetch_add(1);
                }
            };

            {
            if (same_seed_beam)
            {
                const std::vector<Neighbor> beam_batch = retset.select_beam_seed_biased<tsl::robin_set<uint32_t>>(
                    beam_width, same_seed_beam_pool, neighborcache_io_seed);
                for (const Neighbor &nbr : beam_batch)
                    process_beam_neighbor(nbr);
            }
            else
            {
                uint32_t num_seen = 0;
                while (retset.has_unexpanded_node() && frontier.size() + neighborcache_frontier.size() < beam_width &&
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
                    active_seeds.insert(neighborcache_io_seed(id));

                std::vector<Neighbor> extras =
                    retset.collect_unexpanded_with_io_seeds<tsl::robin_set<uint32_t>, tsl::robin_set<uint32_t>>(
                        neighborcache_io_seed, active_seeds, beam_ids);
                const float dist_limit = beam_max_dist + seed_batch_expand_dist_eps;
                extras.erase(
                    std::remove_if(extras.begin(), extras.end(),
                                   [dist_limit](const Neighbor &n) { return n.distance > dist_limit; }),
                    extras.end());
                extras.erase(std::remove_if(extras.begin(), extras.end(),
                                            [&](const Neighbor &n) {
                                                return !neighborcache_disk_cache_lookup_hit(n.id, query_scratch);
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

            if (_record_hop_frontier && stats != nullptr && (!neighborcache_frontier.empty() || !frontier.empty()))
            {
                QueryStats::HopFrontierRecord rec;
                rec.hop = current_iteration;
                rec.neighborcache_nodes = neighborcache_frontier;
                rec.base_nodes = frontier;
                stats->hop_frontier_trace.push_back(std::move(rec));
            }

            if (_hotness_profiler.enabled())
            {
                if (_neighborcache_dyn_enabled || !_neighborcache_dc_map.empty())
                {
                    if (!neighborcache_frontier.empty())
                        _hotness_profiler.on_neighborcache_hop_frontier(neighborcache_frontier);
                }
                else if (!frontier.empty())
                {
                    _hotness_profiler.on_neighborcache_hop_frontier(frontier);
                }
            }

            if (_neighborcache_unified_disk && (!neighborcache_frontier.empty() || !frontier.empty()))
            {
                if (stats != nullptr)
                    stats->n_hops++;

                std::vector<AlignedRead> combined_reqs;
                combined_reqs.reserve(neighborcache_frontier.size() + frontier.size());
                std::vector<uint64_t> miss_sector_ids;
                miss_sector_ids.reserve(frontier.size());

                std::vector<NeighborCacheReadPending> neighborcache_pending;
                std::vector<AlignedRead> neighborcache_io;
                std::unordered_map<uint32_t, std::vector<size_t>> neighborcache_disk_fanout;

                prepare_neighborcache_disk_cache_io(neighborcache_frontier, query_scratch, sector_scratch, sector_scratch_idx,
                                         num_sectors_per_node, neighborcache_pending, neighborcache_io, neighborcache_disk_fanout, stats,
                                         num_ios);
                neighborcache_fallback_unserved_to_base(neighborcache_frontier, neighborcache_pending, neighborcache_disk_fanout, frontier);

                for (const auto &mr : neighborcache_io)
                    combined_reqs.push_back(mr);

                for (uint64_t i = 0; i < frontier.size(); i++)
                {
                    const auto id = frontier[i];
                    std::pair<uint32_t, char *> fnhood;
                    fnhood.first = id;
                    const uint64_t sec = get_node_sector((size_t)id);
                    query_scratch->read_sectors.insert(sec);

                    fnhood.second =
                        sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                    sector_scratch_idx++;
                    combined_reqs.emplace_back(sec * defaults::SECTOR_LEN, read_len, fnhood.second);
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
                }

                complete_neighborcache_disk_cache_io(query_scratch, neighborcache_pending, neighborcache_disk_fanout);
                finalize_neighborcache_pending_nodes(neighborcache_pending, neighborcache_frontier, query_scratch, sector_scratch,
                                           sector_scratch_idx, frontier_nhoods, stats);
            }
            else
            {
            std::vector<NeighborCacheReadPending> neighborcache_pending;
            std::vector<AlignedRead> neighborcache_io;
            std::unordered_map<uint32_t, std::vector<size_t>> neighborcache_disk_fanout;
            std::vector<uint64_t> miss_sector_ids;
            miss_sector_ids.reserve(frontier.size());

            if (!neighborcache_frontier.empty() && (_neighborcache_disk_reader || _neighborcache_dyn_enabled))
            {
                prepare_neighborcache_disk_cache_io(neighborcache_frontier, query_scratch, sector_scratch, sector_scratch_idx,
                                         num_sectors_per_node, neighborcache_pending, neighborcache_io, neighborcache_disk_fanout, stats,
                                         num_ios);
            }
            neighborcache_fallback_unserved_to_base(neighborcache_frontier, neighborcache_pending, neighborcache_disk_fanout, frontier);

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
                    frontier_nhoods.push_back(fnhood);
                }
                if (!neighborcache_io.empty() || !frontier_read_reqs.empty())
                {
                    if (stats != nullptr)
                        stats->n_hops++;
                    const float batch_us = issue_neighborcache_and_base_disk_reads(reader, (_neighborcache_dyn_enabled ? _neighborcache_dyn_reader : _neighborcache_disk_reader), ctx, neighborcache_io,
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
                    complete_neighborcache_disk_cache_io(query_scratch, neighborcache_pending, neighborcache_disk_fanout);
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
                if (!neighborcache_io.empty() || !frontier_read_reqs.empty())
                {
                    if (stats != nullptr)
                        stats->n_hops++;
                    issue_neighborcache_and_base_disk_reads(reader, (_neighborcache_dyn_enabled ? _neighborcache_dyn_reader : _neighborcache_disk_reader), ctx, neighborcache_io, frontier_read_reqs,
                                                    io_timer, stats);
                    complete_neighborcache_disk_cache_io(query_scratch, neighborcache_pending, neighborcache_disk_fanout);
                }
            }
            }
            if (frontier.empty() && !neighborcache_io.empty())
            {
                if (stats != nullptr)
                    stats->n_hops++;
                issue_neighborcache_and_base_disk_reads(reader, (_neighborcache_dyn_enabled ? _neighborcache_dyn_reader : _neighborcache_disk_reader), ctx, neighborcache_io, frontier_read_reqs, io_timer,
                                                stats);
                complete_neighborcache_disk_cache_io(query_scratch, neighborcache_pending, neighborcache_disk_fanout);
            }

            if (!neighborcache_pending.empty())
            {
                finalize_neighborcache_pending_nodes(neighborcache_pending, neighborcache_frontier, query_scratch, sector_scratch,
                                             sector_scratch_idx, frontier_nhoods, stats);
            }

            }
            reorder_frontier_nhoods_by_expand_order(expand_order, frontier_nhoods);

            std::unordered_map<uint32_t, std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_by_id;
            cached_by_id.reserve(cached_nhoods.size());
            for (auto &c : cached_nhoods)
                cached_by_id.emplace(c.first, c);

            std::unordered_map<uint32_t, char *> disk_by_id;
            disk_by_id.reserve(frontier_nhoods.size());
            for (auto &fn : frontier_nhoods)
                disk_by_id[fn.first] = fn.second;

#ifdef USE_BING_INFRA
        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods)
        {
            T *node_fp_coords_copy = nullptr;
            std::pair<uint32_t, uint32_t *> nh_dummy;
            if (_neighborcache_mem_pool == nullptr || !_neighborcache_mem_pool->active() ||
                !_neighborcache_mem_pool->lookup(cached_nhood.first, node_fp_coords_copy, nh_dummy))
            {
                bool seed_page_hit = false;
                if ((_neighborcache_seed_page_lru || _neighborcache_seed_prefetch) && query_scratch != nullptr)
                {
                    const auto seed_it = query_scratch->neighborcache_seed_page_coord_ptr.find(cached_nhood.first);
                    if (seed_it != query_scratch->neighborcache_seed_page_coord_ptr.end() && seed_it->second != nullptr)
                    {
                        node_fp_coords_copy = seed_it->second;
                        seed_page_hit = true;
                    }
                }
                if (!seed_page_hit)
                {
                    auto global_cache_iter = _coord_cache.find(cached_nhood.first);
                    if (global_cache_iter == _coord_cache.end())
                        continue;
                    node_fp_coords_copy = global_cache_iter->second;
                }
            }
            profile_on_acquired(cached_nhood.first);
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
                    nn.nc_parent = cached_nhood.first;
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
            profile_on_acquired(frontier_nhood.first);
            if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active() &&
                (_neighborcache_mem_runtime_admit || _neighborcache_dyn_enabled))
            {
                this->neighborcache_dyn_admit_node(frontier_nhood.first, node_disk_buf, stats, query_scratch,
                                           current_iteration);
                if (this->_neighborcache_dyn_enabled)
                {
                    uint32_t parent = std::numeric_limits<uint32_t>::max();
                    const auto pit = query_scratch->parent_of(frontier_nhood.first);
                    if (pit.has_value())
                        parent = pit.value();
                    this->neighborcache_dyn_note_base_load(frontier_nhood.first, parent, node_disk_buf, query_scratch);
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
            neighborcache_get_expand_neighbors(frontier_nhood.first, node_disk_buf, node_nbrs, nnbrs);
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
                    nn.nc_parent = frontier_nhood.first;
                    retset.insert(nn);
                }
            }

            if (stats != nullptr)
                stats->cpu_us += (float)cpu_timer.elapsed();
        }
#else
            for (uint32_t expand_id : expand_order)
            {
                const auto cached_it = cached_by_id.find(expand_id);
                if (cached_it != cached_by_id.end())
                {
                    auto &cached_nhood = cached_it->second;
                    T *node_fp_coords_copy = nullptr;
                    bool pool_hit = false;
                    if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active())
                    {
                        std::pair<uint32_t, uint32_t *> nh_dummy;
                        pool_hit = _neighborcache_mem_pool->lookup(cached_nhood.first, node_fp_coords_copy, nh_dummy);
                    }
                    if (!pool_hit && (_neighborcache_seed_page_lru || _neighborcache_seed_prefetch) && query_scratch != nullptr)
                    {
                        const auto seed_it = query_scratch->neighborcache_seed_page_coord_ptr.find(cached_nhood.first);
                        if (seed_it != query_scratch->neighborcache_seed_page_coord_ptr.end() && seed_it->second != nullptr)
                        {
                            node_fp_coords_copy = seed_it->second;
                            pool_hit = true;
                        }
                    }
                    if (!pool_hit)
                    {
                        auto global_cache_iter = _coord_cache.find(cached_nhood.first);
                        if (global_cache_iter == _coord_cache.end())
                            continue;
                        node_fp_coords_copy = global_cache_iter->second;
                    }
                    profile_on_acquired(cached_nhood.first);
                    float cur_expanded_dist;
                    if (!_use_disk_index_pq)
                    {
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
                    nn.nc_parent = cached_nhood.first;
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
                profile_on_acquired(expand_id);
                if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active() &&
                    (_neighborcache_mem_runtime_admit || _neighborcache_dyn_enabled))
                {
                    {
                        this->neighborcache_dyn_admit_node(expand_id, node_disk_buf, stats, query_scratch, current_iteration);
                    }
                    if (this->_neighborcache_dyn_enabled)
                    {
                        uint32_t parent = std::numeric_limits<uint32_t>::max();
                        const auto pit = query_scratch->parent_of(expand_id);
                        if (pit.has_value())
                            parent = pit.value();
                        this->neighborcache_dyn_note_base_load(expand_id, parent, node_disk_buf, query_scratch);
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
                full_retset.push_back(Neighbor(expand_id, cur_expanded_dist));
                const uint32_t *node_nbrs = nullptr;
                neighborcache_get_expand_neighbors(expand_id, node_disk_buf, node_nbrs, nnbrs);
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
                    nn.nc_parent = expand_id;
                        retset.insert(nn);
                    }
                }

                if (stats != nullptr)
                    stats->cpu_us += (float)cpu_timer.elapsed();
            }
#endif

        hops++;
    }

    if (_neighborcache_dyn_enabled)
    {
        {
            for (uint32_t sector : query_scratch->neighborcache_dyn_pinned_sectors)
                neighborcache_dyn_release_page_pin(sector);
            query_scratch->neighborcache_dyn_pinned_sectors.clear();
        }
        neighborcache_dyn_on_query_end(stats, query_scratch);
    }
    if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active())
    {
        _neighborcache_mem_pool->on_query_end(stats == nullptr ? 0 : stats->n_disk_reads,
                                      stats == nullptr ? 0.0 : stats->io_us);
    }
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
        stats->n_unique_neighborcache_sectors = (unsigned)query_scratch->read_neighborcache_disk_cache_sectors.size();
        stats->total_us = (float)query_timer.elapsed();
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
int PQFlashIndex<T, LabelT>::plan_neighborcache_memory_cache(double neighborcache_memory_gb, double host_memory_gb, double reserve_gb,
                                                     uint32_t num_threads, uint64_t &out_max_nodes,
                                                     std::string &report) const
{
    out_max_nodes = 0;
    report.clear();
    if (neighborcache_memory_gb <= 0.0)
    {
        report = "neighborcache_memory_gb <= 0; NEIGHBORCACHE memory cache disabled.";
        return 0;
    }

    const uint64_t host_bytes =
        (host_memory_gb > 0.0) ? static_cast<uint64_t>(host_memory_gb * (1024.0 * 1024.0 * 1024.0))
                               : read_host_mem_total_bytes();
    if (host_bytes == 0)
    {
        report = "Failed to determine host memory size (pass --neighborcache_host_memory_gb).";
        return -1;
    }

    const uint64_t reserve_bytes =
        (reserve_gb > 0.0) ? static_cast<uint64_t>(reserve_gb * (1024.0 * 1024.0 * 1024.0)) : 0;
    const uint64_t baseline = estimate_baseline_resident_bytes(num_threads);
    const uint64_t per_node = bytes_per_cached_node();
    const uint64_t requested = static_cast<uint64_t>(neighborcache_memory_gb * (1024.0 * 1024.0 * 1024.0));

    uint64_t residual = 0;
    if (host_bytes > baseline + reserve_bytes)
        residual = host_bytes - baseline - reserve_bytes;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << "NEIGHBORCACHE memory budget:\n"
        << "  host_total          = " << (host_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  baseline_index      = " << (baseline / (1024.0 * 1024.0 * 1024.0))
        << " GB (PQ + pivots + thread scratch estimate)\n"
        << "  reserve             = " << (reserve_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  residual_for_cache  = " << (residual / (1024.0 * 1024.0 * 1024.0)) << " GB\n"
        << "  requested_cache     = " << neighborcache_memory_gb << " GB\n"
        << "  bytes_per_node      = " << per_node << " B"
        << " (aligned_dim=" << _aligned_dim << ", max_degree=" << _max_degree << ")";

    if (requested > residual)
    {
        oss << "\nERROR: requested NEIGHBORCACHE memory cache (" << neighborcache_memory_gb
            << " GB) exceeds residual budget (" << (residual / (1024.0 * 1024.0 * 1024.0))
            << " GB). Reduce --neighborcache_memory_gb or --neighborcache_memory_reserve_gb.";
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
int PQFlashIndex<T, LabelT>::build_neighborcache_memory_node_list(const std::string &profile_prefix, uint64_t max_nodes,
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
        diskann::cerr << "NEIGHBORCACHE profile node_expand size " << node_expand.size() << " != num_points " << _num_points
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
int PQFlashIndex<T, LabelT>::build_neighborcache_disk_node_list(const std::string &profile_prefix, uint64_t max_nodes,
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
        const uint64_t pool_nodes = _neighborcache_mem_pool != nullptr ? _neighborcache_mem_pool->size() : 0;
        const uint64_t exclude = std::max<uint64_t>(memory_tier_exclude_count, pool_nodes);
        const uint64_t request = std::min<uint64_t>(_num_points, max_nodes + std::min<uint64_t>(exclude, _num_points));
        cache_bfs_levels(request, bfs_nodes, false, 1.0);
        node_list.reserve(static_cast<size_t>(max_nodes));
        for (uint32_t id : bfs_nodes)
        {
            if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->contains(id))
                continue;
            node_list.push_back(id);
            if (node_list.size() >= max_nodes)
                break;
        }
        if (node_list.size() != max_nodes)
        {
            diskann::cerr << "NEIGHBORCACHE disk-cache BFS prefill selected only " << node_list.size() << " / " << max_nodes
                          << " nodes after excluding the N-cache." << std::endl;
            return -1;
        }
        diskann::cout << "NEIGHBORCACHE disk-cache node list: BFS prefill selected " << node_list.size()
                      << " nodes after excluding "
                      << (_neighborcache_mem_pool != nullptr ? _neighborcache_mem_pool->size() : 0) << " N-cache nodes." << std::endl;
        return 0;
    }

    std::vector<uint64_t> node_expand;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> edges;
    if (HotnessProfiler::load(profile_prefix, node_expand, edges) != 0)
        return -1;
    if (node_expand.size() != _num_points)
    {
        diskann::cerr << "NEIGHBORCACHE profile node_expand size " << node_expand.size() << " != num_points " << _num_points
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
        if (build_neighborcache_memory_node_list(profile_prefix, memory_tier_exclude_count, mem_tier, 0) != 0)
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
            diskann::cerr << "NEIGHBORCACHE disk-cache: flat hot-node list empty (max_nodes=" << max_nodes << ")."
                          << std::endl;
            return -1;
        }
        diskann::cout << "NEIGHBORCACHE disk-cache node list: flat Top-" << node_list.size()
                      << " by node_expand (k_hops=0, exclude memory-tier=" << exclude_ids.size() << ")."
                      << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    VamanaGraph graph;
    // SIFT100M _mem.index adjacency is too large for many machines. When
    // NEIGHBORCACHE_DCACHE_PROFILE_GRAPH=1, build a lightweight undirected graph from
    // profile directed edges so directed_beam / pct / parent packing can run.
    const char *prof_graph_env = std::getenv("NEIGHBORCACHE_DCACHE_PROFILE_GRAPH");
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
        diskann::cout << "NEIGHBORCACHE disk-cache: profile-edge graph num_points=" << graph.num_points
                      << " directed_edges=" << edges.size() << " (skip " << mem_index_path << ")" << std::endl;
    }
    else if (load_vamana_graph(mem_index_path, graph) != 0)
    {
        diskann::cerr << "NEIGHBORCACHE disk-cache: failed to load Vamana graph from " << mem_index_path << std::endl;
        return -1;
    }

    if (layout == "edge" || layout == "jiang")
    {
        if (compute_edge_disk_cache_list(graph, node_expand, edges, nps, k_hops, max_nodes, exclude_ids, node_list) !=
            0)
            return -1;
        diskann::cout << "NEIGHBORCACHE disk-cache node list: edge-importance packing, k_hops=" << k_hops
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: edge-variant packing layout=" << layout << ", k_hops=" << k_hops
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: edge-replica packing, slots=" << node_list.size()
                      << " (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        return 0;
    }

    if (layout == "edge_star_dup")
    {
        if (compute_edge_star_dup_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids, node_list) !=
            0)
            return -1;
        diskann::cout << "NEIGHBORCACHE disk-cache node list: edge_star_dup packing, slots=" << node_list.size()
                      << " (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        return 0;
    }

    if (layout == "frontier")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 4;
        if (compute_frontier_disk_cache_list(graph, node_expand, edges, profile_prefix, nps, beam_width, max_nodes,
                                             exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "NEIGHBORCACHE disk-cache node list: frontier co-location packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: frontier_page packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: frontier_dup packing, slots=" << node_list.size()
                      << " (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        return 0;
    }

    if (layout == "parent")
    {
        if (compute_profile_parent_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids,
                                                   node_list) != 0)
            return -1;
        diskann::cout << "NEIGHBORCACHE disk-cache node list: profile-parent packing, selected " << node_list.size()
                      << " nodes (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (layout == "parent_star")
    {
        if (compute_parent_star_disk_cache_list(graph, node_expand, edges, nps, max_nodes, exclude_ids, node_list) != 0)
            return -1;
        diskann::cout << "NEIGHBORCACHE disk-cache node list: parent_star packing, selected " << node_list.size()
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: dbeam_estar_split packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: Layout E (directed_beam) packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: Layout E (" << layout << ") packing, selected "
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: child-only (" << layout << ") packing, selected "
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: unlimited child-replica (" << layout << ") packing, slots="
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: directed_seed_replica (Layout E replica), beam_width="
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: seed_replica (" << layout << ") packing, slots="
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: seed_replica_benefit (" << layout << ") packing, slots="
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: seed_replica_budget (" << layout << ") packing, slots="
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: seed_core_unique_fill (" << layout << ") slots="
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: directed_beam_starfill (legacy E), beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: directed_beam_top4first, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: directed_star packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: directed_beam_hybrid packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: directed_beam_inseed packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: directed_beam_tight packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: cooccur_star packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: dbeam_cooccur packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: frontier_topk packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: directed_beam_dual packing, beam_width=" << beam_width
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
        diskann::cout << "NEIGHBORCACHE disk-cache node list: node_top4first packing, k_hops=" << k_hops << ", selected "
                      << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size() << ")." << std::endl;
        append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
        return 0;
    }

    if (compute_hot_node_disk_cache_list(graph, node_expand, edges, nps, k_hops, max_nodes, exclude_ids, node_list) !=
        0)
        return -1;

    diskann::cout << "NEIGHBORCACHE disk-cache node list: hot-node path packing (avg edge weight), k_hops=" << k_hops
                  << ", selected " << node_list.size() << " nodes (exclude memory-tier=" << exclude_ids.size()
                  << ")." << std::endl;
    append_uncounted_nodes_to_disk_list(node_expand, max_nodes, exclude_ids, node_list);
    return 0;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::neighborcache_memory_cached_count() const
{
    if (_neighborcache_mem_pool && _neighborcache_mem_pool->active())
        return _neighborcache_mem_pool->size();
    return _nhood_cache.size();
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::print_neighborcache_memory_cache_stats() const
{
    if (!_neighborcache_mem_pool || !_neighborcache_mem_pool->active())
        return;
    const uint64_t ghost_actual_admissions = _neighborcache_mem_pool->ghost_actual_admissions();
    const uint64_t ghost_post_hits = _neighborcache_mem_pool->ghost_post_admission_hits();
    const uint64_t ghost_reused_admissions = _neighborcache_mem_pool->ghost_reused_admissions();
    const uint64_t ghost_admission_evictions = _neighborcache_mem_pool->ghost_admission_evictions();
    const uint64_t ghost_zero_hit_evictions = _neighborcache_mem_pool->ghost_zero_hit_evictions();
    diskann::cout << "NEIGHBORCACHE ncache admission: policy=" << _neighborcache_mem_pool->admission_policy_name()
                  << " hop_threshold=" << _neighborcache_mem_pool->admission_hop_threshold()
                  << " rejected=" << _neighborcache_mem_pool->admission_rejections()
                  << " second_hits=" << _neighborcache_mem_pool->admission_second_hits()
                  << " low_priority=" << _neighborcache_mem_pool->low_priority_admissions()
                  << " evictions=" << _neighborcache_mem_pool->eviction_count()
                  << " adaptive_queries=" << _neighborcache_mem_pool->adaptive_query_count()
                  << " half_life_queries=" << _neighborcache_mem_pool->adaptive_half_life_queries()
                  << " adaptive_reject="
                  << (_neighborcache_mem_pool->adaptive_reject_enabled() ? "on" : "off")
                  << " fast_miss=" << (_neighborcache_mem_pool->fast_miss_enabled() ? "on" : "off")
                  << " fast_miss_bytes=" << _neighborcache_mem_pool->fast_miss_bitmap_bytes()
                  << " fast_miss_bypasses=" << _neighborcache_mem_pool->fast_miss_bypasses()
                  << " clock=" << (_neighborcache_mem_pool->clock_enabled() ? "on" : "off")
                  << " clock_second_chances=" << _neighborcache_mem_pool->clock_second_chances()
                  << " ewma_hit_rate=" << _neighborcache_mem_pool->adaptive_hit_rate()
                  << " ewma_ghost_reuse=" << _neighborcache_mem_pool->adaptive_ghost_reuse_rate()
                  << " ewma_evictions_per_query=" << _neighborcache_mem_pool->adaptive_evictions_per_query()
                  << " transitions=" << _neighborcache_mem_pool->adaptive_transition_count() << std::endl;
    if (_neighborcache_mem_pool->ghost_stats_enabled())
    {
        diskann::cout << "NEIGHBORCACHE ncache ghost value: actual_admissions=" << ghost_actual_admissions
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
                      << " tracking_bytes=" << _neighborcache_mem_pool->ghost_tracking_bitmap_bytes()
                      << std::endl;
    }
    if (_neighborcache_mem_pool->shadow_enabled())
    {
        const auto shadow_stats = _neighborcache_mem_pool->shadow_stats();
        const uint64_t real_admit_samples = _neighborcache_mem_pool->shadow_real_admit_samples();
        const uint64_t real_admit_ns = _neighborcache_mem_pool->shadow_real_admit_ns();
        const uint64_t query_disk_reads = _neighborcache_mem_pool->shadow_query_disk_reads();
        const uint64_t query_io_ns = _neighborcache_mem_pool->shadow_query_io_ns();
        diskann::cout << "NEIGHBORCACHE ncache shadow cost: control="
                      << (_neighborcache_mem_pool->shadow_control_enabled() ? "on" : "off")
                      << " sample_divisor="
                      << _neighborcache_mem_pool->shadow_sample_divisor()
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
            diskann::cout << "NEIGHBORCACHE ncache shadow: policy=" << shadow.name
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
                          << " dropped=" << _neighborcache_mem_pool->shadow_dropped_samples()
                          << std::endl;
        }
    }
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::neighborcache_mem_pool_contains(uint32_t node_id) const
{
    return _neighborcache_mem_pool && _neighborcache_mem_pool->active() && _neighborcache_mem_pool->contains(node_id);
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::neighborcache_dc_map_contains(uint32_t node_id) const
{
    return neighborcache_disk_cache_lookup_hit(node_id, nullptr);
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::verify_neighborcache_disk_cache_against_base(uint64_t max_reports)
{
    if (_neighborcache_dc_map.empty() || !_neighborcache_disk_reader)
    {
        diskann::cout << "NEIGHBORCACHE verify: disk cache empty or reader missing." << std::endl;
        return 0;
    }

    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;
    _neighborcache_disk_reader->register_thread();

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

    for (const auto &kv : _neighborcache_dc_map)
    {
        const uint32_t node_id = kv.first;
        if (kv.second.empty())
            continue;
        const NeighborCacheDiskLoc &loc = kv.second[0];

        one_id[0] = node_id;
        coord_bufs[0] = base_coord.data();
        nbr_bufs[0] = {0, base_nbr.data()};
        const auto ok = read_nodes(one_id, coord_bufs, nbr_bufs);
        if (!ok[0])
        {
            diskann::cerr << "NEIGHBORCACHE verify: base read failed for node " << node_id << std::endl;
            continue;
        }

        AlignedRead ar;
        ar.offset = static_cast<uint64_t>(loc.sector) * defaults::SECTOR_LEN;
        ar.len = defaults::SECTOR_LEN;
        ar.buf = sidecar_buf;
        std::vector<AlignedRead> sidecar_reqs;
        sidecar_reqs.push_back(ar);
        _neighborcache_disk_reader->read(sidecar_reqs, ctx);

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
                diskann::cerr << "NEIGHBORCACHE verify mismatch node=" << node_id << " sector=" << loc.sector
                              << " slot=" << loc.slot << " nsectors=" << loc.nsectors << " " << reason << std::endl;
                reports++;
            }
        }
    }

    aligned_free(sidecar_buf);
    diskann::cout << "NEIGHBORCACHE verify: checked=" << checked << " mismatches=" << mismatches << std::endl;
    return mismatches > 0 ? -1 : 0;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::clear_neighborcache_memory_cache()
{
    if (_neighborcache_mem_pool)
        _neighborcache_mem_pool->clear();
    _neighborcache_mem_pool.reset();
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
int PQFlashIndex<T, LabelT>::load_neighborcache_memory_pool(const std::string &profile_prefix,
                                                    std::vector<uint32_t> &node_list)
{
    clear_neighborcache_memory_cache();
    if (node_list.empty())
        return 0;

    const char *cold_env = std::getenv("NEIGHBORCACHE_NCACHE_COLD");
    if (cold_env != nullptr && std::strcmp(cold_env, "0") != 0)
    {
        uint64_t capacity = node_list.size();
        if (const char *cap_env = std::getenv("NEIGHBORCACHE_NCACHE_CAP_NODES"))
            capacity = (std::max)(capacity, static_cast<uint64_t>(std::strtoull(cap_env, nullptr, 10)));
        _neighborcache_mem_pool = std::make_unique<NeighborCacheMemoryPool<T>>();
        _neighborcache_mem_pool->init(capacity, _aligned_dim, _max_degree, _num_points);
        _neighborcache_mem_pool->commit_initial_load({}, {});
        diskann::cout << "NEIGHBORCACHE memory pool: cold start, capacity " << capacity << " nodes, 0 preloaded"
                      << std::endl;
        return 0;
    }

    std::vector<uint64_t> node_expand;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> edges;
    if (!profile_prefix.empty() && HotnessProfiler::load(profile_prefix, node_expand, edges) != 0)
        return -1;

    _neighborcache_mem_pool = std::make_unique<NeighborCacheMemoryPool<T>>();
    _neighborcache_mem_pool->init(node_list.size(), _aligned_dim, _max_degree, _num_points);

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
            coord_buffers.push_back(_neighborcache_mem_pool->coord_ptr(slot));
            nbr_buffers.emplace_back(0, _neighborcache_mem_pool->nhood_ptr(slot));
        }

        const auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (!read_status[i])
            {
                diskann::cerr << "NEIGHBORCACHE memory pool: failed to read node " << nodes_to_read[i] << std::endl;
                return -1;
            }
            const uint32_t slot = static_cast<uint32_t>(start_idx + i);
            _neighborcache_mem_pool->nhood_ptr(slot)[0] = nbr_buffers[i].first;
        }
    }

    _neighborcache_mem_pool->commit_initial_load(node_list, node_expand);
    diskann::cout << "NEIGHBORCACHE memory pool loaded " << _neighborcache_mem_pool->size() << " nodes"
                  << (profile_prefix.empty() ? " (BFS around medoid)" : " (profile Top-N)") << std::endl;
    return 0;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::enable_neighborcache_memory_runtime_admit(bool enable)
{
    _neighborcache_mem_runtime_admit = enable;
}

namespace
{
inline void neighborcache_push_storage_nbrs(std::vector<uint32_t> &nbrs, std::vector<uint32_t> &off, const uint32_t *nhood,
                                    uint32_t max_degree)
{
    if (off.empty())
        off.push_back(0);
    const uint32_t n = std::min(nhood[0], max_degree);
    nbrs.insert(nbrs.end(), nhood + 1, nhood + 1 + n);
    off.push_back(static_cast<uint32_t>(nbrs.size()));
}

inline uint64_t neighborcache_env_u64(const char *name, uint64_t def)
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
size_t PQFlashIndex<T, LabelT>::neighborcache_dyn_loc_shard_index(uint32_t node_id) const
{
    return (static_cast<uint64_t>(node_id) * 11400714819323198485ull) &
           (_neighborcache_dyn_loc_shard_count - 1);
}

template <typename T, typename LabelT>
typename PQFlashIndex<T, LabelT>::NeighborCacheDynLocShard &
PQFlashIndex<T, LabelT>::neighborcache_dyn_loc_shard(uint32_t node_id)
{
    return *_neighborcache_dyn_loc_shards[neighborcache_dyn_loc_shard_index(node_id)];
}

template <typename T, typename LabelT>
const typename PQFlashIndex<T, LabelT>::NeighborCacheDynLocShard &
PQFlashIndex<T, LabelT>::neighborcache_dyn_loc_shard(uint32_t node_id) const
{
    return *_neighborcache_dyn_loc_shards[neighborcache_dyn_loc_shard_index(node_id)];
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_loc_clear_unlocked()
{
    for (size_t shard_index = 0; shard_index < _neighborcache_dyn_loc_shard_count; ++shard_index)
    {
        NeighborCacheDynLocShard &shard = *_neighborcache_dyn_loc_shards[shard_index];
        NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::DynamicUnique, true);
        shard.locations.clear();
    }
    if (_neighborcache_canon_slots != nullptr)
    {
        const size_t nslots = _neighborcache_canon_mask + 1;
        for (size_t i = 0; i < nslots; ++i)
            _neighborcache_canon_slots[i].store(0, std::memory_order_relaxed);
        _neighborcache_canon_authoritative.store(1, std::memory_order_release);
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_loc_add_unlocked(uint32_t node_id, const NeighborCacheDiskLoc &loc)
{
    NeighborCacheDynLocShard &shard = neighborcache_dyn_loc_shard(node_id);
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::DynamicUnique, true);
    auto &locations = shard.locations[node_id];
    for (const NeighborCacheDiskLoc &candidate : locations)
        if (candidate.sector == loc.sector && candidate.slot == loc.slot)
            return;
    // Lookup takes the first live location; newest pages carry the current co-access group.
    locations.insert(locations.begin(), loc);
    neighborcache_canon_publish(node_id, loc);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_loc_erase_sector_unlocked(uint32_t node_id, uint32_t sector)
{
    NeighborCacheDynLocShard &shard = neighborcache_dyn_loc_shard(node_id);
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::DynamicUnique, true);
    auto it = shard.locations.find(node_id);
    if (it == shard.locations.end())
        return;
    auto &locations = it.value();
    locations.erase(std::remove_if(locations.begin(), locations.end(),
                                   [sector](const NeighborCacheDiskLoc &loc) { return loc.sector == sector; }),
                    locations.end());
    if (locations.empty())
    {
        shard.locations.erase(it);
        neighborcache_canon_clear_node(node_id);
    }
    else
        neighborcache_canon_publish(node_id, locations.front());
}

static uint64_t neighborcache_canon_hash(uint32_t node_id)
{
    return static_cast<uint64_t>(node_id) * 11400714819323198485ull;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_canon_reset()
{
    uint64_t upper = std::max<uint64_t>(
        1024, _neighborcache_dyn_page_cap * std::max<uint32_t>(_neighborcache_dyn_members_per_page, 1) * 2);
    uint64_t slots = 1ull << 23;
    while (slots > 1024 && (slots >> 1) >= upper)
        slots >>= 1;
    _neighborcache_canon_slots.reset(new std::atomic<uint64_t>[static_cast<size_t>(slots)]());
    _neighborcache_canon_mask = static_cast<size_t>(slots - 1);
    _neighborcache_canon_authoritative.store(1, std::memory_order_release);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_canon_publish(uint32_t node_id, const NeighborCacheDiskLoc &loc)
{
    if (_neighborcache_canon_slots == nullptr ||
        _neighborcache_canon_authoritative.load(std::memory_order_relaxed) == 0)
        return;
    if (!neighborcache_dyn_is_sector(loc.sector) || loc.slot > 15 || loc.nsectors == 0 || loc.nsectors > 7)
    {
        _neighborcache_canon_authoritative.store(0, std::memory_order_release);
        return;
    }
    const uint32_t page = loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
    if (page > 0xffffffu)
    {
        _neighborcache_canon_authoritative.store(0, std::memory_order_release);
        return;
    }
    const uint64_t tag = static_cast<uint64_t>(node_id) + 1;
    const uint64_t desired = tag | (static_cast<uint64_t>(page) << 32) |
                             (static_cast<uint64_t>(loc.slot & 15u) << 56) |
                             (static_cast<uint64_t>(loc.nsectors & 7u) << 60) | (1ull << 63);
    size_t index = static_cast<size_t>(neighborcache_canon_hash(node_id) & _neighborcache_canon_mask);
    const size_t limit = _neighborcache_canon_mask + 1;
    for (size_t step = 0; step < limit; ++step)
    {
        std::atomic<uint64_t> &slot = _neighborcache_canon_slots[index];
        uint64_t cur = slot.load(std::memory_order_acquire);
        const uint32_t cur_tag = static_cast<uint32_t>(cur);
        const bool reusable = cur == 0 || cur_tag == static_cast<uint32_t>(tag) || (cur >> 63) == 0;
        if (reusable)
        {
            if (slot.compare_exchange_weak(cur, desired, std::memory_order_release, std::memory_order_acquire))
                return;
            continue;
        }
        index = (index + 1) & _neighborcache_canon_mask;
    }
    _neighborcache_canon_authoritative.store(0, std::memory_order_release);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_canon_clear_node(uint32_t node_id)
{
    if (_neighborcache_canon_slots == nullptr)
        return;
    const uint64_t tag = static_cast<uint64_t>(node_id) + 1;
    size_t index = static_cast<size_t>(neighborcache_canon_hash(node_id) & _neighborcache_canon_mask);
    const size_t limit = _neighborcache_canon_mask + 1;
    for (size_t step = 0; step < limit; ++step)
    {
        std::atomic<uint64_t> &slot = _neighborcache_canon_slots[index];
        uint64_t cur = slot.load(std::memory_order_acquire);
        if (cur == 0)
            return;
        if (static_cast<uint32_t>(cur) == static_cast<uint32_t>(tag))
        {
            if (slot.compare_exchange_weak(cur, tag, std::memory_order_release, std::memory_order_acquire))
                return;
            continue;
        }
        index = (index + 1) & _neighborcache_canon_mask;
    }
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::neighborcache_canon_lookup(uint32_t node_id, NeighborCacheDiskLoc &out) const
{
    if (_neighborcache_canon_slots == nullptr)
        return -1;
    const bool authoritative = _neighborcache_canon_authoritative.load(std::memory_order_acquire) != 0;
    // A saturated or unrepresentable fast directory is incomplete. Bypass it:
    // probing a full table on every miss can scan millions of slots. The caller
    // falls back to the complete sharded location directory when we return -1.
    if (!authoritative)
        return -1;
    const uint64_t tag = static_cast<uint64_t>(node_id) + 1;
    size_t index = static_cast<size_t>(neighborcache_canon_hash(node_id) & _neighborcache_canon_mask);
    const size_t limit = _neighborcache_canon_mask + 1;
    for (size_t step = 0; step < limit; ++step)
    {
        const uint64_t cur = _neighborcache_canon_slots[index].load(std::memory_order_acquire);
        if (cur == 0)
            return authoritative ? 0 : -1;
        if (static_cast<uint32_t>(cur) == static_cast<uint32_t>(tag))
        {
            if ((cur >> 63) == 0)
                return authoritative ? 0 : -1;
            out.sector = NEIGHBORCACHE_DYN_SECTOR_BASE + static_cast<uint32_t>((cur >> 32) & 0xffffffu);
            out.slot = static_cast<uint16_t>((cur >> 56) & 15u);
            out.nsectors = static_cast<uint16_t>((cur >> 60) & 7u);
            return 1;
        }
        index = (index + 1) & _neighborcache_canon_mask;
    }
    return -1;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_bump_page_gen(uint32_t page_idx)
{
    if (_neighborcache_dyn_page_gen != nullptr && page_idx < _neighborcache_dyn_physical_cap)
        _neighborcache_dyn_page_gen[page_idx].fetch_add(1, std::memory_order_release);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_query_page_prefetch_current(const SSDQueryScratch<T> *query_scratch,
                                                                uint32_t page_idx) const
{
    if (query_scratch == nullptr || _neighborcache_dyn_page_gen == nullptr || page_idx >= _neighborcache_dyn_physical_cap)
        return false;
    const uint32_t gen = _neighborcache_dyn_page_gen[page_idx].load(std::memory_order_acquire);
    if (gen == 0 || (gen & 1u))
        return false;
    const auto it = query_scratch->neighborcache_prefetched_page_gen.find(page_idx);
    return it != query_scratch->neighborcache_prefetched_page_gen.end() && it->second == gen;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_query_page_prefetch_remember(SSDQueryScratch<T> *query_scratch,
                                                                 uint32_t page_idx) const
{
    if (query_scratch == nullptr || _neighborcache_dyn_page_gen == nullptr || page_idx >= _neighborcache_dyn_physical_cap)
        return;
    const uint32_t gen = _neighborcache_dyn_page_gen[page_idx].load(std::memory_order_acquire);
    if (gen != 0 && (gen & 1u) == 0)
        query_scratch->neighborcache_prefetched_page_gen[page_idx] = gen;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_seed_page_unlocked(uint32_t seed_id, uint32_t &page_idx) const
{
    const NeighborCacheDynLocShard &shard = neighborcache_dyn_loc_shard(seed_id);
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::DynamicShared, false);
    const auto it = shard.locations.find(seed_id);
    if (it == shard.locations.end())
        return false;
    for (const NeighborCacheDiskLoc &loc : it.value())
    {
        if (loc.slot == 0 && neighborcache_dyn_is_sector(loc.sector))
        {
            const uint32_t candidate = loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
            if (candidate < _neighborcache_dyn_page_secondary.size() && _neighborcache_dyn_page_secondary[candidate])
                continue;
            if (neighborcache_dyn_page_live(candidate))
            {
                page_idx = candidate;
                return true;
            }
        }
    }
    return false;
}

template <typename T, typename LabelT>
size_t PQFlashIndex<T, LabelT>::neighborcache_dc_loc_count_unlocked(uint32_t node_id) const
{
    if (_neighborcache_dyn_enabled)
    {
        const NeighborCacheDynLocShard &shard = neighborcache_dyn_loc_shard(node_id);
        NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::DynamicShared, false);
        const auto it = shard.locations.find(node_id);
        return it == shard.locations.end() ? 0 : it.value().size();
    }
    const auto it = _neighborcache_dc_map.find(node_id);
    return it == _neighborcache_dc_map.end() ? 0 : it->second.size();
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dc_loc_at_unlocked(uint32_t node_id, size_t index, NeighborCacheDiskLoc &loc) const
{
    if (_neighborcache_dyn_enabled)
    {
        const NeighborCacheDynLocShard &shard = neighborcache_dyn_loc_shard(node_id);
        NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::DynamicShared, false);
        const auto it = shard.locations.find(node_id);
        if (it == shard.locations.end() || index >= it.value().size())
            return false;
        loc = it.value()[index];
        return true;
    }
    const auto it = _neighborcache_dc_map.find(node_id);
    if (it == _neighborcache_dc_map.end() || index >= it->second.size())
        return false;
    loc = it->second[index];
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dc_loc_contains_unlocked(uint32_t node_id, const NeighborCacheDiskLoc &loc) const
{
    bool found = false;
    neighborcache_dc_for_each_loc_unlocked(node_id, [&](const NeighborCacheDiskLoc &candidate, size_t) {
        if (candidate.sector == loc.sector && candidate.slot == loc.slot)
            found = true;
    });
    return found;
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::neighborcache_dc_loc_directory_empty_unlocked() const
{
    if (!_neighborcache_dyn_enabled)
        return _neighborcache_dc_map.empty();
    for (size_t shard_index = 0; shard_index < _neighborcache_dyn_loc_shard_count; ++shard_index)
    {
        const NeighborCacheDynLocShard &shard = *_neighborcache_dyn_loc_shards[shard_index];
        NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::DynamicShared, false);
        if (!shard.locations.empty())
            return false;
    }
    return true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_writer_loop()
{
    neighborcache_pin_background_thread();
    while (true)
    {
        std::shared_ptr<NeighborCacheDynWriteJob> job;
        {
            std::unique_lock<std::mutex> lock(_neighborcache_dyn_write_q_mu);
            _neighborcache_dyn_write_q_cv.wait(lock, [this] {
                return _neighborcache_dyn_writer_stop.load(std::memory_order_acquire) || !_neighborcache_dyn_write_q.empty();
            });
            if (_neighborcache_dyn_writer_stop.load(std::memory_order_acquire) && _neighborcache_dyn_write_q.empty())
                return;
            job = std::move(_neighborcache_dyn_write_q.front());
            _neighborcache_dyn_write_q.pop_front();
        }
        _neighborcache_dyn_write_q_cv.notify_all();
        if (job == nullptr)
            continue;

        // Only writer threads wait for older writes to the same sector. Query
        // threads detach snapshots immediately and never wait for completion.
        if (job->predecessor)
        {
            auto previous = std::move(job->predecessor);
            std::unique_lock<std::mutex> lock(previous->mu);
            previous->cv.wait(lock, [&] { return previous->done; });
        }

        const uint64_t io_started_ns = neighborcache_lock_now_ns();
        if (job->enqueued_ns != 0)
            _neighborcache_dyn_writer_queue_ns.fetch_add(io_started_ns - job->enqueued_ns, std::memory_order_relaxed);
        bool ok = false;
        if (job->op == NeighborCacheDynWriteOp::Sync)
        {
#ifndef _WINDOWS
            ok = _neighborcache_dyn_write_fd >= 0 && ::fdatasync(_neighborcache_dyn_write_fd) == 0;
#else
            std::lock_guard<std::mutex> writer_lock(_neighborcache_dyn_writer_mu);
            _neighborcache_dyn_write_stream.flush();
            ok = _neighborcache_dyn_write_stream.good();
#endif
        }
        else
        {
#ifndef _WINDOWS
            if (_neighborcache_dyn_write_fd >= 0 && job->data && job->nbytes > 0)
            {
                ssize_t written = -1;
                do
                {
                    written = ::pwrite(_neighborcache_dyn_write_fd, job->data.get(), job->nbytes, job->offset);
                } while (written < 0 && errno == EINTR);
                ok = written == static_cast<ssize_t>(job->nbytes);
            }
#else
            if (job->data && job->nbytes > 0)
            {
                std::lock_guard<std::mutex> writer_lock(_neighborcache_dyn_writer_mu);
                _neighborcache_dyn_write_stream.clear();
                _neighborcache_dyn_write_stream.seekp(static_cast<std::streamoff>(job->offset));
                _neighborcache_dyn_write_stream.write(job->data.get(), static_cast<std::streamsize>(job->nbytes));
                _neighborcache_dyn_write_stream.flush();
                ok = _neighborcache_dyn_write_stream.good();
            }
#endif
        }
        _neighborcache_dyn_writer_io_ns.fetch_add(neighborcache_lock_now_ns() - io_started_ns, std::memory_order_relaxed);
        _neighborcache_dyn_writer_completed.fetch_add(1, std::memory_order_relaxed);
        if (!ok)
            _neighborcache_dyn_writer_failed.fetch_add(1, std::memory_order_relaxed);

        if (job->prefetch_page != std::numeric_limits<uint32_t>::max())
        {
            const uint32_t page = job->prefetch_page;
            if (ok)
            {
                // A newer version remains unavailable on disk until its own
                // ordered write completes, even after the cache slot is reused.
                uint64_t version = job->prefetch_version;
                _neighborcache_prefetch_dirty[page].compare_exchange_strong(
                    version, 0, std::memory_order_acq_rel);
                _neighborcache_seed_page_written.fetch_add(1, std::memory_order_relaxed);
            }
            // Failure leaves D unavailable; demand reads fall back to base.
            _neighborcache_prefetch_writeback[page].fetch_sub(1, std::memory_order_acq_rel);
            neighborcache_dyn_release_page_pin(NEIGHBORCACHE_DYN_SECTOR_BASE + page);
        }

        {
            std::lock_guard<std::mutex> done_lock(job->mu);
            job->ok = ok;
            job->done = true;
        }
        job->cv.notify_all();
        if (job->prefetch_page != std::numeric_limits<uint32_t>::max())
        {
            std::lock_guard<std::mutex> lock(_neighborcache_dyn_write_q_mu);
            auto it = _neighborcache_writeback_tail.find(job->prefetch_page);
            if (it != _neighborcache_writeback_tail.end() && it->second.lock() == job)
                _neighborcache_writeback_tail.erase(it);
        }
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::start_neighborcache_dyn_writer()
{
    if (_neighborcache_dyn_writer_started)
        return;
    _neighborcache_dyn_writer_stop.store(false, std::memory_order_release);
    _neighborcache_dyn_writer_threads.clear();
    _neighborcache_dyn_writer_threads.reserve(_neighborcache_dyn_writer_thread_count);
    for (size_t thread = 0; thread < _neighborcache_dyn_writer_thread_count; ++thread)
        _neighborcache_dyn_writer_threads.emplace_back([this] { neighborcache_dyn_writer_loop(); });
    _neighborcache_dyn_writer_started = true;
    if (_neighborcache_dyn_enabled && neighborcache_env_u64("NEIGHBORCACHE_REFRESH_THREAD", 0) != 0 && !_neighborcache_refresh_thread_on)
    {
        _neighborcache_refresh_thread_stop.store(false, std::memory_order_release);
        _neighborcache_refresh_thread_on = true;
        _neighborcache_refresh_thread = std::thread([this] { neighborcache_refresh_thread_loop(); });
        diskann::cout << "NEIGHBORCACHE refresh_thread=on" << std::endl;
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_refresh_thread_loop()
{
    neighborcache_is_refresh_thread() = true;
    neighborcache_pin_background_thread();
    while (!_neighborcache_refresh_thread_stop.load(std::memory_order_acquire))
    {
        if (_neighborcache_refresh_needed.load(std::memory_order_acquire) &&
            _neighborcache_pending_pair_count.load(std::memory_order_acquire) < _neighborcache_pending_pair_cap)
        {
            neighborcache_dyn_request_refresh();
            std::this_thread::yield();
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::microseconds(5));
        }
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::stop_neighborcache_dyn_writer()
{
    if (_neighborcache_refresh_thread_on)
    {
        _neighborcache_refresh_thread_stop.store(true, std::memory_order_release);
        if (_neighborcache_refresh_thread.joinable())
            _neighborcache_refresh_thread.join();
        _neighborcache_refresh_thread_on = false;
    }
    if (!_neighborcache_dyn_writer_started)
        return;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_dyn_write_q_mu);
        _neighborcache_dyn_writer_stop.store(true, std::memory_order_release);
    }
    _neighborcache_dyn_write_q_cv.notify_all();
    for (std::thread &thread : _neighborcache_dyn_writer_threads)
        if (thread.joinable())
            thread.join();
    _neighborcache_dyn_writer_threads.clear();
    _neighborcache_dyn_writer_started = false;

    // Fail any leftover jobs (should be empty after drain; stop waits for empty).
    std::deque<std::shared_ptr<NeighborCacheDynWriteJob>> leftover;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_dyn_write_q_mu);
        leftover.swap(_neighborcache_dyn_write_q);
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
std::shared_ptr<typename PQFlashIndex<T, LabelT>::NeighborCacheDynWriteJob>
PQFlashIndex<T, LabelT>::neighborcache_dyn_writer_submit(NeighborCacheDynWriteOp op, off_t offset, const void *data, size_t nbytes,
                                                       bool nonblocking, uint32_t prefetch_page, uint64_t prefetch_version)
{
    if (!_neighborcache_dyn_writer_started || _neighborcache_dyn_writer_stop.load(std::memory_order_acquire))
        return nullptr;

    auto job = std::make_shared<NeighborCacheDynWriteJob>();
    job->op = op;
    job->offset = offset;
    job->nbytes = nbytes;
    job->enqueued_ns = neighborcache_lock_now_ns();
    job->prefetch_page = prefetch_page;
    job->prefetch_version = prefetch_version;
    if (op == NeighborCacheDynWriteOp::Write)
    {
        if (data == nullptr || nbytes == 0)
            return nullptr;
        void *aligned = nullptr;
#ifndef _WINDOWS
        if (::posix_memalign(&aligned, defaults::SECTOR_LEN, nbytes) != 0)
            return nullptr;
#else
        aligned = std::malloc(nbytes);
        if (aligned == nullptr)
            return nullptr;
#endif
        std::memcpy(aligned, data, nbytes);
        job->data.reset(static_cast<char *>(aligned));
    }

    {
        std::unique_lock<std::mutex> lock(_neighborcache_dyn_write_q_mu, std::defer_lock);
        if (nonblocking)
        {
            if (!lock.try_lock() || _neighborcache_dyn_write_q.size() >= _neighborcache_dyn_write_q_cap)
                return nullptr;
        }
        else
        {
            lock.lock();
            _neighborcache_dyn_write_q_cv.wait(lock, [this] {
                return _neighborcache_dyn_writer_stop.load(std::memory_order_acquire) ||
                       _neighborcache_dyn_write_q.size() < _neighborcache_dyn_write_q_cap;
            });
        }
        if (_neighborcache_dyn_writer_stop.load(std::memory_order_acquire))
            return nullptr;
        if (prefetch_page != std::numeric_limits<uint32_t>::max())
        {
            auto &tail = _neighborcache_writeback_tail[prefetch_page];
            job->predecessor = tail.lock();
            tail = job;
        }
        _neighborcache_dyn_write_q.push_back(job);
        _neighborcache_dyn_writer_enqueued.fetch_add(1, std::memory_order_relaxed);
    }
    _neighborcache_dyn_write_q_cv.notify_all();
    return job;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_writer_submit_and_wait(NeighborCacheDynWriteOp op, off_t offset, const void *data,
                                                               size_t nbytes)
{
    std::shared_ptr<NeighborCacheDynWriteJob> job = neighborcache_dyn_writer_submit(op, offset, data, nbytes);
    if (job == nullptr)
        return false;
    std::unique_lock<std::mutex> done_lock(job->mu);
    job->cv.wait(done_lock, [&job] { return job->done; });
    return job->ok;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::enable_neighborcache_dynamic_3cache(bool enable, const std::string &flush_prefix)
{
    if (!enable)
    {
        stop_neighborcache_dyn_patch_worker();
        stop_neighborcache_dyn_commit_workers();
        stop_neighborcache_dyn_writer();
        _neighborcache_dyn_enabled = false;
        return;
    }
    stop_neighborcache_dyn_patch_worker();
    stop_neighborcache_dyn_commit_workers();
    stop_neighborcache_dyn_writer();
    _neighborcache_dyn_enabled = true;
    g_dyn_probe.reset();
    neighborcache_seed_write_reset();
    const bool import_static_prefill =
        neighborcache_env_u64("NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC", 0) != 0 && !_neighborcache_dc_path.empty() && !_neighborcache_dc_map.empty();

    const uint64_t ncache_n =
        (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active()) ? _neighborcache_mem_pool->capacity() : 0;
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    const uint64_t base_pages =
        (_num_points > 0) ? (1ULL + (_num_points + nps - 1ULL) / nps) : 0ULL;
    const uint64_t default_space_cap = base_pages > 0 ? std::max<uint64_t>((base_pages + 4ULL) / 5ULL, 1024) : 4096;
    // A profile-selected seed fraction is not a fraction of original index pages.
    // Keep the old default, with an explicit budget override for such experiments.
    const uint64_t dcache_space_cap = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES", default_space_cap);
    _neighborcache_dyn_page_cap = std::min(neighborcache_env_u64("NEIGHBORCACHE_DCACHE_CAP", dcache_space_cap), dcache_space_cap);
    const uint64_t requested_mcache_cap =
        neighborcache_env_u64("NEIGHBORCACHE_MCACHE_CAP", _neighborcache_dyn_page_cap > 0 ? _neighborcache_dyn_page_cap * 2ULL : 65536ULL);
    // Explicitly static runs have no M heat table or seed replacement heap.
    _neighborcache_static_no_mcache = neighborcache_env_u64("NEIGHBORCACHE_FREEZE_MAINTENANCE", 2) == 1;
    _neighborcache_freeze_maintenance = _neighborcache_static_no_mcache;
    const uint64_t mcache_cap = _neighborcache_static_no_mcache ? 0 :
        std::max<uint64_t>(requested_mcache_cap, _neighborcache_dyn_page_cap * 2ULL);
    const uint8_t sig_t = static_cast<uint8_t>(neighborcache_env_u64("NEIGHBORCACHE_SEED_T", NeighborCacheMetadataCache::kDefaultSigT));
    if (_neighborcache_static_no_mcache)
        _neighborcache_mcache.clear();
    else
        _neighborcache_mcache.init(mcache_cap, sig_t, _num_points);
    _neighborcache_deferred_ncache_page_write = neighborcache_env_u64("NEIGHBORCACHE_DEFERRED_NCACHE_PAGE_WRITE", 0) != 0;
    _neighborcache_seed_pin_neighbors = neighborcache_env_u64("NEIGHBORCACHE_SEED_PIN_NEIGHBORS", 0) != 0;
    _neighborcache_seed_pin_age_queries = static_cast<uint32_t>(std::min<uint64_t>(
        std::numeric_limits<uint32_t>::max(), neighborcache_env_u64("NEIGHBORCACHE_SEED_PIN_AGE_QUERIES", 192)));
    _neighborcache_seed_pin_writes_per_query = static_cast<uint32_t>(std::min<uint64_t>(
        std::numeric_limits<uint32_t>::max(), neighborcache_env_u64("NEIGHBORCACHE_SEED_PIN_WRITES_PER_QUERY", 2)));
    _neighborcache_seed_page_lru = neighborcache_env_u64("NEIGHBORCACHE_SEED_PAGE_LRU", 1) != 0;
    // The shared prefetch is the seed-page buffer: one LRU, readable and writable.
    _neighborcache_seed_prefetch = _neighborcache_shared_cache.enabled() && _neighborcache_shared_cache_dyn;
    if (_neighborcache_seed_prefetch)
        _neighborcache_seed_page_lru = false;
    _neighborcache_seed_page_cap = 0;
    _neighborcache_seed_page_shards.clear();
    _neighborcache_seed_page_owners.clear();
    if (_neighborcache_seed_page_lru || _neighborcache_seed_prefetch)
    {
        // This path replaces the full-table pin scan.
        _neighborcache_seed_pin_neighbors = false;
        uint64_t pages = neighborcache_env_u64("NEIGHBORCACHE_SEED_PAGE_PAGES", kNeighborCacheSeedPageDefaultPages);
        if (pages < 64)
            pages = 64;
        if (pages > 1048576)
            pages = 1048576;
        _neighborcache_seed_page_cap = _neighborcache_seed_prefetch ? 0 : static_cast<uint32_t>(pages);
        const uint32_t per_shard =
            std::max<uint32_t>(1, static_cast<uint32_t>((pages + kNeighborCacheSeedPageShards - 1) / kNeighborCacheSeedPageShards));
        _neighborcache_seed_page_owners.reserve(kNeighborCacheSeedPageShards);
        if (!_neighborcache_seed_prefetch)
            _neighborcache_seed_page_shards.reserve(kNeighborCacheSeedPageShards);
        for (uint32_t shard_i = 0; shard_i < kNeighborCacheSeedPageShards; ++shard_i)
        {
            if (!_neighborcache_seed_prefetch)
            {
                auto shard = std::make_unique<NeighborCacheSeedPageShard>();
                shard->cap = per_shard;
                _neighborcache_seed_page_shards.push_back(std::move(shard));
            }
            _neighborcache_seed_page_owners.push_back(std::make_unique<NeighborCacheSeedPageOwnerShard>());
        }
        _neighborcache_seed_page_stamp.store(1, std::memory_order_relaxed);
        _neighborcache_seed_page_opens.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_open_miss.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_open_nbrs.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_appends.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_replaces.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_touches.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_hits.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_evicts.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_enqueued.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_dropped.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_written.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_skipped.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_failed.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_nbr_sum.store(0, std::memory_order_relaxed);
        _neighborcache_seed_page_inflight.store(0, std::memory_order_relaxed);
    }
    if (_neighborcache_seed_pin_neighbors && _neighborcache_mem_pool != nullptr)
        _neighborcache_mem_pool->enable_seed_hold_limit(true);
    _neighborcache_dcache_family = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_FAMILY", 0) != 0;
    if (_neighborcache_dcache_family)
    {
        uint64_t cap = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_FAMILY_CAP", 0);
        if (cap == 0 && _neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active())
            cap = _neighborcache_mem_pool->capacity();
        if (cap == 0)
            cap = 65536;
        _neighborcache_dcache_family_cap = cap;
    }
    _neighborcache_coread_pack = neighborcache_env_u64("NEIGHBORCACHE_COREAD_PACK", 0) != 0;
    _neighborcache_coread_pages.store(0, std::memory_order_relaxed);
    _neighborcache_coread_hop2_pages.store(0, std::memory_order_relaxed);
    _neighborcache_coread_enqueued.store(0, std::memory_order_relaxed);
    _neighborcache_coread_write_dropped.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_evict_ncache_on_commit =
        neighborcache_env_u64("NEIGHBORCACHE_DCACHE_EVICT_NCACHE_ON_COMMIT", 0) != 0;
    _neighborcache_dcache_ncache_erases.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_no_seed_replace = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_NO_SEED_REPLACE", 0) != 0;
    _neighborcache_dcache_second_page = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_SECOND_PAGE", 0) != 0;
    _neighborcache_dcache_max_pages = 1;
    if (_neighborcache_dcache_second_page)
    {
        uint64_t max_pages = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_MAX_PAGES", 5);
        if (max_pages < 2)
            max_pages = 2;
        if (max_pages > 5)
            max_pages = 5;
        _neighborcache_dcache_max_pages = static_cast<uint8_t>(max_pages);
    }
    _neighborcache_dcache_pair_ios.store(0, std::memory_order_relaxed);
    _neighborcache_span_adaptive = _neighborcache_dcache_second_page && neighborcache_env_u64("NEIGHBORCACHE_DCACHE_SPAN_ADAPTIVE", 0) != 0;
    if (const char *v = std::getenv("NEIGHBORCACHE_DCACHE_SPAN_MIN_HITS"))
        _neighborcache_span_min_hits = std::strtof(v, nullptr);
    _neighborcache_span_explore = static_cast<uint32_t>(neighborcache_env_u64("NEIGHBORCACHE_DCACHE_SPAN_EXPLORE", 16));
    _neighborcache_span_full_reads.store(0, std::memory_order_relaxed);
    _neighborcache_span_primary_only.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_freeze_after = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_FREEZE_AFTER", 0);
    _neighborcache_member_snapshot_at = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_MEMBER_SNAPSHOT_AT", 0);
    _neighborcache_member_snapshot_done.store(false, std::memory_order_relaxed);
    _neighborcache_member_at_snapshot.clear();
    _neighborcache_member_snapshot_slots.store(0, std::memory_order_relaxed);
    _neighborcache_prefetch_from_early.store(0, std::memory_order_relaxed);
    _neighborcache_prefetch_from_late.store(0, std::memory_order_relaxed);
    for (uint32_t h = 0; h < 128; ++h)
    {
        _neighborcache_early_hop_hist[h].store(0, std::memory_order_relaxed);
        _neighborcache_late_hop_hist[h].store(0, std::memory_order_relaxed);
    }
    _neighborcache_dcache_query_patch = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_QUERY_PATCH", 1) != 0;
    _neighborcache_repack_on_read = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_REPACK_ON_READ", 0) != 0;
    _neighborcache_dcache_query_patch_async = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_QUERY_PATCH_ASYNC", 1) != 0;
    _neighborcache_dcache_query_patch_max = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(64, neighborcache_env_u64("NEIGHBORCACHE_DCACHE_QUERY_PATCH_MAX", 4))));
    _neighborcache_patch_q_cap = static_cast<size_t>(
        std::max<uint64_t>(8, std::min<uint64_t>(4096, neighborcache_env_u64("NEIGHBORCACHE_DCACHE_QUERY_PATCH_Q", 64))));
    _neighborcache_patch_batch_us = static_cast<uint32_t>(
        std::min<uint64_t>(10000, neighborcache_env_u64("NEIGHBORCACHE_DCACHE_QUERY_PATCH_BATCH_US", 100)));
    _neighborcache_patch_worker_no_disk = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_QUERY_PATCH_NO_DISK", 0) != 0;
    _neighborcache_dcache_query_patch_trig.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_query_patch_ok.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_query_patch_drop.store(0, std::memory_order_relaxed);
    _neighborcache_real_io_coaccess = neighborcache_env_u64("NEIGHBORCACHE_REAL_IO_COACCESS", 1) != 0;
    _neighborcache_real_io_direct = neighborcache_env_u64("NEIGHBORCACHE_REAL_IO_DIRECT", 1) != 0;
    _neighborcache_real_io_window_pairs.store(0, std::memory_order_relaxed);
    _neighborcache_real_io_adjacent_pairs.store(0, std::memory_order_relaxed);
    _neighborcache_real_io_window = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(16, neighborcache_env_u64("NEIGHBORCACHE_REAL_IO_WINDOW", 4))));
    _neighborcache_real_io_sample_queries = static_cast<uint32_t>(
        std::max<uint64_t>(1, neighborcache_env_u64("NEIGHBORCACHE_REAL_IO_SAMPLE_QUERIES", 64)));
    _neighborcache_real_io_max_partners = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(128, neighborcache_env_u64("NEIGHBORCACHE_REAL_IO_MAX_PARTNERS", 64))));
    _neighborcache_real_io_queries.store(0, std::memory_order_relaxed);
    _neighborcache_real_io_sampled_queries.store(0, std::memory_order_relaxed);
    _neighborcache_real_io_nodes.store(0, std::memory_order_relaxed);
    _neighborcache_real_io_pairs.store(0, std::memory_order_relaxed);
    _neighborcache_pending_buffer_cap_bytes =
        _neighborcache_deferred_ncache_page_write
            ? neighborcache_env_u64("NEIGHBORCACHE_PENDING_BUFFER_BYTES", 4ULL * 1024ULL * 1024ULL)
            : 0;
    _neighborcache_pending_buffer_payload_bytes = 0;
    _neighborcache_pending_require_full_page =
        _neighborcache_pending_buffer_cap_bytes > 0 &&
        neighborcache_env_u64("NEIGHBORCACHE_PENDING_REQUIRE_FULL_PAGE", 1) != 0;
    _neighborcache_pending_buffer_ready_percent = static_cast<uint32_t>(
        std::min<uint64_t>(100, neighborcache_env_u64("NEIGHBORCACHE_PENDING_BUFFER_READY_PERCENT", 0)));
    _neighborcache_pending_buffer_ready_min_members = static_cast<uint16_t>(
        std::min<uint64_t>(std::numeric_limits<uint16_t>::max(),
                           std::max<uint64_t>(1, neighborcache_env_u64("NEIGHBORCACHE_PENDING_BUFFER_READY_MIN_MEMBERS", 2))));
    _neighborcache_pending_buffer_inserts.store(0, std::memory_order_relaxed);
    _neighborcache_pending_buffer_ready_writes.store(0, std::memory_order_relaxed);
    _neighborcache_pending_buffer_pressure_writes.store(0, std::memory_order_relaxed);
    _neighborcache_pending_buffer_drops.store(0, std::memory_order_relaxed);
    _neighborcache_pending_member_refreshes.store(0, std::memory_order_relaxed);
    _neighborcache_pending_became_full.store(0, std::memory_order_relaxed);
    _neighborcache_monitor_warmup_queries = static_cast<uint32_t>(
        std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                           neighborcache_env_u64("NEIGHBORCACHE_MONITOR_WARMUP_QUERIES", 0)));
    _neighborcache_monitor_interval_queries = static_cast<uint32_t>(
        std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                           std::max<uint64_t>(1, neighborcache_env_u64("NEIGHBORCACHE_MONITOR_INTERVAL_QUERIES", 1000000))));
    _neighborcache_dcache_net_gate_enabled =
        neighborcache_env_u64("NEIGHBORCACHE_DCACHE_NET_GATE", _neighborcache_mcache.adaptive_update_enabled() ? 1 : 0) != 0;
    _neighborcache_dcache_probe_period = static_cast<uint32_t>(
        std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                           std::max<uint64_t>(1, neighborcache_env_u64("NEIGHBORCACHE_DCACHE_PROBE_PERIOD", 100))));
    _neighborcache_dcache_gate_active.store(true, std::memory_order_relaxed);
    _neighborcache_dcache_query_issued.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_query_completed.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_enabled_queries.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_avoided_pages.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_physical_reads.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_served_nodes.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_page_writes.store(0, std::memory_order_relaxed);
    _neighborcache_dcache_gate_transitions.store(0, std::memory_order_relaxed);
    // Each dedicated-page write costs SSD bandwidth that competes with query reads.
    if (const char *v = std::getenv("NEIGHBORCACHE_DCACHE_NET_WRITE_WEIGHT"))
        _neighborcache_dcache_net_write_weight = std::max(0.0, std::strtod(v, nullptr));
    else
        _neighborcache_dcache_net_write_weight = 0.0;
    const auto neighborcache_env_double = [](const char *name, double fallback, double minimum) {
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0')
            return fallback;
        char *end = nullptr;
        const double parsed = std::strtod(value, &end);
        return end == value || !std::isfinite(parsed) ? fallback : std::max(minimum, parsed);
    };
    _neighborcache_dcache_base_read_us =
        neighborcache_env_double("NEIGHBORCACHE_DCACHE_BASE_READ_US", 40.0, 0.0);
    _neighborcache_dcache_read_us =
        neighborcache_env_double("NEIGHBORCACHE_DCACHE_READ_US", 50.0, 0.0);
    _neighborcache_dcache_write_us =
        neighborcache_env_double("NEIGHBORCACHE_DCACHE_WRITE_US", 56.0, 0.0);
    _neighborcache_dcache_cpu_us =
        neighborcache_env_double("NEIGHBORCACHE_DCACHE_CPU_US", 20.0, 0.0);
    _neighborcache_dcache_gate_off_us =
        neighborcache_env_double("NEIGHBORCACHE_DCACHE_GATE_OFF_US", 0.0, -1000000.0);
    _neighborcache_dcache_gate_on_us =
        neighborcache_env_double("NEIGHBORCACHE_DCACHE_GATE_ON_US", 10.0, -1000000.0);
    {
        std::lock_guard<std::mutex> gate_lock(_neighborcache_dcache_gate_mu);
        _neighborcache_dcache_last_enabled_queries = 0;
        _neighborcache_dcache_last_avoided_pages = 0;
        _neighborcache_dcache_last_physical_reads = 0;
        _neighborcache_dcache_last_served_nodes = 0;
        _neighborcache_dcache_last_page_writes = 0;
        _neighborcache_dcache_low_windows = 0;
        _neighborcache_dcache_high_windows = 0;
        _neighborcache_dcache_inactive_stages = 0;
    }
    const auto pending_knob = [](const char *name, uint32_t fallback) {
        return static_cast<uint32_t>(
            std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                               std::max<uint64_t>(1, neighborcache_env_u64(name, fallback))));
    };
    _neighborcache_pending_max_age_queries = pending_knob("NEIGHBORCACHE_PENDING_MAX_AGE_QUERIES", 65536);
    _neighborcache_pending_check_interval = pending_knob("NEIGHBORCACHE_PENDING_CHECK_INTERVAL", 2048);
    _neighborcache_pending_force_batch = pending_knob("NEIGHBORCACHE_PENDING_FORCE_BATCH", 4);
    const uint64_t pending_pair_default = _neighborcache_pending_buffer_cap_bytes > 0 ? 16384 : 4096;
    _neighborcache_pending_pair_cap = static_cast<size_t>(
        std::min<uint64_t>(NEIGHBORCACHE_PENDING_PAIR_CAP_MAX,
                           std::max<uint64_t>(1, neighborcache_env_u64("NEIGHBORCACHE_PENDING_PAIR_CAP", pending_pair_default))));
    const uint64_t requested_heap_shards =
        neighborcache_env_u64("NEIGHBORCACHE_HEAP_SHARDS", NEIGHBORCACHE_DEFAULT_HEAP_SHARDS);
    _neighborcache_heap_shard_count =
        requested_heap_shards >= 1 && requested_heap_shards <= NEIGHBORCACHE_MAX_HEAP_SHARDS &&
                (requested_heap_shards & (requested_heap_shards - 1)) == 0
            ? static_cast<size_t>(requested_heap_shards)
            : NEIGHBORCACHE_DEFAULT_HEAP_SHARDS;
    const size_t heap_slot_capacity = static_cast<size_t>(_neighborcache_mcache.slot_capacity());
    for (auto &shard : _neighborcache_heap_shards)
    {
        NeighborCacheTimedMutexGuard hlock(shard.mu, NeighborCacheLockKind::Heap);
        shard.max_heap.clear();
        shard.min_heap.clear();
    }
    _neighborcache_heap_position.assign(heap_slot_capacity, NeighborCacheMetadataCache::kInvalid);
    _neighborcache_heap_kind.assign(heap_slot_capacity, static_cast<uint8_t>(NeighborCacheHeapKind::None));
    _neighborcache_node_state.reset(new std::atomic<uint8_t>[heap_slot_capacity]);
    _neighborcache_node_state_capacity = heap_slot_capacity;
    for (size_t slot = 0; slot < heap_slot_capacity; ++slot)
        _neighborcache_node_state[slot].store(static_cast<uint8_t>(NeighborCacheNodeState::NonSeed), std::memory_order_relaxed);
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
        _neighborcache_ready_pairs.clear();
        neighborcache_pending_member_clear();
        _neighborcache_deletion_to_pending.clear();
        _neighborcache_pending_buffer_buckets.clear();
        _neighborcache_pending_buffer_buckets.resize(static_cast<size_t>(nps) + 1);
        _neighborcache_reserved_free_pages = 0;
        _neighborcache_pending_pair_count.store(0, std::memory_order_relaxed);
        _neighborcache_deletion_pair_count.store(0, std::memory_order_relaxed);
        _neighborcache_query_epoch.store(0, std::memory_order_relaxed);
        _neighborcache_refresh_needed.store(false, std::memory_order_relaxed);
        _neighborcache_refresh_running.store(false, std::memory_order_relaxed);
        _neighborcache_score_unit.store(1.0f);
        _neighborcache_decay_query_count.store(0, std::memory_order_relaxed);
        _neighborcache_decay_access_count.store(0, std::memory_order_relaxed);
        _neighborcache_decay_cycle_start.store(0, std::memory_order_relaxed);
        _neighborcache_decay_half_life_accesses =
            static_cast<double>(std::max<uint64_t>(1, neighborcache_env_u64("NEIGHBORCACHE_DECAY_HALF_LIFE_ACCESSES", 2800000)));
        _neighborcache_decay_cycles.store(0, std::memory_order_relaxed);
        _neighborcache_decay_early_cycles.store(0, std::memory_order_relaxed);
        _neighborcache_decay_started.store(false, std::memory_order_relaxed);
        _neighborcache_pair_refresh = 0;
        _neighborcache_heap_writes = 0;
        _neighborcache_heap_deletes = 0;
        _neighborcache_ncache_write_trig = 0;
    }

    if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active() && _medoids != nullptr)
    {
        for (size_t i = 0; i < _num_medoids; i++)
            _neighborcache_mem_pool->pin(_medoids[i]);
    }

    _neighborcache_dyn_path = flush_prefix.empty() ? std::string() : (flush_prefix + "_neighborcache_dc.dyn.data");
    if (_neighborcache_dyn_path.empty())
        throw ANNException("Dynamic 3-cache requires a disk-cache file path.", -1, __FUNCSIG__, __FILE__, __LINE__);
    if (_neighborcache_dyn_reader)
    {
        _neighborcache_dyn_reader->close();
        _neighborcache_dyn_reader.reset();
    }
    if (_neighborcache_dyn_write_stream.is_open())
        _neighborcache_dyn_write_stream.close();
#ifndef _WINDOWS
    if (_neighborcache_dyn_write_fd >= 0)
    {
        ::close(_neighborcache_dyn_write_fd);
        _neighborcache_dyn_write_fd = -1;
    }
#endif
    const uint64_t requested_loc_shards =
        neighborcache_env_u64("NEIGHBORCACHE_DYN_LOC_SHARDS", 64);
    _neighborcache_dyn_loc_shard_count =
        requested_loc_shards >= 1 && requested_loc_shards <= NEIGHBORCACHE_DYN_LOC_MAX_SHARDS &&
                (requested_loc_shards & (requested_loc_shards - 1)) == 0
            ? static_cast<size_t>(requested_loc_shards)
            : 64;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        _neighborcache_dyn_page_members.clear();
        _neighborcache_dyn_page_member_count.clear();
        _neighborcache_dyn_members_per_page = static_cast<uint32_t>(nps);
        if (!import_static_prefill)
            _neighborcache_dc_map.clear();
        neighborcache_dyn_loc_clear_unlocked();
        const uint64_t page_multiplier =
            _neighborcache_dcache_second_page ? static_cast<uint64_t>(_neighborcache_dcache_max_pages) : 1ULL;
        // Keep seed_member_loc / canonical through static prefill import; they encode
        // the real seed-group page layout. Clear after import if not prefilling.
        if (!import_static_prefill)
        {
            _neighborcache_dc_seed_member_loc.clear();
            _neighborcache_dc_seed_canonical_loc.clear();
        }
        _neighborcache_dyn_free.clear();
        _neighborcache_dyn_committed_pages = 0;
        _neighborcache_dyn_next_page = 0;
        _neighborcache_dyn_physical_cap = _neighborcache_dyn_page_cap * page_multiplier + std::max<uint64_t>(_max_nthreads, 1) + 1;
        if (_neighborcache_dyn_physical_cap >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / _neighborcache_dyn_members_per_page)
            throw ANNException("Dynamic d-cache member directory is too large.", -1, __FUNCSIG__, __FILE__,
                               __LINE__);
        _neighborcache_dyn_page_members.assign(
            static_cast<size_t>(_neighborcache_dyn_physical_cap) * _neighborcache_dyn_members_per_page, NEIGHBORCACHE_DYN_INVALID_PAGE);
        _neighborcache_dyn_page_member_count.assign(static_cast<size_t>(_neighborcache_dyn_physical_cap), 0);
        _neighborcache_dyn_page_sibling.assign(static_cast<size_t>(_neighborcache_dyn_physical_cap), NEIGHBORCACHE_DYN_INVALID_PAGE);
        _neighborcache_dyn_page_secondary.assign(static_cast<size_t>(_neighborcache_dyn_physical_cap), 0);
        _neighborcache_dyn_page_span_len.assign(static_cast<size_t>(_neighborcache_dyn_physical_cap), 0);
        _neighborcache_dyn_page_span_base.assign(static_cast<size_t>(_neighborcache_dyn_physical_cap), NEIGHBORCACHE_DYN_INVALID_PAGE);
        _neighborcache_dyn_page_readers.reset(new std::atomic<uint32_t>[_neighborcache_dyn_physical_cap]);
        _neighborcache_dyn_page_gen.reset(new std::atomic<uint32_t>[_neighborcache_dyn_physical_cap]);
        _neighborcache_dyn_page_retired.reset(new std::atomic<uint8_t>[_neighborcache_dyn_physical_cap]);
        _neighborcache_span_reads.reset(new std::atomic<uint32_t>[_neighborcache_dyn_physical_cap]);
        _neighborcache_span_hits.reset(new std::atomic<uint32_t>[_neighborcache_dyn_physical_cap]);
        _neighborcache_span_decisions.reset(new std::atomic<uint32_t>[_neighborcache_dyn_physical_cap]);
        for (uint64_t i = 0; i < _neighborcache_dyn_physical_cap; ++i)
        {
            _neighborcache_dyn_page_readers[i].store(0, std::memory_order_relaxed);
            _neighborcache_dyn_page_gen[i].store(0, std::memory_order_relaxed);
            _neighborcache_dyn_page_retired[i].store(0, std::memory_order_relaxed);
            _neighborcache_span_reads[i].store(0, std::memory_order_relaxed);
            _neighborcache_span_hits[i].store(0, std::memory_order_relaxed);
            _neighborcache_span_decisions[i].store(0, std::memory_order_relaxed);
        }
        _neighborcache_prefetch_dirty.reset();
        _neighborcache_prefetch_writeback.reset();
        if (_neighborcache_seed_prefetch)
        {
            _neighborcache_prefetch_dirty.reset(new std::atomic<uint64_t>[_neighborcache_dyn_physical_cap]);
            _neighborcache_prefetch_writeback.reset(new std::atomic<uint32_t>[_neighborcache_dyn_physical_cap]);
            for (uint64_t i = 0; i < _neighborcache_dyn_physical_cap; ++i)
            {
                _neighborcache_prefetch_dirty[i].store(0, std::memory_order_relaxed);
                _neighborcache_prefetch_writeback[i].store(0, std::memory_order_relaxed);
            }
        }
        neighborcache_canon_reset();
        {
            std::lock_guard<std::mutex> rlock(_neighborcache_dyn_reclaim_mu);
            _neighborcache_dyn_reclaim.clear();
        }
    }
    const uint64_t dyn_file_bytes = _neighborcache_dyn_physical_cap * defaults::SECTOR_LEN;
#ifndef _WINDOWS
    _neighborcache_dyn_write_fd =
        ::open(_neighborcache_dyn_path.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_DIRECT | O_LARGEFILE, 0644);
    if (_neighborcache_dyn_write_fd < 0 || ::ftruncate(_neighborcache_dyn_write_fd, static_cast<off_t>(dyn_file_bytes)) != 0)
        throw ANNException("Failed to create dynamic disk-cache file.", -1, __FUNCSIG__, __FILE__, __LINE__);
#else
    _neighborcache_dyn_write_stream.open(_neighborcache_dyn_path,
                                 std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!_neighborcache_dyn_write_stream.is_open())
        throw ANNException("Failed to create dynamic disk-cache file.", -1, __FUNCSIG__, __FILE__, __LINE__);
    _neighborcache_dyn_write_stream.seekp(static_cast<std::streamoff>(dyn_file_bytes - 1));
    const char zero = 0;
    _neighborcache_dyn_write_stream.write(&zero, 1);
    _neighborcache_dyn_write_stream.flush();
    if (!_neighborcache_dyn_write_stream.good())
        throw ANNException("Failed to preallocate dynamic disk-cache file.", -1, __FUNCSIG__, __FILE__, __LINE__);
#endif
    _neighborcache_dyn_write_q_cap = static_cast<size_t>(
        std::max<uint64_t>(1, neighborcache_env_u64("NEIGHBORCACHE_DCACHE_WRITER_QUEUE", 64)));
    _neighborcache_dyn_writer_thread_count = static_cast<size_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(16, neighborcache_env_u64("NEIGHBORCACHE_DCACHE_WRITER_THREADS", 1))));
    _neighborcache_commit_thread_count = static_cast<size_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(16, neighborcache_env_u64(
            "NEIGHBORCACHE_DCACHE_COMMIT_THREADS", _neighborcache_dyn_writer_thread_count))));
    _neighborcache_commit_q_cap = static_cast<size_t>(
        std::max<uint64_t>(8, std::min<uint64_t>(65536, neighborcache_env_u64("NEIGHBORCACHE_DCACHE_COMMIT_QUEUE", 256))));
    _neighborcache_commit_inline = neighborcache_env_u64("NEIGHBORCACHE_DCACHE_COMMIT_INLINE", 0) != 0;
    _neighborcache_dyn_writer_enqueued.store(0, std::memory_order_relaxed);
    _neighborcache_dyn_writer_queue_ns.store(0, std::memory_order_relaxed);
    _neighborcache_dyn_writer_io_ns.store(0, std::memory_order_relaxed);
    _neighborcache_dyn_writer_completed.store(0, std::memory_order_relaxed);
    _neighborcache_dyn_writer_failed.store(0, std::memory_order_relaxed);
    _neighborcache_writeback_queue_drops.store(0, std::memory_order_relaxed);
    _neighborcache_writeback_read_bypasses.store(0, std::memory_order_relaxed);
    if (_neighborcache_seed_prefetch && _neighborcache_shared_cache.enabled())
        _neighborcache_shared_cache.set_evict_hook(&PQFlashIndex<T, LabelT>::neighborcache_prefetch_evict_hook, this);
    _neighborcache_commit_enqueued.store(0, std::memory_order_relaxed);
    _neighborcache_commit_coalesced.store(0, std::memory_order_relaxed);
    _neighborcache_commit_dropped.store(0, std::memory_order_relaxed);
    _neighborcache_commit_completed.store(0, std::memory_order_relaxed);
    _neighborcache_commit_failed.store(0, std::memory_order_relaxed);
    _neighborcache_commit_queue_ns.store(0, std::memory_order_relaxed);
    _neighborcache_commit_work_ns.store(0, std::memory_order_relaxed);
    for (size_t source = 0; source < _neighborcache_commit_source_enqueued.size(); ++source)
    {
        _neighborcache_commit_source_enqueued[source].store(0, std::memory_order_relaxed);
        _neighborcache_commit_source_completed[source].store(0, std::memory_order_relaxed);
    }
    start_neighborcache_dyn_writer();
#ifndef _WINDOWS
    _neighborcache_dyn_reader.reset(new LinuxAlignedFileReader());
#else
    _neighborcache_dyn_reader.reset(new WindowsAlignedFileReader());
#endif
    _neighborcache_dyn_reader->open(_neighborcache_dyn_path);
#pragma omp parallel for num_threads((int)_max_nthreads)
    for (int64_t thread = 0; thread < static_cast<int64_t>(_max_nthreads); ++thread)
    {
#pragma omp critical
        { _neighborcache_dyn_reader->register_thread(); }
    }
    if (!_neighborcache_static_no_mcache)
    {
        start_neighborcache_dyn_patch_worker();
        start_neighborcache_dyn_commit_workers();
    }
    if (import_static_prefill && neighborcache_dyn_import_static_prefill() != 0)
        throw ANNException("Failed to import the static BFS d-cache into dynamic NEIGHBORCACHE.", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    // After dyn import, per-member locs are authoritative. Seed-first canonical-extent
    // IO is for raw static layout; keeping it on with remapped sidecars corrupts reads
    // when a node also appears on another seed's page.
    if (import_static_prefill)
    {
        _neighborcache_seed_first_lookup = false;
        diskann::cout << "NEIGHBORCACHE static prefill: seed-first lookup forced off (use dyn member locs)"
                      << std::endl;
    }
    _neighborcache_mem_runtime_admit = true;
    if (!import_static_prefill)
        _neighborcache_seed_first_lookup = true;
    _neighborcache_stash_cap = _neighborcache_coread_pack
                           ? neighborcache_env_u64("NEIGHBORCACHE_STASH_CAP", 65536)
                           : (_neighborcache_deferred_ncache_page_write
                                  ? 0
                                  : neighborcache_env_u64("NEIGHBORCACHE_STASH_CAP",
                                                  ncache_n > 0 ? std::max<uint64_t>(ncache_n * 2ULL, 4096) : 16384));
    {
        std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
        _neighborcache_payload_stash.clear();
        _neighborcache_stash_lru.clear();
        _neighborcache_pending_buffer_payload_bytes = 0;
        _neighborcache_parent_fetched_by_slot.assign(static_cast<size_t>(_neighborcache_mcache.slot_capacity()),
                                             NeighborCacheRecentChildren{});
        _neighborcache_parent_fetched_owner.assign(static_cast<size_t>(_neighborcache_mcache.slot_capacity()),
                                           NeighborCacheMetadataCache::kInvalid);
    }
    diskann::cout << "NEIGHBORCACHE dynamic 3-cache: m-cache cap=" << mcache_cap << " (2x seed cap) "
                  << "edge_storage=vector(unbounded) T=" << (int)sig_t
                  << " heap+delay-write nps=" << nps << " d-cache pages<=" << _neighborcache_dyn_page_cap
                  << " (space cap=" << dcache_space_cap << ", base pages=" << base_pages << ")"
                  << " file=" << _neighborcache_dyn_path << " physical_pages=" << _neighborcache_dyn_physical_cap
                  << " score=float32 unit=1..16/100x1k-query"
                  << " heap_shards=" << _neighborcache_heap_shard_count
                  << " dcache_net_gate=" << (_neighborcache_dcache_net_gate_enabled ? "on" : "off")
                  << " dcache_probe=1/" << _neighborcache_dcache_probe_period
                  << " deferred_ncache_page_write=" << (_neighborcache_deferred_ncache_page_write ? "on" : "off")
                  << " seed_pin_neighbors=" << (_neighborcache_seed_pin_neighbors ? "on" : "off")
                  << " seed_pin_age=" << _neighborcache_seed_pin_age_queries
                  << " seed_pin_writes_per_query=" << _neighborcache_seed_pin_writes_per_query
                  << " seed_page_lru=" << (_neighborcache_seed_page_lru ? "on" : "off")
                  << " seed_prefetch=" << (_neighborcache_seed_prefetch ? "on" : "off")
                  << " prefetch_writeback=async_detached"
                  << " seed_page_pages=" << _neighborcache_seed_page_cap
                  << " dcache_family=" << (_neighborcache_dcache_family ? "on" : "off")
                  << " dcache_family_cap=" << _neighborcache_dcache_family_cap
                  << " coread_pack=" << (_neighborcache_coread_pack ? "on" : "off")
                  << " canon_slots=" << (_neighborcache_canon_mask + 1)
                  << " dcache_writer=on queue=" << _neighborcache_dyn_write_q_cap
                  << " dcache_ncache_exclusive="
                  << (_neighborcache_dcache_evict_ncache_on_commit ? "on" : "off")
                  << " real_io_coaccess=" << (_neighborcache_real_io_coaccess ? "on" : "off")
                  << " real_io_window=" << _neighborcache_real_io_window
                  << " real_io_sample=1/" << _neighborcache_real_io_sample_queries
                  << " pending_buffer_bytes=" << _neighborcache_pending_buffer_cap_bytes
                  << " pending_buffer_ready=" << _neighborcache_pending_buffer_ready_percent << "%"
                  << "/" << _neighborcache_pending_buffer_ready_min_members << "-members"
                  << " pending_require_full_page=" << (_neighborcache_pending_require_full_page ? "on" : "off")
                  << " pending_pair_cap=" << _neighborcache_pending_pair_cap
                  << " dcache_no_seed_replace=" << (_neighborcache_dcache_no_seed_replace ? "on" : "off")
                  << " dcache_second_page=" << (_neighborcache_dcache_second_page ? "on" : "off")
                  << " dcache_max_pages=" << static_cast<unsigned>(_neighborcache_dcache_max_pages)
                  << " dcache_freeze_after=" << _neighborcache_dcache_freeze_after
                  << " dcache_query_patch=" << (_neighborcache_dcache_query_patch ? "on" : "off")
                  << " repack_on_read=" << (_neighborcache_repack_on_read ? "on" : "off")
                  << " patch_async=" << (_neighborcache_dcache_query_patch_async ? "on" : "off")
                  << " patch_max=" << _neighborcache_dcache_query_patch_max
                  << " patch_q=" << _neighborcache_patch_q_cap
                  << " patch_batch_us=" << _neighborcache_patch_batch_us
                  << " patch_no_disk=" << (_neighborcache_patch_worker_no_disk ? "on" : "off")
                  << " patch_stash_cap=" << _neighborcache_stash_cap
                  << std::endl;
}

template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::neighborcache_dyn_import_static_prefill()
{
    if (_neighborcache_dc_path.empty() || _neighborcache_dc_map.empty() || _neighborcache_dyn_write_fd < 0)
        return -1;

    // Prefer seed_member_loc for page_count: dense-fill / flat padding may extend
    // _neighborcache_dc_map past the real seed-group pages and those extra pages have no seed.
    uint64_t page_count = 0;
    if (!_neighborcache_dc_seed_member_loc.empty())
    {
        for (const auto &seed_members : _neighborcache_dc_seed_member_loc)
            for (const auto &member : seed_members.second)
                page_count = std::max<uint64_t>(page_count, static_cast<uint64_t>(member.second.sector) + 1ULL);
    }
    else
    {
        for (const auto &entry : _neighborcache_dc_map)
        {
            for (const NeighborCacheDiskLoc &loc : entry.second)
                page_count = std::max<uint64_t>(page_count, static_cast<uint64_t>(loc.sector) + 1ULL);
        }
    }
    const uint64_t file_pages = get_file_size(_neighborcache_dc_path) / defaults::SECTOR_LEN;
    if (file_pages > 0 && page_count > file_pages)
    {
        diskann::cerr << "NEIGHBORCACHE BFS prefill: map pages=" << page_count << " exceeds file pages=" << file_pages
                      << " (" << _neighborcache_dc_path << ")." << std::endl;
        return -1;
    }
    if (page_count == 0 || page_count > _neighborcache_dyn_page_cap)
    {
        diskann::cerr << "NEIGHBORCACHE BFS prefill page count " << page_count << " exceeds dynamic capacity "
                      << _neighborcache_dyn_page_cap << "." << std::endl;
        return -1;
    }

#ifndef _WINDOWS
    const int src_fd = ::open(_neighborcache_dc_path.c_str(), O_RDONLY | O_DIRECT | O_LARGEFILE);
    if (src_fd < 0)
    {
        diskann::cerr << "NEIGHBORCACHE BFS prefill could not open " << _neighborcache_dc_path << "." << std::endl;
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
            !neighborcache_dyn_writer_submit_and_wait(NeighborCacheDynWriteOp::Write, static_cast<off_t>(offset), buffer, bytes))
        {
            copy_ok = false;
            break;
        }
        offset += bytes;
    }
    ::free(buffer);
    ::close(src_fd);
    if (!copy_ok || !neighborcache_dyn_writer_submit_and_wait(NeighborCacheDynWriteOp::Sync, 0, nullptr, 0))
    {
        diskann::cerr << "NEIGHBORCACHE BFS prefill failed while copying " << bytes_to_copy << " bytes." << std::endl;
        return -1;
    }
#else
    diskann::cerr << "NEIGHBORCACHE BFS dynamic prefill is not implemented on Windows." << std::endl;
    return -1;
#endif

    std::vector<uint32_t> seeds(static_cast<size_t>(page_count), NEIGHBORCACHE_DYN_INVALID_PAGE);
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        // seed_member_loc / map store unique members only; padded seed replicas may be absent.
        // Reconstruct full pages: place known (sector,slot) members, then fill holes with the page seed.
        std::vector<std::vector<uint32_t>> page_slots(
            static_cast<size_t>(page_count),
            std::vector<uint32_t>(_neighborcache_dyn_members_per_page, NEIGHBORCACHE_DYN_INVALID_PAGE));
        auto place = [&](uint32_t node_id, const NeighborCacheDiskLoc &static_loc) {
            if (static_loc.sector >= page_count || static_loc.slot >= _neighborcache_dyn_members_per_page)
                return;
            page_slots[static_loc.sector][static_loc.slot] = node_id;
        };
        if (!_neighborcache_dc_seed_member_loc.empty())
        {
            // Page owner is the seed-group key, NOT "whoever sits in slot 0".
            // directed_seed_replica continuation pages are child-only in slots 0..n-1
            // and only pad with the group seed — treating slot0 as seed corrupts pads.
            for (const auto &seed_members : _neighborcache_dc_seed_member_loc)
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
            for (const auto &entry : _neighborcache_dc_map)
                for (const NeighborCacheDiskLoc &static_loc : entry.second)
                {
                    place(entry.first, static_loc);
                    if (static_loc.slot == 0 && static_loc.sector < page_count)
                        seeds[static_loc.sector] = entry.first;
                }
        }
        for (const auto &kv : _neighborcache_dc_seed_canonical_loc)
        {
            if (kv.second.sector < page_count && seeds[kv.second.sector] == NEIGHBORCACHE_DYN_INVALID_PAGE)
                seeds[kv.second.sector] = kv.first;
        }
        for (uint64_t page = 0; page < page_count; ++page)
        {
            uint32_t seed = seeds[page];
            if (seed == NEIGHBORCACHE_DYN_INVALID_PAGE)
            {
                for (uint32_t slot = 0; slot < _neighborcache_dyn_members_per_page; ++slot)
                {
                    if (page_slots[page][slot] != NEIGHBORCACHE_DYN_INVALID_PAGE)
                    {
                        seed = page_slots[page][slot];
                        break;
                    }
                }
            }
            if (seed == NEIGHBORCACHE_DYN_INVALID_PAGE)
            {
                diskann::cerr << "NEIGHBORCACHE BFS prefill page " << page << " has no seed." << std::endl;
                return -1;
            }
            seeds[page] = seed;
            if (page_slots[page][0] == NEIGHBORCACHE_DYN_INVALID_PAGE)
                page_slots[page][0] = seed;
            neighborcache_dyn_bump_page_gen(static_cast<uint32_t>(page));
            for (uint32_t slot = 0; slot < _neighborcache_dyn_members_per_page; ++slot)
            {
                if (page_slots[page][slot] == NEIGHBORCACHE_DYN_INVALID_PAGE)
                    page_slots[page][slot] = seed;
                const uint32_t node_id = page_slots[page][slot];
                NeighborCacheDiskLoc dynamic_loc;
                dynamic_loc.sector = static_cast<uint32_t>(NEIGHBORCACHE_DYN_SECTOR_BASE + page);
                dynamic_loc.slot = static_cast<uint16_t>(slot);
                dynamic_loc.nsectors = 1;
                neighborcache_dyn_loc_add_unlocked(node_id, dynamic_loc);
                const size_t member_index =
                    static_cast<size_t>(page) * _neighborcache_dyn_members_per_page + slot;
                _neighborcache_dyn_page_members[member_index] = node_id;
            }
            _neighborcache_dyn_page_member_count[page] = static_cast<uint16_t>(_neighborcache_dyn_members_per_page);
            neighborcache_dyn_bump_page_gen(static_cast<uint32_t>(page));
        }
        _neighborcache_dyn_committed_pages = page_count;
        _neighborcache_dyn_next_page = page_count;
        _neighborcache_dc_map.clear();

        // Remap static sidecar sectors → dynamic tagged sectors so any leftover
        // seed-member strict paths stay consistent with imported dyn pages.
        if (!_neighborcache_dc_seed_member_loc.empty())
        {
            tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, NeighborCacheDiskLoc>> remapped_members;
            remapped_members.reserve(_neighborcache_dc_seed_member_loc.size());
            for (const auto &seed_members : _neighborcache_dc_seed_member_loc)
            {
                auto &out_members = remapped_members[seed_members.first];
                for (const auto &member : seed_members.second)
                {
                    NeighborCacheDiskLoc loc;
                    loc.sector = member.second.sector;
                    loc.slot = member.second.slot;
                    loc.nsectors = member.second.nsectors;
                    if (loc.sector < page_count)
                        loc.sector = static_cast<uint32_t>(NEIGHBORCACHE_DYN_SECTOR_BASE + loc.sector);
                    out_members.emplace(member.first, loc);
                }
            }
            _neighborcache_dc_seed_member_loc = std::move(remapped_members);

            tsl::robin_map<uint32_t, NeighborCacheDiskLoc> remapped_canonical;
            remapped_canonical.reserve(_neighborcache_dc_seed_canonical_loc.size());
            for (const auto &kv : _neighborcache_dc_seed_canonical_loc)
            {
                NeighborCacheDiskLoc loc;
                loc.sector = kv.second.sector;
                loc.slot = kv.second.slot;
                loc.nsectors = kv.second.nsectors;
                if (loc.sector < page_count)
                    loc.sector = static_cast<uint32_t>(NEIGHBORCACHE_DYN_SECTOR_BASE + loc.sector);
                remapped_canonical.emplace(kv.first, loc);
            }
            _neighborcache_dc_seed_canonical_loc = std::move(remapped_canonical);
        }
    }

    if (!_neighborcache_static_no_mcache)
    for (uint32_t seed : seeds)
    {
        const auto tr = _neighborcache_mcache.on_expand(seed, _neighborcache_score_unit);
        if (!tr.present || tr.evicted_id != NeighborCacheMetadataCache::kInvalid)
        {
            diskann::cerr << "NEIGHBORCACHE BFS prefill could not register all page seeds in m-cache." << std::endl;
            return -1;
        }
        {
            NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
            neighborcache_dyn_set_state_unlocked(seed, tr.slot_id, NeighborCacheNodeState::Seed);
        }
        NeighborCacheHeapShard &shard = neighborcache_dyn_heap_shard(tr.slot_id);
        NeighborCacheTimedMutexGuard hlock(shard.mu, NeighborCacheLockKind::Heap);
        neighborcache_dyn_heap_upsert_unlocked(tr.slot_id, tr.score);
    }

    if (_neighborcache_disk_reader)
    {
        _neighborcache_disk_reader->close();
        _neighborcache_disk_reader.reset();
    }
    diskann::cout << "NEIGHBORCACHE dynamic BFS prefill: " << page_count << "/" << _neighborcache_dyn_page_cap
                  << " pages, " << (page_count * _neighborcache_dyn_members_per_page)
                  << " nodes; all pages full before measurement." << std::endl;
    return 0;
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::neighborcache_dynamic_3cache_enabled() const
{
    return _neighborcache_dyn_enabled;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::neighborcache_dynamic_flush_count() const
{
    return _neighborcache_dyn_flush_count;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::neighborcache_dynamic_page_count() const
{
    NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
    return _neighborcache_dyn_committed_pages;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::neighborcache_dynamic_page_capacity() const
{
    return _neighborcache_dyn_page_cap;
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::neighborcache_dynamic_disk_full() const
{
    return neighborcache_dyn_disk_is_full();
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dynamic_begin_measurement()
{
    _neighborcache_monitor_warmup_queries = _neighborcache_query_epoch.load(std::memory_order_acquire);
    _neighborcache_pending_buffer_inserts.store(0, std::memory_order_relaxed);
    _neighborcache_pending_buffer_ready_writes.store(0, std::memory_order_relaxed);
    _neighborcache_pending_buffer_pressure_writes.store(0, std::memory_order_relaxed);
    _neighborcache_pending_buffer_drops.store(0, std::memory_order_relaxed);
    _neighborcache_pending_member_refreshes.store(0, std::memory_order_relaxed);
    _neighborcache_pending_became_full.store(0, std::memory_order_relaxed);
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
        _neighborcache_heap_writes = 0;
        _neighborcache_heap_deletes = 0;
        _neighborcache_ncache_write_trig = 0;
    }
    // Default: when D-cache is already full, freeze M/pair/heap maintenance for
    // the timed run but keep D-page lookups. Override with
    // NEIGHBORCACHE_FREEZE_MAINTENANCE=0 to keep updating, or =1 to force freeze.
    const uint64_t freeze_env = neighborcache_env_u64("NEIGHBORCACHE_FREEZE_MAINTENANCE", 2);
    const bool disk_full = neighborcache_dyn_disk_is_full();
    _neighborcache_freeze_maintenance = _neighborcache_static_no_mcache || freeze_env == 1 || (freeze_env == 2 && disk_full);
    if (_neighborcache_freeze_maintenance)
    {
        stop_neighborcache_dyn_patch_worker();
        neighborcache_dyn_wait_for_commits();
    }
    _neighborcache_dyn_member_bits.clear();
    if (_neighborcache_freeze_maintenance && _num_points > 0)
    {
        const uint64_t words = (_num_points + 63ULL) / 64ULL;
        _neighborcache_dyn_member_bits.assign(static_cast<size_t>(words), 0ULL);
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        for (size_t shard_index = 0; shard_index < _neighborcache_dyn_loc_shard_count; ++shard_index)
        {
            const NeighborCacheDynLocShard &shard = *_neighborcache_dyn_loc_shards[shard_index];
            NeighborCacheTimedSharedMutexGuard shard_lock(shard.mu, NeighborCacheLockKind::DynamicShared, false);
            for (const auto &kv : shard.locations)
            {
                const uint32_t node_id = kv.first;
                if (static_cast<uint64_t>(node_id) >= _num_points)
                    continue;
                bool live = false;
                for (const NeighborCacheDiskLoc &loc : kv.second)
                {
                    if (neighborcache_dyn_is_sector(loc.sector))
                    {
                        const uint32_t idx = loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
                        if (neighborcache_dyn_page_live(idx))
                        {
                            live = true;
                            break;
                        }
                    }
                }
                if (!live)
                    continue;
                _neighborcache_dyn_member_bits[static_cast<size_t>(node_id >> 6)] |=
                    (1ULL << (node_id & 63u));
            }
        }
    }
    diskann::cout << "NEIGHBORCACHE measurement counters reset at query="
                  << _neighborcache_monitor_warmup_queries
                  << " freeze_maintenance=" << (_neighborcache_freeze_maintenance ? "on" : "off")
                  << " disk_full=" << (disk_full ? "yes" : "no")
                  << " member_filter_bits="
                  << (_neighborcache_dyn_member_bits.size() * 64ULL) << std::endl;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::neighborcache_dynamic_clean_seeds() const
{
    NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    uint64_t n = 0;
    for (uint16_t count : _neighborcache_dyn_page_member_count)
        if (count >= nps)
            n++;
    return n;
}

template <typename T, typename LabelT> uint64_t PQFlashIndex<T, LabelT>::neighborcache_dynamic_partial_seeds() const
{
    NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    uint64_t n = 0;
    for (uint16_t count : _neighborcache_dyn_page_member_count)
        if (count > 0 && count < nps)
            n++;
    return n;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::print_neighborcache_dynamic_3cache_stats() const
{
    if (!_neighborcache_dyn_enabled)
        return;
    // Optional post-search snapshot for offline page-neighbor coverage analysis.
    // Called after timed search. Copy under the directory lock, then write outside it.
    // Format: uint32 magic/version/slots, followed by records of
    // uint32 page/secondary/count and `slots` uint32 member IDs (slot 0 is seed).
    if (const char *path = std::getenv("NEIGHBORCACHE_DUMP_DYNAMIC_PAGES"); path && path[0])
    {
        std::vector<uint32_t> records;
        const uint32_t slots = _neighborcache_dyn_members_per_page;
        {
            NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
            for (size_t page = 0; page < _neighborcache_dyn_page_member_count.size(); ++page)
            {
                const uint32_t count = _neighborcache_dyn_page_member_count[page];
                if (count == 0)
                    continue;
                const size_t begin = page * slots;
                // A seed can have several physical copies. Export the primary page
                // selected by the live seed lookup, not every copy as a separate seed.
                uint32_t selected_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
                if (!neighborcache_dyn_seed_page_unlocked(_neighborcache_dyn_page_members[begin], selected_page) ||
                    selected_page != page)
                    continue;
                records.push_back(static_cast<uint32_t>(page));
                records.push_back(page < _neighborcache_dyn_page_secondary.size() && _neighborcache_dyn_page_secondary[page]);
                records.push_back(count);
                for (uint32_t slot = 0; slot < slots; ++slot)
                    records.push_back(slot < count ? _neighborcache_dyn_page_members[begin + slot] : UINT32_MAX);
            }
        }
        const uint32_t header[] = {0x4e435047, 1, slots};
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(header), sizeof(header));
        out.write(reinterpret_cast<const char *>(records.data()), records.size() * sizeof(uint32_t));
        if (!out)
            throw ANNException("Failed to export dynamic page snapshot", -1, __FUNCSIG__, __FILE__, __LINE__);
        diskann::cout << "NEIGHBORCACHE dynamic page snapshot: pages=" << records.size() / (slots + 3)
                      << " slots=" << slots << " path=" << path << std::endl;
    }
    uint64_t dynamic_pages = 0;
    uint64_t dynamic_loc_entries = 0;
    uint64_t dynamic_locs = 0;
    uint64_t dynamic_loc_capacity = 0;
    uint64_t seed_dir_entries = 0;
    uint64_t seed_dir_members = 0;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        dynamic_pages = _neighborcache_dyn_committed_pages;
        for (size_t shard_index = 0; shard_index < _neighborcache_dyn_loc_shard_count; ++shard_index)
        {
            const NeighborCacheDynLocShard &shard = *_neighborcache_dyn_loc_shards[shard_index];
            NeighborCacheTimedSharedMutexGuard shard_lock(shard.mu, NeighborCacheLockKind::DynamicShared, false);
            dynamic_loc_entries += shard.locations.size();
            for (const auto &kv : shard.locations)
            {
                dynamic_locs += kv.second.size();
                dynamic_loc_capacity += kv.second.capacity();
            }
        }
        seed_dir_entries = _neighborcache_dyn_committed_pages;
        for (uint16_t count : _neighborcache_dyn_page_member_count)
            seed_dir_members += count;
    }
    diskann::cout << "NEIGHBORCACHE dynamic 3-cache stats: m-cache=" << _neighborcache_mcache.size() << "/" << _neighborcache_mcache.capacity()
                  << " mcache_edges=" << _neighborcache_mcache.edge_count()
                  << " n-cache="
                  << ((_neighborcache_mem_pool && _neighborcache_mem_pool->active()) ? _neighborcache_mem_pool->size() : 0) << "/"
                  << ((_neighborcache_mem_pool && _neighborcache_mem_pool->active()) ? _neighborcache_mem_pool->capacity() : 0)
                  << " flushes=" << _neighborcache_dyn_flush_count << " clean=" << neighborcache_dynamic_clean_seeds()
                  << " partial=" << neighborcache_dynamic_partial_seeds() << " dyn_pages=" << dynamic_pages
                  << " coread_pages=" << _neighborcache_coread_pages.load(std::memory_order_relaxed)
                  << " hop2_pages=" << _neighborcache_coread_hop2_pages.load(std::memory_order_relaxed)
                  << " coread_enqueued=" << _neighborcache_coread_enqueued.load(std::memory_order_relaxed)
                  << " coread_dropped=" << _neighborcache_coread_write_dropped.load(std::memory_order_relaxed)
                  << std::endl;
    {
        const uint64_t nwrites = g_neighborcache_seed_writes.load(std::memory_order_relaxed);
        const double inv = nwrites > 0 ? 1.0 / static_cast<double>(nwrites) : 0.0;
        diskann::cout << std::fixed << std::setprecision(3);
        diskann::cout << "NEIGHBORCACHE seed-write nbrs: n=" << nwrites
                      << " graph_mean=" << (g_neighborcache_seed_graph_sum.load(std::memory_order_relaxed) * inv)
                      << " graph_median=" << neighborcache_seed_hist_median(g_neighborcache_seed_graph_hist, nwrites)
                      << " followed_mean=" << (g_neighborcache_seed_followed_sum.load(std::memory_order_relaxed) * inv)
                      << " followed_median=" << neighborcache_seed_hist_median(g_neighborcache_seed_followed_hist, nwrites)
                      << " sig_mean=" << (g_neighborcache_seed_sig_sum.load(std::memory_order_relaxed) * inv)
                      << " packed_mean=" << (g_neighborcache_seed_packed_sum.load(std::memory_order_relaxed) * inv)
                      << std::endl;
        diskann::cout << "NEIGHBORCACHE seed-write graph_hist:";
        for (int i = 0; i < kNeighborCacheSeedNbrHist; ++i)
            diskann::cout << ' ' << g_neighborcache_seed_graph_hist[i].load(std::memory_order_relaxed);
        diskann::cout << std::endl;
        diskann::cout << "NEIGHBORCACHE seed-write followed_hist:";
        for (int i = 0; i < kNeighborCacheSeedNbrHist; ++i)
            diskann::cout << ' ' << g_neighborcache_seed_followed_hist[i].load(std::memory_order_relaxed);
        diskann::cout << std::endl;
        if (neighborcache_log_seed_ids())
        {
            const char *path = std::getenv("NEIGHBORCACHE_SEED_WRITE_IDS");
            std::lock_guard<std::mutex> lock(g_neighborcache_seed_id_mu);
            std::ofstream id_out(path, std::ios::binary | std::ios::trunc);
            if (id_out)
                id_out.write(reinterpret_cast<const char *>(g_neighborcache_seed_ids.data()),
                             g_neighborcache_seed_ids.size() * sizeof(uint32_t));
            diskann::cout << "NEIGHBORCACHE seed-write ids: n=" << g_neighborcache_seed_ids.size() << " path=" << path
                          << std::endl;
        }
        if (_neighborcache_seed_pin_neighbors)
            diskann::cout << "NEIGHBORCACHE seed-pin forced_writes=" << _neighborcache_seed_pin_forced.load(std::memory_order_relaxed)
                          << " cold_writes=" << _neighborcache_seed_pin_cold_writes.load(std::memory_order_relaxed)
                          << " early_writes=" << _neighborcache_seed_pin_early_writes.load(std::memory_order_relaxed)
                          << std::endl;
        if (_neighborcache_seed_page_lru || _neighborcache_seed_prefetch)
        {
            uint64_t resident = 0;
            for (const auto &shard : _neighborcache_seed_page_shards)
            {
                if (!shard)
                    continue;
                std::lock_guard<NeighborCacheSpinLock> lock(shard->mu);
                resident += shard->pages.size();
            }
            const uint64_t enqueued = _neighborcache_seed_page_enqueued.load(std::memory_order_relaxed);
            const uint64_t nbr_sum = _neighborcache_seed_page_nbr_sum.load(std::memory_order_relaxed);
            const uint64_t opens = _neighborcache_seed_page_opens.load(std::memory_order_relaxed);
            const uint64_t open_nbrs = _neighborcache_seed_page_open_nbrs.load(std::memory_order_relaxed);
            diskann::cout << "NEIGHBORCACHE seed-page cap=" << _neighborcache_seed_page_cap << " resident=" << resident
                          << " opens=" << opens << " open_miss="
                          << _neighborcache_seed_page_open_miss.load(std::memory_order_relaxed) << " open_nbrs=" << open_nbrs
                          << " open_nbr_mean=" << (opens > 0 ? static_cast<double>(open_nbrs) / static_cast<double>(opens) : 0.0)
                          << " appends=" << _neighborcache_seed_page_appends.load(std::memory_order_relaxed)
                          << " replaces=" << _neighborcache_seed_page_replaces.load(std::memory_order_relaxed)
                          << " touches=" << _neighborcache_seed_page_touches.load(std::memory_order_relaxed)
                          << " hits=" << _neighborcache_seed_page_hits.load(std::memory_order_relaxed)
                          << " evicts=" << _neighborcache_seed_page_evicts.load(std::memory_order_relaxed)
                          << " enqueued=" << enqueued << " dropped="
                          << _neighborcache_seed_page_dropped.load(std::memory_order_relaxed)
                          << " written=" << _neighborcache_seed_page_written.load(std::memory_order_relaxed)
                          << " skipped=" << _neighborcache_seed_page_skipped.load(std::memory_order_relaxed)
                          << " failed=" << _neighborcache_seed_page_failed.load(std::memory_order_relaxed)
                          << " evict_nbr_mean="
                          << (enqueued > 0 ? static_cast<double>(nbr_sum) / static_cast<double>(enqueued) : 0.0)
                          << std::endl;
        }
        if (_neighborcache_dcache_family)
            diskann::cout << "NEIGHBORCACHE dcache-family formed=" << _neighborcache_family_formed.load(std::memory_order_relaxed)
                          << " expanded=" << _neighborcache_family_expanded.load(std::memory_order_relaxed)
                          << " dissolved=" << _neighborcache_family_dissolved.load(std::memory_order_relaxed)
                          << " idle=" << _neighborcache_family_idle.load(std::memory_order_relaxed)
                          << " coalesced=" << _neighborcache_family_coalesced.load(std::memory_order_relaxed)
                          << " dropped=" << _neighborcache_family_dropped.load(std::memory_order_relaxed) << std::endl;
        diskann::cout.unsetf(std::ios::fixed);
    }
    diskann::cout << "NEIGHBORCACHE mcache update: adaptive="
                  << (_neighborcache_mcache.adaptive_update_enabled() ? "on" : "off")
                  << " denominator=" << _neighborcache_mcache.update_denominator()
                  << " maintenance_suspended=" << (_neighborcache_mcache.maintenance_suspended() ? "on" : "off")
                  << " ewma_hit_rate=" << _neighborcache_mcache.adaptive_hit_rate()
                  << " observed=" << _neighborcache_mcache.observed_accesses()
                  << " updates=" << _neighborcache_mcache.performed_updates()
                  << " skipped=" << _neighborcache_mcache.skipped_updates()
                  << " transitions=" << _neighborcache_mcache.adaptive_transition_count()
                  << " bitmap_bytes=" << _neighborcache_mcache.membership_bitmap_bytes() << std::endl;
    const uint64_t dcache_enabled_queries =
        _neighborcache_dcache_enabled_queries.load(std::memory_order_relaxed);
    const uint64_t dcache_avoided = _neighborcache_dcache_avoided_pages.load(std::memory_order_relaxed);
    const uint64_t dcache_reads = _neighborcache_dcache_physical_reads.load(std::memory_order_relaxed);
    diskann::cout << "NEIGHBORCACHE dcache net: gate=" << (_neighborcache_dcache_net_gate_enabled ? "on" : "off")
                  << " active="
                  << (_neighborcache_dcache_gate_active.load(std::memory_order_relaxed) ? "on" : "off")
                  << " probe=1/" << _neighborcache_dcache_probe_period
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
                  << _neighborcache_dcache_gate_transitions.load(std::memory_order_relaxed)
                  << " latency_model_us=" << _neighborcache_dcache_base_read_us << "/"
                  << _neighborcache_dcache_read_us << "/" << _neighborcache_dcache_write_us << "/"
                  << _neighborcache_dcache_cpu_us << std::endl;
    diskann::cout << "NEIGHBORCACHE dcache_locations: shards=" << _neighborcache_dyn_loc_shard_count
                  << " entries=" << dynamic_loc_entries << " locations=" << dynamic_locs
                  << " vector_capacity=" << dynamic_loc_capacity
                  << " loc_bytes=" << sizeof(NeighborCacheDiskLoc) << std::endl;
    diskann::cout << "NEIGHBORCACHE seed_directory: entries=" << seed_dir_entries << " members=" << seed_dir_members
                  << " member_arena_slots=" << _neighborcache_dyn_page_members.size()
                  << " member_arena_bytes=" << _neighborcache_dyn_page_members.capacity() * sizeof(uint32_t)
                  << " count_bytes=" << _neighborcache_dyn_page_member_count.capacity() * sizeof(uint16_t) << std::endl;
    {
        uint64_t paired = 0, first_sum = 0, final_sum = 0, grew = 0, same = 0, shrunk = 0;
        uint64_t first_hist[11] = {};
        uint64_t final_hist[11] = {};
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        for (size_t page = 0; page < _neighborcache_dyn_page_member_count.size(); ++page)
        {
            const uint16_t count = _neighborcache_dyn_page_member_count[page];
            if (count == 0)
                continue;
            if (page < _neighborcache_dyn_page_secondary.size() && _neighborcache_dyn_page_secondary[page])
                continue;
            const size_t begin = page * _neighborcache_dyn_members_per_page;
            if (begin >= _neighborcache_dyn_page_members.size())
                continue;
            const uint32_t seed = _neighborcache_dyn_page_members[begin];
            auto it = _neighborcache_seed_first_fill.find(seed);
            if (it == _neighborcache_seed_first_fill.end())
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
        diskann::cout << "NEIGHBORCACHE dcache_fill: resident_seeds=" << paired
                      << " first_members=" << first_avg << " first_neighbors=" << (first_avg - 1.0)
                      << " first_fill=" << (first_avg / 10.0)
                      << " final_members=" << final_avg << " final_neighbors=" << (final_avg - 1.0)
                      << " final_fill=" << (final_avg / 10.0)
                      << " grew=" << grew << " same=" << same << " shrunk=" << shrunk << std::endl;
        diskann::cout << "NEIGHBORCACHE dcache_fill_hist first";
        for (int n = 1; n <= 10; ++n)
            diskann::cout << " " << n << ":" << first_hist[n];
        diskann::cout << " final";
        for (int n = 1; n <= 10; ++n)
            diskann::cout << " " << n << ":" << final_hist[n];
        diskann::cout << std::endl;
        if (_neighborcache_dcache_second_page)
        {
            uint64_t extra_pages = 0, extra_neighbors = 0;
            uint64_t span_hist[6] = {};
            for (size_t page = 0; page < _neighborcache_dyn_page_member_count.size(); ++page)
            {
                if (!neighborcache_dyn_page_live(static_cast<uint32_t>(page)))
                    continue;
                const bool secondary =
                    page < _neighborcache_dyn_page_secondary.size() && _neighborcache_dyn_page_secondary[page];
                if (secondary)
                {
                    ++extra_pages;
                    const uint16_t count = _neighborcache_dyn_page_member_count[page];
                    if (count > 1)
                        extra_neighbors += static_cast<uint64_t>(count - 1);
                    continue;
                }
                uint32_t n = 1;
                if (page < _neighborcache_dyn_page_span_len.size() && _neighborcache_dyn_page_span_len[page] > 1)
                    n = _neighborcache_dyn_page_span_len[page];
                if (n > 5)
                    n = 5;
                ++span_hist[n];
            }
            diskann::cout << "NEIGHBORCACHE dcache_pages: max=" << static_cast<uint32_t>(_neighborcache_dcache_max_pages)
                          << " p1=" << span_hist[1] << " p2=" << span_hist[2] << " p3=" << span_hist[3]
                          << " p4=" << span_hist[4] << " p5=" << span_hist[5]
                          << " extra_pages=" << extra_pages << " extra_neighbors=" << extra_neighbors
                          << " multi_ios=" << _neighborcache_dcache_pair_ios.load(std::memory_order_relaxed)
                          << " span_adaptive=" << (_neighborcache_span_adaptive ? "on" : "off")
                          << " span_min_hits=" << _neighborcache_span_min_hits << " span_explore=" << _neighborcache_span_explore
                          << " span_full_reads=" << _neighborcache_span_full_reads.load(std::memory_order_relaxed)
                          << " span_primary_only=" << _neighborcache_span_primary_only.load(std::memory_order_relaxed)
                          << std::endl;
        }
    }
    if (_neighborcache_member_snapshot_done.load(std::memory_order_acquire))
    {
        uint64_t neighbor_slots = 0, early_still = 0, late_slots = 0;
        {
            NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
            for (size_t page = 0; page < _neighborcache_dyn_page_member_count.size(); ++page)
            {
                const uint16_t count = _neighborcache_dyn_page_member_count[page];
                if (count <= 1)
                    continue;
                const size_t begin = page * _neighborcache_dyn_members_per_page;
                if (begin + count > _neighborcache_dyn_page_members.size())
                    continue;
                for (size_t mi = 1; mi < count; ++mi)
                {
                    const uint32_t mid = _neighborcache_dyn_page_members[begin + mi];
                    if (mid == NEIGHBORCACHE_DYN_INVALID_PAGE)
                        continue;
                    ++neighbor_slots;
                    if (_neighborcache_member_at_snapshot.find(mid) != _neighborcache_member_at_snapshot.end())
                        ++early_still;
                    else
                        ++late_slots;
                }
            }
        }
        const uint64_t early_hits = _neighborcache_prefetch_from_early.load(std::memory_order_relaxed);
        const uint64_t late_hits = _neighborcache_prefetch_from_late.load(std::memory_order_relaxed);
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
        hop_stats(_neighborcache_early_hop_hist, early_hits, early_mean, early_p50, early_p90);
        hop_stats(_neighborcache_late_hop_hist, late_hits, late_mean, late_p50, late_p90);
        diskann::cout << "NEIGHBORCACHE prefetch_member_age: snapshot_query=" << _neighborcache_member_snapshot_at
                      << " early_neighbor_ids=" << _neighborcache_member_at_snapshot.size()
                      << " early_neighbor_slots=" << _neighborcache_member_snapshot_slots.load(std::memory_order_relaxed)
                      << " final_neighbor_slots=" << neighbor_slots
                      << " early_still=" << early_still << " late_slots=" << late_slots
                      << " prefetch_early=" << early_hits << " prefetch_late=" << late_hits
                      << " early_hop_mean=" << early_mean << " early_hop_p50=" << early_p50
                      << " early_hop_p90=" << early_p90 << " late_hop_mean=" << late_mean
                      << " late_hop_p50=" << late_p50 << " late_hop_p90=" << late_p90 << std::endl;
    }
    const uint64_t ncache_payload_bytes =
        (_neighborcache_mem_pool && _neighborcache_mem_pool->active()) ? _neighborcache_mem_pool->payload_capacity_bytes() : 0;
    uint64_t stash_payload_bytes = 0;
    uint64_t pending_buffer_payload_bytes = 0;
    uint64_t parent_fetched_bytes = 0;
    uint64_t stash_entries = 0;
    uint64_t parent_entries = 0;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_stash_mu);
        stash_entries = _neighborcache_payload_stash.size();
        for (const auto &kv : _neighborcache_payload_stash)
            stash_payload_bytes +=
                kv.second.coords.capacity() * sizeof(T) + kv.second.nbrs.capacity() * sizeof(uint32_t);
        pending_buffer_payload_bytes = _neighborcache_pending_buffer_payload_bytes;
        for (const auto &kids : _neighborcache_parent_fetched_by_slot)
            if (!kids.empty())
                ++parent_entries;
        parent_fetched_bytes = _neighborcache_parent_fetched_by_slot.capacity() * sizeof(NeighborCacheRecentChildren) +
                               _neighborcache_parent_fetched_owner.capacity() * sizeof(uint32_t);
    }
    diskann::cout << "NEIGHBORCACHE ncache_memory: payload_capacity_bytes=" << ncache_payload_bytes
                  << " lock=" << ((_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->spin_lock_enabled()) ? "spin" : "mutex")
                  << " stash_entries=" << stash_entries << "/" << _neighborcache_stash_cap
                  << " stash_payload_capacity_bytes=" << stash_payload_bytes
                  << " parent_fetched_entries=" << parent_entries
                  << " parent_fetched_capacity_bytes=" << parent_fetched_bytes << std::endl;
    diskann::cout << "NEIGHBORCACHE pending_buffer: payload_bytes=" << pending_buffer_payload_bytes << "/"
                  << _neighborcache_pending_buffer_cap_bytes
                  << " entries=" << stash_entries
                  << " inserts=" << _neighborcache_pending_buffer_inserts.load(std::memory_order_relaxed)
                  << " ready_writes="
                  << _neighborcache_pending_buffer_ready_writes.load(std::memory_order_relaxed)
                  << " pressure_writes="
                  << _neighborcache_pending_buffer_pressure_writes.load(std::memory_order_relaxed)
                  << " drops=" << _neighborcache_pending_buffer_drops.load(std::memory_order_relaxed)
                  << " member_refreshes="
                  << _neighborcache_pending_member_refreshes.load(std::memory_order_relaxed)
                  << " became_full=" << _neighborcache_pending_became_full.load(std::memory_order_relaxed)
                  << " dcache_ncache_erases="
                  << _neighborcache_dcache_ncache_erases.load(std::memory_order_relaxed)
                  << " query_patch_trig=" << _neighborcache_dcache_query_patch_trig.load(std::memory_order_relaxed)
                  << " query_patch_ok=" << _neighborcache_dcache_query_patch_ok.load(std::memory_order_relaxed)
                  << " query_patch_drop=" << _neighborcache_dcache_query_patch_drop.load(std::memory_order_relaxed)
                  << std::endl;
    const uint64_t writer_done = _neighborcache_dyn_writer_completed.load(std::memory_order_relaxed);
    diskann::cout << "NEIGHBORCACHE detached_writeback: queue_drops="
                  << _neighborcache_writeback_queue_drops.load(std::memory_order_relaxed)
                  << " demand_bypasses=" << _neighborcache_writeback_read_bypasses.load(std::memory_order_relaxed)
                  << std::endl;
    const uint64_t commit_done = _neighborcache_commit_completed.load(std::memory_order_relaxed);
    const uint64_t commit_failed = _neighborcache_commit_failed.load(std::memory_order_relaxed);
    const uint64_t commit_attempted = commit_done + commit_failed;
    diskann::cout << "NEIGHBORCACHE async_commit: commit_threads=" << _neighborcache_commit_thread_count
                  << " writer_threads=" << _neighborcache_dyn_writer_thread_count
                  << " enqueued=" << _neighborcache_commit_enqueued.load(std::memory_order_relaxed)
                  << " coalesced=" << _neighborcache_commit_coalesced.load(std::memory_order_relaxed)
                  << " dropped=" << _neighborcache_commit_dropped.load(std::memory_order_relaxed)
                  << " completed=" << commit_done << " failed=" << commit_failed << " fail_why(missing/notfull/slot/score/payload/write)="
                  << _neighborcache_commit_fail_why[0].load(std::memory_order_relaxed) << "/"
                  << _neighborcache_commit_fail_why[1].load(std::memory_order_relaxed) << "/"
                  << _neighborcache_commit_fail_why[2].load(std::memory_order_relaxed) << "/"
                  << _neighborcache_commit_fail_why[3].load(std::memory_order_relaxed) << "/"
                  << _neighborcache_commit_fail_why[4].load(std::memory_order_relaxed) << "/"
                  << _neighborcache_commit_fail_why[5].load(std::memory_order_relaxed)
                  << " queue_us="
                  << (commit_attempted == 0
                          ? 0.0
                          : static_cast<double>(_neighborcache_commit_queue_ns.load(std::memory_order_relaxed)) /
                                1000.0 / static_cast<double>(commit_attempted))
                  << " work_us="
                  << (commit_attempted == 0
                          ? 0.0
                          : static_cast<double>(_neighborcache_commit_work_ns.load(std::memory_order_relaxed)) /
                                1000.0 / static_cast<double>(commit_attempted))
                  << " source_enqueued=" << _neighborcache_commit_source_enqueued[0].load(std::memory_order_relaxed)
                  << "/" << _neighborcache_commit_source_enqueued[1].load(std::memory_order_relaxed)
                  << "/" << _neighborcache_commit_source_enqueued[2].load(std::memory_order_relaxed)
                  << " source_completed=" << _neighborcache_commit_source_completed[0].load(std::memory_order_relaxed)
                  << "/" << _neighborcache_commit_source_completed[1].load(std::memory_order_relaxed)
                  << "/" << _neighborcache_commit_source_completed[2].load(std::memory_order_relaxed)
                  << " writer_enqueued=" << _neighborcache_dyn_writer_enqueued.load(std::memory_order_relaxed)
                  << " writer_failed=" << _neighborcache_dyn_writer_failed.load(std::memory_order_relaxed)
                  << " writer_queue_us="
                  << (writer_done == 0
                          ? 0.0
                          : static_cast<double>(_neighborcache_dyn_writer_queue_ns.load(std::memory_order_relaxed)) /
                                1000.0 / static_cast<double>(writer_done))
                  << " writer_io_us="
                  << (writer_done == 0
                          ? 0.0
                          : static_cast<double>(_neighborcache_dyn_writer_io_ns.load(std::memory_order_relaxed)) /
                                1000.0 / static_cast<double>(writer_done))
                  << std::endl;
    diskann::cout << "NEIGHBORCACHE real_io_coaccess: enabled=" << (_neighborcache_real_io_coaccess ? "yes" : "no")
                  << " queries=" << _neighborcache_real_io_queries.load(std::memory_order_relaxed)
                  << " sampled_queries=" << _neighborcache_real_io_sampled_queries.load(std::memory_order_relaxed)
                  << " nodes=" << _neighborcache_real_io_nodes.load(std::memory_order_relaxed)
                  << " directed_pairs=" << _neighborcache_real_io_pairs.load(std::memory_order_relaxed)
                  << " window=" << _neighborcache_real_io_window
                  << " max_partners=" << _neighborcache_real_io_max_partners
                  << " direct_only=" << (_neighborcache_real_io_direct ? "yes" : "no")
                  << " window_pairs=" << _neighborcache_real_io_window_pairs.load(std::memory_order_relaxed)
                  << " adjacent_pairs=" << _neighborcache_real_io_adjacent_pairs.load(std::memory_order_relaxed)
                  << std::endl;
    diskann::cout << "NEIGHBORCACHE dyn probe: ncache_hit=" << g_dyn_probe.ncache_hit
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
    NeighborCacheHeapEntry best_max;
    NeighborCacheHeapEntry best_min;
    for (auto &shard : _neighborcache_heap_shards)
    {
        NeighborCacheTimedMutexGuard hlock(shard.mu, NeighborCacheLockKind::Heap);
        max_n += shard.max_heap.size();
        min_n += shard.min_heap.size();
        heap_storage_bytes +=
            (shard.max_heap.capacity() + shard.min_heap.capacity()) * sizeof(NeighborCacheHeapEntry);
        if (!shard.max_heap.empty() &&
            (!have_max || neighborcache_dyn_heap_better_unlocked(shard.max_heap.front(), best_max, NeighborCacheHeapKind::Max)))
        {
            best_max = shard.max_heap.front();
            have_max = true;
        }
        if (!shard.min_heap.empty() &&
            (!have_min || neighborcache_dyn_heap_better_unlocked(shard.min_heap.front(), best_min, NeighborCacheHeapKind::Min)))
        {
            best_min = shard.min_heap.front();
            have_min = true;
        }
    }
    heap_storage_bytes += _neighborcache_heap_position.capacity() * sizeof(uint32_t) +
                          _neighborcache_heap_kind.capacity() * sizeof(uint8_t) +
                          _neighborcache_node_state_capacity * sizeof(std::atomic<uint8_t>);
    if (have_max)
    {
        max_c = best_max.score;
        max_id = _neighborcache_mcache.node_at(best_max.slot_id);
    }
    if (have_min)
    {
        min_c = best_min.score;
        min_id = _neighborcache_mcache.node_at(best_min.slot_id);
    }
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        pair_n = _neighborcache_ready_pairs.size();
        if (!_neighborcache_ready_pairs.empty())
        {
            ins = _neighborcache_ready_pairs.begin()->second.insertion_id;
            del = _neighborcache_ready_pairs.begin()->second.deletion_id;
        }
        writes = _neighborcache_heap_writes;
        deletes = _neighborcache_heap_deletes;
        trig = _neighborcache_ncache_write_trig;
        refresh = _neighborcache_pair_refresh;
        decay_queries = _neighborcache_decay_query_count.load(std::memory_order_relaxed);
        decay_accesses = _neighborcache_decay_access_count.load(std::memory_order_relaxed);
        decay_cycles = _neighborcache_decay_cycles.load(std::memory_order_relaxed);
        decay_early_cycles = _neighborcache_decay_early_cycles.load(std::memory_order_relaxed);
        decay_started = _neighborcache_decay_started.load(std::memory_order_relaxed);
        score_unit = _neighborcache_score_unit.load();
    }
    diskann::cout << "NEIGHBORCACHE heap: max_top=" << max_id << ":" << max_c << " min_top=" << min_id << ":" << min_c
                  << " max_n=" << max_n << " min_n=" << min_n << " pending=" << pair_n
                  << " ins=" << ins << " del=" << del << " writes=" << writes << " deletes=" << deletes
                  << " trig=" << trig << " pair_refresh=" << refresh << " storage_bytes=" << heap_storage_bytes
                  << std::endl;
    diskann::cout << "NEIGHBORCACHE score: unit=" << score_unit << "/" << NEIGHBORCACHE_SCORE_UNIT_MAX
                  << " half_life_accesses=" << _neighborcache_decay_half_life_accesses << " cycles=" << decay_cycles
                  << " early_cycles=" << decay_early_cycles << " decay_queries=" << decay_queries
                  << " decay_accesses=" << decay_accesses << " accesses_per_query="
                  << (decay_queries == 0 ? 0.0 : static_cast<double>(decay_accesses) / decay_queries)
                  << " decay=" << (decay_started ? "on" : "off")
                  << " edge_saturations=" << _neighborcache_mcache.edge_saturations()
                  << " mcache_evictable=" << _neighborcache_mcache.evictable_size()
                  << " probe pair_set=" << g_dyn_probe.pair_set
                  << " skip_on_disk=" << g_dyn_probe.pair_skip_max_on_disk
                  << " skip_count=" << g_dyn_probe.pair_skip_count << " write_trig=" << g_dyn_probe.write_trig
                  << " timeout_trig=" << g_dyn_probe.timeout_trig << " timeout_ok=" << g_dyn_probe.timeout_ok
                  << " write_ok=" << g_dyn_probe.write_ok << " delete_ok=" << g_dyn_probe.delete_ok << std::endl;
    static constexpr const char *lock_names[] = {"score", "heap", "pair", "mcache", "dyn_shared",
                                                 "dyn_unique", "commit_io"};
    {
        auto &p = neighborcache_hop_io_probe();
        diskann::cout << "NEIGHBORCACHE hop_io:";
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
    neighborcache_io_latency_report();
    diskann::cout << "NEIGHBORCACHE lock sample=1/1024:";
    auto &lock_metrics = neighborcache_lock_metrics();
    for (size_t i = 0; i < static_cast<size_t>(NeighborCacheLockKind::Count); ++i)
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

template <typename T, typename LabelT> uint32_t PQFlashIndex<T, LabelT>::neighborcache_dyn_alloc_page_unlocked()
{
    neighborcache_dyn_drain_reclaim_unlocked();
    if (!_neighborcache_dyn_free.empty())
    {
        const uint32_t idx = _neighborcache_dyn_free.back();
        _neighborcache_dyn_free.pop_back();
        neighborcache_shared_cache_erase_dyn(idx, 1);
        if (_neighborcache_prefetch_dirty)
            _neighborcache_prefetch_dirty[idx].store(0, std::memory_order_release);
        return idx;
    }
    if (_neighborcache_dyn_next_page < _neighborcache_dyn_physical_cap)
        return _neighborcache_dyn_next_page++;
    return NEIGHBORCACHE_DYN_INVALID_PAGE;
}

template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::neighborcache_dyn_alloc_contiguous_unlocked(uint32_t count)
{
    if (count == 0)
        return NEIGHBORCACHE_DYN_INVALID_PAGE;
    if (count == 1)
        return neighborcache_dyn_alloc_page_unlocked();
    neighborcache_dyn_drain_reclaim_unlocked();
    if (_neighborcache_dyn_next_page + count <= _neighborcache_dyn_physical_cap)
    {
        const uint32_t base = _neighborcache_dyn_next_page;
        _neighborcache_dyn_next_page = static_cast<uint32_t>(_neighborcache_dyn_next_page + count);
        return base;
    }
    std::sort(_neighborcache_dyn_free.begin(), _neighborcache_dyn_free.end());
    for (size_t i = 0; i + 1 < _neighborcache_dyn_free.size(); ++i)
    {
        bool run = true;
        for (uint32_t k = 1; k < count; ++k)
        {
            if (i + k >= _neighborcache_dyn_free.size() || _neighborcache_dyn_free[i + k] != _neighborcache_dyn_free[i] + k)
            {
                run = false;
                break;
            }
        }
        if (!run)
            continue;
        const uint32_t base = _neighborcache_dyn_free[i];
        _neighborcache_dyn_free.erase(_neighborcache_dyn_free.begin() + static_cast<std::ptrdiff_t>(i),
                              _neighborcache_dyn_free.begin() + static_cast<std::ptrdiff_t>(i + count));
        neighborcache_shared_cache_erase_dyn(base, count);
        if (_neighborcache_prefetch_dirty)
            for (uint32_t k = 0; k < count; ++k)
                _neighborcache_prefetch_dirty[base + k].store(0, std::memory_order_release);
        return base;
    }
    return NEIGHBORCACHE_DYN_INVALID_PAGE;
}

template <typename T, typename LabelT>
std::vector<uint32_t> PQFlashIndex<T, LabelT>::neighborcache_dyn_now_page_ids(uint32_t seed_id,
                                                                     const NeighborCacheMetadataCache::Snapshot &snap,
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
            has_payload = _neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active() &&
                          _neighborcache_mem_pool->contains(id);
            if (!has_payload && _neighborcache_pending_buffer_cap_bytes > 0)
            {
                std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
                has_payload = _neighborcache_payload_stash.find(id) != _neighborcache_payload_stash.end();
            }
        }
        else
        {
            has_payload = neighborcache_dyn_has_payload(id);
        }
        if (!has_payload)
            return;
        now.push_back(id);
    };

    now.push_back(seed_id);
    used.insert(seed_id);

    std::vector<uint32_t> old_members;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        uint32_t page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
        if (neighborcache_dyn_seed_page_unlocked(seed_id, page_idx))
        {
            const size_t count = _neighborcache_dyn_page_member_count[page_idx];
            const size_t begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
            old_members.assign(_neighborcache_dyn_page_members.begin() + begin,
                               _neighborcache_dyn_page_members.begin() + begin + count);
        }
    }
    std::vector<uint32_t> fetched;
    const uint32_t fetched_slot = _neighborcache_mcache.slot_of(seed_id);
    if (fetched_slot != NeighborCacheMetadataCache::kInvalid)
    {
        const uint32_t slot_id = fetched_slot;
        std::lock_guard<std::mutex> slock(neighborcache_parent_fetched_mu(slot_id));
        if (slot_id < _neighborcache_parent_fetched_by_slot.size() && _neighborcache_parent_fetched_owner[slot_id] == seed_id)
        {
            const auto &kids = _neighborcache_parent_fetched_by_slot[slot_id];
            fetched.assign(kids.begin(), kids.end());
        }
    }

    // Directed W is the primary co-packing signal. Recently fetched children
    // and old members are only fallback fillers when the page still has room.
    for (const auto &sc : snap.significant)
        try_add(sc.first);

    for (auto it = fetched.rbegin(); it != fetched.rend(); ++it)
        try_add(*it);

    if (!_neighborcache_real_io_coaccess)
    {
        for (uint32_t id : old_members)
        {
            if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->contains(id))
                continue;
            try_add(id);
        }
        for (uint32_t id : old_members)
            try_add(id);
    }
    return now;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_hotter_mismatch(const uint32_t *page_members, size_t page_member_count,
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
void PQFlashIndex<T, LabelT>::neighborcache_dyn_retire_secondary_unlocked(uint32_t page_idx)
{
    if (page_idx >= _neighborcache_dyn_page_member_count.size())
        return;
    const uint16_t member_count = _neighborcache_dyn_page_member_count[page_idx];
    const size_t member_begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
    const uint32_t old_sector = static_cast<uint32_t>(NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx);
    for (uint16_t i = 0; i < member_count; ++i)
        neighborcache_dyn_loc_erase_sector_unlocked(_neighborcache_dyn_page_members[member_begin + i], old_sector);
    _neighborcache_dyn_page_member_count[page_idx] = 0;
    if (page_idx < _neighborcache_dyn_page_sibling.size())
        _neighborcache_dyn_page_sibling[page_idx] = NEIGHBORCACHE_DYN_INVALID_PAGE;
    if (page_idx < _neighborcache_dyn_page_secondary.size())
        _neighborcache_dyn_page_secondary[page_idx] = 0;
    if (page_idx < _neighborcache_dyn_page_span_len.size())
        _neighborcache_dyn_page_span_len[page_idx] = 0;
    if (page_idx < _neighborcache_dyn_page_span_base.size())
        _neighborcache_dyn_page_span_base[page_idx] = NEIGHBORCACHE_DYN_INVALID_PAGE;
    neighborcache_dyn_retire_page_unlocked(page_idx);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_invalidate_seed_page(uint32_t seed_id)
{
    uint32_t page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
    if (!neighborcache_dyn_seed_page_unlocked(seed_id, page_idx))
        return;
    uint8_t span = 1;
    if (page_idx < _neighborcache_dyn_page_span_len.size() && _neighborcache_dyn_page_span_len[page_idx] > 1)
        span = _neighborcache_dyn_page_span_len[page_idx];
    const uint16_t member_count = _neighborcache_dyn_page_member_count[page_idx];
    const size_t member_begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
    const uint32_t old_sector = static_cast<uint32_t>(NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx);
    neighborcache_shared_cache_erase_dyn(page_idx, span);

    for (uint16_t i = 0; i < member_count; ++i)
        neighborcache_dyn_loc_erase_sector_unlocked(_neighborcache_dyn_page_members[member_begin + i], old_sector);
    if (neighborcache_dyn_page_live(page_idx))
    {
        if (_neighborcache_dyn_committed_pages > 0)
            _neighborcache_dyn_committed_pages--;
    }
    _neighborcache_dyn_page_member_count[page_idx] = 0;
    if (page_idx < _neighborcache_dyn_page_sibling.size())
        _neighborcache_dyn_page_sibling[page_idx] = NEIGHBORCACHE_DYN_INVALID_PAGE;
    if (page_idx < _neighborcache_dyn_page_secondary.size())
        _neighborcache_dyn_page_secondary[page_idx] = 0;
    if (page_idx < _neighborcache_dyn_page_span_len.size())
        _neighborcache_dyn_page_span_len[page_idx] = 0;
    if (page_idx < _neighborcache_dyn_page_span_base.size())
        _neighborcache_dyn_page_span_base[page_idx] = NEIGHBORCACHE_DYN_INVALID_PAGE;
    neighborcache_dyn_retire_page_unlocked(page_idx);
    for (uint8_t extra = 1; extra < span; ++extra)
    {
        const uint32_t extra_page = page_idx + extra;
        if (extra_page < _neighborcache_dyn_page_member_count.size())
            neighborcache_dyn_retire_secondary_unlocked(extra_page);
    }
    if (!neighborcache_dyn_seed_page_unlocked(seed_id, page_idx))
        _neighborcache_seed_first_fill.erase(seed_id);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_release_page_pin(uint32_t sector)
{
    if (!neighborcache_dyn_is_sector(sector) || _neighborcache_dyn_page_readers == nullptr)
        return;
    const uint32_t page_idx = sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
    if (page_idx >= _neighborcache_dyn_physical_cap)
        return;
    // seq_cst pairs with neighborcache_dyn_retire_page_unlocked: whichever side observes
    // readers==0 && retired==1 last claims the page via CAS, exactly once.
    if (_neighborcache_dyn_page_readers[page_idx].fetch_sub(1, std::memory_order_seq_cst) != 1)
        return;
    if (_neighborcache_dyn_page_retired[page_idx].load(std::memory_order_seq_cst) == 0)
        return;
    uint8_t expected = 1;
    if (!_neighborcache_dyn_page_retired[page_idx].compare_exchange_strong(expected, 0, std::memory_order_seq_cst))
        return;
    std::lock_guard<std::mutex> rlock(_neighborcache_dyn_reclaim_mu);
    _neighborcache_dyn_reclaim.push_back(page_idx);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_retire_page_unlocked(uint32_t page_idx)
{
    _neighborcache_dyn_page_retired[page_idx].store(1, std::memory_order_seq_cst);
    if (_neighborcache_dyn_page_readers[page_idx].load(std::memory_order_seq_cst) != 0)
        return;
    uint8_t expected = 1;
    if (_neighborcache_dyn_page_retired[page_idx].compare_exchange_strong(expected, 0, std::memory_order_seq_cst))
        _neighborcache_dyn_free.push_back(page_idx);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_span_should_read_full(uint32_t base, uint32_t span) const
{
    if (!_neighborcache_span_adaptive || span <= 1)
        return true;
    const uint32_t decision = _neighborcache_span_decisions[base].fetch_add(1, std::memory_order_relaxed) + 1;
    const uint32_t reads = _neighborcache_span_reads[base].load(std::memory_order_relaxed);
    bool full = reads < 4 || (_neighborcache_span_explore > 0 && decision % _neighborcache_span_explore == 0);
    if (!full)
    {
        const double hits = _neighborcache_span_hits[base].load(std::memory_order_relaxed);
        full = hits >= static_cast<double>(_neighborcache_span_min_hits) * reads * (span - 1);
    }
    if (full)
    {
        if (_neighborcache_span_reads[base].fetch_add(1, std::memory_order_relaxed) + 1 >= 64)
        {
            _neighborcache_span_reads[base].store(32, std::memory_order_relaxed);
            _neighborcache_span_hits[base].store(_neighborcache_span_hits[base].load(std::memory_order_relaxed) / 2,
                                         std::memory_order_relaxed);
        }
        _neighborcache_span_full_reads.fetch_add(1, std::memory_order_relaxed);
    }
    else
        _neighborcache_span_primary_only.fetch_add(1, std::memory_order_relaxed);
    return full;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_span_carry_stats(uint32_t from_page, uint32_t to_page)
{
    if (!_neighborcache_span_reads || to_page >= _neighborcache_dyn_physical_cap)
        return;
    uint32_t reads = 0, hits = 0, decisions = 0;
    if (from_page < _neighborcache_dyn_physical_cap)
    {
        reads = _neighborcache_span_reads[from_page].load(std::memory_order_relaxed);
        hits = _neighborcache_span_hits[from_page].load(std::memory_order_relaxed);
        decisions = _neighborcache_span_decisions[from_page].load(std::memory_order_relaxed);
    }
    _neighborcache_span_reads[to_page].store(reads, std::memory_order_relaxed);
    _neighborcache_span_hits[to_page].store(hits, std::memory_order_relaxed);
    _neighborcache_span_decisions[to_page].store(decisions, std::memory_order_relaxed);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_drain_reclaim_unlocked()
{
    std::lock_guard<std::mutex> rlock(_neighborcache_dyn_reclaim_mu);
    if (_neighborcache_dyn_reclaim.empty())
        return;
    _neighborcache_dyn_free.insert(_neighborcache_dyn_free.end(), _neighborcache_dyn_reclaim.begin(), _neighborcache_dyn_reclaim.end());
    _neighborcache_dyn_reclaim.clear();
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_maybe_mark_seed(uint32_t node_id)
{
    (void)node_id;
    if (!_neighborcache_dyn_enabled)
        return;
    neighborcache_mark_refresh_needed();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_note_touch(const NeighborCacheMetadataCache::TouchResult &tr,
                                                   SSDQueryScratch<T> *query_scratch)
{
    if (!_neighborcache_dyn_enabled || !tr.present)
        return;
    if (tr.evicted_id != NeighborCacheMetadataCache::kInvalid)
        neighborcache_dyn_on_mcache_evict(tr.evicted_id, tr.evicted_slot);
    if (_neighborcache_mcache.node_at(tr.slot_id) != tr.node_id || query_scratch == nullptr)
        return;
    query_scratch->mark_neighborcache_heap_dirty(tr.slot_id);
    neighborcache_mark_refresh_needed();
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_ncache_evictable(uint32_t node_id) const
{
    if (_neighborcache_mem_pool == nullptr || !_neighborcache_mem_pool->active())
        return false;
    return _neighborcache_mem_pool->contains_evictable(node_id);
}

template <typename T, typename LabelT>
typename PQFlashIndex<T, LabelT>::NeighborCacheNodeState PQFlashIndex<T, LabelT>::neighborcache_dyn_state_unlocked(uint32_t slot_id) const
{
    return slot_id < _neighborcache_node_state_capacity
               ? static_cast<NeighborCacheNodeState>(_neighborcache_node_state[slot_id].load(std::memory_order_acquire))
               : NeighborCacheNodeState::NonSeed;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_set_state_unlocked(uint32_t node_id, uint32_t slot_id, NeighborCacheNodeState st)
{
    if (slot_id < _neighborcache_node_state_capacity)
        _neighborcache_node_state[slot_id].store(static_cast<uint8_t>(st), std::memory_order_release);
    _neighborcache_mcache.set_evictable(node_id, st == NeighborCacheNodeState::NonSeed);
}

template <typename T, typename LabelT>
size_t PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_shard_index(uint32_t slot_id) const
{
    return static_cast<size_t>(slot_id) & (_neighborcache_heap_shard_count - 1);
}

template <typename T, typename LabelT>
typename PQFlashIndex<T, LabelT>::NeighborCacheHeapShard &PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_shard(uint32_t slot_id)
{
    return _neighborcache_heap_shards[neighborcache_dyn_heap_shard_index(slot_id)];
}

template <typename T, typename LabelT>
const typename PQFlashIndex<T, LabelT>::NeighborCacheHeapShard &PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_shard(
    uint32_t slot_id) const
{
    return _neighborcache_heap_shards[neighborcache_dyn_heap_shard_index(slot_id)];
}

template <typename T, typename LabelT>
std::vector<typename PQFlashIndex<T, LabelT>::NeighborCacheHeapEntry> &PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_for(
    NeighborCacheHeapShard &shard, NeighborCacheHeapKind kind)
{
    return kind == NeighborCacheHeapKind::Max ? shard.max_heap : shard.min_heap;
}

template <typename T, typename LabelT>
const std::vector<typename PQFlashIndex<T, LabelT>::NeighborCacheHeapEntry> &PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_for(
    const NeighborCacheHeapShard &shard, NeighborCacheHeapKind kind) const
{
    return kind == NeighborCacheHeapKind::Max ? shard.max_heap : shard.min_heap;
}

template <typename T, typename LabelT>
typename PQFlashIndex<T, LabelT>::NeighborCacheHeapKind PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_kind_unlocked(
    uint32_t slot_id) const
{
    return slot_id < _neighborcache_heap_kind.size()
               ? static_cast<NeighborCacheHeapKind>(_neighborcache_heap_kind[slot_id])
               : NeighborCacheHeapKind::None;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_set_heap_kind_unlocked(uint32_t slot_id, NeighborCacheHeapKind kind)
{
    if (slot_id < _neighborcache_heap_kind.size())
        _neighborcache_heap_kind[slot_id] = static_cast<uint8_t>(kind);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_better_unlocked(const NeighborCacheHeapEntry &a, const NeighborCacheHeapEntry &b,
                                                             NeighborCacheHeapKind kind) const
{
    if (a.score != b.score)
        return kind == NeighborCacheHeapKind::Max ? a.score > b.score : a.score < b.score;
    return kind == NeighborCacheHeapKind::Max ? a.slot_id > b.slot_id : a.slot_id < b.slot_id;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_swap_unlocked(std::vector<NeighborCacheHeapEntry> &heap, size_t a, size_t b)
{
    if (a == b)
        return;
    std::swap(heap[a], heap[b]);
    _neighborcache_heap_position[heap[a].slot_id] = static_cast<uint32_t>(a);
    _neighborcache_heap_position[heap[b].slot_id] = static_cast<uint32_t>(b);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_sift_up_unlocked(std::vector<NeighborCacheHeapEntry> &heap, size_t index,
                                                              NeighborCacheHeapKind kind)
{
    while (index > 0)
    {
        const size_t parent = (index - 1) / 2;
        if (!neighborcache_dyn_heap_better_unlocked(heap[index], heap[parent], kind))
            break;
        neighborcache_dyn_heap_swap_unlocked(heap, index, parent);
        index = parent;
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_sift_down_unlocked(std::vector<NeighborCacheHeapEntry> &heap, size_t index,
                                                                NeighborCacheHeapKind kind)
{
    while (true)
    {
        const size_t left = index * 2 + 1;
        if (left >= heap.size())
            return;
        const size_t right = left + 1;
        size_t best = left;
        if (right < heap.size() && neighborcache_dyn_heap_better_unlocked(heap[right], heap[left], kind))
            best = right;
        if (!neighborcache_dyn_heap_better_unlocked(heap[best], heap[index], kind))
            return;
        neighborcache_dyn_heap_swap_unlocked(heap, index, best);
        index = best;
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_insert_unlocked(const NeighborCacheHeapEntry &entry, NeighborCacheHeapKind kind)
{
    if (entry.slot_id >= _neighborcache_heap_position.size() || kind == NeighborCacheHeapKind::None)
        return;
    auto &heap = neighborcache_dyn_heap_for(neighborcache_dyn_heap_shard(entry.slot_id), kind);
    neighborcache_dyn_set_heap_kind_unlocked(entry.slot_id, kind);
    _neighborcache_heap_position[entry.slot_id] = static_cast<uint32_t>(heap.size());
    heap.push_back(entry);
    neighborcache_dyn_heap_sift_up_unlocked(heap, heap.size() - 1, kind);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_erase_unlocked(uint32_t slot_id)
{
    if (slot_id >= _neighborcache_heap_position.size())
        return;
    const NeighborCacheHeapKind kind = neighborcache_dyn_heap_kind_unlocked(slot_id);
    if (kind == NeighborCacheHeapKind::None)
        return;
    auto &heap = neighborcache_dyn_heap_for(neighborcache_dyn_heap_shard(slot_id), kind);
    const size_t index = _neighborcache_heap_position[slot_id];
    if (index >= heap.size() || heap[index].slot_id != slot_id)
    {
        neighborcache_dyn_set_heap_kind_unlocked(slot_id, NeighborCacheHeapKind::None);
        _neighborcache_heap_position[slot_id] = NeighborCacheMetadataCache::kInvalid;
        return;
    }
    const NeighborCacheHeapEntry replacement = heap.back();
    heap.pop_back();
    neighborcache_dyn_set_heap_kind_unlocked(slot_id, NeighborCacheHeapKind::None);
    _neighborcache_heap_position[slot_id] = NeighborCacheMetadataCache::kInvalid;
    if (index >= heap.size())
        return;
    heap[index] = replacement;
    _neighborcache_heap_position[replacement.slot_id] = static_cast<uint32_t>(index);
    const size_t parent = index > 0 ? (index - 1) / 2 : 0;
    if (index > 0 &&
        neighborcache_dyn_heap_better_unlocked(heap[index], heap[parent], kind))
        neighborcache_dyn_heap_sift_up_unlocked(heap, index, kind);
    else
        neighborcache_dyn_heap_sift_down_unlocked(heap, index, kind);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_apply_unlocked(uint32_t slot_id, float score, NeighborCacheHeapKind desired)
{
    if (slot_id >= _neighborcache_heap_position.size())
        return;
    if (neighborcache_dyn_heap_kind_unlocked(slot_id) == desired && desired != NeighborCacheHeapKind::None)
    {
        auto &heap = neighborcache_dyn_heap_for(neighborcache_dyn_heap_shard(slot_id), desired);
        const size_t index = _neighborcache_heap_position[slot_id];
        if (index < heap.size() && heap[index].slot_id == slot_id)
        {
            heap[index].score = score;
            const size_t parent = index > 0 ? (index - 1) / 2 : 0;
            if (index > 0 &&
                neighborcache_dyn_heap_better_unlocked(heap[index], heap[parent], desired))
                neighborcache_dyn_heap_sift_up_unlocked(heap, index, desired);
            else
                neighborcache_dyn_heap_sift_down_unlocked(heap, index, desired);
            return;
        }
    }
    neighborcache_dyn_heap_erase_unlocked(slot_id);
    if (desired != NeighborCacheHeapKind::None)
        neighborcache_dyn_heap_insert_unlocked(NeighborCacheHeapEntry{score, slot_id}, desired);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_heap_upsert_unlocked(uint32_t slot_id, float score)
{
    if (slot_id >= _neighborcache_heap_position.size())
        return;
    const uint32_t node_id = _neighborcache_mcache.node_at(slot_id);
    if (node_id == NeighborCacheMetadataCache::kInvalid)
    {
        neighborcache_dyn_heap_erase_unlocked(slot_id);
        return;
    }
    const auto st = neighborcache_dyn_state_unlocked(slot_id);
    NeighborCacheHeapKind desired = NeighborCacheHeapKind::None;
    if (st == NeighborCacheNodeState::Seed)
        desired = NeighborCacheHeapKind::Min;
    else if (st == NeighborCacheNodeState::NonSeed && neighborcache_dyn_ncache_evictable(node_id))
        desired = NeighborCacheHeapKind::Max;
    neighborcache_dyn_heap_apply_unlocked(slot_id, score, desired);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_clean_heap_unlocked(NeighborCacheHeapShard &shard, NeighborCacheHeapKind kind)
{
    auto &heap = neighborcache_dyn_heap_for(shard, kind);
    while (!heap.empty())
    {
        const NeighborCacheHeapEntry entry = heap.front();
        const uint32_t node_id = _neighborcache_mcache.node_at(entry.slot_id);
        if (kind == NeighborCacheHeapKind::Max)
        {
            if (neighborcache_dyn_state_unlocked(entry.slot_id) == NeighborCacheNodeState::NonSeed &&
                node_id != NeighborCacheMetadataCache::kInvalid && neighborcache_dyn_ncache_evictable(node_id))
                break;
        }
        else if (neighborcache_dyn_state_unlocked(entry.slot_id) == NeighborCacheNodeState::Seed &&
                 node_id != NeighborCacheMetadataCache::kInvalid)
        {
            break;
        }
        neighborcache_dyn_heap_erase_unlocked(entry.slot_id);
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_best_heap_entry(NeighborCacheHeapKind kind, NeighborCacheHeapEntry &entry,
                                                        size_t &shard_index)
{
    bool found = false;
    for (size_t index = 0; index < _neighborcache_heap_shard_count; ++index)
    {
        NeighborCacheHeapShard &shard = _neighborcache_heap_shards[index];
        NeighborCacheTimedMutexGuard hlock(shard.mu, NeighborCacheLockKind::Heap);
        neighborcache_dyn_clean_heap_unlocked(shard, kind);
        const auto &heap = neighborcache_dyn_heap_for(shard, kind);
        if (heap.empty())
            continue;
        if (!found || neighborcache_dyn_heap_better_unlocked(heap.front(), entry, kind))
        {
            entry = heap.front();
            shard_index = index;
            found = true;
        }
    }
    return found;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_flush_dirty_heap(SSDQueryScratch<T> *query_scratch)
{
    if (query_scratch == nullptr || query_scratch->neighborcache_heap_dirty_slots.empty())
        return;

    const size_t dirty_count = query_scratch->neighborcache_heap_dirty_slots.size();
    query_scratch->neighborcache_heap_dirty_nodes.resize(dirty_count);
    query_scratch->neighborcache_heap_dirty_scores.resize(dirty_count);
    query_scratch->neighborcache_heap_dirty_eligible.resize(dirty_count);
    for (size_t index = 0; index < dirty_count; ++index)
    {
        const uint32_t slot_id = query_scratch->neighborcache_heap_dirty_slots[index];
        uint32_t node_id = NeighborCacheMetadataCache::kInvalid;
        float score = 0.0f;
        if (!_neighborcache_mcache.node_score_at(slot_id, node_id, score))
        {
            query_scratch->neighborcache_heap_dirty_nodes[index] = NeighborCacheMetadataCache::kInvalid;
            continue;
        }
        query_scratch->neighborcache_heap_dirty_nodes[index] = node_id;
        query_scratch->neighborcache_heap_dirty_scores[index] = score;
        query_scratch->neighborcache_heap_dirty_eligible[index] =
            static_cast<uint8_t>(neighborcache_dyn_ncache_evictable(node_id));
    }
    thread_local std::vector<uint32_t> shard_begin;
    thread_local std::vector<uint32_t> by_shard;
    shard_begin.assign(_neighborcache_heap_shard_count + 1, 0);
    for (size_t index = 0; index < dirty_count; ++index)
        ++shard_begin[neighborcache_dyn_heap_shard_index(query_scratch->neighborcache_heap_dirty_slots[index]) + 1];
    for (size_t s = 0; s < _neighborcache_heap_shard_count; ++s)
        shard_begin[s + 1] += shard_begin[s];
    by_shard.resize(dirty_count);
    {
        thread_local std::vector<uint32_t> cursor;
        cursor.assign(shard_begin.begin(), shard_begin.end() - 1);
        for (size_t index = 0; index < dirty_count; ++index)
            by_shard[cursor[neighborcache_dyn_heap_shard_index(query_scratch->neighborcache_heap_dirty_slots[index])]++] =
                static_cast<uint32_t>(index);
    }
    for (size_t shard_index = 0; shard_index < _neighborcache_heap_shard_count; ++shard_index)
    {
        if (shard_begin[shard_index] == shard_begin[shard_index + 1])
            continue;
        NeighborCacheTimedMutexGuard hlock(_neighborcache_heap_shards[shard_index].mu, NeighborCacheLockKind::Heap);
        for (uint32_t bi = shard_begin[shard_index]; bi < shard_begin[shard_index + 1]; ++bi)
        {
            const size_t index = by_shard[bi];
            const uint32_t slot_id = query_scratch->neighborcache_heap_dirty_slots[index];
            const uint32_t node_id = query_scratch->neighborcache_heap_dirty_nodes[index];
            if (node_id == NeighborCacheMetadataCache::kInvalid)
                continue;
            if (_neighborcache_mcache.node_at(slot_id) != node_id)
                continue;
            NeighborCacheHeapKind desired = NeighborCacheHeapKind::None;
            const NeighborCacheNodeState state = neighborcache_dyn_state_unlocked(slot_id);
            if (state == NeighborCacheNodeState::Seed)
                desired = NeighborCacheHeapKind::Min;
            else if (state == NeighborCacheNodeState::NonSeed && query_scratch->neighborcache_heap_dirty_eligible[index] != 0)
                desired = NeighborCacheHeapKind::Max;
            neighborcache_dyn_heap_apply_unlocked(slot_id, query_scratch->neighborcache_heap_dirty_scores[index], desired);
        }
    }

    std::vector<uint32_t> cancellation_slots;
    if (_neighborcache_deletion_pair_count.load(std::memory_order_acquire) != 0)
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        for (uint32_t slot_id : query_scratch->neighborcache_heap_dirty_slots)
        {
            const auto pending_it = _neighborcache_deletion_to_pending.find(slot_id);
            if (pending_it == _neighborcache_deletion_to_pending.end())
                continue;
            const uint32_t insertion_slot = pending_it->second;
            if (_neighborcache_mcache.score_at(slot_id) >= _neighborcache_mcache.score_at(insertion_slot))
                cancellation_slots.push_back(insertion_slot);
        }
    }
    if (!cancellation_slots.empty())
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
        for (uint32_t insertion_slot : cancellation_slots)
        {
            const auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
            if (pair_it == _neighborcache_ready_pairs.end())
                continue;
            const uint32_t deletion_slot = pair_it->second.deletion_slot;
            if (deletion_slot != NeighborCacheMetadataCache::kInvalid &&
                _neighborcache_mcache.score_at(deletion_slot) >= _neighborcache_mcache.score_at(insertion_slot))
            {
                neighborcache_dyn_clear_ready_pair_unlocked(insertion_slot);
            }
        }
    }
    query_scratch->clear_neighborcache_heap_dirty();
    query_scratch->neighborcache_heap_dirty_nodes.clear();
    query_scratch->neighborcache_heap_dirty_scores.clear();
    query_scratch->neighborcache_heap_dirty_eligible.clear();
}

template <typename T, typename LabelT>
typename PQFlashIndex<T, LabelT>::NeighborCachePendingMemberShard &
PQFlashIndex<T, LabelT>::neighborcache_pending_member_shard(uint32_t node_id)
{
    return *_neighborcache_pending_member_shards[node_id & (NEIGHBORCACHE_PENDING_MEMBER_SHARDS - 1)];
}

template <typename T, typename LabelT>
const typename PQFlashIndex<T, LabelT>::NeighborCachePendingMemberShard &
PQFlashIndex<T, LabelT>::neighborcache_pending_member_shard(uint32_t node_id) const
{
    return *_neighborcache_pending_member_shards[node_id & (NEIGHBORCACHE_PENDING_MEMBER_SHARDS - 1)];
}

template <typename T, typename LabelT>
std::vector<uint32_t> PQFlashIndex<T, LabelT>::neighborcache_pending_member_slots(uint32_t node_id) const
{
    const NeighborCachePendingMemberShard &shard = neighborcache_pending_member_shard(node_id);
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Pair, false);
    const auto it = shard.slots.find(node_id);
    return it == shard.slots.end() ? std::vector<uint32_t>{} : it.value();
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_pending_member_contains(uint32_t node_id) const
{
    const NeighborCachePendingMemberShard &shard = neighborcache_pending_member_shard(node_id);
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Pair, false);
    return shard.slots.find(node_id) != shard.slots.end();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_pending_member_add(uint32_t node_id, uint32_t insertion_slot)
{
    NeighborCachePendingMemberShard &shard = neighborcache_pending_member_shard(node_id);
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Pair, true);
    auto &slots = shard.slots[node_id];
    if (std::find(slots.begin(), slots.end(), insertion_slot) == slots.end())
        slots.push_back(insertion_slot);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_pending_member_remove(uint32_t node_id, uint32_t insertion_slot)
{
    NeighborCachePendingMemberShard &shard = neighborcache_pending_member_shard(node_id);
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Pair, true);
    auto it = shard.slots.find(node_id);
    if (it == shard.slots.end())
        return;
    auto &slots = it.value();
    slots.erase(std::remove(slots.begin(), slots.end(), insertion_slot), slots.end());
    if (slots.empty())
        shard.slots.erase(it);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_pending_member_clear()
{
    for (auto &shard_ptr : _neighborcache_pending_member_shards)
    {
        NeighborCachePendingMemberShard &shard = *shard_ptr;
        NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Pair, true);
        shard.slots.clear();
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_clear_ready_pair_unlocked(uint32_t insertion_slot, bool restore_states)
{
    auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
    if (pair_it == _neighborcache_ready_pairs.end())
        return;
    if (restore_states && (pair_it->second.committing || pair_it->second.queued))
        return;

    NeighborCacheReadyPair pair = std::move(pair_it->second);
    if (pair.retained_members < _neighborcache_pending_buffer_buckets.size())
        _neighborcache_pending_buffer_buckets[pair.retained_members].erase(insertion_slot);
    _neighborcache_ready_pairs.erase(pair_it);
    _neighborcache_pending_pair_count.fetch_sub(1, std::memory_order_release);
    const uint32_t ins = pair.insertion_id;
    const uint32_t del = pair.deletion_id;
    const uint32_t ins_slot = pair.insertion_slot;
    const uint32_t del_slot = pair.deletion_slot;
    if (del_slot == NeighborCacheMetadataCache::kInvalid && _neighborcache_reserved_free_pages > 0)
        --_neighborcache_reserved_free_pages;
    else if (del_slot != NeighborCacheMetadataCache::kInvalid)
    {
        _neighborcache_deletion_to_pending.erase(del_slot);
        _neighborcache_deletion_pair_count.fetch_sub(1, std::memory_order_release);
    }
    for (uint32_t member : pair.snap.member_ids)
        neighborcache_pending_member_remove(member, ins_slot);
    if (ins != std::numeric_limits<uint32_t>::max())
        neighborcache_dyn_release_seed_neighbor_holds(ins);

    if (!restore_states)
        return;
    if (ins != std::numeric_limits<uint32_t>::max() &&
        neighborcache_dyn_state_unlocked(ins_slot) == NeighborCacheNodeState::ReadyToInsertion)
        neighborcache_dyn_set_state_unlocked(ins, ins_slot, NeighborCacheNodeState::NonSeed);
    if (del != std::numeric_limits<uint32_t>::max() &&
        neighborcache_dyn_state_unlocked(del_slot) == NeighborCacheNodeState::ReadyToDeletion)
        neighborcache_dyn_set_state_unlocked(del, del_slot, NeighborCacheNodeState::Seed);
    if (ins_slot != NeighborCacheMetadataCache::kInvalid)
    {
        NeighborCacheTimedMutexGuard hlock(neighborcache_dyn_heap_shard(ins_slot).mu, NeighborCacheLockKind::Heap);
        neighborcache_dyn_heap_upsert_unlocked(ins_slot, _neighborcache_mcache.score_at(ins_slot));
    }
    if (del_slot != NeighborCacheMetadataCache::kInvalid)
    {
        NeighborCacheTimedMutexGuard hlock(neighborcache_dyn_heap_shard(del_slot).mu, NeighborCacheLockKind::Heap);
        neighborcache_dyn_heap_upsert_unlocked(del_slot, _neighborcache_mcache.score_at(del_slot));
    }
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::neighborcache_dcache_updates_frozen() const
{
    return _neighborcache_dcache_freeze_after > 0 &&
           _neighborcache_query_epoch.load(std::memory_order_acquire) >= _neighborcache_dcache_freeze_after;
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::neighborcache_dyn_disk_is_full() const
{
    NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
    return _neighborcache_dyn_committed_pages >= _neighborcache_dyn_page_cap;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_on_mcache_evict(uint32_t node_id, uint32_t slot_id)
{
    if (node_id == NeighborCacheMetadataCache::kInvalid)
        return;
    if (slot_id != NeighborCacheMetadataCache::kInvalid)
    {
        std::lock_guard<std::mutex> slock(neighborcache_parent_fetched_mu(slot_id));
        if (slot_id < _neighborcache_parent_fetched_by_slot.size() &&
            _neighborcache_parent_fetched_owner[slot_id] == node_id)
        {
            _neighborcache_parent_fetched_by_slot[slot_id].clear();
            _neighborcache_parent_fetched_owner[slot_id] = NeighborCacheMetadataCache::kInvalid;
        }
    }
    neighborcache_mark_refresh_needed();
    NeighborCacheNodeState st = NeighborCacheNodeState::NonSeed;
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        st = neighborcache_dyn_state_unlocked(slot_id);
        if (st == NeighborCacheNodeState::NonSeed)
            return;
    }
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
        st = neighborcache_dyn_state_unlocked(slot_id);
        if (st == NeighborCacheNodeState::ReadyToInsertion)
            neighborcache_dyn_clear_ready_pair_unlocked(slot_id);
        else if (st == NeighborCacheNodeState::ReadyToDeletion)
        {
            const auto pending_it = _neighborcache_deletion_to_pending.find(slot_id);
            if (pending_it != _neighborcache_deletion_to_pending.end())
                neighborcache_dyn_clear_ready_pair_unlocked(pending_it->second);
        }
        neighborcache_dyn_set_state_unlocked(node_id, slot_id, NeighborCacheNodeState::NonSeed);
    }
    {
        NeighborCacheTimedMutexGuard hlock(neighborcache_dyn_heap_shard(slot_id).mu, NeighborCacheLockKind::Heap);
        neighborcache_dyn_heap_erase_unlocked(slot_id);
    }
    if (st != NeighborCacheNodeState::Seed && st != NeighborCacheNodeState::ReadyToDeletion)
        return;
    if (neighborcache_dcache_updates_frozen() || (_neighborcache_dcache_no_seed_replace && neighborcache_dyn_disk_is_full()))
        return;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        neighborcache_dyn_invalidate_seed_page(node_id);
    }
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
        _neighborcache_heap_deletes++;
    }
    g_dyn_probe.delete_ok++;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_build_pending_flush(uint32_t seed_id,
                                                           const NeighborCacheMetadataCache::Snapshot &snap,
                                                           NeighborCachePendingFlush &pf) const
{
    pf = NeighborCachePendingFlush{};
    pf.seed_id = seed_id;
    pf.snap = snap;
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    const std::vector<uint32_t> now =
        neighborcache_dyn_now_page_ids(seed_id, snap, _neighborcache_deferred_ncache_page_write);
    std::vector<T> coords;
    std::vector<uint32_t> nbrs;
    for (uint32_t mid : now)
    {
        if (_neighborcache_deferred_ncache_page_write)
        {
            pf.member_ids.push_back(mid);
            continue;
        }
        if (!neighborcache_dyn_copy_member_payload(mid, coords, nbrs))
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
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_refresh_pending_members(uint32_t insertion_slot)
{
    uint32_t seed_id = NeighborCacheMetadataCache::kInvalid;
    std::vector<uint32_t> old_members;
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        const auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
        if (pair_it == _neighborcache_ready_pairs.end() || pair_it->second.committing)
            return false;
        seed_id = pair_it->second.insertion_id;
        old_members = pair_it->second.snap.member_ids;
    }

    NeighborCacheMetadataCache::Snapshot snap;
    if (!_neighborcache_mcache.snapshot(seed_id, snap))
        return false;
    std::vector<uint32_t> refreshed =
        neighborcache_dyn_now_page_ids(seed_id, snap, _neighborcache_deferred_ncache_page_write);
    const size_t page_capacity = static_cast<size_t>(
        (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
    for (uint32_t member_id : old_members)
    {
        if (refreshed.size() >= page_capacity)
            break;
        if (std::find(refreshed.begin(), refreshed.end(), member_id) != refreshed.end())
            continue;
        bool available = _neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->contains(member_id);
        if (!available)
        {
            std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
            available = _neighborcache_payload_stash.find(member_id) != _neighborcache_payload_stash.end();
        }
        if (available)
            refreshed.push_back(member_id);
    }
    if (refreshed.empty() || refreshed.front() != seed_id)
        return false;

    bool became_full = false;
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
        auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
        if (pair_it == _neighborcache_ready_pairs.end() || pair_it.value().committing ||
            pair_it.value().insertion_id != seed_id)
            return false;
        NeighborCacheReadyPair &pair = pair_it.value();
        const size_t previous_count = pair.snap.member_ids.size();
        if (pair.snap.member_ids == refreshed)
            return refreshed.size() >= page_capacity;

        if (pair.retained_members < _neighborcache_pending_buffer_buckets.size())
            _neighborcache_pending_buffer_buckets[pair.retained_members].erase(insertion_slot);
        for (uint32_t member_id : pair.snap.member_ids)
            neighborcache_pending_member_remove(member_id, insertion_slot);

        pair.snap.snap = std::move(snap);
        pair.snap.member_ids = refreshed;
        pair.retained_members = 0;
        {
            std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
            for (uint32_t member_id : pair.snap.member_ids)
                if (_neighborcache_payload_stash.find(member_id) != _neighborcache_payload_stash.end())
                    ++pair.retained_members;
        }
        for (uint32_t member_id : pair.snap.member_ids)
            neighborcache_pending_member_add(member_id, insertion_slot);
        if (pair.retained_members < _neighborcache_pending_buffer_buckets.size())
            _neighborcache_pending_buffer_buckets[pair.retained_members].insert(insertion_slot);
        became_full = previous_count < page_capacity && pair.snap.member_ids.size() >= page_capacity;
    }

    _neighborcache_pending_member_refreshes.fetch_add(1, std::memory_order_relaxed);
    if (became_full)
        _neighborcache_pending_became_full.fetch_add(1, std::memory_order_relaxed);
    neighborcache_dyn_release_unreferenced_buffer_members(old_members);
    return refreshed.size() >= page_capacity;
}

template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::neighborcache_seed_page_shard_index(uint32_t id) const
{
    return static_cast<uint32_t>((static_cast<uint64_t>(id) * 11400714819323198485ull) >> 32) &
           (kNeighborCacheSeedPageShards - 1u);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_seed_page_link(uint32_t seed_id, const std::vector<uint32_t> &nbrs)
{
    auto link_one = [this, seed_id](uint32_t id) {
        if (id == std::numeric_limits<uint32_t>::max())
            return;
        NeighborCacheSeedPageOwnerShard &owners = *_neighborcache_seed_page_owners[neighborcache_seed_page_shard_index(id)];
        std::lock_guard<NeighborCacheSpinLock> lock(owners.mu);
        auto it = owners.by_node.find(id);
        if (it == owners.by_node.end())
        {
            owners.by_node.insert({id, std::vector<uint32_t>(1, seed_id)});
            return;
        }
        std::vector<uint32_t> &listed = it.value();
        for (uint32_t existing : listed)
        {
            if (existing == seed_id)
                return;
        }
        listed.push_back(seed_id);
    };
    link_one(seed_id);
    for (uint32_t nbr : nbrs)
    {
        if (nbr != seed_id)
            link_one(nbr);
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_seed_page_unlink(uint32_t seed_id, const std::vector<uint32_t> &nbrs)
{
    auto unlink_one = [this, seed_id](uint32_t id) {
        if (id == std::numeric_limits<uint32_t>::max())
            return;
        NeighborCacheSeedPageOwnerShard &owners = *_neighborcache_seed_page_owners[neighborcache_seed_page_shard_index(id)];
        std::lock_guard<NeighborCacheSpinLock> lock(owners.mu);
        auto it = owners.by_node.find(id);
        if (it == owners.by_node.end())
            return;
        std::vector<uint32_t> &listed = it.value();
        size_t write = 0;
        for (size_t read = 0; read < listed.size(); ++read)
        {
            if (listed[read] != seed_id)
                listed[write++] = listed[read];
        }
        listed.resize(write);
        if (listed.empty())
            owners.by_node.erase(it);
    };
    unlink_one(seed_id);
    for (uint32_t nbr : nbrs)
    {
        if (nbr != seed_id)
            unlink_one(nbr);
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_seed_page_enqueue(NeighborCachePendingFlush pf)
{
    if (pf.member_ids.size() < 2 || pf.member_ids.front() != pf.seed_id)
        return;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_commit_q_mu);
        if (_neighborcache_commit_stop.load(std::memory_order_acquire) || _neighborcache_seed_page_q.size() >= _neighborcache_seed_page_q_cap)
        {
            _neighborcache_seed_page_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        _neighborcache_seed_page_nbr_sum.fetch_add(pf.member_ids.size() - 1, std::memory_order_relaxed);
        _neighborcache_seed_page_q.push_back(std::move(pf));
        _neighborcache_seed_page_enqueued.fetch_add(1, std::memory_order_relaxed);
    }
    _neighborcache_commit_q_cv.notify_one();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_seed_page_flush_evicted(NeighborCacheSeedPage &page)
{
    _neighborcache_seed_page_evicts.fetch_add(1, std::memory_order_relaxed);
    const std::vector<uint32_t> graph = page.nbrs[0];
    neighborcache_seed_page_unlink(page.seed_id, graph);
    if (page.count < 2)
        return;
    NeighborCachePendingFlush pf;
    pf.seed_id = page.seed_id;
    pf.member_ids.reserve(page.count);
    pf.member_coords.reserve(page.count);
    pf.member_nbrs.reserve(page.count);
    for (uint8_t slot = 0; slot < page.count; ++slot)
    {
        if (page.coords[slot].size() * sizeof(T) < _disk_bytes_per_point)
            continue;
        pf.member_ids.push_back(page.ids[slot]);
        pf.member_coords.push_back(std::move(page.coords[slot]));
        pf.member_nbrs.push_back(std::move(page.nbrs[slot]));
    }
    if (pf.member_ids.size() >= 2 && pf.member_ids.front() == page.seed_id)
        neighborcache_seed_page_enqueue(std::move(pf));
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_prefetch_mark_dirty(uint32_t page_idx)
{
    if (_neighborcache_prefetch_dirty && page_idx < _neighborcache_dyn_physical_cap)
        _neighborcache_prefetch_dirty[page_idx].fetch_add(1, std::memory_order_release);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_prefetch_evict_hook(void *ctx, uint32_t key, const char *page, bool discard)
{
    if (ctx == nullptr)
        return true;
    return static_cast<PQFlashIndex<T, LabelT> *>(ctx)->neighborcache_prefetch_on_evict(key, page, discard);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_prefetch_on_evict(uint32_t key, const char *page, bool discard)
{
    if (page == nullptr || !neighborcache_dyn_is_sector(key))
        return true;
    const uint32_t page_idx = key - static_cast<uint32_t>(NEIGHBORCACHE_DYN_SECTOR_BASE);
    if (_neighborcache_prefetch_dirty == nullptr || page_idx >= _neighborcache_dyn_physical_cap)
        return true;
    // Runs under the cache shard lock: copy to an owned task, then unpublish A
    // immediately. B's admission never waits for A's disk write or queue space.
    if (discard)
    {
        // Do not wait for IO and do not make unwritten disk data readable.
        // Physical reuse is deferred by each outstanding writer/query pin;
        // allocation resets the dirty state only after all pins have gone.
        return true;
    }
    const uint64_t version = _neighborcache_prefetch_dirty[page_idx].load(std::memory_order_acquire);
    if (version == 0)
        return true;
    _neighborcache_prefetch_writeback[page_idx].fetch_add(1, std::memory_order_acq_rel);
    _neighborcache_dyn_page_readers[page_idx].fetch_add(1, std::memory_order_seq_cst);
    const off_t off = static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN);
    auto job = neighborcache_dyn_writer_submit(NeighborCacheDynWriteOp::Write, off, page,
                                               defaults::SECTOR_LEN, true, page_idx, version);
    if (job)
        _neighborcache_seed_page_evicts.fetch_add(1, std::memory_order_relaxed);
    else
    {
        // Bounded, nonblocking cache writeback: overload sacrifices this cache
        // write, never correctness. The dirty marker keeps D unreadable and
        // demands use the immutable base index until the page is rebuilt.
        _neighborcache_writeback_queue_drops.fetch_add(1, std::memory_order_relaxed);
        _neighborcache_prefetch_writeback[page_idx].fetch_sub(1, std::memory_order_acq_rel);
        neighborcache_dyn_release_page_pin(key);
    }
    return true;
}

template <typename T, typename LabelT>
NeighborCacheSpinLock &PQFlashIndex<T, LabelT>::neighborcache_seed_write_lock(uint32_t seed_id)
{
    std::lock_guard<std::mutex> guard(_neighborcache_seed_lock_mu);
    auto it = _neighborcache_seed_locks.find(seed_id);
    if (it == _neighborcache_seed_locks.end())
    {
        _neighborcache_seed_locks.emplace(seed_id, std::make_unique<NeighborCacheSpinLock>());
        it = _neighborcache_seed_locks.find(seed_id);
    }
    return *it.value();
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_seed_prefetch_publish(uint32_t seed_id, const std::vector<uint32_t> &ids,
                                                          const std::vector<std::vector<T>> &coords,
                                                          const std::vector<std::vector<uint32_t>> &nbrs)
{
    if (!_neighborcache_seed_prefetch || ids.empty() || ids.size() != coords.size() || ids.size() != nbrs.size() ||
        ids.front() != seed_id)
        return false;
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    const size_t slots = std::min(ids.size(), std::min(nps, static_cast<uint64_t>(kNeighborCacheSeedPageSlots)));
    if (slots == 0)
        return false;
    std::vector<char> buf(defaults::SECTOR_LEN, 0);
    for (size_t mi = 0; mi < slots; ++mi)
    {
        const uint32_t nnbrs = static_cast<uint32_t>(nbrs[mi].size());
        if (_disk_bytes_per_point > coords[mi].size() * sizeof(T) || nnbrs > _max_degree)
            return false;
        char *slot_dst = buf.data() + static_cast<uint64_t>(mi) * _max_node_len;
        memcpy(slot_dst, coords[mi].data(), _disk_bytes_per_point);
        uint32_t *nhood = reinterpret_cast<uint32_t *>(slot_dst + _disk_bytes_per_point);
        nhood[0] = nnbrs;
        if (nnbrs > 0)
            memcpy(nhood + 1, nbrs[mi].data(), nnbrs * sizeof(uint32_t));
        if (mi == 0)
            write_neighborcache_dc_seed_tail(slot_dst, _max_node_len, _disk_bytes_per_point, nnbrs, 1);
    }

    NeighborCacheSpinLock &seed_lk = neighborcache_seed_write_lock(seed_id);
    std::lock_guard<NeighborCacheSpinLock> seed_guard(seed_lk);
    uint32_t page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
    bool existed = false;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        existed = neighborcache_dyn_seed_page_unlocked(seed_id, page_idx);
    }
    if (existed)
        return true;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        if (neighborcache_dyn_seed_page_unlocked(seed_id, page_idx))
            return true;
        if (_neighborcache_dyn_committed_pages >= _neighborcache_dyn_page_cap)
            return false;
        page_idx = neighborcache_dyn_alloc_page_unlocked();
    }
    if (page_idx == NEIGHBORCACHE_DYN_INVALID_PAGE)
        return false;

    if (!neighborcache_shared_cache_admit_dyn(page_idx, buf.data(), true))
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        _neighborcache_dyn_free.push_back(page_idx);
        return false;
    }
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        uint32_t live = NEIGHBORCACHE_DYN_INVALID_PAGE;
        if (neighborcache_dyn_seed_page_unlocked(seed_id, live))
        {
            neighborcache_shared_cache_erase_dyn(page_idx, 1);
            neighborcache_dyn_retire_page_unlocked(page_idx);
            return true;
        }
        if (_neighborcache_dyn_committed_pages >= _neighborcache_dyn_page_cap)
        {
            neighborcache_shared_cache_erase_dyn(page_idx, 1);
            neighborcache_dyn_retire_page_unlocked(page_idx);
            return false;
        }
        if (page_idx < _neighborcache_dyn_page_secondary.size())
            _neighborcache_dyn_page_secondary[page_idx] = 0;
        if (page_idx < _neighborcache_dyn_page_span_len.size())
            _neighborcache_dyn_page_span_len[page_idx] = 1;
        if (page_idx < _neighborcache_dyn_page_span_base.size())
            _neighborcache_dyn_page_span_base[page_idx] = page_idx;
        neighborcache_dyn_bump_page_gen(page_idx);
        const size_t begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
        for (size_t mi = 0; mi < slots; ++mi)
            _neighborcache_dyn_page_members[begin + mi] = ids[mi];
        _neighborcache_dyn_page_member_count[page_idx] = static_cast<uint16_t>(slots);
        const uint32_t sector = NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx;
        for (size_t mi = 0; mi < slots; ++mi)
        {
            NeighborCacheDiskLoc loc;
            loc.sector = sector;
            loc.slot = static_cast<uint16_t>(mi);
            loc.nsectors = 1;
            neighborcache_dyn_loc_add_unlocked(ids[mi], loc);
        }
        neighborcache_dyn_bump_page_gen(page_idx);
        _neighborcache_dyn_committed_pages++;
    }
    _neighborcache_seed_page_opens.fetch_add(1, std::memory_order_relaxed);
    _neighborcache_seed_page_open_nbrs.fetch_add(slots > 0 ? slots - 1 : 0, std::memory_order_relaxed);
    return true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_seed_prefetch_note(uint32_t node_id)
{
    if (!_neighborcache_seed_prefetch || _neighborcache_seed_page_owners.size() != kNeighborCacheSeedPageShards ||
        node_id == std::numeric_limits<uint32_t>::max() || _neighborcache_mem_pool == nullptr ||
        !_neighborcache_mem_pool->contains(node_id))
        return;
    std::vector<T> coords;
    std::vector<uint32_t> nbrs;
    if (!neighborcache_dyn_copy_member_payload(node_id, coords, nbrs, false) ||
        coords.size() * sizeof(T) < _disk_bytes_per_point || nbrs.size() > _max_degree)
        return;
    std::vector<uint32_t> seeds;
    {
        NeighborCacheSeedPageOwnerShard &owners = *_neighborcache_seed_page_owners[neighborcache_seed_page_shard_index(node_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(owners.mu);
        auto it = owners.by_node.find(node_id);
        if (it == owners.by_node.end() || it.value().empty())
            return;
        seeds = it.value();
    }
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    const uint32_t slot_cap =
        static_cast<uint32_t>(std::min(nps, static_cast<uint64_t>(kNeighborCacheSeedPageSlots)));
    for (uint32_t seed_id : seeds)
    {
        if (seed_id == node_id)
            continue;
        NeighborCacheSpinLock &seed_lk = neighborcache_seed_write_lock(seed_id);
        std::lock_guard<NeighborCacheSpinLock> seed_guard(seed_lk);
        uint32_t page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
        uint16_t count = 0;
        std::vector<uint32_t> members;
        {
            NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
            if (!neighborcache_dyn_seed_page_unlocked(seed_id, page_idx) || !neighborcache_dyn_page_live(page_idx))
                continue;
            count = _neighborcache_dyn_page_member_count[page_idx];
            const size_t begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
            members.assign(_neighborcache_dyn_page_members.begin() + begin,
                           _neighborcache_dyn_page_members.begin() + begin + count);
        }
        bool present = false;
        for (uint32_t id : members)
        {
            if (id == node_id)
            {
                present = true;
                break;
            }
        }
        if (present || count == 0 || count > slot_cap)
            continue;
        const bool replace = count >= slot_cap;
        const uint16_t slot = replace ? static_cast<uint16_t>(count - 1) : count;
        if (replace && slot == 0)
            continue;
        const uint32_t sector = NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx;
        std::vector<char> page(defaults::SECTOR_LEN, 0);
        if (!_neighborcache_shared_cache.get_range(sector, 1, page.data()))
            continue;
        char *slot_dst = page.data() + static_cast<uint64_t>(slot) * _max_node_len;
        const uint32_t nnbrs = static_cast<uint32_t>(nbrs.size());
        memcpy(slot_dst, coords.data(), _disk_bytes_per_point);
        uint32_t *nhood = reinterpret_cast<uint32_t *>(slot_dst + _disk_bytes_per_point);
        nhood[0] = nnbrs;
        if (nnbrs > 0)
            memcpy(nhood + 1, nbrs.data(), nnbrs * sizeof(uint32_t));
        {
            // A reader must see both slot membership and payload from the same
            // publication. Existing IO pins also protect the old slot contents.
            NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
            uint32_t live = NEIGHBORCACHE_DYN_INVALID_PAGE;
            if (!neighborcache_dyn_seed_page_unlocked(seed_id, live) || live != page_idx || !neighborcache_dyn_page_live(page_idx))
                continue;
            // Do not wait on a demand read or detached write. Appends also
            // change the page image: an older in-flight disk read must not
            // re-admit a snapshot that is missing the newly published slots.
            if (_neighborcache_dyn_page_readers[page_idx].load(std::memory_order_acquire) != 0)
                continue;
            if (!neighborcache_shared_cache_admit_dyn(page_idx, page.data(), true))
                continue;
            const size_t begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
            if (replace)
            {
                const uint32_t old_id = _neighborcache_dyn_page_members[begin + slot];
                if (old_id != node_id)
                    neighborcache_dyn_loc_erase_sector_unlocked(old_id, sector);
            }
            _neighborcache_dyn_page_members[begin + slot] = node_id;
            if (!replace)
                _neighborcache_dyn_page_member_count[page_idx] = static_cast<uint16_t>(count + 1);
            NeighborCacheDiskLoc loc;
            loc.sector = sector;
            loc.slot = slot;
            loc.nsectors = 1;
            neighborcache_dyn_loc_add_unlocked(node_id, loc);
            neighborcache_dyn_bump_page_gen(page_idx);
        }
        if (replace)
            _neighborcache_seed_page_replaces.fetch_add(1, std::memory_order_relaxed);
        else
            _neighborcache_seed_page_appends.fetch_add(1, std::memory_order_relaxed);
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_seed_prefetch_lookup(uint32_t node_id, std::vector<T> &coords,
                                                         std::vector<uint32_t> &nbrs)
{
    coords.clear();
    nbrs.clear();
    if (!_neighborcache_seed_prefetch || node_id == std::numeric_limits<uint32_t>::max())
        return false;
    NeighborCacheDiskLoc loc;
    if (!neighborcache_resolve_disk_cache_loc(node_id, nullptr, loc) || !neighborcache_dyn_is_sector(loc.sector))
        return false;
    NeighborCacheTimedSharedMutexGuard page_lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
    const uint32_t page = loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
    if (page >= _neighborcache_dyn_physical_cap || !neighborcache_dyn_page_live(page) ||
        loc.slot >= _neighborcache_dyn_page_member_count[page] ||
        _neighborcache_dyn_page_members[static_cast<size_t>(page) * _neighborcache_dyn_members_per_page + loc.slot] != node_id)
        return false;
    std::vector<char> spill(_max_node_len, 0);
    bool spilled = false;
    const char *node = neighborcache_shared_node_ptr(loc.sector, loc.slot, spill.data(), spilled);
    if (node == nullptr)
        return false;
    const size_t ncoord = _disk_bytes_per_point / sizeof(T);
    coords.resize(ncoord);
    memcpy(coords.data(), node, _disk_bytes_per_point);
    const uint32_t *nhood = reinterpret_cast<const uint32_t *>(node + _disk_bytes_per_point);
    uint32_t nnbrs = nhood[0];
    if (nnbrs > _max_degree)
        nnbrs = _max_degree;
    nbrs.assign(nhood + 1, nhood + 1 + nnbrs);
    _neighborcache_seed_page_hits.fetch_add(1, std::memory_order_relaxed);
    return coords.size() * sizeof(T) >= _disk_bytes_per_point;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_seed_page_open(uint32_t seed_id)
{
    if (_neighborcache_seed_prefetch)
    {
        if (_neighborcache_mem_pool == nullptr)
            return;
        struct Copied
        {
            uint32_t id = 0;
            std::vector<T> coords;
            std::vector<uint32_t> nbrs;
        };
        Copied seed;
        seed.id = seed_id;
        if (!neighborcache_dyn_copy_member_payload(seed_id, seed.coords, seed.nbrs, false) ||
            seed.coords.size() * sizeof(T) < _disk_bytes_per_point)
        {
            _neighborcache_seed_page_open_miss.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const std::vector<uint32_t> graph = seed.nbrs;
        std::vector<Copied> copied;
        copied.reserve(kNeighborCacheSeedPageSlots);
        copied.push_back(std::move(seed));
        auto already = [&copied](uint32_t id) {
            for (const Copied &item : copied)
            {
                if (item.id == id)
                    return true;
            }
            return false;
        };
        for (uint32_t nbr : graph)
        {
            if (copied.size() >= kNeighborCacheSeedPageSlots)
                break;
            if (nbr == seed_id || nbr == std::numeric_limits<uint32_t>::max() || already(nbr))
                continue;
            Copied item;
            item.id = nbr;
            if (!neighborcache_dyn_copy_member_payload(nbr, item.coords, item.nbrs, false) ||
                item.coords.size() * sizeof(T) < _disk_bytes_per_point)
                continue;
            copied.push_back(std::move(item));
        }
        std::vector<uint32_t> ids;
        std::vector<std::vector<T>> coord_rows;
        std::vector<std::vector<uint32_t>> nbr_rows;
        ids.reserve(copied.size());
        coord_rows.reserve(copied.size());
        nbr_rows.reserve(copied.size());
        for (Copied &item : copied)
        {
            ids.push_back(item.id);
            coord_rows.push_back(std::move(item.coords));
            nbr_rows.push_back(std::move(item.nbrs));
        }
        if (neighborcache_seed_prefetch_publish(seed_id, ids, coord_rows, nbr_rows))
            neighborcache_seed_page_link(seed_id, graph);
        return;
    }
    if (!_neighborcache_seed_page_lru || _neighborcache_seed_page_shards.size() != kNeighborCacheSeedPageShards || _neighborcache_mem_pool == nullptr)
        return;
    struct Copied
    {
        uint32_t id = 0;
        std::vector<T> coords;
        std::vector<uint32_t> nbrs;
    };
    Copied seed;
    seed.id = seed_id;
    if (!neighborcache_dyn_copy_member_payload(seed_id, seed.coords, seed.nbrs, false) ||
        seed.coords.size() * sizeof(T) < _disk_bytes_per_point)
    {
        _neighborcache_seed_page_open_miss.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const std::vector<uint32_t> graph = seed.nbrs;
    std::vector<Copied> copied;
    copied.reserve(kNeighborCacheSeedPageSlots);
    copied.push_back(std::move(seed));
    auto already = [&copied](uint32_t id) {
        for (const Copied &item : copied)
        {
            if (item.id == id)
                return true;
        }
        return false;
    };
    for (uint32_t nbr : graph)
    {
        if (copied.size() >= kNeighborCacheSeedPageSlots)
            break;
        if (nbr == seed_id || nbr == std::numeric_limits<uint32_t>::max() || already(nbr))
            continue;
        Copied item;
        item.id = nbr;
        if (!neighborcache_dyn_copy_member_payload(nbr, item.coords, item.nbrs, false) ||
            item.coords.size() * sizeof(T) < _disk_bytes_per_point)
            continue;
        copied.push_back(std::move(item));
    }

    std::vector<NeighborCacheSeedPage> doomed;
    {
        NeighborCacheSeedPageShard &shard = *_neighborcache_seed_page_shards[neighborcache_seed_page_shard_index(seed_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(shard.mu);
        if (shard.pages.find(seed_id) != shard.pages.end())
            return;
        while (shard.pages.size() >= shard.cap && !shard.lru.empty())
        {
            const uint32_t old_id = shard.lru.back();
            shard.lru.pop_back();
            auto old_it = shard.pages.find(old_id);
            if (old_it == shard.pages.end())
                continue;
            doomed.push_back(std::move(old_it.value()));
            shard.pages.erase(old_it);
        }
        NeighborCacheSeedPage page;
        page.seed_id = seed_id;
        page.count = static_cast<uint8_t>(copied.size());
        const uint64_t stamp_base = _neighborcache_seed_page_stamp.fetch_add(copied.size(), std::memory_order_relaxed);
        for (size_t slot = 0; slot < copied.size(); ++slot)
        {
            page.ids[slot] = copied[slot].id;
            page.touch[slot] = stamp_base + slot + 1;
            page.coords[slot] = std::move(copied[slot].coords);
            page.nbrs[slot] = std::move(copied[slot].nbrs);
        }
        shard.lru.push_front(seed_id);
        page.lru_it = shard.lru.begin();
        shard.pages.insert({seed_id, std::move(page)});
    }
    _neighborcache_seed_page_opens.fetch_add(1, std::memory_order_relaxed);
    _neighborcache_seed_page_open_nbrs.fetch_add(copied.size() - 1, std::memory_order_relaxed);
    neighborcache_seed_page_link(seed_id, graph);
    for (NeighborCacheSeedPage &old : doomed)
        neighborcache_seed_page_flush_evicted(old);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_seed_page_note(uint32_t node_id)
{
    if (_neighborcache_seed_prefetch)
    {
        neighborcache_seed_prefetch_note(node_id);
        return;
    }
    if (!_neighborcache_seed_page_lru || _neighborcache_seed_page_shards.size() != kNeighborCacheSeedPageShards || node_id == std::numeric_limits<uint32_t>::max())
        return;
    std::vector<uint32_t> seeds;
    {
        NeighborCacheSeedPageOwnerShard &owners = *_neighborcache_seed_page_owners[neighborcache_seed_page_shard_index(node_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(owners.mu);
        auto it = owners.by_node.find(node_id);
        if (it == owners.by_node.end() || it.value().empty())
            return;
        seeds = it.value();
    }
    std::vector<uint32_t> need_insert;
    need_insert.reserve(seeds.size());
    for (uint32_t seed_id : seeds)
    {
        NeighborCacheSeedPageShard &shard = *_neighborcache_seed_page_shards[neighborcache_seed_page_shard_index(seed_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(shard.mu);
        auto it = shard.pages.find(seed_id);
        if (it == shard.pages.end())
            continue;
        NeighborCacheSeedPage &page = it.value();
        bool present = false;
        for (uint8_t slot = 0; slot < page.count; ++slot)
        {
            if (page.ids[slot] != node_id)
                continue;
            page.touch[slot] = _neighborcache_seed_page_stamp.fetch_add(1, std::memory_order_relaxed) + 1;
            shard.lru.splice(shard.lru.begin(), shard.lru, page.lru_it);
            _neighborcache_seed_page_touches.fetch_add(1, std::memory_order_relaxed);
            present = true;
            break;
        }
        if (!present)
            need_insert.push_back(seed_id);
    }
    if (need_insert.empty())
        return;
    std::vector<T> coords;
    std::vector<uint32_t> nbrs;
    if (!neighborcache_dyn_copy_member_payload(node_id, coords, nbrs, false) || coords.size() * sizeof(T) < _disk_bytes_per_point)
        return;
    for (uint32_t seed_id : need_insert)
    {
        NeighborCacheSeedPageShard &shard = *_neighborcache_seed_page_shards[neighborcache_seed_page_shard_index(seed_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(shard.mu);
        auto it = shard.pages.find(seed_id);
        if (it == shard.pages.end())
            continue;
        NeighborCacheSeedPage &page = it.value();
        if (page.count == 0 || page.count > kNeighborCacheSeedPageSlots)
            continue;
        bool present = false;
        for (uint8_t slot = 0; slot < page.count; ++slot)
        {
            if (page.ids[slot] != node_id)
                continue;
            page.touch[slot] = _neighborcache_seed_page_stamp.fetch_add(1, std::memory_order_relaxed) + 1;
            _neighborcache_seed_page_touches.fetch_add(1, std::memory_order_relaxed);
            present = true;
            break;
        }
        if (present)
        {
            shard.lru.splice(shard.lru.begin(), shard.lru, page.lru_it);
            continue;
        }
        const uint64_t stamp = _neighborcache_seed_page_stamp.fetch_add(1, std::memory_order_relaxed) + 1;
        if (page.count < kNeighborCacheSeedPageSlots)
        {
            const uint8_t slot = page.count++;
            page.ids[slot] = node_id;
            page.touch[slot] = stamp;
            page.coords[slot] = coords;
            page.nbrs[slot] = nbrs;
            _neighborcache_seed_page_appends.fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            uint8_t oldest = 1;
            for (uint8_t slot = 2; slot < page.count; ++slot)
            {
                if (page.touch[slot] < page.touch[oldest])
                    oldest = slot;
            }
            page.ids[oldest] = node_id;
            page.touch[oldest] = stamp;
            page.coords[oldest] = coords;
            page.nbrs[oldest] = nbrs;
            _neighborcache_seed_page_replaces.fetch_add(1, std::memory_order_relaxed);
        }
        shard.lru.splice(shard.lru.begin(), shard.lru, page.lru_it);
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_seed_page_lookup(uint32_t node_id, std::vector<T> &coords, std::vector<uint32_t> &nbrs)
{
    coords.clear();
    nbrs.clear();
    if (_neighborcache_seed_prefetch)
        return neighborcache_seed_prefetch_lookup(node_id, coords, nbrs);
    if (!_neighborcache_seed_page_lru || _neighborcache_seed_page_shards.size() != kNeighborCacheSeedPageShards || node_id == std::numeric_limits<uint32_t>::max())
        return false;
    std::vector<uint32_t> seeds;
    {
        NeighborCacheSeedPageOwnerShard &owners = *_neighborcache_seed_page_owners[neighborcache_seed_page_shard_index(node_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(owners.mu);
        auto it = owners.by_node.find(node_id);
        if (it == owners.by_node.end() || it.value().empty())
            return false;
        seeds = it.value();
    }
    for (uint32_t seed_id : seeds)
    {
        NeighborCacheSeedPageShard &shard = *_neighborcache_seed_page_shards[neighborcache_seed_page_shard_index(seed_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(shard.mu);
        auto it = shard.pages.find(seed_id);
        if (it == shard.pages.end())
            continue;
        NeighborCacheSeedPage &page = it.value();
        for (uint8_t slot = 0; slot < page.count; ++slot)
        {
            if (page.ids[slot] != node_id || page.coords[slot].size() * sizeof(T) < _disk_bytes_per_point)
                continue;
            coords = page.coords[slot];
            nbrs = page.nbrs[slot];
            page.touch[slot] = _neighborcache_seed_page_stamp.fetch_add(1, std::memory_order_relaxed) + 1;
            shard.lru.splice(shard.lru.begin(), shard.lru, page.lru_it);
            _neighborcache_seed_page_hits.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::neighborcache_seed_page_contains(uint32_t node_id)
{
    if (_neighborcache_seed_prefetch)
    {
        if (node_id == std::numeric_limits<uint32_t>::max())
            return false;
        NeighborCacheDiskLoc loc;
        if (!neighborcache_resolve_disk_cache_loc(node_id, nullptr, loc) || !neighborcache_dyn_is_sector(loc.sector))
            return false;
        return neighborcache_dyn_page_live(loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE);
    }
    if (!_neighborcache_seed_page_lru || _neighborcache_seed_page_shards.size() != kNeighborCacheSeedPageShards ||
        node_id == std::numeric_limits<uint32_t>::max())
        return false;
    std::vector<uint32_t> seeds;
    {
        NeighborCacheSeedPageOwnerShard &owners = *_neighborcache_seed_page_owners[neighborcache_seed_page_shard_index(node_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(owners.mu);
        auto it = owners.by_node.find(node_id);
        if (it == owners.by_node.end() || it.value().empty())
            return false;
        seeds = it.value();
    }
    for (uint32_t seed_id : seeds)
    {
        NeighborCacheSeedPageShard &shard = *_neighborcache_seed_page_shards[neighborcache_seed_page_shard_index(seed_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(shard.mu);
        auto it = shard.pages.find(seed_id);
        if (it == shard.pages.end())
            continue;
        const NeighborCacheSeedPage &page = it.value();
        for (uint8_t slot = 0; slot < page.count; ++slot)
        {
            if (page.ids[slot] == node_id && page.coords[slot].size() * sizeof(T) >= _disk_bytes_per_point)
                return true;
        }
    }
    return false;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_seed_page_commit(NeighborCachePendingFlush pf)
{
    if (pf.member_ids.size() < 2 || pf.member_ids.front() != pf.seed_id)
    {
        _neighborcache_seed_page_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    bool existed = false;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        uint32_t existing_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
        existed = neighborcache_dyn_seed_page_unlocked(pf.seed_id, existing_page);
    }
    const bool wrote = neighborcache_dyn_commit_one(pf, existed ? pf.seed_id : NEIGHBORCACHE_DYN_INVALID_PAGE);
    if (wrote)
        _neighborcache_seed_page_written.fetch_add(1, std::memory_order_relaxed);
    else if (existed)
        _neighborcache_seed_page_skipped.fetch_add(1, std::memory_order_relaxed);
    else
        _neighborcache_seed_page_failed.fetch_add(1, std::memory_order_relaxed);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_maybe_refresh_pair()
{
    if (!_neighborcache_dyn_enabled || neighborcache_dcache_updates_frozen() || _neighborcache_coread_pack)
        return;

    bool full = false;
    uint32_t cand_ins = std::numeric_limits<uint32_t>::max();
    uint32_t cand_del = std::numeric_limits<uint32_t>::max();
    uint32_t cand_ins_slot = NeighborCacheMetadataCache::kInvalid;
    uint32_t cand_del_slot = NeighborCacheMetadataCache::kInvalid;
    size_t cand_ins_shard = 0;
    size_t cand_del_shard = 0;
    NeighborCacheHeapEntry max_entry;
    NeighborCacheHeapEntry min_entry;
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        NeighborCacheTimedSharedMutexGuard dlock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        if (_neighborcache_ready_pairs.size() >= _neighborcache_pending_pair_cap)
            return;
        const uint64_t committed_pages = _neighborcache_dyn_committed_pages;
        full = committed_pages >= _neighborcache_dyn_page_cap;
    }
    if (!neighborcache_dyn_best_heap_entry(NeighborCacheHeapKind::Max, max_entry, cand_ins_shard))
        return;
    cand_ins_slot = max_entry.slot_id;
    cand_ins = _neighborcache_mcache.node_at(cand_ins_slot);
    if (cand_ins == NeighborCacheMetadataCache::kInvalid)
        return;
    const auto ins_st = neighborcache_dyn_state_unlocked(cand_ins_slot);
    if (ins_st == NeighborCacheNodeState::Seed || ins_st == NeighborCacheNodeState::ReadyToDeletion)
    {
        g_dyn_probe.pair_skip_max_on_disk++;
        return;
    }
    if (full)
    {
        if (_neighborcache_dcache_no_seed_replace)
        {
            g_dyn_probe.pair_skip_count++;
            return;
        }
        if (!neighborcache_dyn_best_heap_entry(NeighborCacheHeapKind::Min, min_entry, cand_del_shard))
            return;
        cand_del_slot = min_entry.slot_id;
        cand_del = _neighborcache_mcache.node_at(cand_del_slot);
        if (cand_del == NeighborCacheMetadataCache::kInvalid)
            return;
        if (max_entry.score <= min_entry.score || cand_ins == cand_del)
        {
            g_dyn_probe.pair_skip_count++;
            return;
        }
    }

    NeighborCacheMetadataCache::Snapshot snap;
    if (!_neighborcache_mcache.snapshot(cand_ins, snap))
        return;
    NeighborCachePendingFlush pf;
    if (!neighborcache_dyn_build_pending_flush(cand_ins, snap, pf))
    {
        NeighborCacheHeapShard &shard = _neighborcache_heap_shards[cand_ins_shard];
        NeighborCacheTimedMutexGuard hlock(shard.mu, NeighborCacheLockKind::Heap);
        neighborcache_dyn_clean_heap_unlocked(shard, NeighborCacheHeapKind::Max);
        if (!shard.max_heap.empty() && shard.max_heap.front().slot_id == cand_ins_slot &&
            _neighborcache_mcache.node_at(cand_ins_slot) == cand_ins)
            neighborcache_dyn_heap_erase_unlocked(cand_ins_slot);
        return;
    }

    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
        NeighborCacheTimedSharedMutexGuard dlock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        if (_neighborcache_ready_pairs.size() >= _neighborcache_pending_pair_cap)
            return;
        const uint64_t committed_pages = _neighborcache_dyn_committed_pages;
        const bool now_full = committed_pages >= _neighborcache_dyn_page_cap;
        if (now_full != full)
            return;
        float current_max_score = 0.0f;
        {
            NeighborCacheHeapShard &shard = _neighborcache_heap_shards[cand_ins_shard];
            NeighborCacheTimedMutexGuard hlock(shard.mu, NeighborCacheLockKind::Heap);
            neighborcache_dyn_clean_heap_unlocked(shard, NeighborCacheHeapKind::Max);
            if (shard.max_heap.empty() || shard.max_heap.front().slot_id != cand_ins_slot ||
                _neighborcache_mcache.node_at(cand_ins_slot) != cand_ins)
                return;
            current_max_score = shard.max_heap.front().score;
        }
        if (full)
        {
            NeighborCacheHeapShard &shard = _neighborcache_heap_shards[cand_del_shard];
            NeighborCacheTimedMutexGuard hlock(shard.mu, NeighborCacheLockKind::Heap);
            neighborcache_dyn_clean_heap_unlocked(shard, NeighborCacheHeapKind::Min);
            if (shard.min_heap.empty() || shard.min_heap.front().slot_id != cand_del_slot ||
                _neighborcache_mcache.node_at(cand_del_slot) != cand_del ||
                current_max_score <= shard.min_heap.front().score)
                return;
        }
        NeighborCacheReadyPair pair;
        pair.insertion_id = cand_ins;
        pair.deletion_id = cand_del;
        pair.insertion_slot = cand_ins_slot;
        pair.deletion_slot = cand_del_slot;
        pair.created_epoch = _neighborcache_query_epoch.load(std::memory_order_relaxed);
        pair.snap = std::move(pf);
        if (_neighborcache_pending_buffer_cap_bytes > 0)
        {
            std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
            for (uint32_t mid : pair.snap.member_ids)
                if (_neighborcache_payload_stash.find(mid) != _neighborcache_payload_stash.end())
                    ++pair.retained_members;
        }
        for (uint32_t mid : pair.snap.member_ids)
            neighborcache_pending_member_add(mid, cand_ins_slot);
        if (cand_del_slot == NeighborCacheMetadataCache::kInvalid)
            ++_neighborcache_reserved_free_pages;
        else
        {
            _neighborcache_deletion_to_pending[cand_del_slot] = cand_ins_slot;
            _neighborcache_deletion_pair_count.fetch_add(1, std::memory_order_release);
        }
        _neighborcache_ready_pairs[cand_ins_slot] = std::move(pair);
        const uint16_t retained_members = _neighborcache_ready_pairs[cand_ins_slot].retained_members;
        if (retained_members < _neighborcache_pending_buffer_buckets.size())
            _neighborcache_pending_buffer_buckets[retained_members].insert(cand_ins_slot);
        _neighborcache_pending_pair_count.fetch_add(1, std::memory_order_release);
        neighborcache_dyn_set_state_unlocked(cand_ins, cand_ins_slot, NeighborCacheNodeState::ReadyToInsertion);
        {
            NeighborCacheTimedMutexGuard hlock(neighborcache_dyn_heap_shard(cand_ins_slot).mu, NeighborCacheLockKind::Heap);
            neighborcache_dyn_heap_upsert_unlocked(cand_ins_slot, _neighborcache_mcache.score_at(cand_ins_slot));
        }
        if (cand_del != std::numeric_limits<uint32_t>::max())
        {
            neighborcache_dyn_set_state_unlocked(cand_del, cand_del_slot, NeighborCacheNodeState::ReadyToDeletion);
            NeighborCacheTimedMutexGuard hlock(neighborcache_dyn_heap_shard(cand_del_slot).mu, NeighborCacheLockKind::Heap);
            neighborcache_dyn_heap_upsert_unlocked(cand_del_slot, _neighborcache_mcache.score_at(cand_del_slot));
        }
        _neighborcache_pair_refresh++;
        g_dyn_probe.pair_set++;
        if (_neighborcache_seed_pin_neighbors)
            neighborcache_dyn_pin_seed_neighbors(cand_ins, cand_ins_slot);
    }
    if (_neighborcache_seed_page_lru || _neighborcache_seed_prefetch)
        neighborcache_seed_page_open(cand_ins);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_pin_seed_neighbors(uint32_t seed_id, uint32_t insertion_slot)
{
    if (!_neighborcache_seed_pin_neighbors || _neighborcache_mem_pool == nullptr || !_neighborcache_mem_pool->active())
        return;
    std::vector<T> coords;
    std::vector<uint32_t> nbrs;
    if (!_neighborcache_mem_pool->copy_payload(seed_id, coords, nbrs))
        return;

    uint32_t nheld = 0;
    {
        std::lock_guard<NeighborCacheSpinRwLock> lock(_neighborcache_seed_pin_mu);
        _neighborcache_seed_pin_slot[seed_id] = insertion_slot;
        auto &listed = _neighborcache_seed_pin_nbrs[seed_id];
        listed.clear();
        listed.reserve(nbrs.size());
        auto &held = _neighborcache_seed_pin_held[seed_id];
        held.clear();
        for (uint32_t nbr : nbrs)
        {
            if (nbr == seed_id)
                continue;
            if (std::find(listed.begin(), listed.end(), nbr) != listed.end())
                continue;
            listed.push_back(nbr);
            _neighborcache_seed_pin_wait[nbr].insert(seed_id);
            if (_neighborcache_mem_pool->hold_seed_neighbor(nbr, 1))
                held.insert(nbr);
        }
        nheld = static_cast<uint32_t>(held.size());
    }
    std::lock_guard<NeighborCacheSpinLock> seen_lock(_neighborcache_seed_pin_seen_mu);
    _neighborcache_seed_pin_seen_held[seed_id] = nheld;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_pin_admitted_neighbor(uint32_t node_id)
{
    if (!_neighborcache_seed_pin_neighbors || _neighborcache_mem_pool == nullptr)
        return;
    std::vector<uint32_t> seeds;
    {
        std::lock_guard<NeighborCacheSpinRwLock> lock(_neighborcache_seed_pin_mu);
        const auto it = _neighborcache_seed_pin_wait.find(node_id);
        if (it == _neighborcache_seed_pin_wait.end())
            return;
        seeds.reserve(it->second.size());
        for (uint32_t seed_id : it->second)
        {
            if (_neighborcache_seed_pin_held[seed_id].insert(node_id).second)
                seeds.push_back(seed_id);
        }
        if (seeds.empty())
            return;
    }
    if (!_neighborcache_mem_pool->hold_seed_neighbor(node_id, static_cast<uint32_t>(seeds.size())))
    {
        std::lock_guard<NeighborCacheSpinRwLock> lock(_neighborcache_seed_pin_mu);
        for (uint32_t seed_id : seeds)
        {
            auto it = _neighborcache_seed_pin_held.find(seed_id);
            if (it != _neighborcache_seed_pin_held.end())
            {
                auto found = it.value().find(node_id);
                if (found != it.value().end())
                    it.value().erase(found);
            }
        }
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_release_seed_neighbor_holds(uint32_t seed_id)
{
    if (!_neighborcache_seed_pin_neighbors || _neighborcache_mem_pool == nullptr)
        return;
    std::vector<uint32_t> nbrs;
    tsl::robin_set<uint32_t> held;
    bool present = true;
    {
        std::lock_guard<NeighborCacheSpinRwLock> lock(_neighborcache_seed_pin_mu);
        auto nbr_it = _neighborcache_seed_pin_nbrs.find(seed_id);
        if (nbr_it == _neighborcache_seed_pin_nbrs.end() && _neighborcache_seed_pin_held.find(seed_id) == _neighborcache_seed_pin_held.end())
            present = false;
        else if (nbr_it != _neighborcache_seed_pin_nbrs.end())
        {
            nbrs = std::move(nbr_it->second);
            _neighborcache_seed_pin_nbrs.erase(nbr_it);
        }
        auto held_it = _neighborcache_seed_pin_held.find(seed_id);
        if (held_it != _neighborcache_seed_pin_held.end())
        {
            held = std::move(held_it->second);
            _neighborcache_seed_pin_held.erase(held_it);
        }
        _neighborcache_seed_pin_slot.erase(seed_id);
        for (uint32_t nbr : nbrs)
        {
            auto wait_it = _neighborcache_seed_pin_wait.find(nbr);
            if (wait_it == _neighborcache_seed_pin_wait.end())
                continue;
            auto found = wait_it.value().find(seed_id);
            if (found != wait_it.value().end())
                wait_it.value().erase(found);
            if (wait_it->second.empty())
                _neighborcache_seed_pin_wait.erase(wait_it);
        }
    }
    {
        std::lock_guard<NeighborCacheSpinLock> seen_lock(_neighborcache_seed_pin_seen_mu);
        _neighborcache_seed_pin_seen_held.erase(seed_id);
    }
    if (!present)
        return;
    for (uint32_t nbr : held)
        _neighborcache_mem_pool->release_seed_neighbor(nbr);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_pin_take_write_budget(uint32_t budget_epoch)
{
    const uint32_t cap = _neighborcache_seed_pin_writes_per_query;
    if (cap == 0)
        return false;
    uint32_t seen = _neighborcache_seed_pin_budget_epoch.load(std::memory_order_relaxed);
    if (seen != budget_epoch)
    {
        uint32_t expected = seen;
        if (_neighborcache_seed_pin_budget_epoch.compare_exchange_strong(expected, budget_epoch, std::memory_order_acq_rel))
            _neighborcache_seed_pin_budget_used.store(0, std::memory_order_relaxed);
    }
    uint32_t used = _neighborcache_seed_pin_budget_used.load(std::memory_order_relaxed);
    while (used < cap)
    {
        if (_neighborcache_seed_pin_budget_used.compare_exchange_weak(used, used + 1, std::memory_order_acq_rel))
            return true;
    }
    return false;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_pin_refund_write_budget(uint32_t budget_epoch)
{
    if (_neighborcache_seed_pin_budget_epoch.load(std::memory_order_relaxed) != budget_epoch)
        return;
    uint32_t used = _neighborcache_seed_pin_budget_used.load(std::memory_order_relaxed);
    while (used > 0 &&
           !_neighborcache_seed_pin_budget_used.compare_exchange_weak(used, used - 1, std::memory_order_acq_rel))
    {
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_enqueue_seed_holding_node(uint32_t node_id, QueryStats *stats)
{
    if (!_neighborcache_seed_pin_neighbors || node_id == std::numeric_limits<uint32_t>::max())
        return;
    uint32_t best_slot = NeighborCacheMetadataCache::kInvalid;
    size_t best_held = 0;
    size_t best_members = 0;
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        std::shared_lock<NeighborCacheSpinRwLock> lock(_neighborcache_seed_pin_mu);
        const auto wait_it = _neighborcache_seed_pin_wait.find(node_id);
        if (wait_it == _neighborcache_seed_pin_wait.end())
            return;
        for (uint32_t seed_id : wait_it->second)
        {
            const auto slot_it = _neighborcache_seed_pin_slot.find(seed_id);
            if (slot_it == _neighborcache_seed_pin_slot.end())
                continue;
            const auto pair_it = _neighborcache_ready_pairs.find(slot_it->second);
            if (pair_it == _neighborcache_ready_pairs.end() || pair_it->second.committing || pair_it->second.queued ||
                pair_it->second.insertion_id != seed_id)
                continue;
            size_t held = 0;
            const auto held_it = _neighborcache_seed_pin_held.find(seed_id);
            if (held_it != _neighborcache_seed_pin_held.end())
                held = held_it->second.size();
            const size_t members = pair_it->second.snap.member_ids.size();
            if (best_slot == NeighborCacheMetadataCache::kInvalid || held > best_held ||
                (held == best_held && members > best_members))
            {
                best_held = held;
                best_members = members;
                best_slot = slot_it->second;
            }
        }
    }
    if (best_slot == NeighborCacheMetadataCache::kInvalid)
        return;
    const uint32_t budget_epoch = _neighborcache_query_epoch.load(std::memory_order_relaxed);
    if (!neighborcache_dyn_pin_take_write_budget(budget_epoch))
        return;
    neighborcache_dyn_refresh_pending_members(best_slot);
    if (!neighborcache_dyn_enqueue_ready_pair(best_slot, NeighborCacheCommitSource::Evict))
    {
        neighborcache_dyn_pin_refund_write_budget(budget_epoch);
        return;
    }
    _neighborcache_seed_pin_cold_writes.fetch_add(1, std::memory_order_relaxed);
    if (stats != nullptr)
        stats->n_neighborcache_dyn_flushes++;
}

template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::neighborcache_dyn_flush_fullest_pinned_seed()
{
    uint32_t best_slot = NeighborCacheMetadataCache::kInvalid;
    uint32_t best_seed = std::numeric_limits<uint32_t>::max();
    size_t best_held = 0;
    size_t best_members = 0;
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        std::shared_lock<NeighborCacheSpinRwLock> lock(_neighborcache_seed_pin_mu);
        for (const auto &kv : _neighborcache_ready_pairs)
        {
            if (kv.second.committing || kv.second.queued)
                continue;
            const uint32_t seed_id = kv.second.insertion_id;
            size_t held = 0;
            const auto it = _neighborcache_seed_pin_held.find(seed_id);
            if (it != _neighborcache_seed_pin_held.end())
                held = it->second.size();
            const size_t members = kv.second.snap.member_ids.size();
            if (held > best_held || (held == best_held && members > best_members))
            {
                best_held = held;
                best_members = members;
                best_slot = kv.first;
                best_seed = seed_id;
            }
        }
    }
    if (best_slot == NeighborCacheMetadataCache::kInvalid)
        return std::numeric_limits<uint32_t>::max();
    if (!neighborcache_dyn_commit_ready_pair(best_slot))
        return std::numeric_limits<uint32_t>::max();
    _neighborcache_seed_pin_forced.fetch_add(1, std::memory_order_relaxed);
    return best_seed;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_write_settled_pinned_seeds(uint32_t query_epoch, QueryStats *stats)
{
    if (!_neighborcache_seed_pin_neighbors || _neighborcache_seed_pin_age_queries == 0)
        return;
    struct Observed
    {
        uint32_t slot = 0;
        uint32_t seed_id = 0;
        uint32_t age = 0;
        uint32_t held = 0;
    };
    std::vector<Observed> observed;
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        std::shared_lock<NeighborCacheSpinRwLock> lock(_neighborcache_seed_pin_mu);
        observed.reserve(_neighborcache_ready_pairs.size());
        for (const auto &kv : _neighborcache_ready_pairs)
        {
            if (kv.second.committing || kv.second.queued)
                continue;
            const uint32_t seed_id = kv.second.insertion_id;
            uint32_t held = 0;
            const auto held_it = _neighborcache_seed_pin_held.find(seed_id);
            if (held_it != _neighborcache_seed_pin_held.end())
                held = static_cast<uint32_t>(held_it->second.size());
            const uint32_t age =
                query_epoch >= kv.second.created_epoch ? query_epoch - kv.second.created_epoch : 0;
            observed.push_back(Observed{kv.first, seed_id, age, held});
        }
    }
    struct Due
    {
        uint32_t slot = 0;
        uint32_t held = 0;
    };
    std::vector<Due> due;
    {
        std::lock_guard<NeighborCacheSpinLock> seen_lock(_neighborcache_seed_pin_seen_mu);
        for (const Observed &item : observed)
        {
            const auto seen = _neighborcache_seed_pin_seen_held.find(item.seed_id);
            const bool first = seen == _neighborcache_seed_pin_seen_held.end();
            const bool grew = !first && seen->second != item.held;
            if (first || grew)
            {
                _neighborcache_seed_pin_seen_held[item.seed_id] = item.held;
                continue;
            }
            if (item.age < _neighborcache_seed_pin_age_queries)
                continue;
            due.push_back(Due{item.slot, item.held});
        }
    }
    std::sort(due.begin(), due.end(), [](const Due &a, const Due &b) { return a.held > b.held; });
    const uint32_t budget_epoch = query_epoch > 0 ? query_epoch - 1 : 0;
    for (const Due &item : due)
    {
        if (!neighborcache_dyn_pin_take_write_budget(budget_epoch))
            break;
        neighborcache_dyn_refresh_pending_members(item.slot);
        if (!neighborcache_dyn_enqueue_ready_pair(item.slot, NeighborCacheCommitSource::Expired))
        {
            neighborcache_dyn_pin_refund_write_budget(budget_epoch);
            continue;
        }
        _neighborcache_seed_pin_early_writes.fetch_add(1, std::memory_order_relaxed);
        if (stats != nullptr)
            stats->n_neighborcache_dyn_flushes++;
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_request_refresh()
{
    if (!_neighborcache_dyn_enabled)
        return;
    neighborcache_mark_refresh_needed();
    if (_neighborcache_refresh_thread_on && !neighborcache_is_refresh_thread())
        return;
    if (_neighborcache_pending_pair_count.load(std::memory_order_acquire) >= _neighborcache_pending_pair_cap)
        return;
    if (_neighborcache_refresh_running.load(std::memory_order_relaxed))
        return;

    bool expected = false;
    if (!_neighborcache_refresh_running.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        return;

    for (unsigned pass = 0; pass < 4; ++pass)
    {
        _neighborcache_refresh_needed.store(false, std::memory_order_release);
        if (_neighborcache_pending_pair_count.load(std::memory_order_acquire) < _neighborcache_pending_pair_cap)
            neighborcache_dyn_maybe_refresh_pair();
        if (!_neighborcache_refresh_needed.load(std::memory_order_acquire))
            break;
    }
    _neighborcache_refresh_running.store(false, std::memory_order_release);

    if (_neighborcache_refresh_needed.load(std::memory_order_acquire) &&
        _neighborcache_pending_pair_count.load(std::memory_order_acquire) < _neighborcache_pending_pair_cap)
    {
        expected = false;
        if (_neighborcache_refresh_running.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        {
            _neighborcache_refresh_needed.store(false, std::memory_order_release);
            neighborcache_dyn_maybe_refresh_pair();
            _neighborcache_refresh_running.store(false, std::memory_order_release);
        }
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_enqueue_ready_pair(uint32_t insertion_slot, NeighborCacheCommitSource source)
{
    if (!_neighborcache_commit_started || _neighborcache_commit_stop.load(std::memory_order_acquire) ||
        insertion_slot == NeighborCacheMetadataCache::kInvalid || _neighborcache_freeze_maintenance ||
        neighborcache_dcache_updates_frozen())
        return false;

    if (_neighborcache_commit_inline)
    {
        _neighborcache_commit_enqueued.fetch_add(1, std::memory_order_relaxed);
        _neighborcache_commit_source_enqueued[static_cast<size_t>(source)].fetch_add(1, std::memory_order_relaxed);
        if (neighborcache_dyn_commit_ready_pair(insertion_slot))
        {
            _neighborcache_commit_completed.fetch_add(1, std::memory_order_relaxed);
            _neighborcache_commit_source_completed[static_cast<size_t>(source)].fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        _neighborcache_commit_failed.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    auto set_queued = [this, insertion_slot](bool queued) {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
        auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
        if (pair_it == _neighborcache_ready_pairs.end() || pair_it->second.committing)
            return false;
        pair_it.value().queued = queued;
        return true;
    };
    if (!set_queued(true))
        return false;

    bool dropped = false;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_commit_q_mu);
        if (_neighborcache_commit_pending_slots.find(insertion_slot) != _neighborcache_commit_pending_slots.end())
        {
            _neighborcache_commit_coalesced.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        dropped = _neighborcache_commit_q.size() >= _neighborcache_commit_q_cap;
        if (dropped)
            _neighborcache_commit_dropped.fetch_add(1, std::memory_order_relaxed);
    }
    if (dropped)
    {
        set_queued(false);
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(_neighborcache_commit_q_mu);
        if (_neighborcache_commit_pending_slots.find(insertion_slot) != _neighborcache_commit_pending_slots.end())
        {
            _neighborcache_commit_coalesced.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        NeighborCachePairCommitJob job;
        job.insertion_slot = insertion_slot;
        job.source = source;
        job.enqueued_ns = neighborcache_lock_now_ns();
        _neighborcache_commit_q.push_back(job);
        _neighborcache_commit_pending_slots.insert(insertion_slot);
        _neighborcache_commit_enqueued.fetch_add(1, std::memory_order_relaxed);
        _neighborcache_commit_source_enqueued[static_cast<size_t>(source)].fetch_add(1, std::memory_order_relaxed);
    }
    _neighborcache_commit_q_cv.notify_one();
    return true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_commit_loop()
{
    neighborcache_pin_background_thread();
    if (_neighborcache_dyn_reader != nullptr)
        _neighborcache_dyn_reader->register_thread();
    struct Inflight
    {
        NeighborCachePendingFlush pf;
        bool hop2 = false;
        uint32_t page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
        std::shared_ptr<NeighborCacheDynWriteJob> job;
    };
    std::deque<Inflight> inflight;
    const size_t depth = std::max<size_t>(1, _neighborcache_dyn_writer_thread_count);
    auto drop_pending = [this](const NeighborCachePendingFlush &pf) {
        std::lock_guard<NeighborCacheSpinLock> pending_lock(_neighborcache_coread_pending_mu);
        for (uint32_t id : pf.member_ids)
            _neighborcache_coread_pending.erase(id);
        _neighborcache_commit_q_cv.notify_all();
    };
    auto finish_one = [&]() {
        if (inflight.empty())
            return;
        Inflight item = std::move(inflight.front());
        inflight.pop_front();
        bool write_ok = false;
        if (item.job != nullptr)
        {
            std::unique_lock<std::mutex> done_lock(item.job->mu);
            item.job->cv.wait(done_lock, [&item] { return item.job->done; });
            write_ok = item.job->ok;
        }
        const bool committed = neighborcache_dyn_coread_complete_page(item.pf, item.page_idx, write_ok);
        if (committed)
        {
            _neighborcache_coread_pages.fetch_add(1, std::memory_order_relaxed);
            if (item.hop2)
                _neighborcache_coread_hop2_pages.fetch_add(1, std::memory_order_relaxed);
        }
        drop_pending(item.pf);
    };
    while (true)
    {
        NeighborCachePairCommitJob job;
        NeighborCacheCoreadWriteJob coread_job;
        NeighborCachePendingFlush seed_pf;
        NeighborCachePendingFlush family_pf;
        enum class Step
        {
            Coread,
            SeedPage,
            Family,
            Pair,
            Drain,
            Exit
        } step = Step::Drain;
        {
            std::unique_lock<std::mutex> lock(_neighborcache_commit_q_mu);
            _neighborcache_commit_q_cv.wait(lock, [this, &inflight] {
                return _neighborcache_commit_stop.load(std::memory_order_acquire) || !_neighborcache_commit_q.empty() ||
                       !_neighborcache_coread_write_q.empty() || !_neighborcache_seed_page_q.empty() || !_neighborcache_family_q.empty() ||
                       !inflight.empty();
            });
            if (!_neighborcache_coread_write_q.empty() && inflight.size() < depth)
            {
                coread_job = std::move(_neighborcache_coread_write_q.front());
                _neighborcache_coread_write_q.pop_front();
                step = Step::Coread;
            }
            else if (!_neighborcache_seed_page_q.empty())
            {
                seed_pf = std::move(_neighborcache_seed_page_q.front());
                _neighborcache_seed_page_q.pop_front();
                _neighborcache_seed_page_inflight.fetch_add(1, std::memory_order_acq_rel);
                step = Step::SeedPage;
            }
            else if (!_neighborcache_family_q.empty())
            {
                const uint32_t seed_id = _neighborcache_family_q.front();
                _neighborcache_family_q.pop_front();
                auto pending = _neighborcache_family_pending.find(seed_id);
                if (pending == _neighborcache_family_pending.end())
                    continue;
                family_pf = std::move(pending.value());
                _neighborcache_family_pending.erase(pending);
                _neighborcache_family_inflight.fetch_add(1, std::memory_order_acq_rel);
                step = Step::Family;
            }
            else if (!inflight.empty())
                step = Step::Drain;
            else if (_neighborcache_commit_stop.load(std::memory_order_acquire) && _neighborcache_commit_q.empty() &&
                     _neighborcache_seed_page_q.empty() && _neighborcache_family_q.empty())
                step = Step::Exit;
            else if (!_neighborcache_commit_q.empty())
            {
                job = _neighborcache_commit_q.front();
                _neighborcache_commit_q.pop_front();
                step = Step::Pair;
            }
            else
                continue;
        }
        if (step == Step::Exit)
            return;
        if (step == Step::SeedPage)
        {
            neighborcache_seed_page_commit(std::move(seed_pf));
            _neighborcache_seed_page_inflight.fetch_sub(1, std::memory_order_acq_rel);
            _neighborcache_commit_q_cv.notify_all();
            continue;
        }
        if (step == Step::Family)
        {
            neighborcache_family_commit(std::move(family_pf));
            _neighborcache_family_inflight.fetch_sub(1, std::memory_order_acq_rel);
            _neighborcache_commit_q_cv.notify_all();
            continue;
        }
        if (step == Step::Drain)
        {
            finish_one();
            continue;
        }
        if (step == Step::Coread)
        {
            uint32_t page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
            std::shared_ptr<NeighborCacheDynWriteJob> write_job;
            if (!neighborcache_dyn_coread_submit_page(coread_job.pf, page_idx, write_job))
            {
                drop_pending(coread_job.pf);
                continue;
            }
            Inflight item;
            item.pf = std::move(coread_job.pf);
            item.hop2 = coread_job.hop2;
            item.page_idx = page_idx;
            item.job = std::move(write_job);
            inflight.push_back(std::move(item));
            continue;
        }

        const uint64_t work_started_ns = neighborcache_lock_now_ns();
        if (job.enqueued_ns != 0)
            _neighborcache_commit_queue_ns.fetch_add(work_started_ns - job.enqueued_ns, std::memory_order_relaxed);
        const bool committed = neighborcache_dyn_commit_ready_pair(job.insertion_slot);
        _neighborcache_commit_work_ns.fetch_add(neighborcache_lock_now_ns() - work_started_ns, std::memory_order_relaxed);
        if (committed)
        {
            _neighborcache_commit_completed.fetch_add(1, std::memory_order_relaxed);
            _neighborcache_commit_source_completed[static_cast<size_t>(job.source)].fetch_add(1,
                                                                                      std::memory_order_relaxed);
        }
        else
            _neighborcache_commit_failed.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(_neighborcache_commit_q_mu);
            _neighborcache_commit_pending_slots.erase(job.insertion_slot);
        }
        _neighborcache_commit_q_cv.notify_all();
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::start_neighborcache_dyn_commit_workers()
{
    if (_neighborcache_commit_started)
        return;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_commit_q_mu);
        _neighborcache_commit_q.clear();
        _neighborcache_commit_pending_slots.clear();
        _neighborcache_coread_write_q.clear();
        _neighborcache_seed_page_q.clear();
        _neighborcache_seed_page_inflight.store(0, std::memory_order_relaxed);
        _neighborcache_family_q.clear();
        _neighborcache_family_pending.clear();
        _neighborcache_family_inflight.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<NeighborCacheSpinLock> pending(_neighborcache_coread_pending_mu);
            _neighborcache_coread_pending.clear();
        }
        _neighborcache_commit_stop.store(false, std::memory_order_release);
    }
    _neighborcache_commit_threads.clear();
    _neighborcache_commit_threads.reserve(_neighborcache_commit_thread_count);
    for (size_t thread = 0; thread < _neighborcache_commit_thread_count; ++thread)
        _neighborcache_commit_threads.emplace_back([this] { neighborcache_dyn_commit_loop(); });
    _neighborcache_commit_started = true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::stop_neighborcache_dyn_commit_workers()
{
    if (!_neighborcache_commit_started)
        return;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_commit_q_mu);
        _neighborcache_commit_stop.store(true, std::memory_order_release);
    }
    _neighborcache_commit_q_cv.notify_all();
    for (std::thread &thread : _neighborcache_commit_threads)
        if (thread.joinable())
            thread.join();
    _neighborcache_commit_threads.clear();
    {
        std::lock_guard<std::mutex> lock(_neighborcache_commit_q_mu);
        _neighborcache_commit_q.clear();
        _neighborcache_commit_pending_slots.clear();
        _neighborcache_coread_write_q.clear();
        _neighborcache_seed_page_q.clear();
        _neighborcache_family_q.clear();
        _neighborcache_family_pending.clear();
        {
            std::lock_guard<NeighborCacheSpinLock> pending(_neighborcache_coread_pending_mu);
            _neighborcache_coread_pending.clear();
        }
    }
    _neighborcache_commit_started = false;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_wait_for_commits()
{
    if (!_neighborcache_commit_started)
        return;
    std::unique_lock<std::mutex> lock(_neighborcache_commit_q_mu);
    _neighborcache_commit_q_cv.wait(lock, [this] {
        std::lock_guard<NeighborCacheSpinLock> pending(_neighborcache_coread_pending_mu);
        return (_neighborcache_commit_pending_slots.empty() && _neighborcache_coread_write_q.empty() &&
                _neighborcache_coread_pending.empty() && _neighborcache_seed_page_q.empty() &&
                _neighborcache_seed_page_inflight.load(std::memory_order_acquire) == 0 && _neighborcache_family_q.empty() &&
                _neighborcache_family_inflight.load(std::memory_order_acquire) == 0) ||
               _neighborcache_commit_stop.load(std::memory_order_acquire);
    });
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_coread_id_pending(uint32_t id)
{
    std::lock_guard<NeighborCacheSpinLock> lock(_neighborcache_coread_pending_mu);
    return _neighborcache_coread_pending.find(id) != _neighborcache_coread_pending.end();
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_enqueue_coread_write(NeighborCachePendingFlush &pf, bool hop2, bool &handed_off)
{
    handed_off = false;
    if (!_neighborcache_commit_started || _neighborcache_commit_inline)
        return neighborcache_dyn_commit_one(pf);
    {
        std::lock_guard<std::mutex> lock(_neighborcache_commit_q_mu);
        if (_neighborcache_commit_stop.load(std::memory_order_acquire))
            return false;
        if (_neighborcache_coread_write_q.size() >= _neighborcache_coread_write_q_cap)
        {
            _neighborcache_coread_write_dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        NeighborCacheCoreadWriteJob job;
        job.hop2 = hop2;
        job.pf = std::move(pf);
        {
            std::lock_guard<NeighborCacheSpinLock> pending(_neighborcache_coread_pending_mu);
            for (uint32_t id : job.pf.member_ids)
                _neighborcache_coread_pending.insert(id);
        }
        _neighborcache_coread_write_q.push_back(std::move(job));
        _neighborcache_coread_enqueued.fetch_add(1, std::memory_order_relaxed);
        handed_off = true;
    }
    _neighborcache_commit_q_cv.notify_one();
    return true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::flush_neighborcache_coread_writes()
{
    neighborcache_dyn_wait_for_commits();
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::neighborcache_dyn_commit_ready_pair(uint32_t insertion_slot)
{
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
        auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
        if (pair_it == _neighborcache_ready_pairs.end() || pair_it->second.committing)
        {
            _neighborcache_commit_fail_why[0].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        pair_it.value().queued = false;
        const NeighborCacheReadyPair &candidate = pair_it->second;
        const size_t page_capacity = static_cast<size_t>(
            (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
        if (_neighborcache_pending_require_full_page &&
            candidate.snap.member_ids.size() < page_capacity)
        {
            _neighborcache_commit_fail_why[1].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const float insertion_score = _neighborcache_mcache.score_at(candidate.insertion_slot);
        if (_neighborcache_mcache.node_at(candidate.insertion_slot) != candidate.insertion_id ||
            (candidate.deletion_id != std::numeric_limits<uint32_t>::max() &&
             _neighborcache_mcache.node_at(candidate.deletion_slot) != candidate.deletion_id))
        {
            _neighborcache_commit_fail_why[2].fetch_add(1, std::memory_order_relaxed);
            neighborcache_dyn_clear_ready_pair_unlocked(insertion_slot);
            return false;
        }
        if (candidate.deletion_id != std::numeric_limits<uint32_t>::max() &&
            insertion_score <= _neighborcache_mcache.score_at(candidate.deletion_slot))
        {
            _neighborcache_commit_fail_why[3].fetch_add(1, std::memory_order_relaxed);
            neighborcache_dyn_clear_ready_pair_unlocked(insertion_slot);
            return false;
        }
        if (pair_it.value().retained_members < _neighborcache_pending_buffer_buckets.size())
            _neighborcache_pending_buffer_buckets[pair_it.value().retained_members].erase(insertion_slot);
        pair_it.value().committing = true;
    }

    NeighborCacheReadyPair pair;
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        const auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
        if (pair_it == _neighborcache_ready_pairs.end() || !pair_it->second.committing)
            return false;
        pair = pair_it->second;
    }
    NeighborCachePendingFlush pf = pair.snap;
    if (_neighborcache_deferred_ncache_page_write)
    {
        pf.member_coords.clear();
        pf.member_nbrs.clear();
        pf.member_coords.reserve(pf.member_ids.size());
        pf.member_nbrs.reserve(pf.member_ids.size());
        for (uint32_t member_id : pf.member_ids)
        {
            std::vector<T> coords;
            std::vector<uint32_t> nbrs;
            if (!neighborcache_dyn_copy_member_payload(member_id, coords, nbrs))
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
            _neighborcache_commit_fail_why[4].fetch_add(1, std::memory_order_relaxed);
            NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
            auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
            if (pair_it != _neighborcache_ready_pairs.end())
            {
                pair_it.value().committing = false;
                neighborcache_dyn_clear_ready_pair_unlocked(insertion_slot);
            }
            return false;
        }
    }
    const bool wrote = neighborcache_dyn_commit_one(pf, pair.deletion_id);

    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
        auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
        if (pair_it == _neighborcache_ready_pairs.end())
        {
            _neighborcache_commit_fail_why[0].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!wrote)
        {
            _neighborcache_commit_fail_why[5].fetch_add(1, std::memory_order_relaxed);
            pair_it.value().committing = false;
            neighborcache_dyn_clear_ready_pair_unlocked(insertion_slot);
            return false;
        }

        const uint32_t ins = pair.insertion_id;
        const uint32_t del = pair.deletion_id;
        const uint32_t ins_slot = pair.insertion_slot;
        const uint32_t del_slot = pair.deletion_slot;
        const float insertion_score = _neighborcache_mcache.score_at(ins_slot);
        neighborcache_dyn_clear_ready_pair_unlocked(ins_slot, false);
        if (del == std::numeric_limits<uint32_t>::max() && neighborcache_dyn_disk_is_full())
        {
            std::vector<uint32_t> surplus_free_pairs;
            surplus_free_pairs.reserve(_neighborcache_ready_pairs.size());
            for (const auto &kv : _neighborcache_ready_pairs)
                if (kv.second.deletion_slot == NeighborCacheMetadataCache::kInvalid)
                    surplus_free_pairs.push_back(kv.first);
            for (uint32_t slot_id : surplus_free_pairs)
                neighborcache_dyn_clear_ready_pair_unlocked(slot_id);
        }
        if (del != std::numeric_limits<uint32_t>::max())
        {
            neighborcache_dyn_set_state_unlocked(del, del_slot, NeighborCacheNodeState::NonSeed);
            {
                NeighborCacheTimedMutexGuard hlock(neighborcache_dyn_heap_shard(del_slot).mu, NeighborCacheLockKind::Heap);
                neighborcache_dyn_heap_upsert_unlocked(del_slot, _neighborcache_mcache.score_at(del_slot));
            }
            _neighborcache_heap_deletes++;
            g_dyn_probe.delete_ok++;
        }
        neighborcache_dyn_set_state_unlocked(ins, ins_slot, NeighborCacheNodeState::Seed);
        {
            NeighborCacheTimedMutexGuard hlock(neighborcache_dyn_heap_shard(ins_slot).mu, NeighborCacheLockKind::Heap);
            neighborcache_dyn_heap_upsert_unlocked(ins_slot, insertion_score);
        }
        _neighborcache_heap_writes++;
        g_dyn_probe.write_ok++;
    }
    neighborcache_dyn_release_unreferenced_buffer_members(pair.snap.member_ids);
    return true;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_commit_expired_pairs(uint32_t query_epoch, QueryStats *stats)
{
    if (!_neighborcache_dyn_enabled || !_neighborcache_deferred_ncache_page_write)
        return;

    std::vector<uint32_t> expired;
    expired.reserve(_neighborcache_pending_force_batch);
    {
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        for (const auto &entry : _neighborcache_ready_pairs)
        {
            const NeighborCacheReadyPair &pair = entry.second;
            if (!pair.committing && pair.created_epoch <= query_epoch &&
                static_cast<uint32_t>(query_epoch - pair.created_epoch) >= _neighborcache_pending_max_age_queries)
            {
                expired.push_back(entry.first);
                if (expired.size() >= _neighborcache_pending_force_batch)
                    break;
            }
        }
    }

    g_dyn_probe.timeout_trig.fetch_add(expired.size(), std::memory_order_relaxed);
    for (uint32_t insertion_slot : expired)
    {
        if (_neighborcache_pending_require_full_page)
            neighborcache_dyn_refresh_pending_members(insertion_slot);
        if (!neighborcache_dyn_enqueue_ready_pair(insertion_slot, NeighborCacheCommitSource::Expired))
        {
            if (_neighborcache_pending_require_full_page)
            {
                NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
                const auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
                if (pair_it != _neighborcache_ready_pairs.end() && !pair_it->second.committing &&
                    pair_it->second.created_epoch <= query_epoch &&
                    static_cast<uint32_t>(query_epoch - pair_it->second.created_epoch) >=
                        _neighborcache_pending_max_age_queries)
                    neighborcache_dyn_clear_ready_pair_unlocked(insertion_slot);
            }
            continue;
        }
        g_dyn_probe.timeout_ok.fetch_add(1, std::memory_order_relaxed);
        if (stats != nullptr)
            stats->n_neighborcache_dyn_flushes++;
    }
}

template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::neighborcache_dyn_admit_node(uint32_t node_id, const char *node_disk_buf, QueryStats *stats,
                                                      SSDQueryScratch<T> *query_scratch, uint32_t search_hop)
{
    if (_neighborcache_mem_pool == nullptr || !_neighborcache_mem_pool->active() ||
        !(_neighborcache_mem_runtime_admit || _neighborcache_dyn_enabled))
        return NeighborCacheMemoryPool<T>::INVALID_NODE;

    bool admitted_after_rejection = false;
    {
        if (!_neighborcache_mem_pool->should_admit(node_id, search_hop, admitted_after_rejection))
            return NeighborCacheMemoryPool<T>::INVALID_NODE;
    }

    const bool dynamic_maintenance =
        _neighborcache_dyn_enabled && query_scratch != nullptr &&
        query_scratch->neighborcache_dcache_maintenance_enabled;
    if (dynamic_maintenance && !_neighborcache_coread_pack && !_neighborcache_mem_pool->contains(node_id))
    {
        if (_neighborcache_seed_pin_neighbors)
        {
            const uint32_t cold_held = _neighborcache_mem_pool->coldest_held_node(node_id);
            if (cold_held != NeighborCacheMemoryPool<T>::INVALID_NODE)
                neighborcache_dyn_enqueue_seed_holding_node(cold_held, stats);
        }
        uint32_t victim;
        bool hit_hot_prefix = false;
        {
            victim = _neighborcache_mem_pool->peek_lru_victim(node_id, _neighborcache_seed_pin_neighbors ? &hit_hot_prefix : nullptr);
        }
        if (hit_hot_prefix)
        {
            const uint32_t seed_id = neighborcache_dyn_flush_fullest_pinned_seed();
            if (seed_id != std::numeric_limits<uint32_t>::max())
                _neighborcache_mem_pool->demote_lru_tail(seed_id);
            hit_hot_prefix = false;
            victim = _neighborcache_mem_pool->peek_lru_victim(node_id, &hit_hot_prefix);
            if (hit_hot_prefix)
            {
                _neighborcache_mem_pool->drop_coldest_seed_hold(node_id);
                victim = _neighborcache_mem_pool->peek_lru_victim(node_id, nullptr);
            }
        }
        if (victim != NeighborCacheMemoryPool<T>::INVALID_NODE)
            neighborcache_dyn_on_ncache_evict(victim, stats);
    }
    uint32_t evicted;
    {
        evicted = _neighborcache_mem_pool->try_admit(node_id, node_disk_buf, _disk_bytes_per_point, _max_node_len,
                                             search_hop, admitted_after_rejection);
    }
    if (_neighborcache_seed_pin_neighbors && _neighborcache_mem_pool->contains(node_id))
        neighborcache_dyn_pin_admitted_neighbor(node_id);
    if (!_neighborcache_freeze_maintenance && (_neighborcache_seed_page_lru || _neighborcache_seed_prefetch) &&
        _neighborcache_mem_pool->contains(node_id))
        neighborcache_seed_page_note(node_id);
    if (stats != nullptr && evicted != NeighborCacheMemoryPool<T>::INVALID_NODE)
        stats->n_neighborcache_mem_evictions++;
    // Co-read packing does not read the m-cache heap. Skip the second N-cache
    // lookup that only existed to mark that heap dirty.
    if (dynamic_maintenance && !_neighborcache_coread_pack && _neighborcache_mcache.size() != 0 &&
        _neighborcache_mem_pool->contains(node_id) && query_scratch != nullptr)
    {
        const uint32_t slot_id = _neighborcache_mcache.slot_of(node_id);
        if (slot_id != NeighborCacheMetadataCache::kInvalid)
            query_scratch->mark_neighborcache_heap_dirty(slot_id);
    }
    return evicted;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_stash_evict_unlocked()
{
    size_t pinned_rotations = 0;
    while (_neighborcache_payload_stash.size() > _neighborcache_stash_cap && !_neighborcache_stash_lru.empty())
    {
        const uint32_t victim = _neighborcache_stash_lru.back();
        if (_neighborcache_patch_pin_count.find(victim) != _neighborcache_patch_pin_count.end())
        {
            if (++pinned_rotations > _neighborcache_payload_stash.size())
                break;
            _neighborcache_stash_lru.splice(_neighborcache_stash_lru.begin(), _neighborcache_stash_lru, std::prev(_neighborcache_stash_lru.end()));
            continue;
        }
        _neighborcache_stash_lru.pop_back();
        _neighborcache_payload_stash.erase(victim);
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_note_base_load(uint32_t node_id, uint32_t parent, const char *node_disk_buf,
                                                       SSDQueryScratch<T> *query_scratch)
{
    if (!_neighborcache_dyn_enabled || query_scratch == nullptr ||
        !query_scratch->neighborcache_dcache_maintenance_enabled || node_disk_buf == nullptr)
        return;

    const T *coord_src = offset_to_node_coords(const_cast<char *>(node_disk_buf));
    const uint32_t *nhood = offset_to_node_nhood(const_cast<char *>(node_disk_buf));
    if (_neighborcache_real_io_coaccess)
    {
        query_scratch->neighborcache_storage_node_order.push_back(node_id);
        if (query_scratch->neighborcache_record_real_io)
            neighborcache_push_storage_nbrs(query_scratch->neighborcache_storage_nbrs, query_scratch->neighborcache_storage_nbr_off, nhood,
                                    static_cast<uint32_t>(_max_degree));
    }
    const uint32_t nnbrs = nhood[0];
    if (nnbrs > _max_degree)
        return;

    if ((!_neighborcache_deferred_ncache_page_write || _neighborcache_coread_pack) &&
        !(_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active() && _neighborcache_mem_pool->contains(node_id)))
    {
        const size_t ncoords = _disk_bytes_per_point / sizeof(T);
        std::vector<T> coords(coord_src, coord_src + ncoords);
        std::vector<uint32_t> nbrs(nhood + 1, nhood + 1 + nnbrs);
        std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
        {
            auto it = _neighborcache_payload_stash.find(node_id);
            if (it != _neighborcache_payload_stash.end())
            {
                it.value().coords = std::move(coords);
                it.value().nbrs = std::move(nbrs);
                _neighborcache_stash_lru.splice(_neighborcache_stash_lru.begin(), _neighborcache_stash_lru, it.value().lru_it);
                it.value().lru_it = _neighborcache_stash_lru.begin();
            }
            else
            {
                _neighborcache_stash_lru.push_front(node_id);
                NeighborCachePayloadStashEntry ent;
                ent.coords = std::move(coords);
                ent.nbrs = std::move(nbrs);
                ent.lru_it = _neighborcache_stash_lru.begin();
                _neighborcache_payload_stash[node_id] = std::move(ent);
                neighborcache_dyn_stash_evict_unlocked();
            }
        }
    }

    // Co-read packing uses the query-local parent map, not m-cache child lists.
    if (!_neighborcache_coread_pack)
    {
        const uint32_t parent_slot = parent != std::numeric_limits<uint32_t>::max()
                                         ? _neighborcache_mcache.slot_of(parent)
                                         : NeighborCacheMetadataCache::kInvalid;
        if (parent_slot != NeighborCacheMetadataCache::kInvalid && parent_slot < _neighborcache_parent_fetched_by_slot.size())
        {
            std::lock_guard<std::mutex> slock(neighborcache_parent_fetched_mu(parent_slot));
            if (_neighborcache_parent_fetched_owner[parent_slot] != parent)
            {
                _neighborcache_parent_fetched_by_slot[parent_slot].clear();
                _neighborcache_parent_fetched_owner[parent_slot] = parent;
            }
            _neighborcache_parent_fetched_by_slot[parent_slot].touch(node_id);
        }

        if (parent_slot != NeighborCacheMetadataCache::kInvalid && query_scratch != nullptr)
            query_scratch->mark_neighborcache_heap_dirty(parent_slot);
        if (parent != std::numeric_limits<uint32_t>::max())
            neighborcache_dyn_maybe_mark_seed(parent);
    }

    if (_neighborcache_dcache_query_patch && parent != std::numeric_limits<uint32_t>::max() &&
        neighborcache_dyn_note_patch_candidate(node_id, parent, query_scratch))
    {
        const size_t ncoords = _disk_bytes_per_point / sizeof(T);
        std::vector<T> coords(coord_src, coord_src + ncoords);
        std::vector<uint32_t> nbrs(nhood + 1, nhood + 1 + nnbrs);
        std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
        auto it = _neighborcache_payload_stash.find(node_id);
        if (it != _neighborcache_payload_stash.end())
        {
            it.value().coords = std::move(coords);
            it.value().nbrs = std::move(nbrs);
            _neighborcache_stash_lru.splice(_neighborcache_stash_lru.begin(), _neighborcache_stash_lru, it.value().lru_it);
            it.value().lru_it = _neighborcache_stash_lru.begin();
        }
        else
        {
            _neighborcache_stash_lru.push_front(node_id);
            NeighborCachePayloadStashEntry ent;
            ent.coords = std::move(coords);
            ent.nbrs = std::move(nbrs);
            ent.lru_it = _neighborcache_stash_lru.begin();
            _neighborcache_payload_stash[node_id] = std::move(ent);
        }
        neighborcache_dyn_stash_evict_unlocked();
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_has_payload(uint32_t node_id) const
{
    if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active() && _neighborcache_mem_pool->contains(node_id))
        return true;
    {
        std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
        if (_neighborcache_payload_stash.find(node_id) != _neighborcache_payload_stash.end())
            return true;
    }
    NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
    bool found = false;
    neighborcache_dc_for_each_loc_unlocked(node_id, [&](const NeighborCacheDiskLoc &loc, size_t) {
        if (found)
            return;
        if (!neighborcache_dyn_is_sector(loc.sector))
            return;
        const uint32_t idx = loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
        if (neighborcache_dyn_page_live(idx))
            found = true;
    });
    return found;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_extract_overlay_payload(uint32_t node_id, std::vector<T> &coords,
                                                               std::vector<uint32_t> &nbrs) const
{
    NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
    bool found = false;
    neighborcache_dc_for_each_loc_unlocked(node_id, [&](const NeighborCacheDiskLoc &loc, size_t) {
        if (found)
            return;
        if (!neighborcache_dyn_is_sector(loc.sector))
            return;
        const uint32_t idx = loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
        if (!neighborcache_dyn_page_live(idx) || _neighborcache_dyn_reader == nullptr)
            return;
        char *page = nullptr;
        alloc_aligned(reinterpret_cast<void **>(&page), defaults::SECTOR_LEN, defaults::SECTOR_LEN);
        std::vector<AlignedRead> reads;
        reads.emplace_back(static_cast<uint64_t>(idx) * defaults::SECTOR_LEN, defaults::SECTOR_LEN, page);
        // A dirty seed page may never have reached D yet. Maintenance must
        // consult the same prefetch buffer before falling back to its D offset.
        if (!neighborcache_use_shared_cache(loc.sector) ||
            !_neighborcache_shared_cache.get_range(loc.sector, 1, page))
        {
            if (!neighborcache_dyn_disk_readable(loc.sector))
            {
                aligned_free(page);
                return;
            }
            _neighborcache_dyn_reader->read(reads, _neighborcache_dyn_reader->get_ctx());
        }
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
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_copy_member_payload(uint32_t node_id, std::vector<T> &coords,
                                                            std::vector<uint32_t> &nbrs, bool allow_disk) const
{
    if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->copy_payload(node_id, coords, nbrs))
        return true;
    {
        std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
        const auto it = _neighborcache_payload_stash.find(node_id);
        if (it != _neighborcache_payload_stash.end())
        {
            coords = it.value().coords;
            nbrs = it.value().nbrs;
            return true;
        }
    }
    if (!allow_disk)
        return false;
    return neighborcache_dyn_extract_overlay_payload(node_id, coords, nbrs);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_buffer_evicted_member(uint32_t node_id, bool &inserted)
{
    inserted = false;
    if (_neighborcache_pending_buffer_cap_bytes == 0 || _neighborcache_mem_pool == nullptr)
        return false;

    std::vector<T> coords;
    std::vector<uint32_t> nbrs;
    if (!_neighborcache_mem_pool->copy_payload(node_id, coords, nbrs))
        return false;

    std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
    auto existing = _neighborcache_payload_stash.find(node_id);
    if (existing != _neighborcache_payload_stash.end())
    {
        _neighborcache_stash_lru.splice(_neighborcache_stash_lru.begin(), _neighborcache_stash_lru, existing.value().lru_it);
        existing.value().lru_it = _neighborcache_stash_lru.begin();
        return _neighborcache_pending_buffer_payload_bytes > _neighborcache_pending_buffer_cap_bytes;
    }

    _neighborcache_stash_lru.push_front(node_id);
    NeighborCachePayloadStashEntry entry;
    entry.coords = std::move(coords);
    entry.nbrs = std::move(nbrs);
    entry.lru_it = _neighborcache_stash_lru.begin();
    _neighborcache_pending_buffer_payload_bytes +=
        entry.coords.size() * sizeof(T) + entry.nbrs.size() * sizeof(uint32_t);
    _neighborcache_payload_stash[node_id] = std::move(entry);
    _neighborcache_pending_buffer_inserts.fetch_add(1, std::memory_order_relaxed);
    inserted = true;
    return _neighborcache_pending_buffer_payload_bytes > _neighborcache_pending_buffer_cap_bytes;
}

template <typename T, typename LabelT>
uint32_t PQFlashIndex<T, LabelT>::neighborcache_dyn_best_buffered_pair(bool require_ready) const
{
    NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
    for (size_t retained = _neighborcache_pending_buffer_buckets.size(); retained > 1; --retained)
    {
        const auto &bucket = _neighborcache_pending_buffer_buckets[retained - 1];
        for (uint32_t insertion_slot : bucket)
        {
            const auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
            if (pair_it != _neighborcache_ready_pairs.end() && !pair_it->second.committing &&
                pair_it->second.retained_members == retained - 1)
            {
                const size_t page_capacity = static_cast<size_t>(
                    (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
                if (_neighborcache_pending_require_full_page &&
                    pair_it->second.snap.member_ids.size() < page_capacity)
                    continue;
                if (require_ready &&
                    (pair_it->second.retained_members < _neighborcache_pending_buffer_ready_min_members ||
                     static_cast<uint64_t>(pair_it->second.retained_members) * 100 <
                         static_cast<uint64_t>(pair_it->second.snap.member_ids.size()) *
                             _neighborcache_pending_buffer_ready_percent))
                    continue;
                return insertion_slot;
            }
        }
    }
    return NeighborCacheMetadataCache::kInvalid;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_release_unreferenced_buffer_members(
    const std::vector<uint32_t> &member_ids)
{
    if (_neighborcache_pending_buffer_cap_bytes == 0)
        return;

    NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
    std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
    for (uint32_t member_id : member_ids)
    {
        if (neighborcache_pending_member_contains(member_id))
            continue;
        auto stash_it = _neighborcache_payload_stash.find(member_id);
        if (stash_it == _neighborcache_payload_stash.end())
            continue;
        const uint64_t payload_bytes =
            stash_it.value().coords.size() * sizeof(T) +
            stash_it.value().nbrs.size() * sizeof(uint32_t);
        _neighborcache_stash_lru.erase(stash_it.value().lru_it);
        _neighborcache_payload_stash.erase(stash_it);
        _neighborcache_pending_buffer_payload_bytes =
            payload_bytes >= _neighborcache_pending_buffer_payload_bytes
                ? 0
                : _neighborcache_pending_buffer_payload_bytes - payload_bytes;
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_trim_pending_buffer()
{
    if (_neighborcache_pending_buffer_cap_bytes == 0)
        return;

    for (;;)
    {
        uint32_t victim = NeighborCacheMetadataCache::kInvalid;
        {
            std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
            if (_neighborcache_pending_buffer_payload_bytes <= _neighborcache_pending_buffer_cap_bytes ||
                _neighborcache_stash_lru.empty())
                return;
            victim = _neighborcache_stash_lru.back();
        }

        bool cleared_pair = false;
        {
            NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
            const std::vector<uint32_t> dependent_slots = neighborcache_pending_member_slots(victim);
            for (uint32_t insertion_slot : dependent_slots)
            {
                const auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
                if (pair_it != _neighborcache_ready_pairs.end() && (pair_it->second.committing || pair_it->second.queued))
                    return;
            }
            for (uint32_t insertion_slot : dependent_slots)
            {
                if (_neighborcache_ready_pairs.find(insertion_slot) == _neighborcache_ready_pairs.end())
                    continue;
                neighborcache_dyn_clear_ready_pair_unlocked(insertion_slot);
                cleared_pair = true;
            }

            std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
            auto stash_it = _neighborcache_payload_stash.find(victim);
            if (stash_it == _neighborcache_payload_stash.end())
                continue;
            const uint64_t payload_bytes =
                stash_it.value().coords.size() * sizeof(T) +
                stash_it.value().nbrs.size() * sizeof(uint32_t);
            _neighborcache_stash_lru.erase(stash_it.value().lru_it);
            _neighborcache_payload_stash.erase(stash_it);
            _neighborcache_pending_buffer_payload_bytes =
                payload_bytes >= _neighborcache_pending_buffer_payload_bytes
                    ? 0
                    : _neighborcache_pending_buffer_payload_bytes - payload_bytes;
            _neighborcache_pending_buffer_drops.fetch_add(1, std::memory_order_relaxed);
        }
        if (cleared_pair)
            neighborcache_dyn_request_refresh();
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_family_touch_unlocked(uint32_t node_id)
{
    const auto it = _neighborcache_dcache_lru_pos.find(node_id);
    if (it != _neighborcache_dcache_lru_pos.end())
    {
        _neighborcache_dcache_lru.splice(_neighborcache_dcache_lru.begin(), _neighborcache_dcache_lru, it->second);
        return;
    }
    _neighborcache_dcache_lru.push_front(node_id);
    _neighborcache_dcache_lru_pos[node_id] = _neighborcache_dcache_lru.begin();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_family_link_unlocked(uint32_t node_id, uint32_t seed_id)
{
    auto it = _neighborcache_family_of.find(node_id);
    if (it == _neighborcache_family_of.end())
    {
        _neighborcache_family_of.insert({node_id, std::vector<uint32_t>(1, seed_id)});
        return;
    }
    std::vector<uint32_t> &listed = it.value();
    for (uint32_t existing : listed)
    {
        if (existing == seed_id)
            return;
    }
    listed.push_back(seed_id);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_family_unlink_node_unlocked(uint32_t node_id, uint32_t seed_id)
{
    auto it = _neighborcache_family_of.find(node_id);
    if (it == _neighborcache_family_of.end())
        return;
    std::vector<uint32_t> &listed = it.value();
    size_t write = 0;
    for (size_t read = 0; read < listed.size(); ++read)
    {
        if (listed[read] != seed_id)
            listed[write++] = listed[read];
    }
    listed.resize(write);
    if (!listed.empty())
        return;
    _neighborcache_family_of.erase(it);
    const auto pos = _neighborcache_dcache_lru_pos.find(node_id);
    if (pos == _neighborcache_dcache_lru_pos.end())
        return;
    _neighborcache_dcache_lru.erase(pos->second);
    _neighborcache_dcache_lru_pos.erase(pos);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_family_erase_unlocked(uint32_t seed_id)
{
    const auto members = _neighborcache_family_members.find(seed_id);
    if (members == _neighborcache_family_members.end())
        return;
    const std::vector<uint32_t> ids = members->second;
    _neighborcache_family_members.erase(members);
    _neighborcache_family_page.erase(seed_id);
    for (uint32_t node_id : ids)
        neighborcache_family_unlink_node_unlocked(node_id, seed_id);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_family_copy_cached(uint32_t node_id, std::vector<T> &coords,
                                                       std::vector<uint32_t> &nbrs)
{
    if (neighborcache_dyn_copy_member_payload(node_id, coords, nbrs, false))
        return coords.size() * sizeof(T) >= _disk_bytes_per_point;
    if (!_neighborcache_seed_page_lru)
        return false;
    return neighborcache_seed_page_lookup(node_id, coords, nbrs) && coords.size() * sizeof(T) >= _disk_bytes_per_point;
}

template <typename T, typename LabelT> bool PQFlashIndex<T, LabelT>::neighborcache_family_cached(uint32_t node_id)
{
    if (node_id == std::numeric_limits<uint32_t>::max())
        return false;
    if (_neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->contains(node_id))
        return true;
    if (neighborcache_seed_page_contains(node_id))
        return true;
    std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
    return _neighborcache_payload_stash.find(node_id) != _neighborcache_payload_stash.end();
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_family_load_nbrs(uint32_t node_id, std::vector<uint32_t> &nbrs)
{
    nbrs.clear();
    std::vector<T> coords;
    if (neighborcache_dyn_copy_member_payload(node_id, coords, nbrs, false))
        return coords.size() * sizeof(T) >= _disk_bytes_per_point && nbrs.size() <= _max_degree;
    if (!_neighborcache_seed_page_lru || _neighborcache_seed_page_shards.size() != kNeighborCacheSeedPageShards ||
        node_id == std::numeric_limits<uint32_t>::max())
        return false;
    std::vector<uint32_t> seeds;
    {
        NeighborCacheSeedPageOwnerShard &owners = *_neighborcache_seed_page_owners[neighborcache_seed_page_shard_index(node_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(owners.mu);
        auto it = owners.by_node.find(node_id);
        if (it == owners.by_node.end() || it.value().empty())
            return false;
        seeds = it.value();
    }
    for (uint32_t seed_id : seeds)
    {
        NeighborCacheSeedPageShard &shard = *_neighborcache_seed_page_shards[neighborcache_seed_page_shard_index(seed_id)];
        std::lock_guard<NeighborCacheSpinLock> lock(shard.mu);
        auto it = shard.pages.find(seed_id);
        if (it == shard.pages.end())
            continue;
        const NeighborCacheSeedPage &page = it.value();
        for (uint8_t slot = 0; slot < page.count; ++slot)
        {
            if (page.ids[slot] != node_id || page.coords[slot].size() * sizeof(T) < _disk_bytes_per_point ||
                page.nbrs[slot].size() > _max_degree)
                continue;
            nbrs = page.nbrs[slot];
            return true;
        }
    }
    return false;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_family_enqueue(uint32_t seed_id)
{
    uint32_t page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
    std::vector<uint32_t> current;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        if (!neighborcache_dyn_seed_page_unlocked(seed_id, page_idx) || !neighborcache_dyn_page_live(page_idx))
            return;
        const size_t count = _neighborcache_dyn_page_member_count[page_idx];
        const size_t begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
        current.assign(_neighborcache_dyn_page_members.begin() + begin, _neighborcache_dyn_page_members.begin() + begin + count);
    }
    if (current.empty() || current.front() != seed_id)
        return;
    const uint32_t byte_cap = _max_node_len == 0 ? 0 : static_cast<uint32_t>(defaults::SECTOR_LEN / _max_node_len);
    const uint32_t slot_cap = std::min(_neighborcache_dyn_members_per_page, byte_cap);
    if (current.size() >= slot_cap)
        return;

    std::vector<uint32_t> seed_nbrs;
    if (!neighborcache_family_load_nbrs(seed_id, seed_nbrs))
        return;
    tsl::robin_set<uint32_t> have(current.begin(), current.end());
    std::vector<uint32_t> extra;
    extra.reserve(slot_cap);
    for (uint32_t nbr : seed_nbrs)
    {
        if (current.size() + extra.size() >= slot_cap)
            break;
        if (nbr == seed_id || nbr == NEIGHBORCACHE_DYN_INVALID_PAGE || nbr == std::numeric_limits<uint32_t>::max())
            continue;
        if (!have.insert(nbr).second)
            continue;
        if (!neighborcache_family_cached(nbr))
        {
            have.erase(nbr);
            continue;
        }
        extra.push_back(nbr);
    }
    if (extra.empty())
    {
        _neighborcache_family_idle.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    NeighborCachePendingFlush pf;
    pf.seed_id = seed_id;
    for (uint32_t nbr : extra)
    {
        std::vector<T> coords;
        std::vector<uint32_t> nbrs;
        if (!neighborcache_family_copy_cached(nbr, coords, nbrs) || nbrs.size() > _max_degree)
            continue;
        pf.member_ids.push_back(nbr);
        pf.member_coords.push_back(std::move(coords));
        pf.member_nbrs.push_back(std::move(nbrs));
    }
    if (pf.member_ids.empty())
        return;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_commit_q_mu);
        if (_neighborcache_commit_stop.load(std::memory_order_acquire))
        {
            _neighborcache_family_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        auto pending = _neighborcache_family_pending.find(seed_id);
        if (pending != _neighborcache_family_pending.end())
        {
            NeighborCachePendingFlush &dst = pending.value();
            tsl::robin_set<uint32_t> queued(dst.member_ids.begin(), dst.member_ids.end());
            for (size_t i = 0; i < pf.member_ids.size() && dst.member_ids.size() < slot_cap; ++i)
            {
                if (!queued.insert(pf.member_ids[i]).second)
                    continue;
                dst.member_ids.push_back(pf.member_ids[i]);
                dst.member_coords.push_back(std::move(pf.member_coords[i]));
                dst.member_nbrs.push_back(std::move(pf.member_nbrs[i]));
            }
            _neighborcache_family_coalesced.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (_neighborcache_family_q.size() >= _neighborcache_family_q_cap)
        {
            _neighborcache_family_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        _neighborcache_family_q.push_back(seed_id);
        _neighborcache_family_pending.insert({seed_id, std::move(pf)});
    }
    _neighborcache_commit_q_cv.notify_one();
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_family_commit(NeighborCachePendingFlush pf)
{
    std::lock_guard<NeighborCacheSpinLock> seed_guard(neighborcache_seed_write_lock(pf.seed_id));
    uint32_t page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
    std::vector<uint32_t> current;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        if (!neighborcache_dyn_seed_page_unlocked(pf.seed_id, page_idx) || !neighborcache_dyn_page_live(page_idx) ||
            _neighborcache_dyn_reader == nullptr)
            return;
        const size_t count = _neighborcache_dyn_page_member_count[page_idx];
        const size_t begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
        current.assign(_neighborcache_dyn_page_members.begin() + begin, _neighborcache_dyn_page_members.begin() + begin + count);
    }
    if (current.empty() || current.front() != pf.seed_id)
        return;
    const uint32_t byte_cap = _max_node_len == 0 ? 0 : static_cast<uint32_t>(defaults::SECTOR_LEN / _max_node_len);
    const uint32_t slot_cap = std::min(_neighborcache_dyn_members_per_page, byte_cap);
    tsl::robin_set<uint32_t> have(current.begin(), current.end());
    std::vector<size_t> keep;
    keep.reserve(pf.member_ids.size());
    for (size_t i = 0; i < pf.member_ids.size(); ++i)
    {
        if (current.size() + keep.size() >= slot_cap)
            break;
        const uint32_t node_id = pf.member_ids[i];
        if (node_id == pf.seed_id || !have.insert(node_id).second)
            continue;
        if (pf.member_coords[i].size() * sizeof(T) < _disk_bytes_per_point || pf.member_nbrs[i].size() > _max_degree)
            continue;
        keep.push_back(i);
    }
    if (keep.empty())
        return;

    if (!neighborcache_dyn_disk_readable(NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx))
        return;

    char *page = nullptr;
    alloc_aligned(reinterpret_cast<void **>(&page), defaults::SECTOR_LEN, defaults::SECTOR_LEN);
    if (page == nullptr)
        return;
    std::vector<AlignedRead> reads;
    reads.emplace_back(static_cast<uint64_t>(page_idx) * defaults::SECTOR_LEN, defaults::SECTOR_LEN, page);
    _neighborcache_dyn_reader->read(reads, _neighborcache_dyn_reader->get_ctx());
    for (size_t n = 0; n < keep.size(); ++n)
    {
        const size_t src = keep[n];
        char *slot_dst = page + (current.size() + n) * _max_node_len;
        const uint32_t nnbrs = static_cast<uint32_t>(pf.member_nbrs[src].size());
        memcpy(slot_dst, pf.member_coords[src].data(), _disk_bytes_per_point);
        uint32_t *nhood = reinterpret_cast<uint32_t *>(slot_dst + _disk_bytes_per_point);
        nhood[0] = nnbrs;
        if (nnbrs > 0)
            memcpy(nhood + 1, pf.member_nbrs[src].data(), nnbrs * sizeof(uint32_t));
    }
    const bool write_ok = neighborcache_dyn_writer_submit_and_wait(
        NeighborCacheDynWriteOp::Write, static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN), page,
        defaults::SECTOR_LEN);
    aligned_free(page);
    if (!write_ok)
        return;

    std::vector<uint32_t> appended;
    appended.reserve(keep.size());
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        uint32_t live_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
        if (!neighborcache_dyn_seed_page_unlocked(pf.seed_id, live_page) || live_page != page_idx ||
            !neighborcache_dyn_page_live(page_idx))
            return;
        const size_t count = _neighborcache_dyn_page_member_count[page_idx];
        const size_t begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
        if (count != current.size())
            return;
        for (size_t slot = 0; slot < current.size(); ++slot)
        {
            if (_neighborcache_dyn_page_members[begin + slot] != current[slot])
                return;
        }
        for (size_t n = 0; n < keep.size() && current.size() + n < _neighborcache_dyn_members_per_page; ++n)
        {
            const uint32_t node_id = pf.member_ids[keep[n]];
            _neighborcache_dyn_page_members[begin + current.size() + n] = node_id;
            appended.push_back(node_id);
        }
        _neighborcache_dyn_page_member_count[page_idx] = static_cast<uint16_t>(current.size() + appended.size());
        neighborcache_dyn_bump_page_gen(page_idx);
    }
    if (appended.empty())
        return;
    const uint32_t sector = static_cast<uint32_t>(NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx);
    neighborcache_shared_cache_erase_dyn(page_idx, 1);
    for (size_t n = 0; n < appended.size(); ++n)
    {
        NeighborCacheDiskLoc loc;
        loc.sector = sector;
        loc.slot = static_cast<uint16_t>(current.size() + n);
        loc.nsectors = 1;
        neighborcache_dyn_loc_add_unlocked(appended[n], loc);
    }
    if (_neighborcache_mem_pool != nullptr)
    {
        std::vector<uint32_t> demote;
        demote.reserve(current.size() + appended.size());
        {
            std::lock_guard<std::mutex> lock(_neighborcache_family_mu);
            auto still_live = [this](uint32_t node_id) {
                auto it = _neighborcache_family_of.find(node_id);
                return it != _neighborcache_family_of.end() && !it->second.empty();
            };
            for (uint32_t node_id : current)
            {
                if (!still_live(node_id))
                    demote.push_back(node_id);
            }
            for (uint32_t node_id : appended)
            {
                if (!still_live(node_id))
                    demote.push_back(node_id);
            }
        }
        for (uint32_t node_id : demote)
            _neighborcache_mem_pool->demote_lru_tail(node_id);
    }
    _neighborcache_family_expanded.fetch_add(1, std::memory_order_relaxed);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_family_on_evict(uint32_t node_id)
{
    if (!_neighborcache_dcache_family)
        return;
    static thread_local int depth = 0;
    if (depth > 0)
        return;
    struct DepthGuard
    {
        int &depth;
        explicit DepthGuard(int &d) : depth(d)
        {
            ++depth;
        }
        ~DepthGuard()
        {
            --depth;
        }
    } guard(depth);

    std::vector<uint32_t> seeds;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_family_mu);
        auto owner = _neighborcache_family_of.find(node_id);
        if (owner == _neighborcache_family_of.end() || owner->second.empty())
            return;
        seeds = owner->second;
        for (uint32_t seed_id : seeds)
        {
            neighborcache_family_erase_unlocked(seed_id);
            _neighborcache_family_dissolved.fetch_add(1, std::memory_order_relaxed);
        }
    }
    for (uint32_t seed_id : seeds)
        neighborcache_family_enqueue(seed_id);
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_family_note_read(uint32_t sector)
{
    if (!_neighborcache_dcache_family || !neighborcache_dyn_is_sector(sector))
        return;
    const uint32_t page_idx = sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
    std::vector<uint32_t> members;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        if (page_idx >= _neighborcache_dyn_page_member_count.size() || !neighborcache_dyn_page_live(page_idx))
            return;
        const size_t count = _neighborcache_dyn_page_member_count[page_idx];
        const size_t begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
        members.assign(_neighborcache_dyn_page_members.begin() + begin, _neighborcache_dyn_page_members.begin() + begin + count);
    }
    if (members.empty() || members.front() == NEIGHBORCACHE_DYN_INVALID_PAGE)
        return;
    const uint32_t seed_id = members.front();
    std::vector<uint32_t> overflow;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_family_mu);
        tsl::robin_set<uint32_t> keep;
        std::vector<uint32_t> fresh;
        fresh.reserve(members.size());
        for (uint32_t node_id : members)
        {
            if (node_id == NEIGHBORCACHE_DYN_INVALID_PAGE || node_id == std::numeric_limits<uint32_t>::max())
                continue;
            if (!keep.insert(node_id).second)
                continue;
            fresh.push_back(node_id);
        }
        if (fresh.empty() || fresh.front() != seed_id)
            return;
        const auto existing = _neighborcache_family_members.find(seed_id);
        if (existing == _neighborcache_family_members.end())
            _neighborcache_family_formed.fetch_add(1, std::memory_order_relaxed);
        else
        {
            for (uint32_t node_id : existing->second)
            {
                if (keep.find(node_id) == keep.end())
                    neighborcache_family_unlink_node_unlocked(node_id, seed_id);
            }
        }
        for (uint32_t node_id : fresh)
            neighborcache_family_link_unlocked(node_id, seed_id);
        _neighborcache_family_members[seed_id] = fresh;
        _neighborcache_family_page[seed_id] = page_idx;
        for (uint32_t node_id : fresh)
            neighborcache_family_touch_unlocked(node_id);
        while (_neighborcache_dcache_family_cap > 0 && _neighborcache_dcache_lru.size() > _neighborcache_dcache_family_cap)
        {
            const uint32_t cold = _neighborcache_dcache_lru.back();
            _neighborcache_dcache_lru.pop_back();
            _neighborcache_dcache_lru_pos.erase(cold);
            overflow.push_back(cold);
        }
    }
    for (uint32_t cold : overflow)
        neighborcache_family_on_evict(cold);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_on_ncache_evict(uint32_t node_id, QueryStats *stats)
{
    if (!_neighborcache_dyn_enabled || _neighborcache_coread_pack || node_id == NeighborCacheMemoryPool<T>::INVALID_NODE)
        return;
    if (_neighborcache_dcache_family)
        neighborcache_family_on_evict(node_id);

    std::vector<uint32_t> pending_slots;
    std::vector<uint32_t> refresh_slots;
    {
        pending_slots = neighborcache_pending_member_slots(node_id);
        if (!pending_slots.empty())
        {
            if (_neighborcache_pending_require_full_page)
            {
                NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
                const size_t page_capacity = static_cast<size_t>(
                    (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
                for (uint32_t insertion_slot : pending_slots)
                {
                    const auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
                    if (pair_it != _neighborcache_ready_pairs.end() &&
                        pair_it->second.snap.member_ids.size() < page_capacity)
                        refresh_slots.push_back(insertion_slot);
                }
            }
        }
    }
    for (uint32_t insertion_slot : refresh_slots)
        neighborcache_dyn_refresh_pending_members(insertion_slot);
    if (!refresh_slots.empty())
    {
        pending_slots = neighborcache_pending_member_slots(node_id);
    }
    if (!pending_slots.empty())
    {
        g_dyn_probe.write_trig += pending_slots.size();
        {
            NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
            _neighborcache_ncache_write_trig += pending_slots.size();
        }
        if (_neighborcache_pending_buffer_cap_bytes > 0)
        {
            bool inserted = false;
            const bool pressure = neighborcache_dyn_buffer_evicted_member(node_id, inserted);
            if (inserted)
            {
                NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, true);
                const std::vector<uint32_t> member_slots = neighborcache_pending_member_slots(node_id);
                if (!member_slots.empty())
                {
                    for (uint32_t insertion_slot : member_slots)
                    {
                        auto pair_it = _neighborcache_ready_pairs.find(insertion_slot);
                        if (pair_it != _neighborcache_ready_pairs.end() && !pair_it.value().committing &&
                            pair_it.value().retained_members < pair_it.value().snap.member_ids.size())
                        {
                            const uint16_t old_retained = pair_it.value().retained_members;
                            if (old_retained < _neighborcache_pending_buffer_buckets.size())
                                _neighborcache_pending_buffer_buckets[old_retained].erase(insertion_slot);
                            ++pair_it.value().retained_members;
                            if (pair_it.value().retained_members < _neighborcache_pending_buffer_buckets.size())
                                _neighborcache_pending_buffer_buckets[pair_it.value().retained_members].insert(
                                    insertion_slot);
                        }
                    }
                }
            }
            uint32_t best_slot = NeighborCacheMetadataCache::kInvalid;
            bool ready_trigger = false;
            if (_neighborcache_pending_buffer_ready_percent > 0)
            {
                best_slot = neighborcache_dyn_best_buffered_pair(true);
                ready_trigger = best_slot != NeighborCacheMetadataCache::kInvalid;
            }
            if (pressure && best_slot == NeighborCacheMetadataCache::kInvalid)
                best_slot = neighborcache_dyn_best_buffered_pair(false);
            if (best_slot != NeighborCacheMetadataCache::kInvalid &&
                neighborcache_dyn_enqueue_ready_pair(best_slot, NeighborCacheCommitSource::Buffered))
            {
                if (ready_trigger)
                    _neighborcache_pending_buffer_ready_writes.fetch_add(1, std::memory_order_relaxed);
                else
                    _neighborcache_pending_buffer_pressure_writes.fetch_add(1, std::memory_order_relaxed);
                if (stats != nullptr)
                    stats->n_neighborcache_dyn_flushes++;
            }
            if (pressure)
                neighborcache_dyn_trim_pending_buffer();
        }
        else
        {
            for (uint32_t insertion_slot : pending_slots)
            {
                if (neighborcache_dyn_enqueue_ready_pair(insertion_slot, NeighborCacheCommitSource::Evict) && stats != nullptr)
                    stats->n_neighborcache_dyn_flushes++;
            }
        }
        {
            const uint32_t slot_id = _neighborcache_mcache.slot_of(node_id);
            if (slot_id != NeighborCacheMetadataCache::kInvalid)
            {
                NeighborCacheTimedMutexGuard hlock(neighborcache_dyn_heap_shard(slot_id).mu, NeighborCacheLockKind::Heap);
                if (neighborcache_dyn_state_unlocked(slot_id) == NeighborCacheNodeState::NonSeed)
                    neighborcache_dyn_heap_erase_unlocked(slot_id);
            }
        }
        neighborcache_dyn_request_refresh();
        return;
    }

    {
        const uint32_t slot_id = _neighborcache_mcache.slot_of(node_id);
        if (slot_id != NeighborCacheMetadataCache::kInvalid)
        {
            NeighborCacheTimedMutexGuard hlock(neighborcache_dyn_heap_shard(slot_id).mu, NeighborCacheLockKind::Heap);
            if (neighborcache_dyn_state_unlocked(slot_id) == NeighborCacheNodeState::NonSeed)
                neighborcache_dyn_heap_erase_unlocked(slot_id);
        }
    }
    neighborcache_dyn_request_refresh();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_prepare_query(SSDQueryScratch<T> *query_scratch)
{
    if (query_scratch == nullptr)
        return;
    if (!_neighborcache_dcache_net_gate_enabled)
    {
        // Keep D lookups available even when maintenance is frozen/suspended.
        query_scratch->neighborcache_dcache_lookup_enabled = true;
        query_scratch->neighborcache_dcache_maintenance_enabled =
            !_neighborcache_freeze_maintenance && !_neighborcache_mcache.maintenance_suspended();
    }
    else
    {
        const uint64_t ticket = _neighborcache_dcache_query_issued.fetch_add(1, std::memory_order_relaxed);
        const bool active = _neighborcache_dcache_gate_active.load(std::memory_order_acquire);
        const bool probe = !active && ticket % _neighborcache_dcache_probe_period == 0;
        query_scratch->neighborcache_dcache_probe = probe;
        query_scratch->neighborcache_dcache_lookup_enabled = active || probe;
        query_scratch->neighborcache_dcache_maintenance_enabled =
            !_neighborcache_freeze_maintenance &&
            (probe || (active && !_neighborcache_mcache.maintenance_suspended()));
    }

    query_scratch->neighborcache_record_real_io = false;
    if (_neighborcache_real_io_coaccess && query_scratch->neighborcache_dcache_maintenance_enabled &&
        _neighborcache_real_io_sample_queries > 0)
    {
        const uint64_t query_number = _neighborcache_real_io_queries.fetch_add(1, std::memory_order_relaxed);
        query_scratch->neighborcache_record_real_io = (query_number % _neighborcache_real_io_sample_queries) == 0;
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_update_dcache_gate(QueryStats *stats,
                                                           SSDQueryScratch<T> *query_scratch)
{
    if (query_scratch == nullptr || !_neighborcache_dcache_net_gate_enabled)
        return;

    uint32_t avoided_pages = 0;
    for (uint64_t sector : query_scratch->neighborcache_avoided_base_sectors)
    {
        if (query_scratch->read_sectors.find(sector) == query_scratch->read_sectors.end())
            ++avoided_pages;
    }
    if (stats != nullptr)
    {
        stats->n_neighborcache_base_pages_avoided = avoided_pages;
        stats->n_neighborcache_dcache_probe = query_scratch->neighborcache_dcache_probe ? 1u : 0u;
    }
    if (!_neighborcache_dcache_net_gate_enabled)
        return;

    if (query_scratch->neighborcache_dcache_lookup_enabled)
    {
        _neighborcache_dcache_enabled_queries.fetch_add(1, std::memory_order_relaxed);
        _neighborcache_dcache_avoided_pages.fetch_add(avoided_pages, std::memory_order_relaxed);
        _neighborcache_dcache_physical_reads.fetch_add(query_scratch->neighborcache_dyn_physical_reads,
                                                std::memory_order_relaxed);
        _neighborcache_dcache_served_nodes.fetch_add(query_scratch->neighborcache_dyn_served_nodes,
                                              std::memory_order_relaxed);
    }

    constexpr uint64_t stage_queries = 1000;
    const uint64_t completed =
        _neighborcache_dcache_query_completed.fetch_add(1, std::memory_order_relaxed) + 1;
    if (completed % stage_queries != 0)
        return;

    std::lock_guard<std::mutex> gate_lock(_neighborcache_dcache_gate_mu);
    bool active = _neighborcache_dcache_gate_active.load(std::memory_order_relaxed);
    if (!active && ++_neighborcache_dcache_inactive_stages < 10)
        return;
    _neighborcache_dcache_inactive_stages = 0;

    const uint64_t enabled_total = _neighborcache_dcache_enabled_queries.load(std::memory_order_relaxed);
    const uint64_t avoided_total = _neighborcache_dcache_avoided_pages.load(std::memory_order_relaxed);
    const uint64_t reads_total = _neighborcache_dcache_physical_reads.load(std::memory_order_relaxed);
    const uint64_t served_total = _neighborcache_dcache_served_nodes.load(std::memory_order_relaxed);
    const uint64_t enabled_delta = enabled_total - _neighborcache_dcache_last_enabled_queries;
    const uint64_t avoided_delta = avoided_total - _neighborcache_dcache_last_avoided_pages;
    const uint64_t reads_delta = reads_total - _neighborcache_dcache_last_physical_reads;
    const uint64_t served_delta = served_total - _neighborcache_dcache_last_served_nodes;
    _neighborcache_dcache_last_enabled_queries = enabled_total;
    _neighborcache_dcache_last_avoided_pages = avoided_total;
    _neighborcache_dcache_last_physical_reads = reads_total;
    _neighborcache_dcache_last_served_nodes = served_total;
    const uint64_t writes_total = _neighborcache_dcache_page_writes.load(std::memory_order_relaxed);
    const uint64_t writes_delta = writes_total - _neighborcache_dcache_last_page_writes;
    _neighborcache_dcache_last_page_writes = writes_total;
    if (enabled_delta == 0)
        return;

    const double net_per_query =
        (static_cast<double>(avoided_delta) - static_cast<double>(reads_delta) -
         _neighborcache_dcache_net_write_weight * static_cast<double>(writes_delta)) /
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
        _neighborcache_dcache_net_write_weight > 0.0 ? _neighborcache_dcache_net_write_weight : 1.0;
    const double net_us_per_query =
        avoided_per_query * _neighborcache_dcache_base_read_us -
        reads_per_query * _neighborcache_dcache_read_us -
        writes_per_query * _neighborcache_dcache_write_us * write_multiplier -
        _neighborcache_dcache_cpu_us;

    bool transition = false;
    if (active)
    {
        _neighborcache_dcache_low_windows =
            net_us_per_query < _neighborcache_dcache_gate_off_us ? _neighborcache_dcache_low_windows + 1 : 0;
        if (_neighborcache_dcache_low_windows >= 5)
        {
            active = false;
            transition = true;
        }
    }
    else
    {
        const bool profitable = net_us_per_query >= _neighborcache_dcache_gate_on_us;
        _neighborcache_dcache_high_windows = profitable ? _neighborcache_dcache_high_windows + 1 : 0;
        if (_neighborcache_dcache_high_windows >= 2)
        {
            active = true;
            transition = true;
        }
    }
    if (!transition)
        return;

    _neighborcache_dcache_gate_active.store(active, std::memory_order_release);
    _neighborcache_dcache_gate_transitions.fetch_add(1, std::memory_order_relaxed);
    _neighborcache_dcache_low_windows = 0;
    _neighborcache_dcache_high_windows = 0;
    _neighborcache_dcache_inactive_stages = 0;
    std::cout << "NEIGHBORCACHE dcache net gate: queries=" << completed
              << " active=" << (active ? "off->on" : "on->off")
              << " sampled_queries=" << enabled_delta
              << " avoided_pages/query=" << avoided_per_query
              << " dcache_reads/query=" << reads_per_query
              << " page_writes/query=" << writes_per_query
              << " net_pages/query=" << net_per_query
              << " net_us/query=" << net_us_per_query
              << " model_us=" << _neighborcache_dcache_base_read_us << "/"
              << _neighborcache_dcache_read_us << "/" << _neighborcache_dcache_write_us << "/"
              << _neighborcache_dcache_cpu_us
              << " served_nodes/query=" << served_per_query << std::endl;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_log_monitor(uint32_t query_epoch) const
{
    if (_neighborcache_monitor_warmup_queries == 0 || query_epoch < _neighborcache_monitor_warmup_queries ||
        (query_epoch - _neighborcache_monitor_warmup_queries) % _neighborcache_monitor_interval_queries != 0)
        return;

    uint64_t pages = 0;
    uint64_t members = 0;
    uint64_t full_pages = 0;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        pages = _neighborcache_dyn_committed_pages;
        const uint16_t page_capacity = static_cast<uint16_t>(
            (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
        for (uint16_t count : _neighborcache_dyn_page_member_count)
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
        NeighborCacheTimedSharedMutexGuard plock(_neighborcache_pair_mu, NeighborCacheLockKind::Pair, false);
        pending = _neighborcache_ready_pairs.size();
        const size_t page_capacity = static_cast<size_t>(
            (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
        for (const auto &entry : _neighborcache_ready_pairs)
            if (entry.second.snap.member_ids.size() >= page_capacity)
                ++pending_full;
        writes = _neighborcache_heap_writes;
    }
    uint64_t buffer_bytes = 0;
    uint64_t buffer_entries = 0;
    {
        std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
        buffer_bytes = _neighborcache_pending_buffer_payload_bytes;
        buffer_entries = _neighborcache_payload_stash.size();
    }
    diskann::cout << "NEIGHBORCACHE page monitor: query=" << query_epoch
                  << " measured=" << query_epoch - _neighborcache_monitor_warmup_queries
                  << " pages=" << pages
                  << " members=" << members
                  << " avg_members=" << (pages == 0 ? 0.0 : static_cast<double>(members) / pages)
                  << " full_pages=" << full_pages
                  << " pending=" << pending
                  << " pending_full=" << pending_full
                  << " buffer_entries=" << buffer_entries
                  << " buffer_bytes=" << buffer_bytes
                  << " member_refreshes="
                  << _neighborcache_pending_member_refreshes.load(std::memory_order_relaxed)
                  << " became_full=" << _neighborcache_pending_became_full.load(std::memory_order_relaxed)
                  << " writes=" << writes << std::endl;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_record_real_io_coaccess(SSDQueryScratch<T> *query_scratch)
{
    if (!_neighborcache_real_io_coaccess || query_scratch == nullptr ||
        !query_scratch->neighborcache_dcache_maintenance_enabled || !query_scratch->neighborcache_record_real_io)
        return;

    const std::vector<uint32_t> &nodes = query_scratch->neighborcache_storage_node_order;
    if (nodes.size() < 2)
        return;

    _neighborcache_real_io_sampled_queries.fetch_add(1, std::memory_order_relaxed);
    _neighborcache_real_io_nodes.fetch_add(nodes.size(), std::memory_order_relaxed);
    uint64_t recorded_pairs = 0, window_pairs = 0, adjacent_pairs = 0;
    bool touched = false;
    const float unit = _neighborcache_score_unit.load(std::memory_order_relaxed);
    const std::vector<uint32_t> &nbrs = query_scratch->neighborcache_storage_nbrs;
    const std::vector<uint32_t> &off = query_scratch->neighborcache_storage_nbr_off;
    const bool have_nbrs = off.size() == nodes.size() + 1;
    auto lists = [&](size_t a, uint32_t id) {
        return std::find(nbrs.begin() + off[a], nbrs.begin() + off[a + 1], id) != nbrs.begin() + off[a + 1];
    };
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        const size_t end = std::min(nodes.size(), i + static_cast<size_t>(_neighborcache_real_io_window) + 1);
        for (size_t j = i + 1; j < end; ++j)
        {
            if (nodes[i] == nodes[j])
                continue;
            window_pairs++;
            const bool adjacent = have_nbrs && (lists(i, nodes[j]) || lists(j, nodes[i]));
            adjacent_pairs += adjacent ? 1 : 0;
            if (_neighborcache_real_io_direct && !adjacent)
                continue;
            const auto forward =
                _neighborcache_mcache.on_real_io_edge(nodes[i], nodes[j], _neighborcache_real_io_max_partners, unit);
            if (forward.present)
            {
                neighborcache_dyn_note_touch(forward, query_scratch);
                touched = true;
            }
            const auto reverse =
                _neighborcache_mcache.on_real_io_edge(nodes[j], nodes[i], _neighborcache_real_io_max_partners, unit);
            if (reverse.present)
            {
                neighborcache_dyn_note_touch(reverse, query_scratch);
                touched = true;
            }
            recorded_pairs += 2;
        }
    }
    _neighborcache_real_io_pairs.fetch_add(recorded_pairs, std::memory_order_relaxed);
    _neighborcache_real_io_window_pairs.fetch_add(window_pairs, std::memory_order_relaxed);
    _neighborcache_real_io_adjacent_pairs.fetch_add(adjacent_pairs, std::memory_order_relaxed);
    if (touched)
        neighborcache_mark_refresh_needed();
}

template <typename T, typename LabelT>
std::vector<uint32_t> PQFlashIndex<T, LabelT>::neighborcache_dyn_assemble_patch_members(
    uint32_t seed_id, const std::vector<uint32_t> &new_ids, const std::vector<uint32_t> &prior_members,
    const std::vector<std::vector<uint32_t>> &prior_extra, std::vector<std::vector<uint32_t>> *extra_out) const
{
    if (extra_out != nullptr)
        extra_out->clear();
    const size_t page_capacity = static_cast<size_t>((_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
    if (!_neighborcache_dcache_second_page || _neighborcache_dcache_max_pages <= 1)
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
            NeighborCacheMetadataCache::Snapshot snap;
            if (_neighborcache_mcache.snapshot(seed_id, snap))
            {
                const std::vector<uint32_t> filled =
                    neighborcache_dyn_now_page_ids(seed_id, snap, _neighborcache_deferred_ncache_page_write);
                for (uint32_t id : filled)
                    try_add(id);
            }
        }
        return now;
    }

    const uint32_t max_pages = _neighborcache_dcache_max_pages;
    std::vector<std::vector<uint32_t>> pages(1);
    tsl::robin_set<uint32_t> seen;
    auto add_to = [&](std::vector<uint32_t> &dst, uint32_t id) {
        if (dst.size() >= page_capacity || id == NEIGHBORCACHE_DYN_INVALID_PAGE)
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
        NeighborCacheMetadataCache::Snapshot snap;
        if (_neighborcache_mcache.snapshot(seed_id, snap))
        {
            const std::vector<uint32_t> filled =
                neighborcache_dyn_now_page_ids(seed_id, snap, _neighborcache_deferred_ncache_page_write);
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
void PQFlashIndex<T, LabelT>::neighborcache_dyn_patch_pin(const std::vector<uint32_t> &member_ids)
{
    std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
    for (uint32_t id : member_ids)
        _neighborcache_patch_pin_count[id] += 1;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_patch_unpin(const std::vector<uint32_t> &member_ids)
{
    std::lock_guard<std::mutex> slock(_neighborcache_stash_mu);
    for (uint32_t id : member_ids)
    {
        auto it = _neighborcache_patch_pin_count.find(id);
        if (it == _neighborcache_patch_pin_count.end())
            continue;
        if (it.value() <= 1)
            _neighborcache_patch_pin_count.erase(it);
        else
            it.value() -= 1;
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_patch_loop()
{
    neighborcache_pin_background_thread();
    if (_neighborcache_dyn_reader != nullptr)
        _neighborcache_dyn_reader->register_thread();
    while (true)
    {
        std::shared_ptr<NeighborCachePatchJob> job;
        {
            std::unique_lock<std::mutex> lock(_neighborcache_patch_q_mu);
            _neighborcache_patch_q_cv.wait(lock, [this] {
                return _neighborcache_patch_stop.load(std::memory_order_acquire) || !_neighborcache_patch_q.empty();
            });
            if (_neighborcache_patch_stop.load(std::memory_order_acquire) && _neighborcache_patch_q.empty())
                return;
            if (_neighborcache_patch_batch_us > 0 && !_neighborcache_patch_stop.load(std::memory_order_acquire))
            {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(_neighborcache_patch_batch_us));
                lock.lock();
                if (_neighborcache_patch_stop.load(std::memory_order_acquire) && _neighborcache_patch_q.empty())
                    return;
                if (_neighborcache_patch_q.empty())
                    continue;
            }
            job = std::move(_neighborcache_patch_q.front());
            _neighborcache_patch_q.pop_front();
        }
        _neighborcache_patch_q_cv.notify_all();
        if (job == nullptr)
            continue;
        if (_neighborcache_freeze_maintenance || neighborcache_dcache_updates_frozen())
        {
            neighborcache_dyn_patch_unpin(job->new_ids);
            neighborcache_dyn_patch_unpin(job->prior_members);
            for (const auto &page : job->prior_extra)
                neighborcache_dyn_patch_unpin(page);
            continue;
        }
        if (job->ready)
        {
            const bool same = neighborcache_dyn_seed_members_equal(job->seed_id, job->prior_members);
            const bool committed = same && neighborcache_dyn_commit_one(job->ready_flush, job->seed_id);
            if (committed)
            {
                _neighborcache_dcache_query_patch_ok.fetch_add(1, std::memory_order_relaxed);
                g_dyn_probe.write_ok.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }

        std::vector<std::vector<uint32_t>> extra_pages;
        std::vector<uint32_t> now = neighborcache_dyn_assemble_patch_members(
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
                neighborcache_dyn_patch_pin(extra_pins);
        }
        std::vector<NeighborCachePendingFlush> flushes(1 + extra_pages.size());
        flushes[0].seed_id = job->seed_id;
        flushes[0].member_ids = std::move(now);
        for (size_t i = 0; i < extra_pages.size(); ++i)
        {
            flushes[i + 1].seed_id = job->seed_id;
            flushes[i + 1].member_ids = std::move(extra_pages[i]);
        }
        const bool allow_disk = !_neighborcache_patch_worker_no_disk;
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
        auto copy_payloads = [&](NeighborCachePendingFlush &dst) {
            dst.member_coords.reserve(dst.member_ids.size());
            dst.member_nbrs.reserve(dst.member_ids.size());
            for (uint32_t mid : dst.member_ids)
            {
                std::vector<T> coords;
                std::vector<uint32_t> nbrs;
                if (!neighborcache_dyn_copy_member_payload(mid, coords, nbrs, allow_disk))
                    return false;
                dst.member_coords.push_back(std::move(coords));
                dst.member_nbrs.push_back(std::move(nbrs));
            }
            return true;
        };
        if (ok)
        {
            for (NeighborCachePendingFlush &dst : flushes)
            {
                if (!copy_payloads(dst))
                {
                    ok = false;
                    break;
                }
            }
        }
        const bool committed = ok && (use_span ? neighborcache_dyn_commit_span(flushes, job->replacement_seed)
                                               : neighborcache_dyn_commit_one(flushes[0], job->replacement_seed));
        if (committed)
        {
            _neighborcache_dcache_query_patch_ok.fetch_add(1, std::memory_order_relaxed);
            g_dyn_probe.write_ok.fetch_add(1, std::memory_order_relaxed);
        }
        neighborcache_dyn_patch_unpin(job->new_ids);
        neighborcache_dyn_patch_unpin(job->prior_members);
        for (const auto &page : job->prior_extra)
            neighborcache_dyn_patch_unpin(page);
        if (!extra_pins.empty())
            neighborcache_dyn_patch_unpin(extra_pins);
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::start_neighborcache_dyn_patch_worker()
{
    if (_neighborcache_patch_started || !_neighborcache_dcache_query_patch)
        return;
    _neighborcache_patch_stop.store(false, std::memory_order_release);
    _neighborcache_patch_thread = std::thread([this] { neighborcache_dyn_patch_loop(); });
    _neighborcache_patch_started = true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::stop_neighborcache_dyn_patch_worker()
{
    if (!_neighborcache_patch_started)
        return;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_patch_q_mu);
        _neighborcache_patch_stop.store(true, std::memory_order_release);
    }
    _neighborcache_patch_q_cv.notify_all();
    if (_neighborcache_patch_thread.joinable())
        _neighborcache_patch_thread.join();
    _neighborcache_patch_started = false;
    std::deque<std::shared_ptr<NeighborCachePatchJob>> leftover;
    {
        std::lock_guard<std::mutex> lock(_neighborcache_patch_q_mu);
        leftover.swap(_neighborcache_patch_q);
    }
    for (auto &job : leftover)
    {
        if (job != nullptr)
        {
            neighborcache_dyn_patch_unpin(job->new_ids);
            neighborcache_dyn_patch_unpin(job->prior_members);
            for (const auto &page : job->prior_extra)
                neighborcache_dyn_patch_unpin(page);
        }
    }
    if (!leftover.empty())
        _neighborcache_dcache_query_patch_drop.fetch_add(leftover.size(), std::memory_order_relaxed);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_enqueue_patch(uint32_t seed_id, std::vector<uint32_t> new_ids,
                                                      std::vector<uint32_t> prior_members,
                                                      std::vector<std::vector<uint32_t>> prior_extra,
                                                      uint32_t replacement_seed)
{
    if (!_neighborcache_patch_started || _neighborcache_patch_stop.load(std::memory_order_acquire) ||
        _neighborcache_freeze_maintenance || new_ids.empty())
        return false;
    auto job = std::make_shared<NeighborCachePatchJob>();
    job->seed_id = seed_id;
    job->new_ids = std::move(new_ids);
    job->prior_members = std::move(prior_members);
    job->prior_extra = std::move(prior_extra);
    job->replacement_seed = replacement_seed;
    neighborcache_dyn_patch_pin(job->new_ids);
    neighborcache_dyn_patch_pin(job->prior_members);
    for (const auto &page : job->prior_extra)
        neighborcache_dyn_patch_pin(page);
    {
        std::unique_lock<std::mutex> lock(_neighborcache_patch_q_mu);
        for (auto it = _neighborcache_patch_q.begin(); it != _neighborcache_patch_q.end(); ++it)
        {
            if (*it != nullptr && (*it)->seed_id == job->seed_id)
            {
                auto old = *it;
                *it = job;
                lock.unlock();
                if (old != nullptr)
                {
                    neighborcache_dyn_patch_unpin(old->new_ids);
                    neighborcache_dyn_patch_unpin(old->prior_members);
                    for (const auto &page : old->prior_extra)
                        neighborcache_dyn_patch_unpin(page);
                }
                _neighborcache_patch_q_cv.notify_all();
                return true;
            }
        }
        if (_neighborcache_patch_q.size() >= _neighborcache_patch_q_cap)
        {
            lock.unlock();
            neighborcache_dyn_patch_unpin(job->new_ids);
            neighborcache_dyn_patch_unpin(job->prior_members);
            for (const auto &page : job->prior_extra)
                neighborcache_dyn_patch_unpin(page);
            _neighborcache_dcache_query_patch_drop.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        _neighborcache_patch_q.push_back(job);
    }
    _neighborcache_patch_q_cv.notify_all();
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_seed_members_equal(uint32_t seed_id,
                                                             const std::vector<uint32_t> &expected) const
{
    NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
    uint32_t page = NEIGHBORCACHE_DYN_INVALID_PAGE;
    if (!neighborcache_dyn_seed_page_unlocked(seed_id, page) || page >= _neighborcache_dyn_page_member_count.size())
        return false;
    const size_t count = _neighborcache_dyn_page_member_count[page];
    if (count == 0 || count != expected.size())
        return false;
    const size_t begin = static_cast<size_t>(page) * _neighborcache_dyn_members_per_page;
    if (begin + count > _neighborcache_dyn_page_members.size())
        return false;
    for (size_t i = 0; i < count; ++i)
        if (_neighborcache_dyn_page_members[begin + i] != expected[i])
            return false;
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_enqueue_ready_repack(NeighborCachePendingFlush flush,
                                                             std::vector<uint32_t> prior_members)
{
    if (!_neighborcache_patch_started || _neighborcache_patch_stop.load(std::memory_order_acquire) || _neighborcache_freeze_maintenance ||
        flush.member_ids.size() < 2 || flush.member_ids.front() != flush.seed_id)
        return false;
    auto job = std::make_shared<NeighborCachePatchJob>();
    job->ready = true;
    job->seed_id = flush.seed_id;
    job->replacement_seed = flush.seed_id;
    job->prior_members = std::move(prior_members);
    job->ready_flush = std::move(flush);
    {
        std::unique_lock<std::mutex> lock(_neighborcache_patch_q_mu);
        for (auto it = _neighborcache_patch_q.begin(); it != _neighborcache_patch_q.end(); ++it)
        {
            if (*it != nullptr && (*it)->seed_id == job->seed_id)
            {
                auto old = *it;
                *it = job;
                lock.unlock();
                if (old != nullptr && !old->ready)
                {
                    neighborcache_dyn_patch_unpin(old->new_ids);
                    neighborcache_dyn_patch_unpin(old->prior_members);
                    for (const auto &page : old->prior_extra)
                        neighborcache_dyn_patch_unpin(page);
                }
                _neighborcache_patch_q_cv.notify_all();
                return true;
            }
        }
        if (_neighborcache_patch_q.size() >= _neighborcache_patch_q_cap)
        {
            _neighborcache_dcache_query_patch_drop.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        _neighborcache_patch_q.push_back(std::move(job));
    }
    _neighborcache_patch_q_cv.notify_all();
    return true;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_repack_read_page(SSDQueryScratch<T> *query_scratch, uint32_t sector,
                                                         const char *sector_base, QueryStats *stats)
{
    if (!_neighborcache_repack_on_read || query_scratch == nullptr || sector_base == nullptr ||
        _neighborcache_mem_pool == nullptr || !_neighborcache_mem_pool->active() || _nnodes_per_sector == 0)
        return;
    if (query_scratch->neighborcache_repack_used >= _neighborcache_dcache_query_patch_max)
        return;
    if (!query_scratch->neighborcache_repack_sectors.insert(sector).second)
        return;

    const std::vector<uint32_t> *members = nullptr;
    for (const auto &kv : query_scratch->neighborcache_dcache_seed_members)
    {
        if (kv.second.empty())
            continue;
        const auto it = query_scratch->neighborcache_dcache_member_packed_loc.find(kv.second.front());
        if (it == query_scratch->neighborcache_dcache_member_packed_loc.end())
            continue;
        if (static_cast<uint32_t>(it->second >> 32) != sector)
            continue;
        members = &kv.second;
        break;
    }
    if (members == nullptr || members->size() < 2 || members->size() >= _nnodes_per_sector)
        return;
    for (size_t i = 0; i < members->size(); ++i)
    {
        const auto it = query_scratch->neighborcache_dcache_member_packed_loc.find((*members)[i]);
        if (it == query_scratch->neighborcache_dcache_member_packed_loc.end())
            return;
        if (static_cast<uint32_t>(it->second >> 32) != sector || (it->second & 0xFFFFu) != i)
            return;
    }
    if (!neighborcache_dyn_seed_members_equal(members->front(), *members))
        return;

    const size_t ncoords = _disk_bytes_per_point / sizeof(T);
    NeighborCachePendingFlush pf;
    pf.seed_id = members->front();
    pf.member_ids = *members;
    pf.member_coords.reserve(members->size());
    pf.member_nbrs.reserve(members->size());
    for (size_t i = 0; i < members->size(); ++i)
    {
        const char *slot = sector_base + static_cast<uint64_t>(i) * _max_node_len;
        const uint32_t *nhood = offset_to_node_nhood(const_cast<char *>(slot));
        const uint32_t nnbrs = nhood[0];
        if (nnbrs > _max_degree)
            return;
        const T *coord_src = offset_to_node_coords(const_cast<char *>(slot));
        pf.member_coords.emplace_back(coord_src, coord_src + ncoords);
        pf.member_nbrs.emplace_back(nhood + 1, nhood + 1 + nnbrs);
    }

    tsl::robin_set<uint32_t> on_page;
    on_page.reserve(members->size() * 2);
    for (uint32_t id : *members)
        on_page.insert(id);
    tsl::robin_map<uint32_t, float> best;
    for (const auto &nbrs : pf.member_nbrs)
    {
        for (uint32_t id : nbrs)
        {
            if (on_page.find(id) != on_page.end() || static_cast<uint64_t>(id) >= _num_points)
                continue;
            if (!_neighborcache_mem_pool->contains(id))
                continue;
            const float score = _neighborcache_mcache.active() ? _neighborcache_mcache.score(id) : 0.0f;
            const auto it = best.find(id);
            if (it == best.end() || score > it->second)
                best[id] = score;
        }
    }
    if (best.empty())
        return;
    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(best.size());
    for (const auto &kv : best)
        ranked.emplace_back(kv.second, kv.first);
    const size_t room = static_cast<size_t>(_nnodes_per_sector) - pf.member_ids.size();
    const size_t take = std::min(room, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(take), ranked.end(),
                      [](const std::pair<float, uint32_t> &a, const std::pair<float, uint32_t> &b) {
                          if (a.first != b.first)
                              return a.first > b.first;
                          return a.second < b.second;
                      });
    for (size_t i = 0; i < take; ++i)
    {
        std::vector<T> coords;
        std::vector<uint32_t> nbrs;
        if (!_neighborcache_mem_pool->copy_payload(ranked[i].second, coords, nbrs))
            continue;
        pf.member_ids.push_back(ranked[i].second);
        pf.member_coords.push_back(std::move(coords));
        pf.member_nbrs.push_back(std::move(nbrs));
    }
    if (pf.member_ids.size() <= members->size())
        return;
    std::vector<uint32_t> prior = *members;
    if (!neighborcache_dyn_enqueue_ready_repack(std::move(pf), std::move(prior)))
        return;
    query_scratch->neighborcache_repack_used++;
    _neighborcache_dcache_query_patch_trig.fetch_add(1, std::memory_order_relaxed);
    if (stats != nullptr)
        stats->n_neighborcache_dyn_flushes++;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_note_patch_candidate(uint32_t node_id, uint32_t parent,
                                                             SSDQueryScratch<T> *query_scratch)
{
    if (_neighborcache_repack_on_read)
        return false;
    if (!_neighborcache_dcache_query_patch || query_scratch == nullptr || parent == std::numeric_limits<uint32_t>::max())
        return false;
    const auto seed_it = query_scratch->neighborcache_dcache_member_to_seed.find(parent);
    if (seed_it == query_scratch->neighborcache_dcache_member_to_seed.end())
        return false;
    const uint32_t seed_id = seed_it->second;
    if (query_scratch->neighborcache_dcache_member_to_seed.find(node_id) != query_scratch->neighborcache_dcache_member_to_seed.end())
        return false;
    const auto mem_it = query_scratch->neighborcache_dcache_seed_members.find(seed_id);
    if (mem_it != query_scratch->neighborcache_dcache_seed_members.end())
    {
        for (uint32_t mid : mem_it->second)
            if (mid == node_id)
                return false;
    }
    const auto extra_it = query_scratch->neighborcache_dcache_seed_extra.find(seed_id);
    if (extra_it != query_scratch->neighborcache_dcache_seed_extra.end())
    {
        for (const auto &page : extra_it->second)
            for (uint32_t mid : page)
                if (mid == node_id)
                    return false;
    }
    auto &bucket = query_scratch->neighborcache_dcache_patch_new[seed_id];
    if (std::find(bucket.begin(), bucket.end(), node_id) != bucket.end())
        return false;
    bucket.push_back(node_id);
    return true;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_apply_query_patches(QueryStats *stats, SSDQueryScratch<T> *query_scratch)
{
    if (!_neighborcache_dcache_query_patch || query_scratch == nullptr)
        return;
    uint32_t applied = 0;
    tsl::robin_set<uint32_t> enqueued_ids;
    for (const auto &kv : query_scratch->neighborcache_dcache_patch_new)
    {
        if (applied >= _neighborcache_dcache_query_patch_max)
            break;
        const uint32_t seed_id = kv.first;
        const auto &new_ids = kv.second;
        if (new_ids.empty())
            continue;
        const auto mem_it = query_scratch->neighborcache_dcache_seed_members.find(seed_id);
        if (mem_it == query_scratch->neighborcache_dcache_seed_members.end() || mem_it->second.empty())
            continue;
        std::vector<std::vector<uint32_t>> prior_extra;
        const auto extra_it = query_scratch->neighborcache_dcache_seed_extra.find(seed_id);
        if (extra_it != query_scratch->neighborcache_dcache_seed_extra.end())
            prior_extra = extra_it->second;
        _neighborcache_dcache_query_patch_trig.fetch_add(1, std::memory_order_relaxed);
        if (_neighborcache_patch_started)
        {
            if (neighborcache_dyn_enqueue_patch(seed_id, new_ids, mem_it->second, prior_extra, seed_id))
            {
                enqueued_ids.insert(new_ids.begin(), new_ids.end());
                ++applied;
                if (stats != nullptr)
                    stats->n_neighborcache_dyn_flushes++;
            }
            continue;
        }
        std::vector<std::vector<uint32_t>> extra_pages;
        std::vector<uint32_t> now =
            neighborcache_dyn_assemble_patch_members(seed_id, new_ids, mem_it->second, prior_extra, &extra_pages);
        if (now.size() < 2 || now.front() != seed_id)
            continue;
        if (extra_pages.empty() && now == mem_it->second)
            continue;
        std::vector<NeighborCachePendingFlush> flushes(1 + extra_pages.size());
        flushes[0].seed_id = seed_id;
        flushes[0].member_ids = now;
        bool ok = true;
        auto copy_one = [&](NeighborCachePendingFlush &dst) {
            for (uint32_t mid : dst.member_ids)
            {
                std::vector<T> coords;
                std::vector<uint32_t> nbrs;
                if (!neighborcache_dyn_copy_member_payload(mid, coords, nbrs))
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
        for (NeighborCachePendingFlush &dst : flushes)
        {
            if (!copy_one(dst))
            {
                ok = false;
                break;
            }
        }
        const bool committed = ok && (flushes.size() > 1 ? neighborcache_dyn_commit_span(flushes, seed_id)
                                                          : neighborcache_dyn_commit_one(flushes[0], seed_id));
        if (!committed)
            continue;
        _neighborcache_dcache_query_patch_ok.fetch_add(1, std::memory_order_relaxed);
        g_dyn_probe.write_ok.fetch_add(1, std::memory_order_relaxed);
        ++applied;
        if (stats != nullptr)
            stats->n_neighborcache_dyn_flushes++;
    }
    std::vector<uint32_t> release;
    for (const auto &kv : query_scratch->neighborcache_dcache_patch_new)
        for (uint32_t id : kv.second)
            if (enqueued_ids.find(id) == enqueued_ids.end())
                release.push_back(id);
    if (!release.empty())
        neighborcache_dyn_patch_unpin(release);
    query_scratch->neighborcache_dcache_patch_new.clear();
    query_scratch->neighborcache_dcache_seed_members.clear();
    query_scratch->neighborcache_dcache_seed_extra.clear();
    query_scratch->neighborcache_dcache_member_to_seed.clear();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_dyn_pack_coread(SSDQueryScratch<T> *query_scratch)
{
    if (!_neighborcache_coread_pack || query_scratch == nullptr || neighborcache_dcache_updates_frozen() ||
        !query_scratch->neighborcache_dcache_maintenance_enabled)
        return;
    const std::vector<uint32_t> &order = query_scratch->neighborcache_storage_node_order;
    if (order.size() < 2)
        return;
    const size_t nps = static_cast<size_t>((_nnodes_per_sector > 0) ? _nnodes_per_sector : 1);
    if (nps < 2)
        return;

    thread_local std::vector<uint32_t> fresh;
    thread_local tsl::robin_set<uint32_t> seen;
    fresh.clear();
    seen.clear();
    auto on_live_page = [&](uint32_t id) {
        NeighborCacheDiskLoc loc;
        const int canon = neighborcache_canon_lookup(id, loc);
        if (canon == 0)
            return false;
        if (canon == 1 && neighborcache_dyn_is_sector(loc.sector) &&
            neighborcache_dyn_page_live(loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE))
            return true;
        bool on_page = false;
        neighborcache_dc_for_each_loc_unlocked(id, [&](const NeighborCacheDiskLoc &candidate, size_t) {
            if (!on_page && neighborcache_dyn_is_sector(candidate.sector) &&
                neighborcache_dyn_page_live(candidate.sector - NEIGHBORCACHE_DYN_SECTOR_BASE))
                on_page = true;
        });
        return on_page;
    };
    fresh.reserve(order.size());
    for (uint32_t id : order)
    {
        if (!seen.insert(id).second)
            continue;
        if (!on_live_page(id) && !neighborcache_coread_id_pending(id))
            fresh.push_back(id);
    }

    thread_local tsl::robin_set<uint32_t> fresh_set;
    fresh_set.clear();
    fresh_set.reserve(fresh.size());
    for (uint32_t id : fresh)
        fresh_set.insert(id);
    tsl::robin_map<uint32_t, std::vector<uint32_t>> kids;
    for (uint32_t child : fresh)
    {
        const auto parent_it = query_scratch->parent_of(child);
        if (!parent_it.has_value() || parent_it.value() == child)
            continue;
        kids[parent_it.value()].push_back(child);
    }

    thread_local tsl::robin_set<uint32_t> used;
    used.clear();
    auto already_on_page = [&](uint32_t id) {
        if (neighborcache_coread_id_pending(id))
            return true;
        return on_live_page(id);
    };
    auto commit_page = [&](uint32_t seed_id, const std::vector<uint32_t> &rest, bool hop2) {
        if (_neighborcache_dyn_committed_pages >= _neighborcache_dyn_page_cap)
            return false;
        NeighborCachePendingFlush pf;
        pf.seed_id = seed_id;
        auto push = [&](uint32_t id) {
            if (pf.member_ids.size() >= nps)
                return;
            std::vector<T> coords;
            std::vector<uint32_t> nbrs;
            if (!neighborcache_dyn_copy_member_payload(id, coords, nbrs, false))
                return;
            pf.member_ids.push_back(id);
            pf.member_coords.push_back(std::move(coords));
            pf.member_nbrs.push_back(std::move(nbrs));
        };
        {
            push(seed_id);
            for (uint32_t id : rest)
                push(id);
        }
        if (pf.member_ids.size() < 2 || pf.member_ids.front() != seed_id)
            return false;
        const std::vector<uint32_t> packed_ids = pf.member_ids;
        bool handed_off = false;
        if (!neighborcache_dyn_enqueue_coread_write(pf, hop2, handed_off))
            return false;
        for (uint32_t id : packed_ids)
            used.insert(id);
        if (!handed_off)
        {
            _neighborcache_coread_pages.fetch_add(1, std::memory_order_relaxed);
            if (hop2)
                _neighborcache_coread_hop2_pages.fetch_add(1, std::memory_order_relaxed);
        }
        return true;
    };
    auto pack_seed = [&](uint32_t seed_id) {
        if (used.find(seed_id) != used.end() || already_on_page(seed_id))
            return;
        const auto hop1_it = kids.find(seed_id);
        if (hop1_it == kids.end())
            return;
        thread_local std::vector<std::pair<uint32_t, bool>> items;
        thread_local tsl::robin_set<uint32_t> seen_item;
        items.clear();
        seen_item.clear();
        seen_item.insert(seed_id);
        for (uint32_t child : hop1_it->second)
        {
            if (used.find(child) != used.end() || !seen_item.insert(child).second)
                continue;
            items.emplace_back(child, false);
            const auto hop2_it = kids.find(child);
            if (hop2_it == kids.end())
                continue;
            for (uint32_t grand : hop2_it->second)
            {
                if (used.find(grand) != used.end() || !seen_item.insert(grand).second)
                    continue;
                items.emplace_back(grand, true);
            }
        }
        for (size_t pos = 0; pos < items.size();)
        {
            std::vector<uint32_t> rest;
            bool hop2 = false;
            while (pos < items.size() && rest.size() + 1 < nps)
            {
                if (used.find(items[pos].first) != used.end())
                {
                    ++pos;
                    continue;
                }
                rest.push_back(items[pos].first);
                hop2 = hop2 || items[pos].second;
                ++pos;
            }
            if (rest.empty() || !commit_page(seed_id, rest, hop2))
                break;
        }
    };
    for (const auto &kv : kids)
    {
        if (fresh_set.find(kv.first) == fresh_set.end())
            pack_seed(kv.first);
    }
    for (uint32_t id : fresh)
        pack_seed(id);

    thread_local std::vector<uint32_t> left;
    left.clear();
    left.reserve(fresh.size());
    for (uint32_t id : fresh)
    {
        if (used.find(id) == used.end())
            left.push_back(id);
    }
    for (size_t base = 0; base < left.size();)
    {
        if (_neighborcache_dyn_committed_pages >= _neighborcache_dyn_page_cap)
            break;
        const size_t window_end = std::min(left.size(), base + nps);
        std::vector<uint32_t> rest;
        uint32_t seed_id = left[base];
        for (size_t j = base + 1; j < window_end; ++j)
            rest.push_back(left[j]);
        base = window_end;
        commit_page(seed_id, rest, false);
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_dyn_on_query_end(QueryStats *stats, SSDQueryScratch<T> *query_scratch)
{
    if (_neighborcache_coread_pack && !neighborcache_dcache_updates_frozen())
        neighborcache_dyn_pack_coread(query_scratch);
    if (_neighborcache_dcache_query_patch && !neighborcache_dcache_updates_frozen())
    {
        neighborcache_dyn_apply_query_patches(stats, query_scratch);
    }
    {
        neighborcache_dyn_record_real_io_coaccess(query_scratch);
    }
    if (!_neighborcache_freeze_maintenance)
        _neighborcache_mcache.on_query_end();
    {
        if (query_scratch != nullptr && query_scratch->neighborcache_dcache_maintenance_enabled)
            neighborcache_dyn_flush_dirty_heap(query_scratch);
        else if (query_scratch != nullptr)
            query_scratch->clear_neighborcache_heap_dirty();
    }
    neighborcache_dyn_update_dcache_gate(stats, query_scratch);
    if (query_scratch == nullptr || !query_scratch->neighborcache_dcache_maintenance_enabled)
        return;
    const uint32_t query_epoch = _neighborcache_query_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (query_epoch % 100000 == 0)
    {
        uint64_t snap_pages = 0, snap_members = 0, extra_pages = 0, extra_neighbors = 0;
        uint64_t span_hist[6] = {};
        {
            NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
            for (size_t page = 0; page < _neighborcache_dyn_page_member_count.size(); ++page)
            {
                const uint16_t count = _neighborcache_dyn_page_member_count[page];
                if (count == 0)
                    continue;
                if (page < _neighborcache_dyn_page_secondary.size() && _neighborcache_dyn_page_secondary[page])
                {
                    ++extra_pages;
                    if (count > 1)
                        extra_neighbors += static_cast<uint64_t>(count - 1);
                    continue;
                }
                ++snap_pages;
                snap_members += count;
                uint32_t n = 1;
                if (page < _neighborcache_dyn_page_span_len.size() && _neighborcache_dyn_page_span_len[page] > 1)
                    n = _neighborcache_dyn_page_span_len[page];
                if (n > 5)
                    n = 5;
                ++span_hist[n];
            }
        }
        const double snap_avg =
            snap_pages == 0 ? 0.0 : static_cast<double>(snap_members) / static_cast<double>(snap_pages);
        diskann::cout << "NEIGHBORCACHE dcache_fill_snap query=" << query_epoch << " pages=" << snap_pages
                      << " avg_members=" << snap_avg << " avg_neighbors=" << (snap_avg - 1.0)
                      << " extra_pages=" << extra_pages << " extra_neighbors=" << extra_neighbors
                      << " p1=" << span_hist[1] << " p2=" << span_hist[2] << " p3=" << span_hist[3]
                      << " p4=" << span_hist[4] << " p5=" << span_hist[5] << std::endl;
    }
    if (_neighborcache_member_snapshot_at > 0 && query_epoch == _neighborcache_member_snapshot_at &&
        !_neighborcache_member_snapshot_done.load(std::memory_order_relaxed))
    {
        std::vector<uint32_t> ids;
        {
            NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
            for (size_t page = 0; page < _neighborcache_dyn_page_member_count.size(); ++page)
            {
                const uint16_t count = _neighborcache_dyn_page_member_count[page];
                if (count <= 1)
                    continue;
                const size_t begin = page * _neighborcache_dyn_members_per_page;
                if (begin + count > _neighborcache_dyn_page_members.size())
                    continue;
                for (size_t mi = 1; mi < count; ++mi)
                {
                    const uint32_t mid = _neighborcache_dyn_page_members[begin + mi];
                    if (mid != NEIGHBORCACHE_DYN_INVALID_PAGE)
                        ids.push_back(mid);
                }
            }
        }
        _neighborcache_member_snapshot_slots.store(ids.size(), std::memory_order_relaxed);
        for (uint32_t id : ids)
            _neighborcache_member_at_snapshot.insert(id);
        _neighborcache_member_snapshot_done.store(true, std::memory_order_release);
        diskann::cout << "NEIGHBORCACHE member_snapshot query=" << query_epoch << " neighbor_slots=" << ids.size()
                      << " neighbor_ids=" << _neighborcache_member_at_snapshot.size() << std::endl;
    }
    if (_neighborcache_seed_pin_neighbors)
        neighborcache_dyn_write_settled_pinned_seeds(query_epoch, stats);
    if (query_epoch % _neighborcache_pending_check_interval == 0)
        neighborcache_dyn_commit_expired_pairs(query_epoch, stats);
    neighborcache_dyn_log_monitor(query_epoch);
    uint64_t page_count = 0;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        page_count = _neighborcache_dyn_committed_pages;
    }
    if (stats != nullptr)
        stats->n_neighborcache_dyn_pages = static_cast<unsigned>(
            std::min<uint64_t>(page_count, static_cast<uint64_t>(std::numeric_limits<unsigned>::max())));
    neighborcache_dyn_request_refresh();
    const bool disk_full = page_count >= _neighborcache_dyn_page_cap;
    if (!_neighborcache_decay_started.load(std::memory_order_acquire))
    {
        if (!disk_full)
            return;
        bool expected = false;
        _neighborcache_decay_started.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
    }

    _neighborcache_decay_query_count.fetch_add(1, std::memory_order_relaxed);
    const uint64_t accesses = query_scratch->neighborcache_expand_count;
    const uint64_t now = _neighborcache_decay_access_count.fetch_add(accesses, std::memory_order_acq_rel) + accesses;
    const uint64_t elapsed = now - std::min(now, _neighborcache_decay_cycle_start.load(std::memory_order_acquire));
    const float unit = static_cast<float>(std::exp2(static_cast<double>(elapsed) / _neighborcache_decay_half_life_accesses));
    const bool saturated = _neighborcache_mcache.edge_saturation_pending();
    if (unit < NEIGHBORCACHE_SCORE_UNIT_MAX && !(saturated && unit >= 2.0f))
    {
        _neighborcache_score_unit.store(unit, std::memory_order_release);
        return;
    }

    std::unique_lock<std::mutex> decay_lock(_neighborcache_decay_mu, std::try_to_lock);
    if (!decay_lock.owns_lock())
        return;
    const uint64_t cycle_start = _neighborcache_decay_cycle_start.load(std::memory_order_acquire);
    if (cycle_start > now)
        return;
    const float divisor = std::min(
        NEIGHBORCACHE_SCORE_UNIT_MAX,
        static_cast<float>(std::exp2(static_cast<double>(now - cycle_start) / _neighborcache_decay_half_life_accesses)));
    if (divisor < 2.0f)
        return;
    const float factor = 1.0f / divisor;
    std::vector<std::unique_ptr<NeighborCacheTimedMutexGuard>> heap_locks;
    heap_locks.reserve(_neighborcache_heap_shard_count);
    for (size_t shard_index = 0; shard_index < _neighborcache_heap_shard_count; ++shard_index)
        heap_locks.emplace_back(
            std::make_unique<NeighborCacheTimedMutexGuard>(_neighborcache_heap_shards[shard_index].mu, NeighborCacheLockKind::Heap));
    _neighborcache_mcache.scale_scores(factor, _neighborcache_score_unit, 1.0f);
    for (size_t shard_index = 0; shard_index < _neighborcache_heap_shard_count; ++shard_index)
    {
        NeighborCacheHeapShard &shard = _neighborcache_heap_shards[shard_index];
        for (NeighborCacheHeapEntry &entry : shard.max_heap)
            entry.score *= factor;
        for (NeighborCacheHeapEntry &entry : shard.min_heap)
            entry.score *= factor;
    }
    _neighborcache_decay_cycle_start.store(now, std::memory_order_release);
    _neighborcache_decay_cycles.fetch_add(1, std::memory_order_relaxed);
    if (divisor < NEIGHBORCACHE_SCORE_UNIT_MAX)
        _neighborcache_decay_early_cycles.fetch_add(1, std::memory_order_relaxed);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_commit_span(std::vector<NeighborCachePendingFlush> &pages,
                                                    uint32_t replacement_seed)
{
    const uint64_t nps = (_nnodes_per_sector > 0) ? _nnodes_per_sector : 1;
    if (pages.size() < 2 || pages.size() > _neighborcache_dcache_max_pages)
        return false;
    const uint32_t seed_id = pages[0].seed_id;
    for (NeighborCachePendingFlush &pf : pages)
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
    auto fill_buf = [&](const NeighborCachePendingFlush &pf, std::vector<char> &buf) -> bool {
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
                write_neighborcache_dc_seed_tail(slot_dst, _max_node_len, _disk_bytes_per_point, nnbrs, span);
        }
        return true;
    };
    std::vector<std::vector<char>> bufs(pages.size());
    for (size_t i = 0; i < pages.size(); ++i)
        if (!fill_buf(pages[i], bufs[i]))
            return false;

    uint32_t base = NEIGHBORCACHE_DYN_INVALID_PAGE;
    uint32_t expected_victim_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        if (replacement_seed != NEIGHBORCACHE_DYN_INVALID_PAGE)
        {
            if (!neighborcache_dyn_seed_page_unlocked(replacement_seed, expected_victim_page))
                return false;
        }
        else if (_neighborcache_dyn_committed_pages >= _neighborcache_dyn_page_cap)
            return false;
        base = neighborcache_dyn_alloc_contiguous_unlocked(span);
    }
    if (base == NEIGHBORCACHE_DYN_INVALID_PAGE || static_cast<uint64_t>(base) + span > _neighborcache_dyn_physical_cap)
        return false;
    auto write_page = [&](uint32_t page_idx, const std::vector<char> &buf) {
#ifndef _WINDOWS
        char *aligned_page = nullptr;
        alloc_aligned(reinterpret_cast<void **>(&aligned_page), defaults::SECTOR_LEN, defaults::SECTOR_LEN);
        memcpy(aligned_page, buf.data(), defaults::SECTOR_LEN);
        const bool write_ok = neighborcache_dyn_writer_submit_and_wait(
            NeighborCacheDynWriteOp::Write, static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN),
            aligned_page, defaults::SECTOR_LEN);
        aligned_free(aligned_page);
        return write_ok;
#else
        return neighborcache_dyn_writer_submit_and_wait(
            NeighborCacheDynWriteOp::Write, static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN), buf.data(),
            defaults::SECTOR_LEN);
#endif
    };
    bool write_ok = true;
    {
        NeighborCacheTimedRegion commit_io(NeighborCacheLockKind::CommitIo);
        for (uint32_t i = 0; i < span; ++i)
            write_ok = write_ok && write_page(base + i, bufs[i]);
    }
    auto free_span = [&]() {
        for (uint32_t i = 0; i < span; ++i)
            _neighborcache_dyn_free.push_back(base + i);
    };
    if (!write_ok)
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        free_span();
        return false;
    }
    uint32_t seed_graph_in = 0, seed_followed_in = 0, seed_sig_n = 0, seed_packed_n = 0;
    const bool note_seed_write =
        !pages.empty() && !pages[0].member_nbrs.empty() &&
        neighborcache_seed_write_count(_neighborcache_mem_pool.get(), pages[0].seed_id, pages[0].member_ids, pages[0].member_nbrs.front(),
                               pages[0].snap.significant, seed_graph_in, seed_followed_in, seed_sig_n, seed_packed_n);
    NeighborCacheSpinLock *seed_lk = _neighborcache_seed_prefetch ? &neighborcache_seed_write_lock(pages[0].seed_id) : nullptr;
    if (seed_lk != nullptr)
        seed_lk->lock();
    struct SeedUnlock
    {
        NeighborCacheSpinLock *p;
        ~SeedUnlock()
        {
            if (p != nullptr)
                p->unlock();
        }
    } seed_unlock{seed_lk};
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        bool valid = true;
        if (replacement_seed == NEIGHBORCACHE_DYN_INVALID_PAGE)
            valid = _neighborcache_dyn_committed_pages < _neighborcache_dyn_page_cap;
        else
        {
            uint32_t victim_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
            valid = neighborcache_dyn_seed_page_unlocked(replacement_seed, victim_page) && victim_page == expected_victim_page;
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
            auto first_it = _neighborcache_seed_first_fill.find(seed_id);
            if (first_it != _neighborcache_seed_first_fill.end())
            {
                kept_first = first_it->second;
                keep_first = true;
            }
        }
        neighborcache_span_carry_stats(replacement_seed == seed_id ? expected_victim_page : NEIGHBORCACHE_DYN_INVALID_PAGE, base);
        if (replacement_seed != NEIGHBORCACHE_DYN_INVALID_PAGE)
            neighborcache_dyn_invalidate_seed_page(replacement_seed);
        if (keep_first)
            _neighborcache_seed_first_fill[seed_id] = kept_first;
        else if (_neighborcache_seed_first_fill.find(seed_id) == _neighborcache_seed_first_fill.end())
            _neighborcache_seed_first_fill[seed_id] = static_cast<uint16_t>(pages[0].member_ids.size());

        for (uint32_t i = 0; i < span; ++i)
        {
            const uint32_t page_idx = base + i;
            const NeighborCachePendingFlush &pf = pages[i];
            if (page_idx < _neighborcache_dyn_page_secondary.size())
                _neighborcache_dyn_page_secondary[page_idx] = i == 0 ? 0 : 1;
            if (page_idx < _neighborcache_dyn_page_span_len.size())
                _neighborcache_dyn_page_span_len[page_idx] = span;
            if (page_idx < _neighborcache_dyn_page_span_base.size())
                _neighborcache_dyn_page_span_base[page_idx] = base;
            if (page_idx < _neighborcache_dyn_page_sibling.size())
                _neighborcache_dyn_page_sibling[page_idx] = NEIGHBORCACHE_DYN_INVALID_PAGE;
            neighborcache_dyn_bump_page_gen(page_idx);
            const size_t member_begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
            std::copy(pf.member_ids.begin(), pf.member_ids.end(), _neighborcache_dyn_page_members.begin() + member_begin);
            _neighborcache_dyn_page_member_count[page_idx] = static_cast<uint16_t>(pf.member_ids.size());
            // Publish locations last. Acquiring a location shard after this
            // release also publishes the immutable page metadata above.
            const uint32_t sector = NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx;
            neighborcache_shared_cache_erase_dyn(page_idx, 1);
            neighborcache_shared_cache_admit_dyn(page_idx, bufs[i].data());
            for (size_t mi = 0; mi < pf.member_ids.size(); ++mi)
            {
                NeighborCacheDiskLoc loc;
                loc.sector = sector;
                loc.slot = static_cast<uint16_t>(mi);
                loc.nsectors = 1;
                neighborcache_dyn_loc_add_unlocked(pf.member_ids[mi], loc);
            }
            neighborcache_dyn_bump_page_gen(page_idx);
        }
        _neighborcache_dyn_committed_pages++;
        _neighborcache_dyn_flush_count++;
        if (note_seed_write)
        {
            neighborcache_seed_write_tally(seed_graph_in, seed_followed_in, seed_sig_n, seed_packed_n);
            neighborcache_seed_write_note_id(pages[0].seed_id);
        }
        _neighborcache_dcache_page_writes.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_coread_submit_page(NeighborCachePendingFlush &pf, uint32_t &page_idx,
                                                          std::shared_ptr<NeighborCacheDynWriteJob> &job)
{
    page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
    job = nullptr;
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
            write_neighborcache_dc_seed_tail(slot_dst, _max_node_len, _disk_bytes_per_point, nnbrs, 1);
    }

    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        uint32_t existing_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
        if (neighborcache_dyn_seed_page_unlocked(pf.seed_id, existing_page))
        {
            const size_t existing_count = _neighborcache_dyn_page_member_count[existing_page];
            const uint32_t *existing_members =
                _neighborcache_dyn_page_members.data() + static_cast<size_t>(existing_page) * _neighborcache_dyn_members_per_page;
            if (existing_count > 0 && !neighborcache_dyn_hotter_mismatch(existing_members, existing_count, pf.member_ids))
                return false;
        }
        if (_neighborcache_dyn_committed_pages >= _neighborcache_dyn_page_cap)
            return false;
        page_idx = neighborcache_dyn_alloc_page_unlocked();
    }
    if (page_idx == NEIGHBORCACHE_DYN_INVALID_PAGE)
        return false;

#ifndef _WINDOWS
    char *aligned_page = nullptr;
    alloc_aligned(reinterpret_cast<void **>(&aligned_page), defaults::SECTOR_LEN, defaults::SECTOR_LEN);
    memcpy(aligned_page, buf.data(), defaults::SECTOR_LEN);
    job = neighborcache_dyn_writer_submit(NeighborCacheDynWriteOp::Write,
                                  static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN), aligned_page,
                                  defaults::SECTOR_LEN);
    aligned_free(aligned_page);
#else
    job = neighborcache_dyn_writer_submit(NeighborCacheDynWriteOp::Write,
                                  static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN), buf.data(),
                                  defaults::SECTOR_LEN);
#endif
    if (job == nullptr)
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        _neighborcache_dyn_free.push_back(page_idx);
        page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
        return false;
    }
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_coread_complete_page(NeighborCachePendingFlush &pf, uint32_t page_idx, bool write_ok)
{
    if (page_idx == NEIGHBORCACHE_DYN_INVALID_PAGE)
        return false;
    if (!write_ok)
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        _neighborcache_dyn_free.push_back(page_idx);
        return false;
    }
    uint32_t seed_graph_in = 0, seed_followed_in = 0, seed_sig_n = 0, seed_packed_n = 0;
    const bool note_seed_write =
        !pf.member_nbrs.empty() &&
        neighborcache_seed_write_count(_neighborcache_mem_pool.get(), pf.seed_id, pf.member_ids, pf.member_nbrs.front(),
                               pf.snap.significant, seed_graph_in, seed_followed_in, seed_sig_n, seed_packed_n);
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        bool valid = _neighborcache_dyn_committed_pages < _neighborcache_dyn_page_cap;
        uint32_t existing_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
        if (neighborcache_dyn_seed_page_unlocked(pf.seed_id, existing_page))
        {
            const size_t existing_count = _neighborcache_dyn_page_member_count[existing_page];
            const uint32_t *existing_members =
                _neighborcache_dyn_page_members.data() + static_cast<size_t>(existing_page) * _neighborcache_dyn_members_per_page;
            if (existing_count > 0 && !neighborcache_dyn_hotter_mismatch(existing_members, existing_count, pf.member_ids))
                valid = false;
        }
        if (!valid)
        {
            _neighborcache_dyn_free.push_back(page_idx);
            return false;
        }
        if (_neighborcache_seed_first_fill.find(pf.seed_id) == _neighborcache_seed_first_fill.end())
            _neighborcache_seed_first_fill[pf.seed_id] = static_cast<uint16_t>(pf.member_ids.size());
        neighborcache_span_carry_stats(NEIGHBORCACHE_DYN_INVALID_PAGE, page_idx);
        neighborcache_dyn_bump_page_gen(page_idx);
        const size_t member_begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
        std::copy(pf.member_ids.begin(), pf.member_ids.end(), _neighborcache_dyn_page_members.begin() + member_begin);
        _neighborcache_dyn_page_member_count[page_idx] = static_cast<uint16_t>(pf.member_ids.size());
        const uint32_t sector = NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx;
        neighborcache_shared_cache_erase_dyn(page_idx, 1);
        for (size_t mi = 0; mi < pf.member_ids.size(); mi++)
        {
            NeighborCacheDiskLoc loc;
            loc.sector = sector;
            loc.slot = static_cast<uint16_t>(mi);
            loc.nsectors = 1;
            neighborcache_dyn_loc_add_unlocked(pf.member_ids[mi], loc);
        }
        neighborcache_dyn_bump_page_gen(page_idx);
        _neighborcache_dyn_committed_pages++;
        _neighborcache_dyn_flush_count++;
        if (note_seed_write)
        {
            neighborcache_seed_write_tally(seed_graph_in, seed_followed_in, seed_sig_n, seed_packed_n);
            neighborcache_seed_write_note_id(pf.seed_id);
        }
        _neighborcache_dcache_page_writes.fetch_add(1, std::memory_order_relaxed);
    }
    if (_neighborcache_dcache_evict_ncache_on_commit && _neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active())
    {
        uint64_t erased = 0;
        for (uint32_t member_id : pf.member_ids)
            erased += _neighborcache_mem_pool->erase(member_id) ? 1ULL : 0ULL;
        _neighborcache_dcache_ncache_erases.fetch_add(erased, std::memory_order_relaxed);
    }
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_commit_one(NeighborCachePendingFlush &pf, uint32_t replacement_seed)
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
            write_neighborcache_dc_seed_tail(slot_dst, _max_node_len, _disk_bytes_per_point, nnbrs, 1);
    }

    uint32_t page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
    uint32_t expected_victim_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        uint32_t existing_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
        if (neighborcache_dyn_seed_page_unlocked(pf.seed_id, existing_page))
        {
            const size_t existing_count = _neighborcache_dyn_page_member_count[existing_page];
            const uint32_t *existing_members =
                _neighborcache_dyn_page_members.data() +
                static_cast<size_t>(existing_page) * _neighborcache_dyn_members_per_page;
            if (existing_count > 0 &&
                !neighborcache_dyn_hotter_mismatch(existing_members, existing_count, pf.member_ids))
                return false;
        }
        if (replacement_seed == NEIGHBORCACHE_DYN_INVALID_PAGE && _neighborcache_dyn_committed_pages >= _neighborcache_dyn_page_cap)
            return false;
        if (replacement_seed != NEIGHBORCACHE_DYN_INVALID_PAGE)
        {
            if (!neighborcache_dyn_seed_page_unlocked(replacement_seed, expected_victim_page))
                return false;
        }
        page_idx = neighborcache_dyn_alloc_page_unlocked();
    }
    if (page_idx == NEIGHBORCACHE_DYN_INVALID_PAGE)
        return false;
    bool write_ok = false;
    {
        NeighborCacheTimedRegion commit_io(NeighborCacheLockKind::CommitIo);
#ifndef _WINDOWS
        char *aligned_page = nullptr;
        alloc_aligned(reinterpret_cast<void **>(&aligned_page), defaults::SECTOR_LEN, defaults::SECTOR_LEN);
        memcpy(aligned_page, buf.data(), defaults::SECTOR_LEN);
        write_ok = neighborcache_dyn_writer_submit_and_wait(
            NeighborCacheDynWriteOp::Write,
            static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN), aligned_page,
            defaults::SECTOR_LEN);
        aligned_free(aligned_page);
#else
        write_ok = neighborcache_dyn_writer_submit_and_wait(
            NeighborCacheDynWriteOp::Write,
            static_cast<off_t>(page_idx) * static_cast<off_t>(defaults::SECTOR_LEN), buf.data(),
            defaults::SECTOR_LEN);
#endif
    }
    if (!write_ok)
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        _neighborcache_dyn_free.push_back(page_idx);
        return false;
    }
    uint32_t seed_graph_in = 0, seed_followed_in = 0, seed_sig_n = 0, seed_packed_n = 0;
    const bool note_seed_write =
        !pf.member_nbrs.empty() &&
        neighborcache_seed_write_count(_neighborcache_mem_pool.get(), pf.seed_id, pf.member_ids, pf.member_nbrs.front(),
                               pf.snap.significant, seed_graph_in, seed_followed_in, seed_sig_n, seed_packed_n);
    NeighborCacheSpinLock *seed_lk = _neighborcache_seed_prefetch ? &neighborcache_seed_write_lock(pf.seed_id) : nullptr;
    if (seed_lk != nullptr)
        seed_lk->lock();
    struct SeedUnlock
    {
        NeighborCacheSpinLock *p;
        ~SeedUnlock()
        {
            if (p != nullptr)
                p->unlock();
        }
    } seed_unlock{seed_lk};
    {
        NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true);
        bool valid = true;
        uint32_t existing_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
        if (neighborcache_dyn_seed_page_unlocked(pf.seed_id, existing_page))
        {
            const size_t existing_count = _neighborcache_dyn_page_member_count[existing_page];
            const uint32_t *existing_members =
                _neighborcache_dyn_page_members.data() +
                static_cast<size_t>(existing_page) * _neighborcache_dyn_members_per_page;
            if (existing_count > 0 &&
                !neighborcache_dyn_hotter_mismatch(existing_members, existing_count, pf.member_ids))
                valid = false;
        }
        if (replacement_seed == NEIGHBORCACHE_DYN_INVALID_PAGE)
            valid = valid && _neighborcache_dyn_committed_pages < _neighborcache_dyn_page_cap;
        else
        {
            uint32_t victim_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
            valid = valid && neighborcache_dyn_seed_page_unlocked(replacement_seed, victim_page) &&
                    victim_page == expected_victim_page;
        }
        if (!valid)
        {
            _neighborcache_dyn_free.push_back(page_idx);
            return false;
        }

        // Copy-on-write: publish the new physical page before retiring the old one.
        uint16_t kept_first = 0;
        bool keep_first = false;
        if (replacement_seed == pf.seed_id)
        {
            auto first_it = _neighborcache_seed_first_fill.find(pf.seed_id);
            if (first_it != _neighborcache_seed_first_fill.end())
            {
                kept_first = first_it->second;
                keep_first = true;
            }
        }
        neighborcache_span_carry_stats(replacement_seed == pf.seed_id ? expected_victim_page : NEIGHBORCACHE_DYN_INVALID_PAGE,
                               page_idx);
        if (replacement_seed != NEIGHBORCACHE_DYN_INVALID_PAGE)
            neighborcache_dyn_invalidate_seed_page(replacement_seed);
        if (keep_first)
            _neighborcache_seed_first_fill[pf.seed_id] = kept_first;
        else if (_neighborcache_seed_first_fill.find(pf.seed_id) == _neighborcache_seed_first_fill.end())
            _neighborcache_seed_first_fill[pf.seed_id] = static_cast<uint16_t>(pf.member_ids.size());

        neighborcache_dyn_bump_page_gen(page_idx);
        const size_t member_begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
        std::copy(pf.member_ids.begin(), pf.member_ids.end(), _neighborcache_dyn_page_members.begin() + member_begin);
        _neighborcache_dyn_page_member_count[page_idx] = static_cast<uint16_t>(pf.member_ids.size());
        // Publish locations only after the page table is complete.
        const uint32_t sector = NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx;
        neighborcache_shared_cache_erase_dyn(page_idx, 1);
        neighborcache_shared_cache_admit_dyn(page_idx, buf.data());
        for (size_t mi = 0; mi < pf.member_ids.size(); mi++)
        {
            NeighborCacheDiskLoc loc;
            loc.sector = sector;
            loc.slot = static_cast<uint16_t>(mi);
            loc.nsectors = 1;
            const uint32_t mid = pf.member_ids[mi];
            neighborcache_dyn_loc_add_unlocked(mid, loc);
        }
        neighborcache_dyn_bump_page_gen(page_idx);
        _neighborcache_dyn_committed_pages++;
        _neighborcache_dyn_flush_count++;
        if (note_seed_write)
        {
            neighborcache_seed_write_tally(seed_graph_in, seed_followed_in, seed_sig_n, seed_packed_n);
            neighborcache_seed_write_note_id(pf.seed_id);
        }
        _neighborcache_dcache_page_writes.fetch_add(1, std::memory_order_relaxed);
    }
    if (_neighborcache_dcache_evict_ncache_on_commit && _neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active())
    {
        uint64_t erased = 0;
        for (uint32_t member_id : pf.member_ids)
            erased += _neighborcache_mem_pool->erase(member_id) ? 1ULL : 0ULL;
        _neighborcache_dcache_ncache_erases.fetch_add(erased, std::memory_order_relaxed);
    }
    return true;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::reload_neighborcache_memory_cache(const std::string &profile_prefix, double neighborcache_memory_gb,
                                                       double host_memory_gb, double reserve_gb, uint32_t num_threads,
                                                       uint64_t &evicted_nodes, std::string &report)
{
    evicted_nodes = 0;
    report.clear();
    const uint64_t before = neighborcache_memory_cached_count();

    uint64_t max_nodes = 0;
    if (plan_neighborcache_memory_cache(neighborcache_memory_gb, host_memory_gb, reserve_gb, num_threads, max_nodes, report) != 0)
        return -1;

    clear_neighborcache_memory_cache();
    std::vector<uint32_t> node_list;
    if (build_neighborcache_memory_node_list(profile_prefix, max_nodes, node_list, 0) != 0)
        return -1;
    if (!node_list.empty())
    {
        if (load_neighborcache_memory_pool(profile_prefix, node_list) != 0)
            return -1;
    }

    const uint64_t after = neighborcache_memory_cached_count();
    if (before > after)
        evicted_nodes = before - after;

    std::ostringstream oss;
    oss << "NEIGHBORCACHE memory reload: cached " << after << " nodes";
    if (evicted_nodes > 0)
        oss << ", evicted " << evicted_nodes << " from DRAM cache";
    oss << ".";
    report = oss.str();
    diskann::cout << report << std::endl;
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::plan_neighborcache_disk_cache(double base_ratio, uint64_t &out_max_nodes,
                                                   std::string &report, bool allow_replica_slots) const
{
    out_max_nodes = 0;
    report.clear();
    if (base_ratio <= 0.0)
    {
        report = "neighborcache_disk_cache_ratio <= 0; NEIGHBORCACHE disk cache disabled.";
        return 0;
    }
    const double max_ratio = allow_replica_slots ? 10.0 : 1.0;
    if (base_ratio > max_ratio)
    {
        report = "ERROR: neighborcache_disk_cache_ratio exceeds allowed maximum.";
        return -1;
    }
    if (_disk_index_file.empty() || _max_node_len == 0)
    {
        report = "ERROR: disk index not loaded; cannot size NEIGHBORCACHE disk cache.";
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
        report = "ERROR: NEIGHBORCACHE disk-cache budget too small for one node.";
        return -1;
    }
    out_max_nodes = nodes;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(4);
    oss << "NEIGHBORCACHE disk-cache plan:\n"
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
int PQFlashIndex<T, LabelT>::reload_neighborcache_disk_cache(const std::string &profile_prefix, double base_ratio,
                                                     const std::string &output_prefix, uint64_t rank_skip,
                                                     uint64_t &evicted_nodes, std::string &report,
                                                     bool unified_single_file, uint32_t k_hops,
                                                     const std::string &layout)
{
    evicted_nodes = 0;
    report.clear();
    const uint64_t before = _neighborcache_dc_map.size();

    uint64_t max_nodes = 0;
    if (plan_neighborcache_disk_cache(base_ratio, max_nodes, report) != 0)
        return -1;

    if (build_and_load_neighborcache_disk_cache(profile_prefix, max_nodes, output_prefix, rank_skip, unified_single_file,
                                        k_hops, layout) != 0)
        return -1;

    const uint64_t after = _neighborcache_dc_map.size();
    if (before > after)
        evicted_nodes = before - after;

    std::ostringstream oss;
    oss << "NEIGHBORCACHE disk-cache reload: " << after << " nodes in map";
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
int PQFlashIndex<T, LabelT>::load_neighborcache_disk_cache_from_prefix(const std::string &output_prefix)
{
    if (_neighborcache_disk_reader)
    {
        _neighborcache_disk_reader->close();
        _neighborcache_disk_reader.reset();
    }
    _neighborcache_unified_disk = false;
    _neighborcache_region_byte_offset = 0;
    _neighborcache_dc_map.clear();
    _neighborcache_dc_seed_page_nbrs.clear();
    _neighborcache_dc_seed_member_loc.clear();
    _neighborcache_dc_seed_canonical_loc.clear();
    _neighborcache_seed_only_layout = false;
    _neighborcache_seed_only_expand = false;
    _neighborcache_seed_first_lookup = false;
    _neighborcache_child_only_layout = false;
    _neighborcache_dc_num_nodes = 0;
    _neighborcache_dc_path.clear();

    const std::string data_path = output_prefix + "_neighborcache_dc.data";
    const std::string nodes_path = output_prefix + "_neighborcache_dc.nodes";
    if (!file_exists(data_path) || !file_exists(nodes_path))
    {
        diskann::cerr << "NEIGHBORCACHE disk-cache reuse: missing " << data_path << " or " << nodes_path << std::endl;
        return -1;
    }

    uint32_t *nodes = nullptr;
    size_t npts = 0, nd = 0;
    diskann::load_bin<uint32_t>(nodes_path, nodes, npts, nd);
    std::vector<uint32_t> node_list(nodes, nodes + npts);
    delete[] nodes;

    if (node_list.empty() || _nnodes_per_sector == 0)
    {
        diskann::cerr << "NEIGHBORCACHE disk-cache reuse: empty node list or nnodes_per_sector=0." << std::endl;
        return -1;
    }

    const uint64_t nps = _nnodes_per_sector;
    uint64_t slot_in_sector = 0;
    uint32_t cur_sector = 0;
    _neighborcache_dc_map.reserve(node_list.size());
    for (uint32_t id : node_list)
    {
        NeighborCacheDiskLoc loc;
        loc.sector = cur_sector;
        loc.slot = static_cast<uint16_t>(slot_in_sector);
        loc.nsectors = 1;
        _neighborcache_dc_map[id].push_back(loc);
        slot_in_sector++;
        if (slot_in_sector == nps)
        {
            slot_in_sector = 0;
            cur_sector++;
        }
    }

    const std::string seed_pages_path = output_prefix + "_neighborcache_dc.seed_pages";
    const std::string seed_nbrs_path = output_prefix + "_neighborcache_dc.seed_nbrs";
    const std::string seed_member_locs_path = output_prefix + "_neighborcache_dc.seed_member_locs";
    std::vector<std::pair<uint32_t, uint8_t>> seed_pages_sidecar;
    if (load_seed_pages_sidecar(seed_pages_path, seed_pages_sidecar) != 0)
        return -1;
    if (load_seed_nbrs_sidecar(seed_nbrs_path, _neighborcache_dc_seed_page_nbrs) == 0 && !_neighborcache_dc_seed_page_nbrs.empty())
    {
        _neighborcache_seed_only_expand =
            (std::getenv("NEIGHBORCACHE_SEED_ONLY_EXPAND") != nullptr &&
             std::strcmp(std::getenv("NEIGHBORCACHE_SEED_ONLY_EXPAND"), "0") != 0);
        if (std::getenv("NEIGHBORCACHE_SEED_ONLY_LOOKUP") != nullptr &&
            std::strcmp(std::getenv("NEIGHBORCACHE_SEED_ONLY_LOOKUP"), "0") != 0)
        {
            _neighborcache_seed_only_layout = true;
            std::unordered_set<uint32_t> seed_ids;
            seed_ids.reserve(_neighborcache_dc_seed_page_nbrs.size());
            for (const auto &kv : _neighborcache_dc_seed_page_nbrs)
                seed_ids.insert(kv.first);
            tsl::robin_map<uint32_t, std::vector<NeighborCacheDiskLoc>> seed_only_map;
            seed_only_map.reserve(seed_ids.size());
            for (const auto &kv : _neighborcache_dc_map)
            {
                if (seed_ids.count(kv.first) > 0)
                    seed_only_map[kv.first] = kv.second;
            }
            _neighborcache_dc_map = std::move(seed_only_map);
        }
    }
    if (load_seed_member_locs_sidecar(seed_member_locs_path, _neighborcache_dc_seed_member_loc, _neighborcache_dc_seed_canonical_loc) !=
        0)
    {
        diskann::cerr << "NEIGHBORCACHE disk-cache reuse: failed to load " << seed_member_locs_path << std::endl;
        return -1;
    }
    for (const auto &entry : seed_pages_sidecar)
    {
        const uint32_t seed_id = entry.first;
        const uint8_t total_pages = entry.second;
        const auto member_it = _neighborcache_dc_seed_member_loc.find(seed_id);
        if (member_it == _neighborcache_dc_seed_member_loc.end() || member_it->second.empty())
            continue;

        uint32_t first_sector = std::numeric_limits<uint32_t>::max();
        for (const auto &member : member_it->second)
            first_sector = std::min(first_sector, member.second.sector);
        if (first_sector != std::numeric_limits<uint32_t>::max())
        {
            NeighborCacheDiskLoc canonical;
            canonical.sector = first_sector;
            canonical.slot = 0;
            canonical.nsectors = total_pages;
            _neighborcache_dc_seed_canonical_loc[seed_id] = canonical;
        }
    }
    for (const auto &seed_members : _neighborcache_dc_seed_member_loc)
    {
        if (seed_members.second.find(seed_members.first) == seed_members.second.end())
        {
            _neighborcache_child_only_layout = true;
            break;
        }
    }
    const bool record_driven_seed_access =
        (std::getenv("NEIGHBORCACHE_RECORD_DRIVEN_SEED_ACCESS") != nullptr &&
         std::strcmp(std::getenv("NEIGHBORCACHE_RECORD_DRIVEN_SEED_ACCESS"), "0") != 0);
    // Seed-group pages are padded with seed replicas, so the sequential node-list
    // locations above are wrong for them; the member-loc sidecar has the real ones.
    if (_neighborcache_child_only_layout || record_driven_seed_access || !_neighborcache_dc_seed_member_loc.empty())
    {
        _neighborcache_dc_map.clear();
        for (const auto &seed_members : _neighborcache_dc_seed_member_loc)
        {
            for (const auto &member : seed_members.second)
                _neighborcache_dc_map[member.first].push_back(member.second);
        }
    }
    update_neighborcache_seed_first_lookup_flag();

    _neighborcache_dc_path = data_path;
    _neighborcache_dc_num_nodes = node_list.size();

#ifndef _WINDOWS
    _neighborcache_disk_reader.reset(new LinuxAlignedFileReader());
#else
    _neighborcache_disk_reader.reset(new WindowsAlignedFileReader());
#endif
    _neighborcache_disk_reader->open(data_path);
#pragma omp parallel for num_threads((int)_max_nthreads)
    for (int64_t thread = 0; thread < (int64_t)_max_nthreads; thread++)
    {
#pragma omp critical
        {
            _neighborcache_disk_reader->register_thread();
        }
    }

    diskann::cout << "NEIGHBORCACHE disk-cache loaded (reuse): " << _neighborcache_dc_num_nodes << " nodes -> " << data_path
                  << " (map size=" << _neighborcache_dc_map.size()
                  << ", seed-first lookup=" << (_neighborcache_seed_first_lookup ? "on" : "off") << ")" << std::endl;
    return 0;
}

template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::build_and_load_neighborcache_disk_cache(const std::string &profile_prefix, uint64_t max_nodes,
                                                             const std::string &output_prefix, uint64_t rank_skip,
                                                             bool unified_single_file, uint32_t k_hops,
                                                             const std::string &layout)
{
    if (_neighborcache_disk_reader)
    {
        _neighborcache_disk_reader->close();
        _neighborcache_disk_reader.reset();
    }
    _neighborcache_unified_disk = false;
    _neighborcache_region_byte_offset = 0;
    _neighborcache_dc_map.clear();
    _neighborcache_dc_seed_page_nbrs.clear();
    _neighborcache_dc_seed_member_loc.clear();
    _neighborcache_dc_seed_canonical_loc.clear();
    _neighborcache_seed_only_layout = false;
    _neighborcache_seed_only_expand = false;
    _neighborcache_seed_first_lookup = false;
    _neighborcache_child_only_layout = false;
    _neighborcache_dc_num_nodes = 0;
    _neighborcache_dc_path.clear();

    if (max_nodes == 0)
        return 0;

    std::vector<uint32_t> node_list;
    std::vector<SeedPageGroup> seed_groups;
    const std::string layout_norm = normalize_disk_cache_layout(layout);
    _neighborcache_child_only_layout =
        is_directed_child_only_layout(layout_norm) || is_directed_child_replica_layout(layout_norm);
    _neighborcache_seed_only_layout = (std::getenv("NEIGHBORCACHE_SEED_ONLY_LOOKUP") != nullptr &&
                               std::strcmp(std::getenv("NEIGHBORCACHE_SEED_ONLY_LOOKUP"), "0") != 0);
    _neighborcache_seed_only_expand =
        (std::getenv("NEIGHBORCACHE_SEED_ONLY_EXPAND") != nullptr &&
         std::strcmp(std::getenv("NEIGHBORCACHE_SEED_ONLY_EXPAND"), "0") != 0);
    std::vector<SeedPageGroup> *seed_groups_ptr =
        (is_directed_beam_pct_layout(layout_norm) || is_directed_child_only_layout(layout_norm) ||
         is_directed_child_replica_layout(layout_norm) || is_directed_seed_replica_layout(layout_norm) ||
         diskann::is_directed_seed_replica_benefit_layout(layout_norm) ||
         diskann::is_directed_seed_replica_budget_layout(layout_norm) ||
         diskann::is_directed_seed_core_unique_fill_layout(layout_norm))
            ? &seed_groups
            : nullptr;
    if (build_neighborcache_disk_node_list(profile_prefix, max_nodes, rank_skip, k_hops, layout, node_list, seed_groups_ptr) !=
        0)
        return -1;

    const std::string data_path =
        unified_single_file ? (output_prefix + "_disk_neighborcache_unified.index") : (output_prefix + "_neighborcache_dc.data");
    const std::string nodes_path = output_prefix + "_neighborcache_dc.nodes";
    const std::string seed_pages_path = output_prefix + "_neighborcache_dc.seed_pages";
    const std::string seed_nbrs_path = output_prefix + "_neighborcache_dc.seed_nbrs";
    const std::string seed_member_locs_path = output_prefix + "_neighborcache_dc.seed_member_locs";

    // Persist node order for debugging / reload.
    diskann::save_bin<uint32_t>(nodes_path, node_list.data(), node_list.size(), 1);

    if (unified_single_file)
    {
        if (copy_file_binary(_disk_index_file, data_path) != 0)
        {
            diskann::cerr << "Failed to copy base index into unified NEIGHBORCACHE file: " << data_path << std::endl;
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
        _neighborcache_region_byte_offset = aligned_sz;
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
        diskann::cout << "NEIGHBORCACHE disk-cache space budget: sectors<=" << space_budget_sectors << " ("
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
        diskann::cerr << "Failed to open NEIGHBORCACHE disk-cache for write: " << data_path << std::endl;
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
            write_neighborcache_dc_seed_tail(slot_dst, _max_node_len, _disk_bytes_per_point, nnbrs, seed_total_pages);

        if (index_in_map)
        {
            NeighborCacheDiskLoc loc;
            loc.sector = sector_id;
            loc.slot = slot_id;
            loc.nsectors = nsectors;
            _neighborcache_dc_map[node_id].push_back(loc);
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

        const std::vector<std::pair<uint32_t, uint32_t>> copack_pairs = load_neighborcache_copack_pairs();
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
                NeighborCacheDiskLoc canonical;
                canonical.sector = page_sector;
                canonical.slot = 0;
                canonical.nsectors = group.total_pages;
                _neighborcache_dc_seed_canonical_loc[group.seed] = canonical;
            }
            for (size_t si = 0; si < page_nodes.size(); ++si)
            {
                const uint32_t member_id = page_nodes[si];
                NeighborCacheDiskLoc member_loc;
                member_loc.sector = page_sector;
                member_loc.slot = static_cast<uint16_t>(si);
                member_loc.nsectors = (pi == 0 && member_id == group.seed) ? group.total_pages : 1;
                _neighborcache_dc_seed_member_loc[group.seed][member_id] = member_loc;
                if (member_id == group.seed && pi == 0)
                    _neighborcache_dc_seed_canonical_loc[group.seed] = member_loc;
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
                                      << " for NEIGHBORCACHE disk-cache (pct80 group)." << std::endl;
                        return -1;
                    }
                    const uint32_t node_id = batch[i];
                    const uint16_t slot_id = static_cast<uint16_t>(begin + i);
                    const bool is_seed = (pi == 0 && node_id == group.seed);
                    const bool index_in_map = !_neighborcache_seed_only_layout || is_seed;
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
                    NeighborCacheDiskLoc member_loc;
                    member_loc.sector = page_sector;
                    member_loc.slot = static_cast<uint16_t>(si);
                    member_loc.nsectors = 1;
                    _neighborcache_dc_seed_member_loc[group.seed][member_id] = member_loc;
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
                                          << " for NEIGHBORCACHE disk-cache (pct cold tail)." << std::endl;
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
            _neighborcache_dc_seed_page_nbrs[group.seed] = std::move(page_nbrs);
        };

        auto apply_copack_canonical = [&](uint32_t seed_a, uint32_t seed_b, uint32_t extent_base) {
            NeighborCacheDiskLoc canon;
            canon.sector = extent_base;
            canon.slot = 0;
            canon.nsectors = 2;
            _neighborcache_dc_seed_canonical_loc[seed_a] = canon;
            _neighborcache_dc_seed_canonical_loc[seed_b] = canon;
            auto fix_seed = [&](uint32_t seed_id) {
                const auto outer = _neighborcache_dc_seed_member_loc.find(seed_id);
                if (outer == _neighborcache_dc_seed_member_loc.end())
                    return;
                const auto inner = outer->second.find(seed_id);
                if (inner == outer->second.end())
                    return;
                NeighborCacheDiskLoc updated = inner->second;
                updated.nsectors = 2;
                _neighborcache_dc_seed_member_loc[seed_id][seed_id] = updated;
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

        diskann::cout << "NEIGHBORCACHE disk-cache seed-group write: groups_written=" << groups_written
                      << " groups_skipped_for_space_budget=" << groups_skipped_budget
                      << " sectors=" << cur_sector << "/" << space_budget_sectors
                      << " padded_slots_emitted=" << group_nodes_emitted << std::endl;

        save_seed_pages_sidecar(seed_pages_path, seed_pages_sidecar);
        if (!_neighborcache_dc_seed_page_nbrs.empty())
            save_seed_nbrs_sidecar(seed_nbrs_path, _neighborcache_dc_seed_page_nbrs);
        if (!_neighborcache_dc_seed_member_loc.empty())
            save_seed_member_locs_sidecar(seed_member_locs_path, _neighborcache_dc_seed_member_loc);
        update_neighborcache_seed_first_lookup_flag();
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
        NeighborCacheDiskLoc loc;
        loc.sector = cur_sector;
        loc.slot = 0;
        loc.nsectors = static_cast<uint16_t>(secs_per_node);
        _neighborcache_dc_map[node_id].push_back(loc);
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
                diskann::cerr << "Failed to read base node " << batch[i] << " for NEIGHBORCACHE disk-cache." << std::endl;
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
        (!_neighborcache_dc_seed_member_loc.empty() && flat_write_begin >= node_list.size() && cur_sector > 0);
    if (skip_dense_budget_fill)
    {
        diskann::cout << "NEIGHBORCACHE disk-cache: skip dense budget fill after seed-groups "
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
                                  << " while padding NEIGHBORCACHE disk-cache to space budget." << std::endl;
                    return -1;
                }
                write_flat_node(batch[i], coord_ptrs[i], nbr_ptrs[i].first, nbr_ptrs[i].second);
                ++replica_slots;
            }
        }
        if (replica_slots > 0)
        {
            diskann::cout << "NEIGHBORCACHE disk-cache space fill: wrote " << replica_slots
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

    diskann::cout << "NEIGHBORCACHE disk-cache packed file sectors=" << cur_sector
                  << " bytes=" << (static_cast<uint64_t>(cur_sector) * defaults::SECTOR_LEN)
                  << " space_budget_sectors=" << space_budget_sectors << std::endl;

    _neighborcache_dc_path = data_path;
    _neighborcache_dc_num_nodes = node_list.size();

    if (unified_single_file)
    {
        _neighborcache_unified_disk = true;
        reader->close();
        reader->open(data_path);
        _disk_index_file = data_path;
        diskann::cout << "NEIGHBORCACHE unified disk-cache ready: " << _neighborcache_dc_num_nodes << " nodes appended at offset "
                      << (_neighborcache_region_byte_offset / (1024.0 * 1024.0)) << " MiB -> " << data_path
                      << " (map size=" << _neighborcache_dc_map.size() << ", seed-first lookup="
                      << (_neighborcache_seed_first_lookup ? "on" : "off") << ", single-fd merged IO)" << std::endl;
        return 0;
    }

#ifndef _WINDOWS
    _neighborcache_disk_reader.reset(new LinuxAlignedFileReader());
#else
    _neighborcache_disk_reader.reset(new WindowsAlignedFileReader());
#endif
    _neighborcache_disk_reader->open(data_path);
#pragma omp parallel for num_threads((int)_max_nthreads)
    for (int64_t thread = 0; thread < (int64_t)_max_nthreads; thread++)
    {
#pragma omp critical
        {
            _neighborcache_disk_reader->register_thread();
        }
    }

    diskann::cout << "NEIGHBORCACHE disk-cache ready: " << _neighborcache_dc_num_nodes << " nodes -> " << data_path
                  << " (map size=" << _neighborcache_dc_map.size() << ", seed-first lookup="
                  << (_neighborcache_seed_first_lookup ? "on" : "off");
    if (_neighborcache_seed_only_layout || _neighborcache_seed_only_expand)
        diskann::cout << " (seed-only lookup=" << (_neighborcache_seed_only_layout ? "on" : "off")
                      << ", page-local expand=" << (_neighborcache_seed_only_expand ? "on" : "off") << ")";
    diskann::cout << ")" << std::endl;
    return 0;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::update_neighborcache_seed_first_lookup_flag()
{
    const bool record_driven_seed_access =
        (std::getenv("NEIGHBORCACHE_RECORD_DRIVEN_SEED_ACCESS") != nullptr &&
         std::strcmp(std::getenv("NEIGHBORCACHE_RECORD_DRIVEN_SEED_ACCESS"), "0") != 0);
    const char *env = std::getenv("NEIGHBORCACHE_SEED_FIRST_LOOKUP");
    if (env != nullptr)
    {
        _neighborcache_seed_first_lookup = (std::strcmp(env, "0") != 0);
    }
    else
    {
        _neighborcache_seed_first_lookup = !_neighborcache_dc_seed_member_loc.empty();
    }
    if (record_driven_seed_access)
    {
        _neighborcache_seed_first_lookup = true;
        diskann::cout << "NEIGHBORCACHE record-driven self-seed access: on" << std::endl;
    }
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_loc_in_query_cache(const SSDQueryScratch<T> *query_scratch,
                                                        uint32_t base_sector,
                                                        uint16_t nsectors) const
{
    if (!_query_sector_cache_enabled || query_scratch == nullptr || nsectors == 0)
        return false;
    if (neighborcache_use_shared_cache(base_sector))
        return _neighborcache_shared_cache.contains_range(base_sector, nsectors);
    for (uint16_t si = 0; si < nsectors; ++si)
    {
        if (query_scratch->neighborcache_sector_cache.find(base_sector + static_cast<uint32_t>(si)) ==
            query_scratch->neighborcache_sector_cache.end())
            return false;
    }
    return true;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_resolve_disk_cache_loc(uint32_t node_id,
                                                           const SSDQueryScratch<T> *query_scratch,
                                                           NeighborCacheDiskLoc &out_loc) const
{
    // The dynamic location directory is independently sharded. Publishing or
    // retiring a page takes the affected location shard locks, so lookup does
    // not need the global page-allocation mutex.
    if (_neighborcache_dyn_enabled)
    {
        if (g_neighborcache_demand_probe != nullptr)
            g_neighborcache_demand_probe->attempted = true;
        g_dyn_probe.resolve_call++;
        NeighborCacheDiskLoc canon_loc;
        const int canon = neighborcache_canon_lookup(node_id, canon_loc);
        if (canon == 1 && neighborcache_dyn_is_sector(canon_loc.sector))
        {
            const uint32_t idx = canon_loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
            if (neighborcache_dyn_page_live(idx))
            {
                out_loc = canon_loc;
                g_dyn_probe.map_hit++;
                return neighborcache_demand_probe_hit();
            }
        }
        else if (canon == 0)
        {
            g_dyn_probe.resolve_miss++;
            return false;
        }
        auto loc_from_dir = [&](uint32_t seed, uint32_t member) -> bool {
            NeighborCacheTimedSharedMutexGuard page_lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
            uint32_t page_idx = NEIGHBORCACHE_DYN_INVALID_PAGE;
            if (!neighborcache_dyn_seed_page_unlocked(seed, page_idx))
                return false;
            const size_t count = _neighborcache_dyn_page_member_count[page_idx];
            const size_t begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
            for (size_t k = 0; k < count; k++)
            {
                if (_neighborcache_dyn_page_members[begin + k] != member)
                    continue;
                out_loc.sector = NEIGHBORCACHE_DYN_SECTOR_BASE + page_idx;
                out_loc.slot = static_cast<uint16_t>(k);
                out_loc.nsectors = 1;
                return neighborcache_demand_probe_hit();
            }
            return false;
        };
        // Default: loc-map, then self-seed page. Parent page only if
        // NEIGHBORCACHE_RESOLVE_PARENT=1 (parent-first hurt the small dynamic d-cache).
        {
            bool found = false;
            neighborcache_dc_for_each_loc_unlocked(node_id, [&](const NeighborCacheDiskLoc &loc, size_t) {
                if (found)
                    return;
                if (!neighborcache_dyn_is_sector(loc.sector))
                    return;
                const uint32_t idx = loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
                if (neighborcache_dyn_page_live(idx))
                {
                    out_loc = loc;
                    g_dyn_probe.map_hit++;
                    found = true;
                }
            });
            if (found)
                return neighborcache_demand_probe_hit();
        }
        if (loc_from_dir(node_id, node_id))
        {
            g_dyn_probe.self_hit++;
            return neighborcache_demand_probe_hit();
        }
        static const bool resolve_parent =
            neighborcache_env_u64("NEIGHBORCACHE_RESOLVE_PARENT", 0) != 0;
        if (resolve_parent && query_scratch != nullptr)
        {
            const auto parent_it = query_scratch->parent_of(node_id);
            if (!parent_it.has_value())
                g_dyn_probe.parent_absent++;
            else if (loc_from_dir(parent_it.value(), node_id))
            {
                g_dyn_probe.parent_hit++;
                return neighborcache_demand_probe_hit();
            }
            else
            {
                uint32_t parent_page = NEIGHBORCACHE_DYN_INVALID_PAGE;
                if (!neighborcache_dyn_seed_page_unlocked(parent_it.value(), parent_page))
                    g_dyn_probe.parent_no_page++;
                else
                    g_dyn_probe.parent_not_member++;
            }
        }
        else if (query_scratch == nullptr ||
                 !query_scratch->parent_of(node_id).has_value())
            g_dyn_probe.parent_absent++;
        g_dyn_probe.resolve_miss++;
        return false;
    }
    if (_neighborcache_dc_seed_member_loc.empty())
        return false;

    const bool record_driven_seed_access =
        (std::getenv("NEIGHBORCACHE_RECORD_DRIVEN_SEED_ACCESS") != nullptr &&
         std::strcmp(std::getenv("NEIGHBORCACHE_RECORD_DRIVEN_SEED_ACCESS"), "0") != 0);
    if (record_driven_seed_access)
    {
        if (query_scratch != nullptr)
        {
            const auto map_it = _neighborcache_dc_map.find(node_id);
            if (map_it != _neighborcache_dc_map.end())
            {
                for (const NeighborCacheDiskLoc &loc : map_it->second)
                {
                    if (neighborcache_loc_in_query_cache(query_scratch, loc.sector, loc.nsectors))
                    {
                        out_loc = loc;
                        return true;
                    }
                }
            }
        }
        const auto seed_it = _neighborcache_dc_seed_member_loc.find(node_id);
        const bool own_extent_contains_record =
            seed_it != _neighborcache_dc_seed_member_loc.end() && seed_it->second.find(node_id) != seed_it->second.end();
        const auto canonical_it = _neighborcache_dc_seed_canonical_loc.find(node_id);
        if (own_extent_contains_record && canonical_it != _neighborcache_dc_seed_canonical_loc.end())
        {
            out_loc = canonical_it->second;
            return true;
        }
        const auto map_it = _neighborcache_dc_map.find(node_id);
        if (map_it != _neighborcache_dc_map.end() && !map_it->second.empty())
        {
            out_loc = map_it->second.front();
            return true;
        }
        return false;
    }

    const bool cache_replica_fallback =
        (std::getenv("NEIGHBORCACHE_CACHE_REPLICA_FALLBACK") != nullptr &&
         std::strcmp(std::getenv("NEIGHBORCACHE_CACHE_REPLICA_FALLBACK"), "0") != 0);
    if (cache_replica_fallback && query_scratch != nullptr)
    {
        const auto map_it = _neighborcache_dc_map.find(node_id);
        if (map_it != _neighborcache_dc_map.end())
        {
            for (const NeighborCacheDiskLoc &loc : map_it->second)
            {
                if (neighborcache_loc_in_query_cache(query_scratch, loc.sector, loc.nsectors))
                {
                    out_loc = loc;
                    return true;
                }
            }
        }
    }

    // Parent-seed first, then self-seed: for path A->B, look up B in A's seed
    // group; if absent, try B's own seed group when B is a seed; otherwise miss
    // and fall through to the base index.
    uint32_t miss_parent = std::numeric_limits<uint32_t>::max();
    if (query_scratch != nullptr)
    {
        const auto parent_it = query_scratch->parent_of(node_id);
        if (parent_it.has_value())
        {
            const uint32_t parent_seed = parent_it.value();
            miss_parent = parent_seed;
            const auto member_it = _neighborcache_dc_seed_member_loc.find(parent_seed);
            if (member_it != _neighborcache_dc_seed_member_loc.end())
            {
                const auto loc_it = member_it->second.find(node_id);
                if (loc_it != member_it->second.end())
                {
                    out_loc = loc_it->second;
                    return true;
                }
            }
            if (_neighborcache_child_only_layout)
            {
                const auto global_it = _neighborcache_dc_map.find(node_id);
                if (global_it != _neighborcache_dc_map.end() && !global_it->second.empty())
                {
                    out_loc = global_it->second.front();
                    return true;
                }
            }
            // Parent seed miss: fall through to self-seed below.
        }
    }

    // Self-seed fallback (also covers the medoid, which has no graph parent).
    const auto seed_it = _neighborcache_dc_seed_member_loc.find(node_id);
    if (seed_it != _neighborcache_dc_seed_member_loc.end())
    {
        const auto self_it = seed_it->second.find(node_id);
        if (self_it != seed_it->second.end())
        {
            out_loc = self_it->second;
            return true;
        }
    }

    if (g_neighborcache_miss_diag)
    {
        const bool in_dcache = _neighborcache_dc_map.find(node_id) != _neighborcache_dc_map.end();
        if (!in_dcache)
            g_neighborcache_miss_not_in_dcache.fetch_add(1, std::memory_order_relaxed);
        else if (miss_parent == std::numeric_limits<uint32_t>::max())
            g_neighborcache_miss_no_parent.fetch_add(1, std::memory_order_relaxed);
        else
        {
            const bool parent_is_seed = _neighborcache_dc_seed_member_loc.find(miss_parent) != _neighborcache_dc_seed_member_loc.end();
            const bool parent_in_ncache = _neighborcache_mem_pool != nullptr && _neighborcache_mem_pool->active() &&
                                          _neighborcache_mem_pool->contains(miss_parent);
            if (parent_in_ncache)
                g_neighborcache_miss_parent_ncache.fetch_add(1, std::memory_order_relaxed);
            else if (!parent_is_seed)
                g_neighborcache_miss_parent_not_seed.fetch_add(1, std::memory_order_relaxed);
            else
                g_neighborcache_miss_parent_lacks.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return false;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_dyn_query_local_dcache_hit(uint32_t node_id,
                                                               SSDQueryScratch<T> *query_scratch) const
{
    if (query_scratch == nullptr)
        return false;
    const auto it = query_scratch->neighborcache_dcache_member_packed_loc.find(node_id);
    if (it == query_scratch->neighborcache_dcache_member_packed_loc.end())
        return false;
    query_scratch->neighborcache_dcache_loc_cache[node_id] = it->second;
    if (_neighborcache_span_adaptive)
    {
        NeighborCacheTimedSharedMutexGuard page_lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        const uint32_t sector = static_cast<uint32_t>(it->second >> 32);
        if (neighborcache_dyn_is_sector(sector))
        {
            const uint32_t page = sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
            if (page < _neighborcache_dyn_page_secondary.size() && _neighborcache_dyn_page_secondary[page])
            {
                const uint32_t sb = _neighborcache_dyn_page_span_base[page];
                if (sb < _neighborcache_dyn_physical_cap &&
                    query_scratch->neighborcache_dyn_full_span.find(sb) != query_scratch->neighborcache_dyn_full_span.end())
                    _neighborcache_span_hits[sb].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    return true;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_pack_dyn_page_into_query_prefetch(uint32_t sector,
                                                                       SSDQueryScratch<T> *query_scratch) const
{
    if (query_scratch == nullptr || !neighborcache_dyn_is_sector(sector) || _neighborcache_dyn_members_per_page == 0)
        return;
    const uint32_t page_idx_fast = sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
    if (!_neighborcache_dcache_second_page && neighborcache_query_page_prefetch_current(query_scratch, page_idx_fast))
        return;
    NeighborCacheTimedSharedMutexGuard page_lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
    const uint32_t page_idx = sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
    if (page_idx >= _neighborcache_dyn_physical_cap || page_idx >= _neighborcache_dyn_page_member_count.size())
        return;
    uint32_t base = page_idx;
    uint32_t span = 1;
    if (_neighborcache_dcache_second_page && page_idx < _neighborcache_dyn_page_span_len.size())
    {
        const uint8_t sl = _neighborcache_dyn_page_span_len[page_idx];
        const uint32_t sb = page_idx < _neighborcache_dyn_page_span_base.size() ? _neighborcache_dyn_page_span_base[page_idx]
                                                                        : NEIGHBORCACHE_DYN_INVALID_PAGE;
        if (sl > 1 && sb != NEIGHBORCACHE_DYN_INVALID_PAGE &&
            static_cast<uint64_t>(sb) + sl <= _neighborcache_dyn_physical_cap)
        {
            base = sb;
            span = sl;
        }
    }
    bool full = true;
    if (span > 1)
    {
        const bool known = query_scratch->neighborcache_dyn_full_span.find(base) != query_scratch->neighborcache_dyn_full_span.end();
        full = known || neighborcache_span_should_read_full(base, span);
        if (full)
            query_scratch->neighborcache_dyn_full_span.insert(base);
    }
    auto collect_page = [&](uint32_t page, std::vector<uint32_t> &members) {
        if (!neighborcache_dyn_page_live(page))
            return;
        const size_t begin = static_cast<size_t>(page) * _neighborcache_dyn_members_per_page;
        const size_t count = _neighborcache_dyn_page_member_count[page];
        if (count == 0 || begin + count > _neighborcache_dyn_page_members.size())
            return;
        members.assign(_neighborcache_dyn_page_members.begin() + begin, _neighborcache_dyn_page_members.begin() + begin + count);
    };
    auto pack_page = [&](uint32_t page, std::vector<uint32_t> &members) {
        if (!neighborcache_dyn_page_live(page))
            return;
        const size_t begin = static_cast<size_t>(page) * _neighborcache_dyn_members_per_page;
        const size_t count = _neighborcache_dyn_page_member_count[page];
        if (count == 0 || begin + count > _neighborcache_dyn_page_members.size())
            return;
        const uint32_t page_sector = NEIGHBORCACHE_DYN_SECTOR_BASE + page;
        members.reserve(count);
        for (size_t mi = 0; mi < count; ++mi)
        {
            const uint32_t mid = _neighborcache_dyn_page_members[begin + mi];
            if (mid == NEIGHBORCACHE_DYN_INVALID_PAGE)
                continue;
            const uint64_t packed = (static_cast<uint64_t>(page_sector) << 32) | (static_cast<uint64_t>(1) << 16) |
                                    static_cast<uint64_t>(mi);
            query_scratch->neighborcache_dcache_member_packed_loc[mid] = packed;
            members.push_back(mid);
        }
    };
    std::vector<std::vector<uint32_t>> packed(span);
    for (uint32_t i = 0; i < span; ++i)
    {
        if (full || base + i == page_idx)
        {
            pack_page(base + i, packed[i]);
            neighborcache_query_page_prefetch_remember(query_scratch, base + i);
        }
        else
            collect_page(base + i, packed[i]);
    }
    if (!_neighborcache_dcache_query_patch || packed[0].empty())
        return;
    const uint32_t seed_id = packed[0].front();
    if (seed_id == NEIGHBORCACHE_DYN_INVALID_PAGE)
        return;
    if (query_scratch->neighborcache_dcache_seed_members.find(seed_id) != query_scratch->neighborcache_dcache_seed_members.end())
        return;
    for (uint32_t i = 0; i < span; ++i)
    {
        if (!full && base + i != page_idx)
            continue;
        for (uint32_t mid : packed[i])
            query_scratch->neighborcache_dcache_member_to_seed[mid] = seed_id;
    }
    query_scratch->neighborcache_dcache_seed_members[seed_id] = std::move(packed[0]);
    std::vector<std::vector<uint32_t>> extra;
    for (uint32_t i = 1; i < span; ++i)
    {
        if (packed[i].size() >= 2)
            extra.push_back(std::move(packed[i]));
    }
    if (!extra.empty())
        query_scratch->neighborcache_dcache_seed_extra[seed_id] = std::move(extra);
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_disk_cache_lookup_hit(uint32_t node_id,
                                                          SSDQueryScratch<T> *query_scratch) const
{
    if (_neighborcache_dyn_enabled)
    {
        const bool lookup_enabled =
            query_scratch != nullptr
                ? query_scratch->neighborcache_dcache_lookup_enabled
                : (_neighborcache_dcache_net_gate_enabled
                       ? _neighborcache_dcache_gate_active.load(std::memory_order_acquire)
                       : !_neighborcache_mcache.maintenance_suspended());
        if (!lookup_enabled)
            return false;
        if (neighborcache_dyn_query_local_dcache_hit(node_id, query_scratch))
            return true;
    }
    if (_neighborcache_dyn_enabled || _neighborcache_seed_first_lookup)
    {
        if (!_neighborcache_dyn_member_bits.empty() && static_cast<uint64_t>(node_id) < _num_points)
        {
            const uint64_t word =
                _neighborcache_dyn_member_bits[static_cast<size_t>(node_id >> 6)];
            if ((word & (1ULL << (node_id & 63u))) == 0)
                return false;
        }
        NeighborCacheDiskLoc loc;
        if (!neighborcache_resolve_disk_cache_loc(node_id, query_scratch, loc))
            return false;
        if (query_scratch != nullptr)
        {
            const uint64_t packed =
                (static_cast<uint64_t>(loc.sector) << 32) |
                (static_cast<uint64_t>(loc.nsectors) << 16) |
                static_cast<uint64_t>(loc.slot);
            query_scratch->neighborcache_dcache_loc_cache[node_id] = packed;
        }
        return true;
    }

    NeighborCacheTimedSharedMutexGuard lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
    const auto it = _neighborcache_dc_map.find(node_id);
    return it != _neighborcache_dc_map.end() && !it->second.empty();
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_get_expand_neighbors(uint32_t expand_id, char *node_disk_buf,
                                                         const uint32_t *&out_nbrs, uint64_t &out_nnbrs) const
{
    if (_neighborcache_seed_only_expand)
    {
        const auto it = _neighborcache_dc_seed_page_nbrs.find(expand_id);
        if (it != _neighborcache_dc_seed_page_nbrs.end() && !it->second.empty())
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

struct NeighborCacheSectorSpan
{
    uint32_t base = 0;
    uint16_t nsectors = 1;
};

static std::vector<NeighborCacheSectorSpan> coalesce_sectors_to_spans(std::vector<uint32_t> sectors)
{
    if (sectors.empty())
        return {};
    std::sort(sectors.begin(), sectors.end());
    sectors.erase(std::unique(sectors.begin(), sectors.end()), sectors.end());
    std::vector<NeighborCacheSectorSpan> spans;
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
void PQFlashIndex<T, LabelT>::prepare_neighborcache_disk_cache_io(
    const std::vector<uint32_t> &neighborcache_ids_in, SSDQueryScratch<T> *query_scratch, char *sector_scratch,
    uint64_t &sector_scratch_idx, size_t num_sectors_per_node, std::vector<NeighborCacheReadPending> &pending,
    std::vector<AlignedRead> &neighborcache_io, std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout_groups,
    QueryStats *stats, uint32_t &num_ios)
{
    pending.clear();
    neighborcache_io.clear();
    disk_fanout_groups.clear();
    if (neighborcache_ids_in.empty())
        return;

    static const bool disable_multiread = (std::getenv("NEIGHBORCACHE_DISABLE_MULTIREAD") != nullptr);
    static const bool record_driven_seed_access =
        (std::getenv("NEIGHBORCACHE_RECORD_DRIVEN_SEED_ACCESS") != nullptr &&
         std::strcmp(std::getenv("NEIGHBORCACHE_RECORD_DRIVEN_SEED_ACCESS"), "0") != 0);
    static const bool adaptive_parent_or_self =
        record_driven_seed_access && std::getenv("NEIGHBORCACHE_ADAPTIVE_PARENT_OR_SELF") != nullptr &&
        std::strcmp(std::getenv("NEIGHBORCACHE_ADAPTIVE_PARENT_OR_SELF"), "0") != 0;

    // Fast path: a node reached from parent p is replicated on p's seed page, so look it up
    // there and serve it in place from the shared page cache. Only the misses go through
    // replica selection and IO planning below.
    const std::vector<uint32_t> *ids = &neighborcache_ids_in;
    thread_local std::vector<uint32_t> slow_ids;
    if (_neighborcache_seed_first_lookup && !record_driven_seed_access && !_neighborcache_dyn_enabled && query_scratch != nullptr &&
        _query_sector_cache_enabled && _neighborcache_shared_cache.enabled() && _nnodes_per_sector > 0)
    {
        slow_ids.clear();
        for (uint32_t id : neighborcache_ids_in)
        {
            const auto parent_it = query_scratch->parent_of(id);
            const uint32_t seed = parent_it.has_value() ? parent_it.value() : id;
            const char *np = nullptr;
            NeighborCacheDiskLoc loc{};
            const auto members_it = _neighborcache_dc_seed_member_loc.find(seed);
            if (members_it != _neighborcache_dc_seed_member_loc.end())
            {
                const auto member_it = members_it->second.find(id);
                if (member_it != members_it->second.end() && neighborcache_use_shared_cache(member_it->second.sector))
                {
                    loc = member_it->second;
                    char *spill = sector_scratch + sector_scratch_idx * defaults::SECTOR_LEN;
                    bool spilled = false;
                    np = neighborcache_shared_node_ptr(loc.sector, loc.slot, spill, spilled);
                    if (spilled)
                        sector_scratch_idx++;
                }
            }
            if (np == nullptr)
            {
                slow_ids.push_back(id);
                continue;
            }
            NeighborCacheReadPending mp;
            mp.id = id;
            mp.loc = loc;
            mp.io_base_sector = loc.sector;
            mp.io_nsectors = 1;
            mp.from_query_cache = true;
            mp.node_ptr = np;
            pending.push_back(mp);
            if (stats != nullptr)
            {
                stats->n_sector_cache_hits++;
                stats->n_neighborcache_sector_cache_hits++;
                if (query_scratch->neighborcache_fetched_sectors.find(loc.sector) ==
                    query_scratch->neighborcache_fetched_sectors.end())
                    stats->n_neighborcache_shared_cross_hits++;
            }
        }
        if (slow_ids.empty())
            return;
        ids = &slow_ids;
    }
    const std::vector<uint32_t> &neighborcache_ids = *ids;

    const size_t read_len = num_sectors_per_node * defaults::SECTOR_LEN;

    // Batch replica selection: greedy set-cover on sectors to minimize distinct reads.
    struct PendingChoice
    {
        uint32_t id = 0;
        size_t loc_idx = 0;
        NeighborCacheDiskLoc loc;
        uint32_t io_base_sector = 0;
        uint16_t io_nsectors = 1;
        bool dynamic_page_pinned = false;
    };
    std::vector<PendingChoice> choices;
    choices.reserve(neighborcache_ids.size());
    std::unordered_set<uint32_t> chosen_ids;
    std::vector<uint32_t> unresolved;
    unresolved.reserve(neighborcache_ids.size());

    if (_neighborcache_seed_first_lookup &&
        !(_neighborcache_dyn_enabled && _neighborcache_dc_seed_member_loc.empty() && _neighborcache_dc_seed_canonical_loc.empty()))
    {
        static const bool cache_replica_fallback =
            (std::getenv("NEIGHBORCACHE_CACHE_REPLICA_FALLBACK") != nullptr &&
             std::strcmp(std::getenv("NEIGHBORCACHE_CACHE_REPLICA_FALLBACK"), "0") != 0);
        static const bool adaptive_extent =
            (std::getenv("NEIGHBORCACHE_ADAPTIVE_EXTENT") != nullptr &&
             std::strcmp(std::getenv("NEIGHBORCACHE_ADAPTIVE_EXTENT"), "0") != 0);
        static const bool parent_single_page =
            (std::getenv("NEIGHBORCACHE_PARENT_SINGLE_PAGE") != nullptr &&
             std::strcmp(std::getenv("NEIGHBORCACHE_PARENT_SINGLE_PAGE"), "0") != 0);

        struct SeedFirstChoiceInput
        {
            uint32_t id = 0;
            uint32_t seed_id = 0;
            NeighborCacheDiskLoc loc{};
            bool replica_cache_only = false;
        };
        std::vector<SeedFirstChoiceInput> sf_inputs;
        sf_inputs.reserve(neighborcache_ids.size());

        for (uint32_t id : neighborcache_ids)
        {
            NeighborCacheDiskLoc loc;
            if (!neighborcache_resolve_disk_cache_loc(id, query_scratch, loc) || !chosen_ids.insert(id).second)
                continue;

            uint32_t seed_id = id;
            if (!record_driven_seed_access && query_scratch != nullptr)
            {
                const auto parent_it = query_scratch->parent_of(id);
                if (parent_it.has_value())
                    seed_id = parent_it.value();
            }

            NeighborCacheDiskLoc strict_loc;
            bool has_strict_loc = false;
            if (record_driven_seed_access)
            {
                const auto canonical_it = _neighborcache_dc_seed_canonical_loc.find(id);
                if (canonical_it != _neighborcache_dc_seed_canonical_loc.end())
                {
                    strict_loc = canonical_it->second;
                    has_strict_loc = true;
                }
            }
            else if (query_scratch != nullptr)
            {
                const auto parent_it = query_scratch->parent_of(id);
                if (parent_it.has_value())
                {
                    const auto member_it = _neighborcache_dc_seed_member_loc.find(parent_it.value());
                    if (member_it != _neighborcache_dc_seed_member_loc.end())
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
                    const auto seed_it = _neighborcache_dc_seed_member_loc.find(id);
                    if (seed_it != _neighborcache_dc_seed_member_loc.end())
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
                neighborcache_loc_in_query_cache(query_scratch, loc.sector, loc.nsectors);
            const bool shared_child_only_loc = _neighborcache_child_only_layout && !has_strict_loc;
            const bool selected_non_strict_loc =
                has_strict_loc &&
                (loc.sector != strict_loc.sector || loc.slot != strict_loc.slot ||
                 loc.nsectors != strict_loc.nsectors);
            const bool replica_cache_only =
                record_already_available || shared_child_only_loc ||
                ((cache_replica_fallback || _neighborcache_child_only_layout) && selected_non_strict_loc);

            sf_inputs.push_back({id, seed_id, loc, replica_cache_only});
        }

        if (adaptive_parent_or_self && query_scratch != nullptr)
        {
            auto is_leaf_like_node = [&](uint32_t node_id) -> bool {
                const auto seed_it = _neighborcache_dc_seed_member_loc.find(node_id);
                if (seed_it == _neighborcache_dc_seed_member_loc.end())
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
                const auto parent_it = query_scratch->parent_of(input.id);
                if (!parent_it.has_value())
                    continue;
                const auto members_it = _neighborcache_dc_seed_member_loc.find(parent_it.value());
                if (members_it == _neighborcache_dc_seed_member_loc.end() ||
                    members_it->second.find(input.id) == members_it->second.end())
                    continue;
                inputs_by_parent[parent_it.value()].push_back(input_idx);
            }

            for (const auto &parent_group : inputs_by_parent)
            {
                const uint32_t parent_seed = parent_group.first;
                const std::vector<size_t> &input_indices = parent_group.second;

                const auto canonical_it = _neighborcache_dc_seed_canonical_loc.find(parent_seed);
                const auto members_it = _neighborcache_dc_seed_member_loc.find(parent_seed);
                if (canonical_it == _neighborcache_dc_seed_canonical_loc.end() ||
                    members_it == _neighborcache_dc_seed_member_loc.end())
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
                    const NeighborCacheDiskLoc &self_loc = sf_inputs[input_idx].loc;
                    if (!neighborcache_loc_in_query_cache(query_scratch, self_loc.sector, self_loc.nsectors))
                    {
                        const uint64_t read_key = (static_cast<uint64_t>(self_loc.sector) << 16) |
                                                  static_cast<uint64_t>(self_loc.nsectors);
                        uncached_self_reads.insert(read_key);
                    }
                }
                const size_t parent_reads =
                    neighborcache_loc_in_query_cache(query_scratch, canonical_it->second.sector,
                                             canonical_it->second.nsectors)
                        ? 0
                        : 1;
                if (parent_reads >= uncached_self_reads.size())
                    continue;
                assign_parent_extent(internal_indices);
            }
        }

        std::unordered_map<uint32_t, std::vector<NeighborCacheSectorSpan>> seed_spans;
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
            const auto canonical_it = _neighborcache_dc_seed_canonical_loc.find(in.seed_id);
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
                        else if (canonical_it != _neighborcache_dc_seed_canonical_loc.end())
                        {
                            // Disjoint needed pages: one full-extent read beats many IOs.
                            io_base = canonical_it->second.sector;
                            io_nsectors = canonical_it->second.nsectors;
                        }
                    }
                }
                else if (canonical_it != _neighborcache_dc_seed_canonical_loc.end())
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
        if (query_scratch != nullptr && !query_scratch->neighborcache_dcache_loc_cache.empty())
        {
            for (uint32_t id : neighborcache_ids)
            {
                const auto cit = query_scratch->neighborcache_dcache_loc_cache.find(id);
                if (cit == query_scratch->neighborcache_dcache_loc_cache.end() || !chosen_ids.insert(id).second)
                    continue;
                NeighborCacheDiskLoc loc;
                loc.sector = static_cast<uint32_t>(cit->second >> 32);
                loc.nsectors = static_cast<uint16_t>((cit->second >> 16) & 0xffffu);
                loc.slot = static_cast<uint16_t>(cit->second & 0xffffu);
                if (!neighborcache_dyn_is_sector(loc.sector))
                    continue;
                choices.push_back({id, 0, loc, loc.sector, loc.nsectors});
            }
            for (uint32_t id : neighborcache_ids)
            {
                if (chosen_ids.find(id) == chosen_ids.end())
                    unresolved.push_back(id);
            }
        }
        else
            unresolved.assign(neighborcache_ids.begin(), neighborcache_ids.end());
    }

    std::unordered_set<uint32_t> need_ids(unresolved.begin(), unresolved.end());

    // Under freeze, page/loc tables are immutable: skip shared lock when every
    // neighborcache id already has a cached loc (common after lookup).
    struct OptionalDynSharedLock
    {
        NeighborCacheTimedSharedMutexGuard<std::shared_mutex> *guard = nullptr;
        alignas(NeighborCacheTimedSharedMutexGuard<std::shared_mutex>) unsigned char
            storage[sizeof(NeighborCacheTimedSharedMutexGuard<std::shared_mutex>)];
        OptionalDynSharedLock(std::shared_mutex &mu, bool take)
        {
            if (take)
                guard = new (storage) NeighborCacheTimedSharedMutexGuard<std::shared_mutex>(mu, NeighborCacheLockKind::DynamicShared, false);
        }
        ~OptionalDynSharedLock()
        {
            if (guard)
                guard->~NeighborCacheTimedSharedMutexGuard<std::shared_mutex>();
        }
        void unlock()
        {
            if (guard)
            {
                guard->~NeighborCacheTimedSharedMutexGuard<std::shared_mutex>();
                guard = nullptr;
            }
        }
    } loc_lock(_neighborcache_dyn_mu, !_neighborcache_freeze_maintenance || !need_ids.empty());
    while (!need_ids.empty())
    {
        std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, size_t>>> sector_hits;
        for (uint32_t id : need_ids)
        {
            neighborcache_dc_for_each_loc_unlocked(id, [&](const NeighborCacheDiskLoc &loc, size_t li) {
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
            const int cache_bonus = neighborcache_loc_in_query_cache(query_scratch, kv.first, 1) ? 1 : 0;
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
            NeighborCacheDiskLoc selected_loc;
            if (!neighborcache_dc_loc_at_unlocked(id_loc.first, id_loc.second, selected_loc))
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
                const size_t loc_count = neighborcache_dc_loc_count_unlocked(id);
                if (loc_count == 0)
                {
                    ++id_it;
                    continue;
                }
                size_t cover_li = 0;
                bool covered = false;
                NeighborCacheDiskLoc cover_loc;
                neighborcache_dc_for_each_loc_unlocked(id, [&](const NeighborCacheDiskLoc &loc, size_t li) {
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
                    stats->n_neighborcache_setcover_grouped++;
                id_it = need_ids.erase(id_it);
            }
        }
    }
    choices.erase(std::remove_if(choices.begin(), choices.end(),
                                 [&](PendingChoice &pc) {
                                     if (!neighborcache_dyn_is_sector(pc.loc.sector))
                                         return false;
                                     const uint32_t page_idx = pc.loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
                                     if (page_idx >= _neighborcache_dyn_physical_cap ||
                                         !neighborcache_dyn_page_live(page_idx))
                                         return true;
                                     if (!neighborcache_dc_loc_contains_unlocked(pc.id, pc.loc))
                                         return true;
                                     // Freeze keeps pages live for the whole run; skip pin/unpin.
                                     if (!_neighborcache_freeze_maintenance &&
                                         (_neighborcache_shared_cache_dyn ||
                                          !neighborcache_loc_in_query_cache(query_scratch, pc.loc.sector, 1)))
                                     {
                                         _neighborcache_dyn_page_readers[page_idx].fetch_add(1, std::memory_order_acq_rel);
                                         query_scratch->neighborcache_dyn_pinned_sectors.push_back(pc.loc.sector);
                                         pc.dynamic_page_pinned = true;
                                     }
                                     return false;
                                 }),
                  choices.end());
    if (_neighborcache_dcache_second_page)
    {
        for (PendingChoice &pc : choices)
        {
            if (!neighborcache_dyn_is_sector(pc.loc.sector))
                continue;
            const uint32_t page_idx = pc.loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
            if (page_idx >= _neighborcache_dyn_page_span_len.size())
                continue;
            const uint8_t sl = _neighborcache_dyn_page_span_len[page_idx];
            const uint32_t sb = page_idx < _neighborcache_dyn_page_span_base.size() ? _neighborcache_dyn_page_span_base[page_idx]
                                                                            : NEIGHBORCACHE_DYN_INVALID_PAGE;
            if (sl <= 1 || sb == NEIGHBORCACHE_DYN_INVALID_PAGE ||
                static_cast<uint64_t>(sb) + sl > _neighborcache_dyn_physical_cap)
                continue;
            if (_neighborcache_span_adaptive &&
                query_scratch->neighborcache_dyn_full_span.find(sb) == query_scratch->neighborcache_dyn_full_span.end())
                continue;
            pc.io_base_sector = NEIGHBORCACHE_DYN_SECTOR_BASE + sb;
            pc.io_nsectors = sl;
            pc.loc.nsectors = sl;
            for (uint32_t extra = 0; extra < sl; ++extra)
            {
                const uint32_t ep = sb + extra;
                if (ep == page_idx || !neighborcache_dyn_page_live(ep))
                    continue;
                const uint32_t es = NEIGHBORCACHE_DYN_SECTOR_BASE + ep;
                if (!_neighborcache_freeze_maintenance &&
                    (_neighborcache_shared_cache_dyn || !neighborcache_loc_in_query_cache(query_scratch, es, 1)))
                {
                    _neighborcache_dyn_page_readers[ep].fetch_add(1, std::memory_order_acq_rel);
                    query_scratch->neighborcache_dyn_pinned_sectors.push_back(es);
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
        const NeighborCacheDiskLoc loc = pc.loc;
        const bool dynamic_page = neighborcache_dyn_is_sector(loc.sector);
        const bool pair_read = dynamic_page && pc.io_nsectors > 1;
        const uint32_t io_base =
            pair_read ? pc.io_base_sector
                      : (dynamic_page || disable_multiread ? loc.sector
                                                           : multiread_root_for_sector(pc.io_base_sector));
        const uint16_t io_nsectors =
            pair_read ? pc.io_nsectors
                      : (dynamic_page || disable_multiread ? uint16_t{1} : pc.io_nsectors);
        pending.push_back({pc.id, nullptr, loc, io_base, io_nsectors, pc.dynamic_page_pinned});
        const size_t pidx = pending.size() - 1;
        if (stats != nullptr && neighborcache_dyn_is_sector(loc.sector))
            stats->n_neighborcache_dyn_hits++;

        bool served = false;
        if (_query_sector_cache_enabled && _nnodes_per_sector > 0 && neighborcache_use_shared_cache(loc.sector))
        {
            char *spill = sector_scratch + sector_scratch_idx * defaults::SECTOR_LEN;
            bool spilled = false;
            const char *np = neighborcache_shared_node_ptr(loc.sector, loc.slot, spill, spilled);
            if (np != nullptr)
            {
                if (spilled)
                    sector_scratch_idx++;
                pending[pidx].node_ptr = np;
                pending[pidx].io_base_sector = loc.sector;
                pending[pidx].io_nsectors = 1;
                pending[pidx].from_query_cache = true;
                if (stats != nullptr)
                {
                    stats->n_sector_cache_hits++;
                    stats->n_neighborcache_sector_cache_hits++;
                    if (query_scratch->neighborcache_fetched_sectors.find(loc.sector) ==
                        query_scratch->neighborcache_fetched_sectors.end())
                        stats->n_neighborcache_shared_cross_hits++;
                }
                served = true;
            }
        }
        else if (_query_sector_cache_enabled)
        {
            const uint32_t cache_base = pair_read ? io_base : pc.io_base_sector;
            const uint16_t cache_nsectors =
                pair_read ? io_nsectors : (disable_multiread ? uint16_t{1} : pc.io_nsectors);
            char *read_buf = sector_scratch + sector_scratch_idx * defaults::SECTOR_LEN;
            if (neighborcache_cache_get_range(query_scratch, cache_base, cache_nsectors, read_buf))
            {
                sector_scratch_idx += cache_nsectors;
                pending[pidx].sec_buf = read_buf;
                pending[pidx].io_base_sector = cache_base;
                pending[pidx].from_query_cache = true;
                if (stats != nullptr)
                {
                    stats->n_sector_cache_hits++;
                    stats->n_neighborcache_sector_cache_hits++;
                    if (neighborcache_use_shared_cache(cache_base) &&
                        query_scratch->neighborcache_fetched_sectors.find(cache_base) ==
                            query_scratch->neighborcache_fetched_sectors.end())
                        stats->n_neighborcache_shared_cross_hits++;
                }
                served = true;
            }
        }
        if (!served)
        {
            // A has left prefetch, but its owned write task may not be complete
            // yet. Do not schedule a read of stale/uninitialized D bytes.
            // The caller routes this unserved ID to the original index.
            if (!neighborcache_dyn_disk_readable(io_base, io_nsectors))
            {
                pending.pop_back();
                if (stats != nullptr)
                    stats->n_neighborcache_dyn_hits--;
                _neighborcache_writeback_read_bypasses.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            const uint32_t group_sec = io_base;
            if (group_sec != loc.sector && stats != nullptr)
                stats->n_neighborcache_io_avoided++;
            disk_fanout_groups[group_sec].push_back(pidx);
            if (stats != nullptr)
            {
                stats->n_neighborcache_dc_hits++;
                stats->n_4k++;
                stats->n_ios++;
            }
            num_ios++;
        }
        query_scratch->read_neighborcache_disk_cache_sectors.insert(loc.sector);
        if (pair_read)
            query_scratch->read_neighborcache_disk_cache_sectors.insert(io_base);
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
        const uint64_t off = neighborcache_dyn_is_sector(disk_cache_sec)
                                 ? static_cast<uint64_t>(disk_cache_sec - NEIGHBORCACHE_DYN_SECTOR_BASE) *
                                       defaults::SECTOR_LEN
                                 : (_neighborcache_unified_disk ? _neighborcache_region_byte_offset : 0) +
                                       static_cast<uint64_t>(disk_cache_sec) * defaults::SECTOR_LEN;
        neighborcache_io.emplace_back(off, byte_len, read_buf);
        if (neighborcache_dyn_is_sector(disk_cache_sec))
        {
            query_scratch->neighborcache_dyn_physical_reads += group_nsectors;
            if (group_nsectors > 1)
                _neighborcache_dcache_pair_ios.fetch_add(1, std::memory_order_relaxed);
        }
        if (stats != nullptr)
        {
            stats->n_disk_reads++;
            stats->n_neighborcache_disk_reads++;
            if (neighborcache_dyn_is_sector(disk_cache_sec))
                stats->n_neighborcache_dyn_disk_reads++;
            if (group_nsectors > 1)
                stats->n_neighborcache_multiread_ios++;
            if (_record_hop_frontier && !stats->hop_frontier_trace.empty())
                stats->hop_frontier_trace.back().physical_reads.push_back(
                    {true, disk_cache_sec, group_nsectors});
        }
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::complete_neighborcache_disk_cache_io(
    SSDQueryScratch<T> *query_scratch, std::vector<NeighborCacheReadPending> &pending,
    const std::unordered_map<uint32_t, std::vector<size_t>> &disk_fanout_groups)
{
    for (const auto &kv : disk_fanout_groups)
    {
        const std::vector<size_t> &indices = kv.second;
        if (indices.empty())
            continue;
        const char *primary = pending[indices[0]].sec_buf;
        if (_query_sector_cache_enabled)
        {
            for (uint16_t si = 0; si < pending[indices[0]].io_nsectors; ++si)
                neighborcache_cache_put(query_scratch, kv.first + static_cast<uint32_t>(si),
                                primary + static_cast<size_t>(si) * defaults::SECTOR_LEN);
        }
        for (size_t idx : indices)
            pending[idx].sec_buf = const_cast<char *>(primary);
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::finalize_neighborcache_pending_nodes(const std::vector<NeighborCacheReadPending> &pending,
                                                           const std::vector<uint32_t> &neighborcache_order,
                                                           SSDQueryScratch<T> *query_scratch, char *sector_scratch,
                                                           uint64_t &sector_scratch_idx,
                                                           std::vector<std::pair<uint32_t, char *>> &frontier_nhoods,
                                                           QueryStats *stats)
{
    std::unordered_map<uint32_t, char *> ready;
    ready.reserve(pending.size());
    for (const auto &mp : pending)
    {
        const char *node = mp.node_ptr;
        char *sec_buf = mp.sec_buf;
        uint32_t io_base = mp.io_base_sector;
        if (node == nullptr && sec_buf == nullptr && _query_sector_cache_enabled)
        {
            char *buf = sector_scratch + sector_scratch_idx * defaults::SECTOR_LEN;
            if (_nnodes_per_sector > 0 && neighborcache_use_shared_cache(mp.loc.sector))
            {
                bool spilled = false;
                node = neighborcache_shared_node_ptr(mp.loc.sector, mp.loc.slot, buf, spilled);
                if (spilled)
                    sector_scratch_idx++;
            }
            else if (neighborcache_cache_get_range(query_scratch, mp.loc.sector, 1, buf))
            {
                sector_scratch_idx++;
                sec_buf = buf;
                io_base = mp.loc.sector;
            }
        }
        if (node == nullptr && sec_buf == nullptr)
            continue;

        char *out_buf = sec_buf;
        if (_nnodes_per_sector > 0)
        {
            const uint64_t base_slot = mp.id % _nnodes_per_sector;
            const char *packed = node;
            uint32_t unpack_base = mp.loc.sector;
            if (packed == nullptr)
            {
                unpack_base = io_base;
                if (mp.loc.sector < io_base || mp.loc.sector >= io_base + static_cast<uint32_t>(mp.io_nsectors))
                    unpack_base = mp.loc.sector;
                const uint64_t sector_off =
                    (mp.loc.sector >= unpack_base)
                        ? static_cast<uint64_t>(mp.loc.sector - unpack_base) * defaults::SECTOR_LEN
                        : 0;
                packed = sec_buf + sector_off + static_cast<uint64_t>(mp.loc.slot) * _max_node_len;
                if (_neighborcache_repack_on_read && neighborcache_dyn_is_sector(mp.loc.sector))
                    neighborcache_dyn_repack_read_page(query_scratch, mp.loc.sector, sec_buf + sector_off, stats);
            }
            // NeighborCache pages use loc.slot (dense page layout); offset_to_node(out_buf, id) adds
            // (id % nps) * max_node_len, so bias the base so it lands on the packed node in place.
            out_buf = const_cast<char *>(packed) - base_slot * _max_node_len;
            if (neighborcache_debug_verify_enabled())
            {
                static const int fd = ::open(_disk_index_file.c_str(), O_RDONLY);
                static std::atomic<int> reported{0};
                std::vector<char> sec(defaults::SECTOR_LEN);
                const uint64_t off = get_node_sector(mp.id) * defaults::SECTOR_LEN;
                if (::pread(fd, sec.data(), defaults::SECTOR_LEN, static_cast<off_t>(off)) ==
                        static_cast<ssize_t>(defaults::SECTOR_LEN) &&
                    memcmp(sec.data() + base_slot * _max_node_len, packed, _max_node_len) != 0 &&
                    reported.fetch_add(1) < 30)
                {
                    int file_eq_packed = -1, file_eq_base = -1;
                    auto *mr = dynamic_cast<LinuxAlignedFileReader *>(_neighborcache_disk_reader.get());
                    if (mr != nullptr)
                    {
                        char *fb = nullptr;
                        alloc_aligned((void **)&fb, defaults::SECTOR_LEN, defaults::SECTOR_LEN);
                        const uint64_t foff = (_neighborcache_unified_disk ? _neighborcache_region_byte_offset : 0) +
                                              static_cast<uint64_t>(mp.loc.sector) * defaults::SECTOR_LEN;
                        if (::pread(mr->get_file_desc(), fb, defaults::SECTOR_LEN, static_cast<off_t>(foff)) ==
                            static_cast<ssize_t>(defaults::SECTOR_LEN))
                        {
                            const char *fslot = fb + static_cast<uint64_t>(mp.loc.slot) * _max_node_len;
                            file_eq_packed = memcmp(fslot, packed, _max_node_len) == 0;
                            file_eq_base = memcmp(fslot, sec.data() + base_slot * _max_node_len, _max_node_len) == 0;
                        }
                        aligned_free(fb);
                    }
                    diskann::cout << "VERIFY_MISMATCH file_slot==served:" << file_eq_packed
                                  << " file_slot==base:" << file_eq_base << " id=" << mp.id
                                  << " loc.sector=" << mp.loc.sector
                                  << " loc.slot=" << mp.loc.slot << " loc.nsectors=" << mp.loc.nsectors
                                  << " io_base=" << mp.io_base_sector << " io_nsectors=" << mp.io_nsectors
                                  << " unpack_base=" << unpack_base << " from_cache=" << mp.from_query_cache
                                  << " sec_buf_null_before=" << (mp.sec_buf == nullptr) << std::endl;
                }
            }
        }
        ready[mp.id] = out_buf;
        if (neighborcache_dyn_is_sector(mp.loc.sector))
        {
            if (_neighborcache_dcache_net_gate_enabled)
                query_scratch->neighborcache_avoided_base_sectors.insert(get_node_sector(mp.id));
            query_scratch->neighborcache_dyn_served_nodes++;
            if (_neighborcache_real_io_coaccess)
            {
                query_scratch->neighborcache_storage_node_order.push_back(mp.id);
                if (query_scratch->neighborcache_record_real_io)
                    neighborcache_push_storage_nbrs(query_scratch->neighborcache_storage_nbrs, query_scratch->neighborcache_storage_nbr_off,
                                            offset_to_node_nhood(offset_to_node(out_buf, mp.id)),
                                            static_cast<uint32_t>(_max_degree));
            }
            // Prefetch buffer: register all members on this served page for same-query hits.
            if (neighborcache_dyn_is_sector(mp.loc.sector) && _neighborcache_dyn_members_per_page > 0)
            {
                const uint32_t page_idx = mp.loc.sector - NEIGHBORCACHE_DYN_SECTOR_BASE;
                if (!neighborcache_query_page_prefetch_current(query_scratch, page_idx))
                {
                    NeighborCacheTimedSharedMutexGuard page_lock(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
                    if (page_idx < _neighborcache_dyn_physical_cap && !neighborcache_query_page_prefetch_current(query_scratch, page_idx))
                    {
                        const size_t begin = static_cast<size_t>(page_idx) * _neighborcache_dyn_members_per_page;
                        const size_t count = _neighborcache_dyn_page_member_count[page_idx];
                        for (size_t mi = 0; mi < count; ++mi)
                        {
                            const uint32_t mid = _neighborcache_dyn_page_members[begin + mi];
                            if (mid == NEIGHBORCACHE_DYN_INVALID_PAGE)
                                continue;
                            const uint64_t packed = (static_cast<uint64_t>(mp.loc.sector) << 32) |
                                                   (static_cast<uint64_t>(1) << 16) |
                                                   static_cast<uint64_t>(mi);
                            query_scratch->neighborcache_dcache_member_packed_loc[mid] = packed;
                        }
                        neighborcache_query_page_prefetch_remember(query_scratch, page_idx);
                    }
                }
            }
            neighborcache_family_note_read(mp.loc.sector);
        }
    }

    for (uint32_t id : neighborcache_order)
    {
        const auto it = ready.find(id);
        if (it == ready.end())
        {
            if (stats != nullptr)
                stats->n_neighborcache_finalize_skipped++;
            continue;
        }
        frontier_nhoods.emplace_back(id, it->second);
    }
    if (neighborcache_debug_verify_enabled() && sector_scratch_idx > defaults::MAX_N_SECTOR_READS)
    {
        static std::atomic<int> overflow_reported{0};
        if (overflow_reported.fetch_add(1) < 10)
            diskann::cout << "SCRATCH_OVERFLOW used=" << sector_scratch_idx << " pending=" << pending.size()
                          << std::endl;
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
    const char *pages_env = std::getenv("NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES");
    if (enable && pages_env != nullptr && !_neighborcache_shared_cache.enabled())
    {
        const uint64_t pages = std::strtoull(pages_env, nullptr, 10);
        if (pages > 0 && _neighborcache_shared_cache.init(pages, std::max<uint64_t>(1, _max_nthreads)))
        {
            const char *dyn_env = std::getenv("NEIGHBORCACHE_SHARED_CACHE_DYN");
            _neighborcache_shared_cache_dyn = dyn_env != nullptr && std::strcmp(dyn_env, "0") != 0;
            diskann::cout << "NEIGHBORCACHE shared D-cache page cache: " << _neighborcache_shared_cache.capacity_pages() << " pages ("
                          << _neighborcache_shared_cache.capacity_pages() * defaults::SECTOR_LEN / (1024.0 * 1024.0)
                          << " MB), " << (_neighborcache_shared_cache.thread_private() ? "thread-private LRUs" : "shared by all threads")
                          << (_neighborcache_shared_cache_dyn ? ", including online D-cache pages." : ".") << std::endl;
        }
    }
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::print_neighborcache_shared_cache_stats() const
{
    if (g_neighborcache_miss_diag)
        diskann::cout << "NEIGHBORCACHE dcache miss diag: not_in_dcache=" << g_neighborcache_miss_not_in_dcache.load()
                      << " no_parent=" << g_neighborcache_miss_no_parent.load()
                      << " parent_in_ncache=" << g_neighborcache_miss_parent_ncache.load()
                      << " parent_not_seed=" << g_neighborcache_miss_parent_not_seed.load()
                      << " parent_lacks_node=" << g_neighborcache_miss_parent_lacks.load() << std::endl;
    if (!_neighborcache_shared_cache.enabled())
        return;
    diskann::cout << "NEIGHBORCACHE shared page cache: capacity=" << _neighborcache_shared_cache.capacity_pages()
                  << " hits=" << _neighborcache_shared_cache.hits() << " misses=" << _neighborcache_shared_cache.misses()
                  << " inserts=" << _neighborcache_shared_cache.inserts() << " evictions=" << _neighborcache_shared_cache.evictions()
                  << " skipped_puts=" << _neighborcache_shared_cache.skipped_puts()
                  << std::endl;
}

template <typename T, typename LabelT>
bool PQFlashIndex<T, LabelT>::neighborcache_cache_get_range(SSDQueryScratch<T> *query_scratch, uint32_t base,
                                                    uint16_t nsectors, char *dst)
{
    if (!_query_sector_cache_enabled || query_scratch == nullptr || nsectors == 0)
        return false;
    if (neighborcache_use_shared_cache(base))
        return _neighborcache_shared_cache.get_range(base, nsectors, dst);
    for (uint16_t si = 0; si < nsectors; ++si)
    {
        const auto cit = query_scratch->neighborcache_sector_cache.find(base + si);
        if (cit == query_scratch->neighborcache_sector_cache.end())
            return false;
        memcpy(dst + static_cast<size_t>(si) * defaults::SECTOR_LEN, cit->second.data(), defaults::SECTOR_LEN);
    }
    return true;
}

template <typename T, typename LabelT>
const char *PQFlashIndex<T, LabelT>::neighborcache_shared_node_ptr(uint32_t sector, uint32_t slot, char *spill, bool &spilled)
{
    spilled = false;
    const uint64_t off = static_cast<uint64_t>(slot) * _max_node_len;
    const uint32_t s0 = sector + static_cast<uint32_t>(off / defaults::SECTOR_LEN);
    const uint64_t o0 = off % defaults::SECTOR_LEN;
    if (neighborcache_dyn_is_sector(sector))
    {
        std::array<char, defaults::SECTOR_LEN * 2> copy;
        const uint16_t n = o0 + _max_node_len <= defaults::SECTOR_LEN ? 1 : 2;
        if (!_neighborcache_shared_cache.get_range(s0, n, copy.data())) return nullptr;
        memcpy(spill, copy.data() + o0, _max_node_len);
        spilled = true;
        return spill;
    }
    const char *p0 = _neighborcache_shared_cache.get_ptr(s0);
    if (p0 == nullptr)
        return nullptr;
    if (o0 + _max_node_len <= defaults::SECTOR_LEN)
        return p0 + o0;
    const char *p1 = _neighborcache_shared_cache.get_ptr(s0 + 1);
    if (p1 == nullptr)
        return nullptr;
    const uint64_t head = defaults::SECTOR_LEN - o0;
    memcpy(spill, p0 + o0, head);
    memcpy(spill + head, p1, _max_node_len - head);
    spilled = true;
    return spill;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::neighborcache_cache_put(SSDQueryScratch<T> *query_scratch, uint32_t sector, const char *src)
{
    if (neighborcache_use_shared_cache(sector))
    {
        // An earlier D read must not re-admit stale bytes after a newer dirty
        // version has been detached from prefetch.
        if (!neighborcache_dyn_disk_readable(sector))
            return;
        const std::atomic<uint64_t> *disk_dirty =
            _neighborcache_prefetch_dirty && neighborcache_dyn_is_sector(sector)
                ? &_neighborcache_prefetch_dirty[sector - NEIGHBORCACHE_DYN_SECTOR_BASE]
                : nullptr;
        _neighborcache_shared_cache.put(sector, src, false, nullptr, disk_dirty);
        query_scratch->neighborcache_fetched_sectors.insert(sector);
        return;
    }
    memcpy(query_scratch->neighborcache_sector_cache[sector].data(), src, defaults::SECTOR_LEN);
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

// The cache-only facade instantiates the methods it uses in its own TU.
#ifndef NEIGHBORCACHE_BACKEND_ONLY
// instantiations
template class PQFlashIndex<uint8_t>;
template class PQFlashIndex<int8_t>;
template class PQFlashIndex<float>;
template class PQFlashIndex<uint8_t, uint16_t>;
template class PQFlashIndex<int8_t, uint16_t>;
template class PQFlashIndex<float, uint16_t>;
#endif

} // namespace diskann
