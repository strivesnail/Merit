#include "merit_metadata_cache.h"
#include "merit_lock_metrics.h"

#include <algorithm>
#include <random>

namespace diskann
{

MeritMetadataCache::MeritMetadataCache()
{
    for (auto &shard : _shards)
        shard = std::make_unique<Shard>();
}

size_t MeritMetadataCache::shard_index(uint32_t node_id)
{
    return static_cast<size_t>(node_id % kShardCount);
}

void MeritMetadataCache::clear()
{
    _capacity.store(0, std::memory_order_relaxed);

    std::vector<std::unique_ptr<MeritTimedMutexGuard>> locks;
    locks.reserve(kShardCount);
    for (auto &shard : _shards)
        locks.emplace_back(std::make_unique<MeritTimedMutexGuard>(shard->mu, MeritLockKind::Metadata));

    for (auto &shard_ptr : _shards)
    {
        Shard &shard = *shard_ptr;
        shard.entries.clear();
        shard.global_lru.clear();
        shard.nonseed_lru.clear();
        shard.scores.clear();
        shard.slot_to_node.clear();
        shard.free_slots.clear();
        shard.capacity = 0;
        shard.tick = 0;
    }
    _slot_stride.store(0, std::memory_order_relaxed);
    _total_size.store(0, std::memory_order_relaxed);
    _total_edges.store(0, std::memory_order_relaxed);
    _total_evictable.store(0, std::memory_order_relaxed);
}

void MeritMetadataCache::init(uint64_t capacity, uint32_t edge_k, uint8_t sig_threshold)
{
    clear();
    const uint64_t effective_capacity = std::min<uint64_t>(capacity, static_cast<uint64_t>(kInvalid));
    const uint32_t slot_stride =
        effective_capacity == 0 ? 0 : static_cast<uint32_t>((effective_capacity + kShardCount - 1) / kShardCount);
    const uint64_t base_capacity = effective_capacity / kShardCount;
    const uint64_t extra_shards = effective_capacity % kShardCount;

    std::vector<std::unique_ptr<MeritTimedMutexGuard>> locks;
    locks.reserve(kShardCount);
    for (auto &shard : _shards)
        locks.emplace_back(std::make_unique<MeritTimedMutexGuard>(shard->mu, MeritLockKind::Metadata));

    _edge_k.store(edge_k, std::memory_order_relaxed);
    _sig_t.store(sig_threshold == 0 ? kDefaultSigT : sig_threshold, std::memory_order_relaxed);
    _slot_stride.store(slot_stride, std::memory_order_relaxed);
    for (size_t shard_id = 0; shard_id < kShardCount; ++shard_id)
    {
        Shard &shard = *_shards[shard_id];
        shard.capacity = base_capacity + (shard_id < extra_shards ? 1 : 0);
        shard.scores.assign(static_cast<size_t>(shard.capacity), 0.0f);
        shard.slot_to_node.assign(static_cast<size_t>(shard.capacity), kInvalid);
        shard.free_slots.reserve(static_cast<size_t>(shard.capacity));
        for (uint32_t slot = static_cast<uint32_t>(shard.capacity); slot-- > 0;)
            shard.free_slots.push_back(slot);
    }
    _capacity.store(effective_capacity, std::memory_order_relaxed);
}

uint64_t MeritMetadataCache::size() const
{
    return _total_size.load(std::memory_order_relaxed);
}

uint64_t MeritMetadataCache::edge_count() const
{
    return _total_edges.load(std::memory_order_relaxed);
}

uint64_t MeritMetadataCache::evictable_size() const
{
    return _total_evictable.load(std::memory_order_relaxed);
}

void MeritMetadataCache::touch_unlocked(Shard &shard, Entry &e)
{
    shard.global_lru.splice(shard.global_lru.begin(), shard.global_lru, e.global_lru_it);
    e.global_lru_it = shard.global_lru.begin();
    if (e.evictable)
    {
        shard.nonseed_lru.splice(shard.nonseed_lru.begin(), shard.nonseed_lru, e.nonseed_lru_it);
        e.nonseed_lru_it = shard.nonseed_lru.begin();
    }
    e.tick = ++shard.tick;
}

uint32_t MeritMetadataCache::evict_one_unlocked(Shard &shard, uint32_t &evicted_slot)
{
    evicted_slot = kInvalid;
    if (shard.nonseed_lru.empty())
        return kInvalid;

    const uint32_t victim = shard.nonseed_lru.back();
    auto eit = shard.entries.find(victim);
    if (eit == shard.entries.end())
        return kInvalid;

    Entry &entry = eit.value();
    evicted_slot = entry.slot_id;
    const uint32_t local_slot = entry.local_slot;
    const bool was_evictable = entry.evictable;
    const uint64_t removed_edges = entry.edges.size();
    shard.global_lru.erase(entry.global_lru_it);
    if (was_evictable)
        shard.nonseed_lru.erase(entry.nonseed_lru_it);
    shard.entries.erase(eit);
    if (local_slot != kInvalid && local_slot < shard.slot_to_node.size())
    {
        shard.slot_to_node[local_slot] = kInvalid;
        shard.scores[local_slot] = 0.0f;
        shard.free_slots.push_back(local_slot);
    }
    _total_size.fetch_sub(1, std::memory_order_relaxed);
    _total_edges.fetch_sub(removed_edges, std::memory_order_relaxed);
    if (was_evictable)
        _total_evictable.fetch_sub(1, std::memory_order_relaxed);
    return victim;
}

MeritMetadataCache::TouchResult MeritMetadataCache::touch_or_insert_unlocked(Shard &shard, size_t shard_id,
                                                                            uint32_t node_id, float score_delta)
{
    TouchResult r;
    auto it = shard.entries.find(node_id);
    if (it != shard.entries.end())
    {
        Entry &e = it.value();
        if (score_delta > 0.0f)
            shard.scores[e.local_slot] += score_delta;
        touch_unlocked(shard, e);
        r.node_id = node_id;
        r.slot_id = e.slot_id;
        r.score = shard.scores[e.local_slot];
        r.present = true;
        return r;
    }

    if (shard.capacity == 0)
        return r;

    if (shard.entries.size() >= shard.capacity)
    {
        uint32_t evicted_slot = kInvalid;
        const uint32_t ev = evict_one_unlocked(shard, evicted_slot);
        if (ev == kInvalid)
            return r;
        r.evicted_id = ev;
        r.evicted_slot = evicted_slot;
    }

    if (shard.free_slots.empty())
        return r;
    const uint32_t local_slot = shard.free_slots.back();
    shard.free_slots.pop_back();
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    const uint32_t slot_id =
        static_cast<uint32_t>(static_cast<uint64_t>(shard_id) * slot_stride + local_slot);
    shard.global_lru.push_front(node_id);
    shard.nonseed_lru.push_front(node_id);
    Entry e;
    e.node_id = node_id;
    e.slot_id = slot_id;
    e.local_slot = local_slot;
    shard.scores[local_slot] = std::max(score_delta, 0.0f);
    e.tick = ++shard.tick;
    e.global_lru_it = shard.global_lru.begin();
    e.nonseed_lru_it = shard.nonseed_lru.begin();
    e.evictable = true;
    shard.entries.insert({node_id, std::move(e)});
    shard.slot_to_node[local_slot] = node_id;
    _total_size.fetch_add(1, std::memory_order_relaxed);
    _total_evictable.fetch_add(1, std::memory_order_relaxed);
    r.node_id = node_id;
    r.slot_id = slot_id;
    r.score = shard.scores[local_slot];
    r.present = true;
    return r;
}

MeritMetadataCache::TouchResult MeritMetadataCache::on_expand(uint32_t node_id, float score_unit)
{
    const size_t shard_id = shard_index(node_id);
    Shard &shard = *_shards[shard_id];
    MeritTimedMutexGuard lock(shard.mu, MeritLockKind::Metadata);
    return touch_or_insert_unlocked(shard, shard_id, node_id, score_unit);
}

MeritMetadataCache::TouchResult MeritMetadataCache::on_edge(uint32_t parent, uint32_t child)
{
    thread_local std::mt19937 rng{std::random_device{}()};
    TouchResult r;
    if ((rng() % 10u) != 0u)
        return r;

    const size_t shard_id = shard_index(parent);
    Shard &shard = *_shards[shard_id];
    MeritTimedMutexGuard lock(shard.mu, MeritLockKind::Metadata);
    // Edge heat and node heat are independent. Expanding parent already added
    // its node score; this call only ensures the parent metadata is resident.
    r = touch_or_insert_unlocked(shard, shard_id, parent, 0.0f);
    if (!r.present)
        return r;

    auto eit = shard.entries.find(parent);
    if (eit == shard.entries.end())
        return r;
    Entry &e = eit.value();

    auto wit = e.edge_ix.find(child);
    if (wit != e.edge_ix.end())
    {
        auto lit = wit.value();
        if (lit->w < 255)
            lit->w = static_cast<uint8_t>(lit->w + 1);
        e.edges.splice(e.edges.begin(), e.edges, lit);
        wit.value() = e.edges.begin();
        return r;
    }

    const uint32_t edge_k = _edge_k.load(std::memory_order_relaxed);
    if (edge_k > 0 && e.edges.size() >= edge_k)
    {
        const Edge dropped = e.edges.back();
        e.edge_ix.erase(dropped.child);
        e.edges.pop_back();
        _total_edges.fetch_sub(1, std::memory_order_relaxed);
    }

    e.edges.push_front(Edge{child, 1});
    e.edge_ix[child] = e.edges.begin();
    _total_edges.fetch_add(1, std::memory_order_relaxed);
    return r;
}

bool MeritMetadataCache::contains(uint32_t node_id) const
{
    const Shard &shard = *_shards[shard_index(node_id)];
    MeritTimedMutexGuard lock(shard.mu, MeritLockKind::Metadata);
    return shard.entries.find(node_id) != shard.entries.end();
}

uint32_t MeritMetadataCache::slot_of(uint32_t node_id) const
{
    const Shard &shard = *_shards[shard_index(node_id)];
    MeritTimedMutexGuard lock(shard.mu, MeritLockKind::Metadata);
    const auto it = shard.entries.find(node_id);
    return it == shard.entries.end() ? kInvalid : it.value().slot_id;
}

uint32_t MeritMetadataCache::node_at(uint32_t slot_id) const
{
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    if (slot_stride == 0 || slot_id == kInvalid)
        return kInvalid;
    const size_t shard_id = slot_id / slot_stride;
    const uint32_t local_slot = slot_id % slot_stride;
    if (shard_id >= kShardCount)
        return kInvalid;
    const Shard &shard = *_shards[shard_id];
    MeritTimedMutexGuard lock(shard.mu, MeritLockKind::Metadata);
    return local_slot < shard.slot_to_node.size() ? shard.slot_to_node[local_slot] : kInvalid;
}

float MeritMetadataCache::score(uint32_t node_id) const
{
    const Shard &shard = *_shards[shard_index(node_id)];
    MeritTimedMutexGuard lock(shard.mu, MeritLockKind::Metadata);
    const auto it = shard.entries.find(node_id);
    return it == shard.entries.end() ? 0.0f : shard.scores[it.value().local_slot];
}

float MeritMetadataCache::score_at(uint32_t slot_id) const
{
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    if (slot_stride == 0 || slot_id == kInvalid)
        return 0.0f;
    const size_t shard_id = slot_id / slot_stride;
    const uint32_t local_slot = slot_id % slot_stride;
    if (shard_id >= kShardCount)
        return 0.0f;
    const Shard &shard = *_shards[shard_id];
    MeritTimedMutexGuard lock(shard.mu, MeritLockKind::Metadata);
    if (local_slot >= shard.slot_to_node.size() || shard.slot_to_node[local_slot] == kInvalid)
        return 0.0f;
    return shard.scores[local_slot];
}

bool MeritMetadataCache::set_evictable(uint32_t node_id, bool evictable)
{
    Shard &shard = *_shards[shard_index(node_id)];
    MeritTimedMutexGuard lock(shard.mu, MeritLockKind::Metadata);
    auto it = shard.entries.find(node_id);
    if (it == shard.entries.end())
        return false;
    Entry &e = it.value();
    if (e.evictable == evictable)
        return true;
    if (evictable)
    {
        shard.nonseed_lru.push_front(node_id);
        e.nonseed_lru_it = shard.nonseed_lru.begin();
        _total_evictable.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        shard.nonseed_lru.erase(e.nonseed_lru_it);
        _total_evictable.fetch_sub(1, std::memory_order_relaxed);
    }
    e.evictable = evictable;
    return true;
}

void MeritMetadataCache::scale_scores(float factor, std::vector<std::pair<uint32_t, float>> &scaled)
{
    std::vector<std::unique_ptr<MeritTimedMutexGuard>> locks;
    locks.reserve(kShardCount);
    for (auto &shard : _shards)
        locks.emplace_back(std::make_unique<MeritTimedMutexGuard>(shard->mu, MeritLockKind::Metadata));

    scaled.clear();
    scaled.reserve(static_cast<size_t>(_total_size.load(std::memory_order_relaxed)));
    for (auto &shard_ptr : _shards)
    {
        Shard &shard = *shard_ptr;
        for (auto it = shard.entries.begin(); it != shard.entries.end(); ++it)
        {
            Entry &entry = it.value();
            shard.scores[entry.local_slot] *= factor;
            scaled.emplace_back(entry.slot_id, shard.scores[entry.local_slot]);
        }
    }
}

bool MeritMetadataCache::snapshot(uint32_t node_id, Snapshot &out) const
{
    const Shard &shard = *_shards[shard_index(node_id)];
    MeritTimedMutexGuard lock(shard.mu, MeritLockKind::Metadata);
    const auto it = shard.entries.find(node_id);
    if (it == shard.entries.end())
        return false;
    const Entry &e = it.value();
    out.slot_id = e.slot_id;
    out.score = shard.scores[e.local_slot];
    out.significant.clear();
    out.significant.reserve(e.edges.size());
    const uint8_t sig_t = _sig_t.load(std::memory_order_relaxed);
    for (const Edge &ed : e.edges)
    {
        if (ed.w >= sig_t)
            out.significant.emplace_back(ed.child, ed.w);
    }
    std::sort(out.significant.begin(), out.significant.end(),
              [](const std::pair<uint32_t, uint8_t> &a, const std::pair<uint32_t, uint8_t> &b) {
                  if (a.second != b.second)
                      return a.second > b.second;
                  return a.first < b.first;
              });
    return true;
}

} // namespace diskann
