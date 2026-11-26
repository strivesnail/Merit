#include "neighborcache_metadata_cache.h"
#include "common_includes.h"
#include "neighborcache_lock_metrics.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace diskann
{

NeighborCacheMetadataCache::NeighborCacheMetadataCache()
{
    _shards.push_back(std::make_unique<Shard>());
}

size_t NeighborCacheMetadataCache::shard_index(uint32_t node_id) const
{
    return shard_count() == 1 ? 0 : static_cast<size_t>(node_id % kShardCount);
}

void NeighborCacheMetadataCache::clear()
{
    _capacity.store(0, std::memory_order_relaxed);

    std::vector<std::unique_ptr<NeighborCacheTimedSharedMutexGuard<std::shared_mutex>>> locks;
    locks.reserve(shard_count());
    for (auto &shard : _shards)
        locks.emplace_back(
            std::make_unique<NeighborCacheTimedSharedMutexGuard<std::shared_mutex>>(shard->mu, NeighborCacheLockKind::Metadata, true, _shared));

    for (auto &shard_ptr : _shards)
    {
        Shard &shard = *shard_ptr;
        shard.node_to_slot.clear();
        shard.scores.clear();
        shard.slot_to_node.clear();
        std::vector<std::vector<Edge>>().swap(shard.edges_by_slot);
        shard.seed_members_by_slot = {};
        shard.edge_locks.clear();
        shard.evictable.clear();
        shard.next_unused_slot = 0;
        shard.nonseed_prev.clear();
        shard.nonseed_next.clear();
        shard.nonseed_head = kInvalid;
        shard.nonseed_tail = kInvalid;
        shard.capacity = 0;
    }
    _slot_stride.store(0, std::memory_order_relaxed);
    _total_size.store(0, std::memory_order_relaxed);
    _total_edges.store(0, std::memory_order_relaxed);
    _total_evictable.store(0, std::memory_order_relaxed);
    _heat_decay_pending.store(false, std::memory_order_relaxed);
    _edge_saturations.store(0, std::memory_order_relaxed);
    _membership_words.reset();
    _membership_word_count = 0;
    _node_count = 0;
    _stable_slots = false;
}

void NeighborCacheMetadataCache::init(uint64_t capacity, uint8_t sig_threshold, uint64_t node_count,
                                      bool index_ids_only, Access access)
{
    clear();
    _shared = access != Access::SingleOwner;
    _shards.resize(access == Access::Sharded ? kShardCount : 1);
    for (auto &shard : _shards)
        if (!shard)
            shard = std::make_unique<Shard>();
    const char *edge_heat_decay = std::getenv("NEIGHBORCACHE_EDGE_HEAT_DECAY");
    _edge_heat_decay = edge_heat_decay == nullptr || std::strtoull(edge_heat_decay, nullptr, 10) != 0;
    const char *max_edges = std::getenv("NEIGHBORCACHE_MCACHE_MAX_EDGES");
    _max_edges = max_edges == nullptr
                     ? 64
                     : static_cast<size_t>(std::max<unsigned long long>(
                           1, std::min<unsigned long long>(256, std::strtoull(max_edges, nullptr, 10))));
    _node_count = node_count;
    const char *lookup_bitmap = std::getenv("NEIGHBORCACHE_MCACHE_LOOKUP_BITMAP");
    const bool accelerate_lookups = lookup_bitmap != nullptr && std::strcmp(lookup_bitmap, "0") != 0;
    if (accelerate_lookups && _node_count > 0)
    {
        _membership_word_count = (_node_count + 63) / 64;
        _membership_words = std::make_unique<std::atomic<uint64_t>[]>(_membership_word_count);
        for (uint64_t index = 0; index < _membership_word_count; ++index)
            _membership_words[index].store(0, std::memory_order_relaxed);
    }
    const uint64_t effective_capacity = std::min<uint64_t>(capacity, static_cast<uint64_t>(kInvalid));
    // Both IDs and capacity are distributed across shards in the same order.
    _stable_slots = index_ids_only && node_count != 0 && effective_capacity >= node_count;
    const char *eager_edges = std::getenv("NEIGHBORCACHE_MCACHE_EAGER_EDGE_SLOTS");
    const bool eager_edge_slots = eager_edges != nullptr && std::strcmp(eager_edges, "0") != 0;
    const uint32_t slot_stride =
        effective_capacity == 0 ? 0 : static_cast<uint32_t>((effective_capacity + shard_count() - 1) / shard_count());
    const uint64_t base_capacity = effective_capacity / shard_count();
    const uint64_t extra_shards = effective_capacity % shard_count();

    std::vector<std::unique_ptr<NeighborCacheTimedSharedMutexGuard<std::shared_mutex>>> locks;
    locks.reserve(shard_count());
    for (auto &shard : _shards)
        locks.emplace_back(
            std::make_unique<NeighborCacheTimedSharedMutexGuard<std::shared_mutex>>(shard->mu, NeighborCacheLockKind::Metadata, true, _shared));

    _sig_t.store(sig_threshold == 0 ? kDefaultSigT : sig_threshold, std::memory_order_relaxed);
    _slot_reciprocal = slot_stride == 0 ? 0 : UINT64_MAX / slot_stride;
    _slot_stride.store(slot_stride, std::memory_order_relaxed);
    for (size_t shard_id = 0; shard_id < shard_count(); ++shard_id)
    {
        Shard &shard = *_shards[shard_id];
        shard.capacity = base_capacity + (shard_id < extra_shards ? 1 : 0);
        shard.node_to_slot.max_load_factor(access == Access::SingleOwner ? 0.65f : 0.5f);
        shard.scores.assign(static_cast<size_t>(shard.capacity), 0);
        shard.slot_to_node.assign(static_cast<size_t>(shard.capacity), kInvalid);
        if (eager_edge_slots)
            shard.edges_by_slot.resize(static_cast<size_t>(shard.capacity));
        shard.evictable.assign(static_cast<size_t>(shard.capacity), 0);
        shard.next_unused_slot = 0;
        if (!_stable_slots)
        {
            shard.nonseed_prev.assign(static_cast<size_t>(shard.capacity), kInvalid);
            shard.nonseed_next.assign(static_cast<size_t>(shard.capacity), kInvalid);
        }
        shard.nonseed_head = kInvalid;
        shard.nonseed_tail = kInvalid;
    }
    _capacity.store(effective_capacity, std::memory_order_relaxed);
}

bool NeighborCacheMetadataCache::membership_contains(uint32_t node_id) const
{
    if (_membership_words == nullptr || node_id >= _node_count)
        return false;
    const uint64_t word = _membership_words[node_id >> 6].load(std::memory_order_acquire);
    return (word & (uint64_t{1} << (node_id & 63))) != 0;
}

void NeighborCacheMetadataCache::membership_set(uint32_t node_id)
{
    if (_membership_words == nullptr || node_id >= _node_count)
        return;
    _membership_words[node_id >> 6].fetch_or(uint64_t{1} << (node_id & 63), std::memory_order_release);
}

void NeighborCacheMetadataCache::membership_clear(uint32_t node_id)
{
    if (_membership_words == nullptr || node_id >= _node_count)
        return;
    _membership_words[node_id >> 6].fetch_and(~(uint64_t{1} << (node_id & 63)), std::memory_order_release);
}

uint64_t NeighborCacheMetadataCache::size() const
{
    return _total_size.load(std::memory_order_relaxed);
}

uint64_t NeighborCacheMetadataCache::edge_count() const
{
    return _total_edges.load(std::memory_order_relaxed);
}

uint64_t NeighborCacheMetadataCache::edge_descriptor_capacity_bytes() const
{
    uint64_t bytes = 0;
    for (const auto &shard : _shards)
    {
        NeighborCacheTimedSharedMutexGuard lock(shard->mu, NeighborCacheLockKind::Metadata, false, _shared);
        bytes += shard->edges_by_slot.capacity() * sizeof(std::vector<Edge>);
    }
    return bytes;
}

uint64_t NeighborCacheMetadataCache::heat_capacity_bytes() const
{
    uint64_t bytes = 0;
    for (const auto &shard : _shards)
        bytes += shard->scores.capacity() * sizeof(Heat);
    return bytes;
}

uint64_t NeighborCacheMetadataCache::evictable_size() const
{
    return _total_evictable.load(std::memory_order_relaxed);
}

void NeighborCacheMetadataCache::nonseed_unlink_unlocked(Shard &shard, uint32_t local_slot)
{
    const uint32_t prev = shard.nonseed_prev[local_slot];
    const uint32_t next = shard.nonseed_next[local_slot];
    if (prev != kInvalid)
        shard.nonseed_next[prev] = next;
    else
        shard.nonseed_head = next;
    if (next != kInvalid)
        shard.nonseed_prev[next] = prev;
    else
        shard.nonseed_tail = prev;
    shard.nonseed_prev[local_slot] = kInvalid;
    shard.nonseed_next[local_slot] = kInvalid;
}

void NeighborCacheMetadataCache::nonseed_push_front_unlocked(Shard &shard, uint32_t local_slot)
{
    shard.nonseed_prev[local_slot] = kInvalid;
    shard.nonseed_next[local_slot] = shard.nonseed_head;
    if (shard.nonseed_head != kInvalid)
        shard.nonseed_prev[shard.nonseed_head] = local_slot;
    else
        shard.nonseed_tail = local_slot;
    shard.nonseed_head = local_slot;
}

void NeighborCacheMetadataCache::touch_unlocked(Shard &shard, uint32_t local_slot)
{
    if (shard.evictable[local_slot] && shard.nonseed_head != local_slot)
    {
        nonseed_unlink_unlocked(shard, local_slot);
        nonseed_push_front_unlocked(shard, local_slot);
    }
}

uint32_t NeighborCacheMetadataCache::evict_one_unlocked(Shard &shard, size_t shard_id, uint32_t &evicted_slot,
                                                uint32_t &reclaimed_local_slot)
{
    evicted_slot = kInvalid;
    reclaimed_local_slot = kInvalid;
    if (shard.nonseed_tail == kInvalid)
        return kInvalid;

    const uint32_t victim_local_slot = shard.nonseed_tail;
    const uint32_t victim = shard.slot_to_node[victim_local_slot].load(std::memory_order_relaxed);
    auto eit = shard.node_to_slot.find(victim);
    if (eit == shard.node_to_slot.end())
        return kInvalid;

    const uint32_t local_slot = eit->second;
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    evicted_slot = static_cast<uint32_t>(static_cast<uint64_t>(shard_id) * slot_stride + local_slot);
    const bool was_evictable = shard.evictable[local_slot] != 0;
    const uint64_t removed_edges = shard.edges_by_slot[local_slot].size();
    if (was_evictable)
        nonseed_unlink_unlocked(shard, local_slot);
    membership_clear(victim);
    shard.node_to_slot.erase(eit);
    if (local_slot != kInvalid && local_slot < shard.capacity)
    {
        shard.slot_to_node[local_slot].store(kInvalid, std::memory_order_release);
        shard.scores[local_slot] = 0;
        std::vector<Edge>().swap(shard.edges_by_slot[local_slot]);
        shard.evictable[local_slot] = 0;
        reclaimed_local_slot = local_slot;
    }
    _total_size.fetch_sub(1, std::memory_order_relaxed);
    _total_edges.fetch_sub(removed_edges, std::memory_order_relaxed);
    if (was_evictable)
        _total_evictable.fetch_sub(1, std::memory_order_relaxed);
    return victim;
}

NeighborCacheMetadataCache::TouchResult NeighborCacheMetadataCache::touch_existing_unlocked(
    Shard &shard, size_t shard_id, uint32_t node_id, uint32_t local_slot, uint8_t heat_increment)
{
    Heat &score = shard.scores[local_slot];
    TouchResult r;
    if (heat_increment != 0)
    {
        if (heat_increment > kMaxHeat - score)
        {
            r.heat_update_deferred = true;
            _heat_decay_pending.store(true, std::memory_order_release);
        }
        else
        {
            score += heat_increment;
            if (score == kMaxHeat)
                _heat_decay_pending.store(true, std::memory_order_release);
        }
    }
    if (!_stable_slots)
        touch_unlocked(shard, local_slot);
    r.node_id = node_id;
    r.slot_id = static_cast<uint32_t>(static_cast<uint64_t>(shard_id) *
        _slot_stride.load(std::memory_order_relaxed) + local_slot);
    r.score = score;
    r.present = true;
    return r;
}

NeighborCacheMetadataCache::TouchResult NeighborCacheMetadataCache::insert_unlocked(
    Shard &shard, size_t shard_id, uint32_t node_id, uint8_t heat_increment)
{
    TouchResult r;
    if (shard.capacity == 0 || (_stable_slots && node_id >= _node_count))
        return r;

    uint32_t local_slot = kInvalid;
    if (shard.node_to_slot.size() >= shard.capacity)
    {
        if (_stable_slots)
            return r;
        uint32_t evicted_slot = kInvalid;
        const uint32_t ev = evict_one_unlocked(shard, shard_id, evicted_slot, local_slot);
        if (ev == kInvalid)
            return r;
        r.evicted_id = ev;
        r.evicted_slot = evicted_slot;
    }
    else
    {
        if (shard.next_unused_slot >= shard.capacity)
            return r;
        local_slot = shard.next_unused_slot;
        // The exclusive shard lock protects descriptor growth and slot reuse.
        // Grow descriptors only as slots become live; preserve stable slot IDs
        // and cap geometric growth at the configured logical capacity.
        if (local_slot >= shard.edges_by_slot.size())
        {
            if (local_slot >= shard.edges_by_slot.capacity())
            {
                const size_t next = static_cast<size_t>(std::min<uint64_t>(
                    shard.capacity, std::max<uint64_t>(64, shard.edges_by_slot.capacity() * 2ULL)));
                shard.edges_by_slot.reserve(next);
            }
            shard.edges_by_slot.resize(static_cast<size_t>(local_slot) + 1);
        }
        if (_shared && local_slot >= shard.edge_locks.size())
            shard.edge_locks.resize(static_cast<size_t>(local_slot) + 1);
        ++shard.next_unused_slot;
    }
    if (local_slot == kInvalid)
        return r;
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    const uint32_t slot_id =
        static_cast<uint32_t>(static_cast<uint64_t>(shard_id) * slot_stride + local_slot);
    shard.scores[local_slot] = heat_increment;
    shard.evictable[local_slot] = 1;
    shard.node_to_slot.insert({node_id, local_slot});
    if (!_stable_slots)
        nonseed_push_front_unlocked(shard, local_slot);
    shard.slot_to_node[local_slot].store(node_id, std::memory_order_release);
    _total_size.fetch_add(1, std::memory_order_relaxed);
    _total_evictable.fetch_add(1, std::memory_order_relaxed);
    // Publish membership only once the record is fully initialized.
    membership_set(node_id);
    r.node_id = node_id;
    r.slot_id = slot_id;
    r.score = heat_increment;
    r.present = true;
    r.inserted = true;
    return r;
}

NeighborCacheMetadataCache::TouchResult NeighborCacheMetadataCache::on_expand(uint32_t node_id, uint8_t heat_increment)
{
    const size_t shard_id = shard_index(node_id);
    Shard &shard = *_shards[shard_id];
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, true, _shared);
    const auto it = shard.node_to_slot.find(node_id);
    if (it != shard.node_to_slot.end())
        return touch_existing_unlocked(shard, shard_id, node_id, it->second, heat_increment);
    return insert_unlocked(shard, shard_id, node_id, heat_increment);
}

NeighborCacheMetadataCache::TouchResult NeighborCacheMetadataCache::on_edge(uint32_t parent, uint32_t child)
{
    return on_real_io_edge(parent, child, _max_edges);
}

NeighborCacheMetadataCache::TouchResult NeighborCacheMetadataCache::on_real_io_edge(uint32_t parent, uint32_t child,
                                                                    size_t max_edges)
{
    if (parent == child || max_edges == 0)
        return {};

    const size_t shard_id = shard_index(parent);
    Shard &shard = *_shards[shard_id];
    const uint32_t stride = _slot_stride.load(std::memory_order_relaxed);
    const auto update_existing = [&](uint32_t slot) {
        update_edge_unlocked(shard, slot, child, max_edges);
        TouchResult result;
        result.node_id = parent;
        result.slot_id = static_cast<uint32_t>(shard_id * stride + slot);
        result.score = shard.scores[slot];
        result.present = true;
        return result;
    };
    if (_shared)
    {
        NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, false, true);
        const auto it = shard.node_to_slot.find(parent);
        if (it != shard.node_to_slot.end())
        {
            const uint32_t slot = it->second;
            std::lock_guard<NeighborCacheSpinLock> record(shard.edge_locks[slot]);
            return update_existing(slot);
        }
    }

    // An insertion can race the shared-to-exclusive handoff. Neither hit path
    // changes recency: only node expansion refreshes an existing parent's LRU.
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, true, _shared);
    const auto it = shard.node_to_slot.find(parent);
    if (it != shard.node_to_slot.end())
        return update_existing(it->second);
    auto result = insert_unlocked(shard, shard_id, parent, 0);
    if (result.present)
        update_edge_unlocked(shard, result.slot_id - shard_id * stride, child, max_edges);
    return result;
}

void NeighborCacheMetadataCache::update_edge_unlocked(Shard &shard, uint32_t local_slot,
                                                     uint32_t child, size_t max_edges)
{
    if (!shard.seed_members_by_slot.empty())
    {
        auto seed = shard.seed_members_by_slot.find(static_cast<uint32_t>(local_slot));
        if (seed != shard.seed_members_by_slot.end())
        {
            auto &members = seed.value();
            for (size_t i = 1; i < members.count; ++i)
            {
                auto &edge = members.edges[i];
                if (edge.child() != child)
                    continue;
                if (edge.heat() == 0)
                    _total_edges.fetch_add(1, std::memory_order_relaxed);
                count_edge_unlocked(edge);
                return;
            }
        }
    }
    std::vector<Edge> &edges = shard.edges_by_slot[local_slot];
    for (Edge &edge : edges)
    {
        if (edge.child() == child)
        {
            count_edge_unlocked(edge);
            return;
        }
    }

    if (edges.size() < max_edges)
    {
        edges.emplace_back(child, 1);
        _total_edges.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    auto weakest = std::min_element(edges.begin(), edges.end(),
                                    [](const Edge &a, const Edge &b) { return a.heat() < b.heat(); });
    const uint8_t inherited_heat = weakest->heat();
    *weakest = Edge(child, inherited_heat);
    count_edge_unlocked(*weakest);
}

bool NeighborCacheMetadataCache::seed_member_slot(uint32_t seed_id, uint32_t page, uint32_t node_id, uint16_t &slot) const
{
    const Shard &shard = *_shards[shard_index(seed_id)];
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, false, _shared);
    const auto entry = shard.node_to_slot.find(seed_id);
    if (entry == shard.node_to_slot.end())
        return false;
    const auto seed = shard.seed_members_by_slot.find(entry->second);
    if (seed == shard.seed_members_by_slot.end() || seed->second.page != page)
        return false;
    const auto &members = seed->second;
    for (uint16_t i = 0; i < members.count; ++i)
    {
        if (members.edges[i].child() == node_id)
        {
            slot = i;
            return true;
        }
    }
    return false;
}

void NeighborCacheMetadataCache::set_seed_members(uint32_t seed_id, uint32_t page, const uint32_t *ids, size_t count)
{
    if (!_stable_slots || count > kMaxSeedPageMembers || (count != 0 && (ids == nullptr || ids[0] != seed_id)))
        throw std::logic_error("Invalid M-cache seed membership");
    Shard &shard = *_shards[shard_index(seed_id)];
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, true, _shared);
    const auto entry = shard.node_to_slot.find(seed_id);
    if (entry == shard.node_to_slot.end())
        throw std::logic_error("Published seed has no M-cache slot");
    const uint32_t slot = entry->second;
    auto &outside = shard.edges_by_slot[slot];
    auto seed = shard.seed_members_by_slot.find(slot);
    if (seed != shard.seed_members_by_slot.end() && seed->second.page != page)
        throw std::logic_error("M-cache seed page changed during publication");
    if (seed != shard.seed_members_by_slot.end() && seed->second.count == count &&
        std::equal(seed->second.edges.begin(), seed->second.edges.begin() + count, ids,
                   [](const Edge &edge, uint32_t id) { return edge.child() == id; }))
        return;

    SeedMembers next;
    next.page = page;
    next.count = static_cast<uint8_t>(count);
    for (size_t i = 0; i < count; ++i)
        next.edges[i] = Edge(ids[i], 0);
    const auto keep_inside = [&](const Edge &edge) {
        for (size_t i = 1; i < count; ++i)
        {
            if (next.edges[i].child() != edge.child())
                continue;
            next.edges[i] = edge;
            return true;
        }
        return false;
    };
    outside.erase(std::remove_if(outside.begin(), outside.end(), keep_inside), outside.end());
    if (seed != shard.seed_members_by_slot.end())
    {
        for (size_t i = 1; i < seed->second.count; ++i)
        {
            const Edge &edge = seed->second.edges[i];
            if (!keep_inside(edge) && edge.heat() != 0)
                outside.push_back(edge);
        }
        if (count == 0)
            shard.seed_members_by_slot.erase(seed);
        else
            seed.value() = next;
    }
    else if (count != 0)
        shard.seed_members_by_slot.emplace(slot, next);
}

bool NeighborCacheMetadataCache::contains(uint32_t node_id) const
{
    // Only a definite absence bypasses the shard lock. IDs beyond the bitmap
    // retain the ordinary lookup path, and hits still validate the live map.
    if (_membership_words != nullptr && node_id < _node_count && !membership_contains(node_id))
        return false;
    const Shard &shard = *_shards[shard_index(node_id)];
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, false, _shared);
    return shard.node_to_slot.find(node_id) != shard.node_to_slot.end();
}

uint32_t NeighborCacheMetadataCache::slot_of(uint32_t node_id) const
{
    // Only a definite absence bypasses the shard lock. IDs beyond the bitmap
    // retain the ordinary lookup path, and hits still validate the live map.
    if (_membership_words != nullptr && node_id < _node_count && !membership_contains(node_id))
        return kInvalid;
    const Shard &shard = *_shards[shard_index(node_id)];
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, false, _shared);
    const auto it = shard.node_to_slot.find(node_id);
    if (it == shard.node_to_slot.end())
        return kInvalid;
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    return static_cast<uint32_t>(static_cast<uint64_t>(shard_index(node_id)) * slot_stride + it->second);
}

std::pair<uint32_t, uint32_t> NeighborCacheMetadataCache::split_slot(uint32_t slot_id, uint32_t stride) const
{
    if (shard_count() == 1)
        return {0, slot_id};
#if defined(__SIZEOF_INT128__)
    // For 32-bit n and d, (n+1)*d < 2^64. Scaling n+1 by a reciprocal
    // rounded strictly down gives floor(n/d), including at exact boundaries.
    const uint64_t next = uint64_t{slot_id} + 1;
    const uint32_t quotient = static_cast<uint32_t>((static_cast<__uint128_t>(next) * _slot_reciprocal) >> 64);
#else
    const uint32_t quotient = slot_id / stride;
#endif
    return {quotient, slot_id - quotient * stride};
}

uint32_t NeighborCacheMetadataCache::node_at(uint32_t slot_id) const
{
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    if (slot_stride == 0 || slot_id == kInvalid)
        return kInvalid;
    const auto [shard_id, local_slot] = split_slot(slot_id, slot_stride);
    if (shard_id >= shard_count())
        return kInvalid;
    const Shard &shard = *_shards[shard_id];
    const auto *entry = shard.slot_to_node.find(local_slot);
    return entry == nullptr ? kInvalid : entry->load(std::memory_order_acquire);
}

bool NeighborCacheMetadataCache::node_score_at(uint32_t slot_id, uint32_t &node_id, Heat &score) const
{
    node_id = kInvalid;
    score = 0;
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    if (slot_stride == 0 || slot_id == kInvalid)
        return false;
    const auto [shard_id, local_slot] = split_slot(slot_id, slot_stride);
    if (shard_id >= shard_count())
        return false;
    const Shard &shard = *_shards[shard_id];
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, false, _shared);
    if (local_slot >= shard.next_unused_slot ||
        shard.slot_to_node[local_slot].load(std::memory_order_relaxed) == kInvalid)
        return false;
    node_id = shard.slot_to_node[local_slot].load(std::memory_order_relaxed);
    score = shard.scores[local_slot];
    return true;
}

NeighborCacheMetadataCache::Heat NeighborCacheMetadataCache::score(uint32_t node_id) const
{
    // Only a definite absence bypasses the shard lock. IDs beyond the bitmap
    // retain the ordinary lookup path, and hits still validate the live map.
    if (_membership_words != nullptr && node_id < _node_count && !membership_contains(node_id))
        return 0;
    const Shard &shard = *_shards[shard_index(node_id)];
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, false, _shared);
    const auto it = shard.node_to_slot.find(node_id);
    return it == shard.node_to_slot.end() ? 0 : shard.scores[it->second];
}

NeighborCacheMetadataCache::Heat NeighborCacheMetadataCache::score_at(uint32_t slot_id) const
{
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    if (slot_stride == 0 || slot_id == kInvalid)
        return 0;
    const auto [shard_id, local_slot] = split_slot(slot_id, slot_stride);
    if (shard_id >= shard_count())
        return 0;
    const Shard &shard = *_shards[shard_id];
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, false, _shared);
    if (local_slot >= shard.next_unused_slot ||
        shard.slot_to_node[local_slot].load(std::memory_order_relaxed) == kInvalid)
        return 0;
    return shard.scores[local_slot];
}

bool NeighborCacheMetadataCache::set_evictable(uint32_t node_id, bool evictable)
{
    return set_evictable_from_slot(node_id, kInvalid, evictable);
}

bool NeighborCacheMetadataCache::set_evictable_from_slot(uint32_t node_id, uint32_t slot_id, bool evictable)
{
    // Only a definite absence bypasses the shard lock. IDs beyond the bitmap
    // retain the ordinary lookup path, and hits still validate the live map.
    if (_membership_words != nullptr && node_id < _node_count && !membership_contains(node_id))
        return false;
    const size_t shard_id = shard_index(node_id);
    Shard &shard = *_shards[shard_id];
    NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, true, _shared);
    uint32_t local_slot = local_slot_hint(shard, shard_id, node_id, slot_id);
    if (local_slot == kInvalid)
    {
        const auto it = shard.node_to_slot.find(node_id);
        if (it == shard.node_to_slot.end())
            return false;
        local_slot = it->second;
    }
    if ((shard.evictable[local_slot] != 0) == evictable)
        return true;
    if (evictable)
    {
        if (!_stable_slots)
            nonseed_push_front_unlocked(shard, local_slot);
        _total_evictable.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        if (!_stable_slots)
            nonseed_unlink_unlocked(shard, local_slot);
        _total_evictable.fetch_sub(1, std::memory_order_relaxed);
    }
    shard.evictable[local_slot] = evictable ? 1 : 0;
    return true;
}

void NeighborCacheMetadataCache::halve_heat()
{
    std::vector<std::unique_ptr<NeighborCacheTimedSharedMutexGuard<std::shared_mutex>>> locks;
    locks.reserve(shard_count());
    for (auto &shard : _shards)
        locks.emplace_back(
            std::make_unique<NeighborCacheTimedSharedMutexGuard<std::shared_mutex>>(shard->mu, NeighborCacheLockKind::Metadata, true, _shared));

    uint64_t removed_edges = 0;
    for (auto &shard_ptr : _shards)
    {
        Shard &shard = *shard_ptr;
        for (auto it = shard.node_to_slot.begin(); it != shard.node_to_slot.end(); ++it)
        {
            const uint32_t local_slot = it->second;
            shard.scores[local_slot] /= 2;
            if (_edge_heat_decay)
            {
                std::vector<Edge> &edges = shard.edges_by_slot[local_slot];
                size_t kept = 0;
                for (Edge &edge : edges)
                {
                    edge.set_heat(edge.heat() / 2);
                    if (edge.heat() != 0)
                        edges[kept++] = edge;
                }
                removed_edges += edges.size() - kept;
                edges.resize(kept);
                if (!shard.seed_members_by_slot.empty())
                {
                    auto seed = shard.seed_members_by_slot.find(local_slot);
                    if (seed != shard.seed_members_by_slot.end())
                    {
                        auto &members = seed.value();
                        for (size_t i = 1; i < members.count; ++i)
                        {
                            auto &edge = members.edges[i];
                            const uint8_t heat = edge.heat();
                            edge.set_heat(heat / 2);
                            removed_edges += heat == 1;
                        }
                    }
                }
            }
        }
    }
    if (removed_edges != 0)
        _total_edges.fetch_sub(removed_edges, std::memory_order_relaxed);
    _heat_decay_pending.store(false, std::memory_order_release);
}

uint64_t NeighborCacheMetadataCache::edge_saturations() const
{
    return _edge_saturations.load(std::memory_order_relaxed);
}

void NeighborCacheMetadataCache::count_edge_unlocked(Edge &edge)
{
    const uint8_t heat = edge.heat();
    if (heat == std::numeric_limits<uint8_t>::max())
        _edge_saturations.fetch_add(1, std::memory_order_relaxed);
    else
        edge.set_heat(heat + 1);
}

void NeighborCacheMetadataCache::copy_snapshot_edges_unlocked(const Shard &shard, uint32_t local_slot,
                                                                  Snapshot &out, bool include_all_edges) const
{
    std::unique_lock<NeighborCacheSpinLock> record;
    if (_shared)
        record = std::unique_lock<NeighborCacheSpinLock>(shard.edge_locks[local_slot]);
    out.score = shard.scores[local_slot];
    out.significant.clear();
    const auto &edges = shard.edges_by_slot[local_slot];
    out.significant.reserve(edges.size());
    const uint8_t threshold = _sig_t.load(std::memory_order_relaxed);
    for (const Edge &edge : edges)
    {
        const uint8_t heat = edge.heat();
        if (include_all_edges || heat >= threshold)
            out.significant.emplace_back(edge.child(), heat);
    }
    if (!shard.seed_members_by_slot.empty())
    {
        const auto seed = shard.seed_members_by_slot.find(local_slot);
        if (seed != shard.seed_members_by_slot.end())
        {
            const auto &members = seed->second;
            for (size_t i = 1; i < members.count; ++i)
            {
                const Edge &edge = members.edges[i];
                const uint8_t heat = edge.heat();
                if (heat != 0 && (include_all_edges || heat >= threshold))
                    out.significant.emplace_back(edge.child(), heat);
            }
        }
    }
}

void NeighborCacheMetadataCache::sort_snapshot_edges(Snapshot &out)
{
    std::sort(out.significant.begin(), out.significant.end(), [](const auto &a, const auto &b) {
        if (a.second != b.second)
            return a.second > b.second;
        return a.first < b.first;
    });
}

uint32_t NeighborCacheMetadataCache::local_slot_hint(const Shard &shard, size_t shard_id, uint32_t node_id,
                                                       uint32_t slot_id) const
{
    if (slot_id == kInvalid || node_id == kInvalid)
        return kInvalid;
    const uint32_t stride = _slot_stride.load(std::memory_order_relaxed);
    const uint64_t base = static_cast<uint64_t>(shard_id) * stride;
    if (slot_id < base || slot_id - base >= stride)
        return kInvalid;
    const uint32_t local_slot = static_cast<uint32_t>(slot_id - base);
    if (local_slot >= shard.next_unused_slot ||
        shard.slot_to_node[local_slot].load(std::memory_order_relaxed) != node_id)
        return kInvalid;
    return local_slot;
}

bool NeighborCacheMetadataCache::snapshot_from_slot(uint32_t node_id, uint32_t slot_id, Snapshot &out,
                                                    bool include_all_edges) const
{
    if (node_id == kInvalid)
        return false;
    const size_t shard_id = shard_index(node_id);
    const Shard &shard = *_shards[shard_id];
    {
        NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, false, _shared);
        uint32_t local_slot = local_slot_hint(shard, shard_id, node_id, slot_id);
        if (local_slot == kInvalid)
        {
            // A shared cache may have recycled this hint since heap selection.
            if (!_shared && _shards.size() == 1)
                return false;
            const auto it = shard.node_to_slot.find(node_id);
            if (it == shard.node_to_slot.end())
                return false;
            local_slot = it->second;
        }
        const uint32_t stride = _slot_stride.load(std::memory_order_relaxed);
        out.slot_id = static_cast<uint32_t>(static_cast<uint64_t>(shard_id) * stride + local_slot);
        copy_snapshot_edges_unlocked(shard, local_slot, out, include_all_edges);
    }
    sort_snapshot_edges(out);
    return true;
}

bool NeighborCacheMetadataCache::snapshot(uint32_t node_id, Snapshot &out, bool include_all_edges) const
{
    // Only a definite absence bypasses the shard lock. IDs beyond the bitmap
    // retain the ordinary lookup path, and hits still validate the live map.
    if (_membership_words != nullptr && node_id < _node_count && !membership_contains(node_id))
        return false;
    const Shard &shard = *_shards[shard_index(node_id)];
    {
        NeighborCacheTimedSharedMutexGuard lock(shard.mu, NeighborCacheLockKind::Metadata, false, _shared);
        const auto it = shard.node_to_slot.find(node_id);
        if (it == shard.node_to_slot.end())
            return false;
        const uint32_t local_slot = it->second;
        const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
        out.slot_id =
            static_cast<uint32_t>(static_cast<uint64_t>(shard_index(node_id)) * slot_stride + local_slot);
        copy_snapshot_edges_unlocked(shard, local_slot, out, include_all_edges);
    }
    sort_snapshot_edges(out);
    return true;
}

} // namespace diskann
