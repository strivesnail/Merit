#include "merit_metadata_cache.h"
#include "common_includes.h"
#include "merit_lock_metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <thread>

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

    std::vector<std::unique_ptr<MeritTimedSharedMutexGuard>> locks;
    locks.reserve(kShardCount);
    for (auto &shard : _shards)
        locks.emplace_back(
            std::make_unique<MeritTimedSharedMutexGuard>(shard->mu, MeritLockKind::Metadata, true));

    for (auto &shard_ptr : _shards)
    {
        Shard &shard = *shard_ptr;
        shard.node_to_slot.clear();
        shard.scores.clear();
        shard.slot_to_node.reset();
        shard.edges_by_slot.clear();
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
    _adaptive_update_enabled = false;
    _membership_words.reset();
    _membership_word_count = 0;
    _node_count = 0;
    _update_denominator.store(1, std::memory_order_relaxed);
    for (size_t index = 0; index < kShardCount; ++index)
    {
        _adaptive_counters[index].observed_hits.store(0, std::memory_order_relaxed);
        _adaptive_counters[index].observed_misses.store(0, std::memory_order_relaxed);
        _adaptive_counters[index].performed_updates.store(0, std::memory_order_relaxed);
        _adaptive_counters[index].skipped_updates.store(0, std::memory_order_relaxed);
    }
    _adaptive_query_count.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> adaptive_lock(_adaptive_mu);
        _adaptive_last_hits = 0;
        _adaptive_last_misses = 0;
        _adaptive_ewma_hit_rate = 0.0;
        _adaptive_low_stages = 0;
        _adaptive_high_stages = 0;
        _adaptive_transitions = 0;
        _adaptive_ewma_initialized = false;
    }
}

void MeritMetadataCache::init(uint64_t capacity, uint8_t sig_threshold, uint64_t node_count)
{
    clear();
    const char *adaptive_update = std::getenv("MERIT_MCACHE_ADAPTIVE_UPDATE");
    _adaptive_update_enabled =
        adaptive_update != nullptr && std::strtoull(adaptive_update, nullptr, 10) != 0;
    const char *edge_heat_decay = std::getenv("MERIT_EDGE_HEAT_DECAY");
    _edge_heat_decay = edge_heat_decay == nullptr || std::strtoull(edge_heat_decay, nullptr, 10) != 0;
    const char *max_edges = std::getenv("MERIT_MCACHE_MAX_EDGES");
    _max_edges = max_edges == nullptr
                     ? 64
                     : static_cast<size_t>(std::max<unsigned long long>(
                           1, std::min<unsigned long long>(256, std::strtoull(max_edges, nullptr, 10))));
    _node_count = node_count;
    if (_adaptive_update_enabled && _node_count > 0)
    {
        _membership_word_count = (_node_count + 63) / 64;
        _membership_words = std::make_unique<std::atomic<uint64_t>[]>(_membership_word_count);
        for (uint64_t index = 0; index < _membership_word_count; ++index)
            _membership_words[index].store(0, std::memory_order_relaxed);
    }
    const uint64_t effective_capacity = std::min<uint64_t>(capacity, static_cast<uint64_t>(kInvalid));
    const uint32_t slot_stride =
        effective_capacity == 0 ? 0 : static_cast<uint32_t>((effective_capacity + kShardCount - 1) / kShardCount);
    const uint64_t base_capacity = effective_capacity / kShardCount;
    const uint64_t extra_shards = effective_capacity % kShardCount;

    std::vector<std::unique_ptr<MeritTimedSharedMutexGuard>> locks;
    locks.reserve(kShardCount);
    for (auto &shard : _shards)
        locks.emplace_back(
            std::make_unique<MeritTimedSharedMutexGuard>(shard->mu, MeritLockKind::Metadata, true));

    _sig_t.store(sig_threshold == 0 ? kDefaultSigT : sig_threshold, std::memory_order_relaxed);
    _slot_stride.store(slot_stride, std::memory_order_relaxed);
    for (size_t shard_id = 0; shard_id < kShardCount; ++shard_id)
    {
        Shard &shard = *_shards[shard_id];
        shard.capacity = base_capacity + (shard_id < extra_shards ? 1 : 0);
        shard.scores.assign(static_cast<size_t>(shard.capacity), 0.0f);
        shard.slot_to_node.reset(new std::atomic<uint32_t>[static_cast<size_t>(shard.capacity)]);
        for (uint64_t slot = 0; slot < shard.capacity; ++slot)
            shard.slot_to_node[slot].store(kInvalid, std::memory_order_relaxed);
        shard.edges_by_slot.resize(static_cast<size_t>(shard.capacity));
        shard.evictable.assign(static_cast<size_t>(shard.capacity), 0);
        shard.next_unused_slot = 0;
        shard.nonseed_prev.assign(static_cast<size_t>(shard.capacity), kInvalid);
        shard.nonseed_next.assign(static_cast<size_t>(shard.capacity), kInvalid);
        shard.nonseed_head = kInvalid;
        shard.nonseed_tail = kInvalid;
    }
    _capacity.store(effective_capacity, std::memory_order_relaxed);
}

bool MeritMetadataCache::membership_contains(uint32_t node_id) const
{
    if (_membership_words == nullptr || node_id >= _node_count)
        return false;
    const uint64_t word = _membership_words[node_id >> 6].load(std::memory_order_acquire);
    return (word & (uint64_t{1} << (node_id & 63))) != 0;
}

void MeritMetadataCache::membership_set(uint32_t node_id)
{
    if (_membership_words == nullptr || node_id >= _node_count)
        return;
    _membership_words[node_id >> 6].fetch_or(uint64_t{1} << (node_id & 63), std::memory_order_release);
}

void MeritMetadataCache::membership_clear(uint32_t node_id)
{
    if (_membership_words == nullptr || node_id >= _node_count)
        return;
    _membership_words[node_id >> 6].fetch_and(~(uint64_t{1} << (node_id & 63)), std::memory_order_release);
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

void MeritMetadataCache::nonseed_unlink_unlocked(Shard &shard, uint32_t local_slot)
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

void MeritMetadataCache::nonseed_push_front_unlocked(Shard &shard, uint32_t local_slot)
{
    shard.nonseed_prev[local_slot] = kInvalid;
    shard.nonseed_next[local_slot] = shard.nonseed_head;
    if (shard.nonseed_head != kInvalid)
        shard.nonseed_prev[shard.nonseed_head] = local_slot;
    else
        shard.nonseed_tail = local_slot;
    shard.nonseed_head = local_slot;
}

void MeritMetadataCache::touch_unlocked(Shard &shard, uint32_t local_slot)
{
    if (shard.evictable[local_slot] && shard.nonseed_head != local_slot)
    {
        nonseed_unlink_unlocked(shard, local_slot);
        nonseed_push_front_unlocked(shard, local_slot);
    }
}

uint32_t MeritMetadataCache::evict_one_unlocked(Shard &shard, size_t shard_id, uint32_t &evicted_slot,
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
        shard.scores[local_slot] = 0.0f;
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

MeritMetadataCache::TouchResult MeritMetadataCache::touch_or_insert_unlocked(Shard &shard, size_t shard_id,
                                                                            uint32_t node_id, float score_delta)
{
    TouchResult r;
    auto it = shard.node_to_slot.find(node_id);
    if (it != shard.node_to_slot.end())
    {
        const uint32_t local_slot = it->second;
        if (score_delta > 0.0f)
            shard.scores[local_slot] += score_delta;
        touch_unlocked(shard, local_slot);
        const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
        r.node_id = node_id;
        r.slot_id = static_cast<uint32_t>(static_cast<uint64_t>(shard_id) * slot_stride + local_slot);
        r.score = shard.scores[local_slot];
        r.present = true;
        return r;
    }

    if (shard.capacity == 0)
        return r;

    uint32_t local_slot = kInvalid;
    if (shard.node_to_slot.size() >= shard.capacity)
    {
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
        local_slot = shard.next_unused_slot++;
    }
    if (local_slot == kInvalid)
        return r;
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    const uint32_t slot_id =
        static_cast<uint32_t>(static_cast<uint64_t>(shard_id) * slot_stride + local_slot);
    shard.scores[local_slot] = std::max(score_delta, 0.0f);
    shard.evictable[local_slot] = 1;
    membership_set(node_id);
    shard.node_to_slot.insert({node_id, local_slot});
    nonseed_push_front_unlocked(shard, local_slot);
    shard.slot_to_node[local_slot].store(node_id, std::memory_order_release);
    _total_size.fetch_add(1, std::memory_order_relaxed);
    _total_evictable.fetch_add(1, std::memory_order_relaxed);
    r.node_id = node_id;
    r.slot_id = slot_id;
    r.score = shard.scores[local_slot];
    r.present = true;
    r.inserted = true;
    return r;
}

MeritMetadataCache::TouchResult MeritMetadataCache::on_expand(uint32_t node_id,
                                                              const std::atomic<float> &score_unit)
{
    const size_t shard_id = shard_index(node_id);
    Shard &shard = *_shards[shard_id];
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Metadata, true);
    return touch_or_insert_unlocked(shard, shard_id, node_id, score_unit.load(std::memory_order_acquire));
}

MeritMetadataCache::UpdateDecision MeritMetadataCache::plan_expand_update(uint32_t node_id, bool force_update,
                                                                           bool suppress_update)
{
    UpdateDecision decision;
    if (!_adaptive_update_enabled)
    {
        decision.update = !suppress_update;
        return decision;
    }

    const size_t counter_shard = shard_index(node_id);
    decision.observed_hit = membership_contains(node_id);
    if (decision.observed_hit)
        _adaptive_counters[counter_shard].observed_hits.fetch_add(1, std::memory_order_relaxed);
    else
        _adaptive_counters[counter_shard].observed_misses.fetch_add(1, std::memory_order_relaxed);

    const uint32_t denominator = _update_denominator.load(std::memory_order_relaxed);
    if (force_update)
        decision.update = true;
    else if (suppress_update)
        decision.update = false;
    else if (denominator > 1)
    {
        thread_local uint32_t state =
            static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())) ^ 0x9e3779b9u;
        state = state * 1664525u + 1013904223u;
        decision.update = state % denominator == 0;
    }

    if (decision.update)
        _adaptive_counters[counter_shard].performed_updates.fetch_add(1, std::memory_order_relaxed);
    else
        _adaptive_counters[counter_shard].skipped_updates.fetch_add(1, std::memory_order_relaxed);
    return decision;
}

void MeritMetadataCache::on_query_end()
{
    if (!_adaptive_update_enabled)
        return;

    constexpr uint64_t stage_queries = 1000;
    constexpr uint64_t half_life_queries = 25000;
    constexpr double low_hit_rate = 0.20;
    constexpr double high_hit_rate = 0.40;
    const uint64_t query_count = _adaptive_query_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (query_count % stage_queries != 0)
        return;

    std::lock_guard<std::mutex> lock(_adaptive_mu);
    uint64_t hits = 0;
    uint64_t misses = 0;
    for (size_t index = 0; index < kShardCount; ++index)
    {
        hits += _adaptive_counters[index].observed_hits.load(std::memory_order_relaxed);
        misses += _adaptive_counters[index].observed_misses.load(std::memory_order_relaxed);
    }
    const uint64_t hit_delta = hits - _adaptive_last_hits;
    const uint64_t miss_delta = misses - _adaptive_last_misses;
    _adaptive_last_hits = hits;
    _adaptive_last_misses = misses;
    const uint64_t access_delta = hit_delta + miss_delta;
    if (access_delta == 0)
        return;

    const double hit_rate = static_cast<double>(hit_delta) / static_cast<double>(access_delta);
    const double decay =
        std::pow(0.5, static_cast<double>(stage_queries) / static_cast<double>(half_life_queries));
    if (!_adaptive_ewma_initialized)
    {
        _adaptive_ewma_hit_rate = hit_rate;
        _adaptive_ewma_initialized = true;
    }
    else
    {
        _adaptive_ewma_hit_rate =
            decay * _adaptive_ewma_hit_rate + (1.0 - decay) * hit_rate;
    }

    _adaptive_low_stages = hit_rate < low_hit_rate ? _adaptive_low_stages + 1 : 0;
    _adaptive_high_stages = hit_rate > high_hit_rate ? _adaptive_high_stages + 1 : 0;
    const uint32_t half_life_stages =
        static_cast<uint32_t>(half_life_queries / stage_queries);
    const uint32_t recovery_stages = std::max<uint32_t>(1, half_life_stages / 4);
    const uint32_t current = _update_denominator.load(std::memory_order_relaxed);
    uint32_t next = current;
    if (_adaptive_low_stages >= half_life_stages)
    {
        if (current == 1)
            next = 10;
        else
            next = 100;
        _adaptive_low_stages = 0;
        _adaptive_high_stages = 0;
    }
    else if (_adaptive_high_stages >= recovery_stages)
    {
        if (current == 100)
            next = 10;
        else
            next = 1;
        _adaptive_low_stages = 0;
        _adaptive_high_stages = 0;
    }
    if (next != current)
    {
        _update_denominator.store(next, std::memory_order_relaxed);
        ++_adaptive_transitions;
        std::cout << "MERIT mcache adaptive: queries=" << query_count
                  << " update=1/" << current << "->1/" << next
                  << " window_hit_rate=" << hit_rate
                  << " ewma_hit_rate=" << _adaptive_ewma_hit_rate
                  << " maintenance=" << (next >= 100 ? "suspended" : "active") << std::endl;
    }
}

double MeritMetadataCache::adaptive_hit_rate() const
{
    std::lock_guard<std::mutex> lock(_adaptive_mu);
    return _adaptive_ewma_hit_rate;
}

uint64_t MeritMetadataCache::observed_accesses() const
{
    uint64_t total = 0;
    for (size_t index = 0; index < kShardCount; ++index)
    {
        total += _adaptive_counters[index].observed_hits.load(std::memory_order_relaxed);
        total += _adaptive_counters[index].observed_misses.load(std::memory_order_relaxed);
    }
    return total;
}

uint64_t MeritMetadataCache::performed_updates() const
{
    uint64_t total = 0;
    for (size_t index = 0; index < kShardCount; ++index)
        total += _adaptive_counters[index].performed_updates.load(std::memory_order_relaxed);
    return total;
}

uint64_t MeritMetadataCache::skipped_updates() const
{
    uint64_t total = 0;
    for (size_t index = 0; index < kShardCount; ++index)
        total += _adaptive_counters[index].skipped_updates.load(std::memory_order_relaxed);
    return total;
}

uint64_t MeritMetadataCache::adaptive_transition_count() const
{
    std::lock_guard<std::mutex> lock(_adaptive_mu);
    return _adaptive_transitions;
}

MeritMetadataCache::TouchResult MeritMetadataCache::on_edge(uint32_t parent, uint32_t child, float unit)
{
    thread_local std::mt19937 rng{std::random_device{}()};
    TouchResult r;
    const uint32_t update_denominator = _update_denominator.load(std::memory_order_relaxed);
    if (_adaptive_update_enabled && update_denominator > 1 && (rng() % update_denominator) != 0u)
        return r;
    if ((rng() % 10u) != 0u)
        return r;

    const size_t shard_id = shard_index(parent);
    Shard &shard = *_shards[shard_id];
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Metadata, true);
    r = touch_or_insert_unlocked(shard, shard_id, parent, 0.0f);
    if (!r.present)
        return r;

    auto eit = shard.node_to_slot.find(parent);
    if (eit == shard.node_to_slot.end())
        return r;
    std::vector<Edge> &edges = shard.edges_by_slot[eit->second];
    const uint8_t increment = edge_increment(unit);

    for (Edge &edge : edges)
    {
        if (edge.child() == child)
        {
            add_edge_heat_unlocked(edge, increment);
            return r;
        }
    }

    if (edges.size() < _max_edges)
    {
        edges.emplace_back(child, increment);
        _total_edges.fetch_add(1, std::memory_order_relaxed);
        return r;
    }

    auto weakest = std::min_element(edges.begin(), edges.end(),
                                    [](const Edge &a, const Edge &b) { return a.heat() < b.heat(); });
    const uint8_t inherited_heat = weakest->heat();
    *weakest = Edge(child, inherited_heat);
    add_edge_heat_unlocked(*weakest, increment);
    return r;
}

MeritMetadataCache::TouchResult MeritMetadataCache::on_real_io_edge(uint32_t parent, uint32_t child,
                                                                    size_t max_edges, float unit)
{
    TouchResult r;
    if (parent == child || max_edges == 0)
        return r;

    const size_t shard_id = shard_index(parent);
    Shard &shard = *_shards[shard_id];
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Metadata, true);
    r = touch_or_insert_unlocked(shard, shard_id, parent, 0.0f);
    if (!r.present)
        return r;

    const auto parent_it = shard.node_to_slot.find(parent);
    if (parent_it == shard.node_to_slot.end())
        return r;
    std::vector<Edge> &edges = shard.edges_by_slot[parent_it->second];
    const uint8_t increment = edge_increment(unit);
    for (Edge &edge : edges)
    {
        if (edge.child() == child)
        {
            add_edge_heat_unlocked(edge, increment);
            return r;
        }
    }

    if (edges.size() < max_edges)
    {
        edges.emplace_back(child, increment);
        _total_edges.fetch_add(1, std::memory_order_relaxed);
        return r;
    }

    auto weakest = std::min_element(edges.begin(), edges.end(),
                                    [](const Edge &a, const Edge &b) { return a.heat() < b.heat(); });
    const uint8_t inherited_heat = weakest->heat();
    *weakest = Edge(child, inherited_heat);
    add_edge_heat_unlocked(*weakest, increment);
    return r;
}

bool MeritMetadataCache::contains(uint32_t node_id) const
{
    const Shard &shard = *_shards[shard_index(node_id)];
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Metadata, false);
    return shard.node_to_slot.find(node_id) != shard.node_to_slot.end();
}

uint32_t MeritMetadataCache::slot_of(uint32_t node_id) const
{
    const Shard &shard = *_shards[shard_index(node_id)];
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Metadata, false);
    const auto it = shard.node_to_slot.find(node_id);
    if (it == shard.node_to_slot.end())
        return kInvalid;
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    return static_cast<uint32_t>(static_cast<uint64_t>(shard_index(node_id)) * slot_stride + it->second);
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
    return local_slot < shard.capacity ? shard.slot_to_node[local_slot].load(std::memory_order_acquire) : kInvalid;
}

bool MeritMetadataCache::node_score_at(uint32_t slot_id, uint32_t &node_id, float &score) const
{
    node_id = kInvalid;
    score = 0.0f;
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    if (slot_stride == 0 || slot_id == kInvalid)
        return false;
    const size_t shard_id = slot_id / slot_stride;
    const uint32_t local_slot = slot_id % slot_stride;
    if (shard_id >= kShardCount)
        return false;
    const Shard &shard = *_shards[shard_id];
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Metadata, false);
    if (local_slot >= shard.capacity ||
        shard.slot_to_node[local_slot].load(std::memory_order_relaxed) == kInvalid)
        return false;
    node_id = shard.slot_to_node[local_slot].load(std::memory_order_relaxed);
    score = shard.scores[local_slot];
    return true;
}

float MeritMetadataCache::score(uint32_t node_id) const
{
    const Shard &shard = *_shards[shard_index(node_id)];
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Metadata, false);
    const auto it = shard.node_to_slot.find(node_id);
    return it == shard.node_to_slot.end() ? 0.0f : shard.scores[it->second];
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
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Metadata, false);
    if (local_slot >= shard.capacity ||
        shard.slot_to_node[local_slot].load(std::memory_order_relaxed) == kInvalid)
        return 0.0f;
    return shard.scores[local_slot];
}

bool MeritMetadataCache::set_evictable(uint32_t node_id, bool evictable)
{
    Shard &shard = *_shards[shard_index(node_id)];
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Metadata, true);
    auto it = shard.node_to_slot.find(node_id);
    if (it == shard.node_to_slot.end())
        return false;
    const uint32_t local_slot = it->second;
    if ((shard.evictable[local_slot] != 0) == evictable)
        return true;
    if (evictable)
    {
        nonseed_push_front_unlocked(shard, local_slot);
        _total_evictable.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        nonseed_unlink_unlocked(shard, local_slot);
        _total_evictable.fetch_sub(1, std::memory_order_relaxed);
    }
    shard.evictable[local_slot] = evictable ? 1 : 0;
    return true;
}

void MeritMetadataCache::scale_scores(float factor, std::atomic<float> &score_unit, float next_score_unit)
{
    std::vector<std::unique_ptr<MeritTimedSharedMutexGuard>> locks;
    locks.reserve(kShardCount);
    for (auto &shard : _shards)
        locks.emplace_back(
            std::make_unique<MeritTimedSharedMutexGuard>(shard->mu, MeritLockKind::Metadata, true));

    for (size_t shard_id = 0; shard_id < kShardCount; ++shard_id)
    {
        Shard &shard = *_shards[shard_id];
        for (auto it = shard.node_to_slot.begin(); it != shard.node_to_slot.end(); ++it)
        {
            const uint32_t local_slot = it->second;
            shard.scores[local_slot] *= factor;
        }
    }
    if (_edge_heat_decay)
    {
        // Fixed-point floor so that heat-1 edges reach zero for any divisor >= 2.
        const uint32_t mul = static_cast<uint32_t>(factor * 65536.0f);
        uint64_t removed_edges = 0;
        for (size_t shard_id = 0; shard_id < kShardCount; ++shard_id)
        {
            Shard &shard = *_shards[shard_id];
            for (auto it = shard.node_to_slot.begin(); it != shard.node_to_slot.end(); ++it)
            {
                std::vector<Edge> &edges = shard.edges_by_slot[it->second];
                size_t kept = 0;
                for (size_t index = 0; index < edges.size(); ++index)
                {
                    edges[index].set_heat(static_cast<uint8_t>((edges[index].heat() * mul) >> 16));
                    if (edges[index].heat() != 0)
                        edges[kept++] = edges[index];
                }
                removed_edges += edges.size() - kept;
                edges.resize(kept);
            }
        }
        if (removed_edges != 0)
            _total_edges.fetch_sub(removed_edges, std::memory_order_relaxed);
        _edge_saturation_pending.store(false, std::memory_order_relaxed);
    }
    score_unit.store(next_score_unit, std::memory_order_release);
}

uint64_t MeritMetadataCache::edge_saturations() const
{
    return _edge_saturations.load(std::memory_order_relaxed);
}

bool MeritMetadataCache::edge_saturation_pending() const
{
    return _edge_saturation_pending.load(std::memory_order_relaxed);
}

uint8_t MeritMetadataCache::edge_increment(float unit) const
{
    if (!_edge_heat_decay || unit <= 1.0f)
        return 1;
    thread_local std::mt19937 rng{std::random_device{}()};
    const float whole = std::floor(unit);
    const float frac = unit - whole;
    const bool round_up = frac > 0.0f && static_cast<float>(rng() >> 8) * (1.0f / 16777216.0f) < frac;
    return static_cast<uint8_t>(std::min(255.0f, whole + (round_up ? 1.0f : 0.0f)));
}

void MeritMetadataCache::add_edge_heat_unlocked(Edge &edge, uint8_t increment)
{
    const uint32_t next = static_cast<uint32_t>(edge.heat()) + increment;
    if (next > std::numeric_limits<uint8_t>::max())
    {
        edge.set_heat(std::numeric_limits<uint8_t>::max());
        _edge_saturations.fetch_add(1, std::memory_order_relaxed);
        if (_edge_heat_decay)
            _edge_saturation_pending.store(true, std::memory_order_relaxed);
        return;
    }
    edge.set_heat(static_cast<uint8_t>(next));
}

bool MeritMetadataCache::snapshot(uint32_t node_id, Snapshot &out) const
{
    const Shard &shard = *_shards[shard_index(node_id)];
    MeritTimedSharedMutexGuard lock(shard.mu, MeritLockKind::Metadata, false);
    const auto it = shard.node_to_slot.find(node_id);
    if (it == shard.node_to_slot.end())
        return false;
    const uint32_t local_slot = it->second;
    const uint32_t slot_stride = _slot_stride.load(std::memory_order_relaxed);
    out.slot_id =
        static_cast<uint32_t>(static_cast<uint64_t>(shard_index(node_id)) * slot_stride + local_slot);
    out.score = shard.scores[local_slot];
    out.significant.clear();
    const std::vector<Edge> &edges = shard.edges_by_slot[local_slot];
    out.significant.reserve(edges.size());
    const uint8_t sig_t = _sig_t.load(std::memory_order_relaxed);
    for (const Edge &ed : edges)
    {
        const uint8_t heat = ed.heat();
        if (heat >= sig_t)
            out.significant.emplace_back(ed.child(), heat);
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
