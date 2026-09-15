#include "merit_metadata_cache.h"

#include <algorithm>
#include <random>

namespace diskann
{

void MeritMetadataCache::clear()
{
    std::lock_guard<std::mutex> lock(_mu);
    _entries.clear();
    _global_lru.clear();
    _nonseed_lru.clear();
    _scores.clear();
    _slot_to_node.clear();
    _free_slots.clear();
    _capacity = 0;
    _tick = 0;
}

void MeritMetadataCache::init(uint64_t capacity, uint32_t edge_k, uint8_t sig_threshold)
{
    clear();
    std::lock_guard<std::mutex> lock(_mu);
    _capacity = std::min<uint64_t>(capacity, static_cast<uint64_t>(kInvalid));
    _edge_k = edge_k;
    _sig_t = sig_threshold == 0 ? kDefaultSigT : sig_threshold;
    _scores.assign(static_cast<size_t>(_capacity), 0.0f);
    _slot_to_node.assign(static_cast<size_t>(_capacity), kInvalid);
    _free_slots.reserve(static_cast<size_t>(_capacity));
    for (uint32_t slot = static_cast<uint32_t>(_capacity); slot-- > 0;)
        _free_slots.push_back(slot);
}

uint64_t MeritMetadataCache::size() const
{
    std::lock_guard<std::mutex> lock(_mu);
    return _entries.size();
}

uint64_t MeritMetadataCache::evictable_size() const
{
    std::lock_guard<std::mutex> lock(_mu);
    return _nonseed_lru.size();
}

void MeritMetadataCache::touch_unlocked(Entry &e)
{
    _global_lru.splice(_global_lru.begin(), _global_lru, e.global_lru_it);
    e.global_lru_it = _global_lru.begin();
    if (e.evictable)
    {
        _nonseed_lru.splice(_nonseed_lru.begin(), _nonseed_lru, e.nonseed_lru_it);
        e.nonseed_lru_it = _nonseed_lru.begin();
    }
    e.tick = ++_tick;
}

uint32_t MeritMetadataCache::evict_one_unlocked()
{
    if (_nonseed_lru.empty())
        return kInvalid;
    const uint32_t victim = _nonseed_lru.back();
    auto eit = _entries.find(victim);
    if (eit == _entries.end())
    {
        _nonseed_lru.pop_back();
        return kInvalid;
    }
    const uint32_t slot = eit.value().slot_id;
    _global_lru.erase(eit.value().global_lru_it);
    _nonseed_lru.erase(eit.value().nonseed_lru_it);
    _entries.erase(eit);
    if (slot != kInvalid && slot < _slot_to_node.size())
    {
        _slot_to_node[slot] = kInvalid;
        _free_slots.push_back(slot);
    }
    return victim;
}

MeritMetadataCache::TouchResult MeritMetadataCache::touch_or_insert_unlocked(uint32_t node_id, float score_delta)
{
    TouchResult r;
    auto it = _entries.find(node_id);
    if (it != _entries.end())
    {
        Entry &e = it.value();
        if (score_delta > 0.0f)
            _scores[e.slot_id] += score_delta;
        touch_unlocked(e);
        r.node_id = node_id;
        r.slot_id = e.slot_id;
        r.score = _scores[e.slot_id];
        r.present = true;
        return r;
    }

    if (_capacity == 0)
        return r;

    if (_entries.size() >= _capacity)
    {
        const uint32_t ev = evict_one_unlocked();
        if (ev == kInvalid)
            return r;
        r.evicted_id = ev;
        r.evicted_slot = _free_slots.empty() ? kInvalid : _free_slots.back();
    }

    if (_free_slots.empty())
        return r;
    const uint32_t slot_id = _free_slots.back();
    _free_slots.pop_back();
    _global_lru.push_front(node_id);
    _nonseed_lru.push_front(node_id);
    Entry e;
    e.node_id = node_id;
    e.slot_id = slot_id;
    _scores[slot_id] = std::max(score_delta, 0.0f);
    e.tick = ++_tick;
    e.global_lru_it = _global_lru.begin();
    e.nonseed_lru_it = _nonseed_lru.begin();
    e.evictable = true;
    _entries.insert({node_id, std::move(e)});
    _slot_to_node[slot_id] = node_id;
    r.node_id = node_id;
    r.slot_id = slot_id;
    r.score = _scores[slot_id];
    r.present = true;
    return r;
}

MeritMetadataCache::TouchResult MeritMetadataCache::on_expand(uint32_t node_id, float score_unit)
{
    std::lock_guard<std::mutex> lock(_mu);
    return touch_or_insert_unlocked(node_id, score_unit);
}

MeritMetadataCache::TouchResult MeritMetadataCache::on_edge(uint32_t parent, uint32_t child)
{
    thread_local std::mt19937 rng{std::random_device{}()};
    TouchResult r;
    if ((rng() % 10u) != 0u)
        return r;

    std::lock_guard<std::mutex> lock(_mu);
    // Edge heat and node heat are independent. Expanding parent already added
    // its node score; this call only ensures the parent metadata is resident.
    r = touch_or_insert_unlocked(parent, 0.0f);
    if (!r.present)
        return r;

    auto eit = _entries.find(parent);
    if (eit == _entries.end())
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

    if (_edge_k > 0 && e.edges.size() >= _edge_k)
    {
        const Edge dropped = e.edges.back();
        e.edge_ix.erase(dropped.child);
        e.edges.pop_back();
    }

    e.edges.push_front(Edge{child, 1});
    e.edge_ix[child] = e.edges.begin();
    return r;
}

bool MeritMetadataCache::contains(uint32_t node_id) const
{
    std::lock_guard<std::mutex> lock(_mu);
    return _entries.find(node_id) != _entries.end();
}

uint32_t MeritMetadataCache::slot_of(uint32_t node_id) const
{
    std::lock_guard<std::mutex> lock(_mu);
    const auto it = _entries.find(node_id);
    return it == _entries.end() ? kInvalid : it.value().slot_id;
}

uint32_t MeritMetadataCache::node_at(uint32_t slot_id) const
{
    std::lock_guard<std::mutex> lock(_mu);
    return slot_id < _slot_to_node.size() ? _slot_to_node[slot_id] : kInvalid;
}

float MeritMetadataCache::score(uint32_t node_id) const
{
    std::lock_guard<std::mutex> lock(_mu);
    const auto it = _entries.find(node_id);
    return it == _entries.end() ? 0.0f : _scores[it.value().slot_id];
}

float MeritMetadataCache::score_at(uint32_t slot_id) const
{
    std::lock_guard<std::mutex> lock(_mu);
    if (slot_id >= _slot_to_node.size() || _slot_to_node[slot_id] == kInvalid)
        return 0.0f;
    return _scores[slot_id];
}

bool MeritMetadataCache::set_evictable(uint32_t node_id, bool evictable)
{
    std::lock_guard<std::mutex> lock(_mu);
    auto it = _entries.find(node_id);
    if (it == _entries.end())
        return false;
    Entry &e = it.value();
    if (e.evictable == evictable)
        return true;
    if (evictable)
    {
        _nonseed_lru.push_front(node_id);
        e.nonseed_lru_it = _nonseed_lru.begin();
    }
    else
    {
        _nonseed_lru.erase(e.nonseed_lru_it);
    }
    e.evictable = evictable;
    return true;
}

void MeritMetadataCache::scale_scores(float factor, std::vector<std::pair<uint32_t, float>> &scaled)
{
    std::lock_guard<std::mutex> lock(_mu);
    scaled.clear();
    scaled.reserve(_entries.size());
    for (auto it = _entries.begin(); it != _entries.end(); ++it)
    {
        Entry &entry = it.value();
        _scores[entry.slot_id] *= factor;
        scaled.emplace_back(entry.slot_id, _scores[entry.slot_id]);
    }
}

bool MeritMetadataCache::snapshot(uint32_t node_id, Snapshot &out) const
{
    std::lock_guard<std::mutex> lock(_mu);
    const auto it = _entries.find(node_id);
    if (it == _entries.end())
        return false;
    const Entry &e = it.value();
    out.slot_id = e.slot_id;
    out.score = _scores[e.slot_id];
    out.significant.clear();
    out.significant.reserve(e.edges.size());
    for (const Edge &ed : e.edges)
    {
        if (ed.w >= _sig_t)
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
