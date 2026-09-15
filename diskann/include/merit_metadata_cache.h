#pragma once

#include "tsl/robin_map.h"

#include <cstdint>
#include <limits>
#include <list>
#include <mutex>
#include <vector>

namespace diskann
{

// Runtime m-cache: float node scores + directed-edge weights.
// All entries share one recency LRU. Non-seeds are also linked into a filtered
// LRU, so the globally coldest evictable non-seed is found in O(1).
class MeritMetadataCache
{
  public:
    static constexpr uint8_t kDefaultSigT = 2;
    static constexpr uint32_t kDefaultEdgeK = 0;
    static constexpr uint32_t kInvalid = std::numeric_limits<uint32_t>::max();

    struct Snapshot
    {
        uint32_t slot_id = kInvalid;
        float score = 0.0f;
        std::vector<std::pair<uint32_t, uint8_t>> significant;
    };

    struct TouchResult
    {
        uint32_t node_id = kInvalid;
        uint32_t slot_id = kInvalid;
        float score = 0.0f;
        uint32_t evicted_id = kInvalid;
        uint32_t evicted_slot = kInvalid;
        bool present = false;
    };

    MeritMetadataCache() = default;

    void clear();
    void init(uint64_t capacity, uint32_t edge_k = kDefaultEdgeK, uint8_t sig_threshold = kDefaultSigT);

    bool active() const
    {
        return _capacity > 0;
    }

    uint64_t capacity() const
    {
        return _capacity;
    }

    uint64_t size() const;
    uint64_t evictable_size() const;

    uint8_t sig_threshold() const
    {
        return _sig_t;
    }

    TouchResult on_expand(uint32_t node_id, float score_unit);
    TouchResult on_edge(uint32_t parent, uint32_t child);

    bool contains(uint32_t node_id) const;
    uint32_t slot_of(uint32_t node_id) const;
    uint32_t node_at(uint32_t slot_id) const;
    float score(uint32_t node_id) const;
    float score_at(uint32_t slot_id) const;
    bool set_evictable(uint32_t node_id, bool evictable);
    void scale_scores(float factor, std::vector<std::pair<uint32_t, float>> &scaled);
    bool snapshot(uint32_t node_id, Snapshot &out) const;

  private:
    struct Edge
    {
        uint32_t child = 0;
        uint8_t w = 0;
    };

    struct Entry
    {
        uint32_t node_id = 0;
        uint32_t slot_id = kInvalid;
        uint64_t tick = 0;
        std::list<Edge> edges;
        tsl::robin_map<uint32_t, std::list<Edge>::iterator> edge_ix;
        std::list<uint32_t>::iterator global_lru_it;
        std::list<uint32_t>::iterator nonseed_lru_it;
        bool evictable = true;
    };

    void touch_unlocked(Entry &entry);
    uint32_t evict_one_unlocked();
    TouchResult touch_or_insert_unlocked(uint32_t node_id, float score_delta);

    uint64_t _capacity = 0;
    uint32_t _edge_k = 0;
    uint8_t _sig_t = kDefaultSigT;
    uint64_t _tick = 0;

    std::list<uint32_t> _global_lru;  // all entries; front = MRU
    std::list<uint32_t> _nonseed_lru; // evictable entries in global recency order
    std::vector<float> _scores;        // contiguous float32 score[slot]
    std::vector<uint32_t> _slot_to_node;
    std::vector<uint32_t> _free_slots;
    tsl::robin_map<uint32_t, Entry> _entries;
    mutable std::mutex _mu;
};

} // namespace diskann
