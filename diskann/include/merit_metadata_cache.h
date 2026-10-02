#pragma once

#include "tsl/robin_map.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace diskann
{

class MeritMetadataCache
{
  public:
    static constexpr size_t kShardCount = 64;
    static constexpr uint8_t kDefaultSigT = 2;
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
        bool inserted = false;
    };

    struct UpdateDecision
    {
        bool update = true;
        bool observed_hit = false;
    };

    MeritMetadataCache();

    void clear();
    void init(uint64_t capacity, uint8_t sig_threshold = kDefaultSigT, uint64_t node_count = 0);

    bool active() const
    {
        return _capacity.load(std::memory_order_relaxed) > 0;
    }

    uint64_t capacity() const
    {
        return _capacity.load(std::memory_order_relaxed);
    }

    uint64_t slot_capacity() const
    {
        return static_cast<uint64_t>(_slot_stride.load(std::memory_order_relaxed)) * kShardCount;
    }

    uint64_t size() const;
    uint64_t edge_count() const;
    uint64_t evictable_size() const;

    uint8_t sig_threshold() const
    {
        return _sig_t.load(std::memory_order_relaxed);
    }

    TouchResult on_expand(uint32_t node_id, const std::atomic<float> &score_unit);
    TouchResult on_edge(uint32_t parent, uint32_t child);
    TouchResult on_real_io_edge(uint32_t parent, uint32_t child, size_t max_edges = 64);
    UpdateDecision plan_expand_update(uint32_t node_id, bool force_update = false,
                                      bool suppress_update = false);
    void on_query_end();

    bool adaptive_update_enabled() const
    {
        return _adaptive_update_enabled;
    }
    uint32_t update_denominator() const
    {
        return _update_denominator.load(std::memory_order_relaxed);
    }
    bool maintenance_suspended() const
    {
        return _adaptive_update_enabled &&
               _update_denominator.load(std::memory_order_relaxed) >= 100;
    }
    double adaptive_hit_rate() const;
    uint64_t observed_accesses() const;
    uint64_t performed_updates() const;
    uint64_t skipped_updates() const;
    uint64_t adaptive_transition_count() const;
    uint64_t membership_bitmap_bytes() const
    {
        return _membership_word_count * sizeof(uint64_t);
    }

    bool contains(uint32_t node_id) const;
    uint32_t slot_of(uint32_t node_id) const;
    uint32_t node_at(uint32_t slot_id) const;
    bool node_score_at(uint32_t slot_id, uint32_t &node_id, float &score) const;
    float score(uint32_t node_id) const;
    float score_at(uint32_t slot_id) const;
    bool set_evictable(uint32_t node_id, bool evictable);
    void scale_scores(float factor, std::atomic<float> &score_unit, float next_score_unit);
    bool snapshot(uint32_t node_id, Snapshot &out) const;

  private:
    class Edge
    {
      public:
        Edge() = default;
        Edge(uint32_t child, uint8_t heat)
        {
            std::memcpy(_bytes.data(), &child, sizeof(child));
            _bytes[sizeof(child)] = heat;
        }

        uint32_t child() const
        {
            uint32_t value = 0;
            std::memcpy(&value, _bytes.data(), sizeof(value));
            return value;
        }

        uint8_t heat() const
        {
            return _bytes[sizeof(uint32_t)];
        }

        void bump_heat()
        {
            uint8_t &value = _bytes[sizeof(uint32_t)];
            if (value < std::numeric_limits<uint8_t>::max())
                ++value;
        }

      private:
        std::array<uint8_t, sizeof(uint32_t) + sizeof(uint8_t)> _bytes{};
    };
    static_assert(sizeof(Edge) == sizeof(uint32_t) + sizeof(uint8_t), "Edge must remain tightly packed");

    struct Shard
    {
        mutable std::shared_mutex mu;
        uint64_t capacity = 0;
        std::vector<float> scores;
        std::unique_ptr<std::atomic<uint32_t>[]> slot_to_node;
        std::vector<std::vector<Edge>> edges_by_slot;
        std::vector<uint8_t> evictable;
        uint32_t next_unused_slot = 0;
        std::vector<uint32_t> nonseed_prev;
        std::vector<uint32_t> nonseed_next;
        uint32_t nonseed_head = kInvalid;
        uint32_t nonseed_tail = kInvalid;
        tsl::robin_map<uint32_t, uint32_t> node_to_slot;
    };

    struct alignas(64) AdaptiveCounters
    {
        std::atomic<uint64_t> observed_hits{0};
        std::atomic<uint64_t> observed_misses{0};
        std::atomic<uint64_t> performed_updates{0};
        std::atomic<uint64_t> skipped_updates{0};
    };

    static size_t shard_index(uint32_t node_id);
    static void nonseed_unlink_unlocked(Shard &shard, uint32_t local_slot);
    static void nonseed_push_front_unlocked(Shard &shard, uint32_t local_slot);
    void touch_unlocked(Shard &shard, uint32_t local_slot);
    bool membership_contains(uint32_t node_id) const;
    void membership_set(uint32_t node_id);
    void membership_clear(uint32_t node_id);
    uint32_t evict_one_unlocked(Shard &shard, size_t shard_id, uint32_t &evicted_slot,
                                uint32_t &reclaimed_local_slot);
    TouchResult touch_or_insert_unlocked(Shard &shard, size_t shard_id, uint32_t node_id, float score_delta);

    std::array<std::unique_ptr<Shard>, kShardCount> _shards;
    std::atomic<uint64_t> _capacity{0};
    std::atomic<uint64_t> _total_size{0};
    std::atomic<uint64_t> _total_edges{0};
    std::atomic<uint64_t> _total_evictable{0};
    std::atomic<uint32_t> _slot_stride{0};
    std::atomic<uint8_t> _sig_t{kDefaultSigT};
    bool _adaptive_update_enabled = false;
    uint64_t _node_count = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> _membership_words;
    uint64_t _membership_word_count = 0;
    std::atomic<uint32_t> _update_denominator{1};
    std::array<AdaptiveCounters, kShardCount> _adaptive_counters{};
    std::atomic<uint64_t> _adaptive_query_count{0};
    mutable std::mutex _adaptive_mu;
    uint64_t _adaptive_last_hits = 0;
    uint64_t _adaptive_last_misses = 0;
    double _adaptive_ewma_hit_rate = 0.0;
    uint32_t _adaptive_low_stages = 0;
    uint32_t _adaptive_high_stages = 0;
    uint64_t _adaptive_transitions = 0;
    bool _adaptive_ewma_initialized = false;
};

} // namespace diskann
