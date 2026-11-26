#pragma once

#include "neighborcache_lazy_array.h"
#include "neighborcache_spin_lock.h"
#include "tsl/robin_map.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace diskann
{

class NeighborCacheMetadataCache
{
  public:
    static constexpr size_t kShardCount = 64;
    static constexpr size_t kMaxSeedPageMembers = 10;
    static constexpr uint8_t kDefaultSigT = 2;
    static constexpr uint32_t kInvalid = std::numeric_limits<uint32_t>::max();

    enum class Access
    {
        Sharded,
        SingleShard,
        SingleOwner
    };

    using Heat = uint16_t;
    static constexpr Heat kMaxHeat = std::numeric_limits<Heat>::max();

    struct Snapshot
    {
        uint32_t slot_id = kInvalid;
        Heat score = 0;
        std::vector<std::pair<uint32_t, uint8_t>> significant;
    };

    struct TouchResult
    {
        uint32_t node_id = kInvalid;
        uint32_t slot_id = kInvalid;
        Heat score = 0;
        uint32_t evicted_id = kInvalid;
        uint32_t evicted_slot = kInvalid;
        bool present = false;
        bool inserted = false;
        bool heat_update_deferred = false;
    };

    NeighborCacheMetadataCache();

    void clear();
    void init(uint64_t capacity, uint8_t sig_threshold = kDefaultSigT, uint64_t node_count = 0,
              bool index_ids_only = false, Access access = Access::Sharded);

    // With room for every admitted ID, a published slot cannot be recycled.
    bool has_stable_slots() const { return _stable_slots; }
    size_t shard_count() const { return _shards.size(); }

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
        return static_cast<uint64_t>(_slot_stride.load(std::memory_order_relaxed)) * shard_count();
    }

    uint64_t size() const;
    uint64_t edge_count() const;
    uint64_t edge_descriptor_capacity_bytes() const;
    uint64_t heat_capacity_bytes() const;
    uint64_t evictable_size() const;

    uint8_t sig_threshold() const
    {
        return _sig_t.load(std::memory_order_relaxed);
    }

    TouchResult on_expand(uint32_t node_id, uint8_t heat_increment = 1);
    TouchResult on_edge(uint32_t parent, uint32_t child);
    TouchResult on_real_io_edge(uint32_t parent, uint32_t child, size_t max_edges = 64);
    uint64_t membership_bitmap_bytes() const
    {
        return _membership_word_count * sizeof(uint64_t);
    }

    bool contains(uint32_t node_id) const;
    uint32_t slot_of(uint32_t node_id) const;
    uint32_t node_at(uint32_t slot_id) const;
    bool node_score_at(uint32_t slot_id, uint32_t &node_id, Heat &score) const;
    Heat score(uint32_t node_id) const;
    Heat score_at(uint32_t slot_id) const;
    bool set_evictable(uint32_t node_id, bool evictable);
    bool set_evictable_from_slot(uint32_t node_id, uint32_t slot_id, bool evictable);
    void halve_heat();
    bool heat_decay_pending() const { return _heat_decay_pending.load(std::memory_order_acquire); }
    uint64_t edge_saturations() const;
    bool snapshot(uint32_t node_id, Snapshot &out, bool include_all_edges = false) const;
    bool snapshot_from_slot(uint32_t node_id, uint32_t slot_id, Snapshot &out, bool include_all_edges) const;

    bool seed_member_slot(uint32_t seed_id, uint32_t page, uint32_t node_id, uint16_t &slot) const;
    void set_seed_members(uint32_t seed_id, uint32_t page, const uint32_t *ids, size_t count);

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

        void set_heat(uint8_t heat)
        {
            _bytes[sizeof(uint32_t)] = heat;
        }

      private:
        std::array<uint8_t, sizeof(uint32_t) + sizeof(uint8_t)> _bytes{};
    };
    static_assert(sizeof(Edge) == sizeof(uint32_t) + sizeof(uint8_t), "Edge must remain tightly packed");

    struct SeedMembers
    {
        uint32_t page = kInvalid;
        uint8_t count = 0;
        // Ordered by record slot. Zero heat retains membership through decay.
        std::array<Edge, kMaxSeedPageMembers> edges{};
    };

    template <typename T> using SlotArray = NeighborCacheLazyArray<T, true>;

    struct Shard
    {
        mutable std::shared_mutex mu;
        uint64_t capacity = 0;
        SlotArray<Heat> scores;
        SlotArray<std::atomic<uint32_t>> slot_to_node;
        std::vector<std::vector<Edge>> edges_by_slot; // Tracked neighbors outside the seed page.
        tsl::robin_map<uint32_t, SeedMembers> seed_members_by_slot;
        SlotArray<uint8_t> evictable;
        // The shard lock protects slot lifetime. Edge writers and snapshot readers
        // sharing that lock additionally serialize per slot, always shard first.
        mutable std::deque<NeighborCacheSpinLock> edge_locks;
        uint32_t next_unused_slot = 0;
        SlotArray<uint32_t> nonseed_prev;
        SlotArray<uint32_t> nonseed_next;
        uint32_t nonseed_head = kInvalid;
        uint32_t nonseed_tail = kInvalid;
        tsl::robin_map<uint32_t, uint32_t> node_to_slot;
    };

    std::pair<uint32_t, uint32_t> split_slot(uint32_t slot_id, uint32_t stride) const;
    size_t shard_index(uint32_t node_id) const;
    uint32_t local_slot_hint(const Shard &shard, size_t shard_id, uint32_t node_id, uint32_t slot_id) const;
    static void nonseed_unlink_unlocked(Shard &shard, uint32_t local_slot);
    static void nonseed_push_front_unlocked(Shard &shard, uint32_t local_slot);
    void touch_unlocked(Shard &shard, uint32_t local_slot);
    bool membership_contains(uint32_t node_id) const;
    void membership_set(uint32_t node_id);
    void membership_clear(uint32_t node_id);
    uint32_t evict_one_unlocked(Shard &shard, size_t shard_id, uint32_t &evicted_slot,
                                uint32_t &reclaimed_local_slot);
    TouchResult insert_unlocked(Shard &shard, size_t shard_id, uint32_t node_id, uint8_t heat_increment);
    TouchResult touch_existing_unlocked(Shard &shard, size_t shard_id, uint32_t node_id,
                                       uint32_t local_slot, uint8_t heat_increment);
    void copy_snapshot_edges_unlocked(const Shard &shard, uint32_t local_slot, Snapshot &out,
                                      bool include_all_edges) const;
    static void sort_snapshot_edges(Snapshot &out);
    void update_edge_unlocked(Shard &shard, uint32_t local_slot, uint32_t child, size_t max_edges);
    void count_edge_unlocked(Edge &edge);

    std::vector<std::unique_ptr<Shard>> _shards;
    bool _shared = true;
    std::atomic<uint64_t> _capacity{0};
    std::atomic<uint64_t> _total_size{0};
    std::atomic<uint64_t> _total_edges{0};
    std::atomic<uint64_t> _total_evictable{0};
    std::atomic<uint32_t> _slot_stride{0};
    uint64_t _slot_reciprocal = 0;
    std::atomic<uint8_t> _sig_t{kDefaultSigT};
    bool _edge_heat_decay = true;
    size_t _max_edges = 64;
    std::atomic<uint64_t> _edge_saturations{0};
    std::atomic<bool> _heat_decay_pending{false};
    uint64_t _node_count = 0;
    bool _stable_slots = false;
    std::unique_ptr<std::atomic<uint64_t>[]> _membership_words;
    uint64_t _membership_word_count = 0;
};

} // namespace diskann
