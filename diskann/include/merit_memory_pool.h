#pragma once

#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#include <cstdint>
#include <limits>
#include <list>
#include <mutex>
#include <utility>
#include <vector>

namespace diskann
{

// MERIT n-cache: fixed slots holding vector + neighbor IDs, LRU eviction.
// Not the static DiskANN _nhood_cache / load_cache_list path.
template <typename T> class MeritMemoryPool
{
  public:
    static constexpr uint32_t INVALID_NODE = std::numeric_limits<uint32_t>::max();

    MeritMemoryPool() = default;

    void clear();

    // Allocate backing storage for up to capacity nodes (coords + nhood layout matches load_cache_list).
    void init(uint64_t capacity, uint64_t aligned_dim, uint64_t max_degree);

    bool active() const
    {
        return _capacity > 0 && _coords_buf != nullptr;
    }

    uint64_t capacity() const
    {
        return _capacity;
    }

    uint64_t size() const
    {
        return _id_to_slot.size();
    }

    uint64_t eviction_count() const
    {
        return _eviction_count;
    }

    bool contains(uint32_t node_id) const;

    // Same shape as _nhood_cache value: {nnbrs, pointer to [nnbrs][nbr ids...]}.
    // Touches LRU (MRU).
    bool lookup(uint32_t node_id, T *&coords, std::pair<uint32_t, uint32_t *> &nhood);

    void bump(uint32_t node_id, uint64_t delta = 1);

    // If the pool is full, the node that try_admit would evict; else INVALID_NODE.
    uint32_t peek_lru_victim() const;

    // Copy coords + neighbor IDs out (does not touch LRU).
    bool copy_payload(uint32_t node_id, std::vector<T> &coords, std::vector<uint32_t> &nbrs) const;

    // Copy node from disk layout buffer (full node record). Returns evicted node id if one was replaced.
    uint32_t try_admit(uint32_t node_id, const char *node_disk_buf, uint64_t disk_bytes_per_point,
                       uint64_t max_node_len_for_coords);

    // After bulk read into slot indices 0..node_ids.size()-1.
    void commit_initial_load(const std::vector<uint32_t> &node_ids, const std::vector<uint64_t> &importance);

    // Pinned ids are never chosen as the LRU victim (medoid).
    void pin(uint32_t node_id);
    bool is_pinned(uint32_t node_id) const;

    T *coord_ptr(uint32_t slot)
    {
        return _coords_buf + slot * _aligned_dim;
    }

    uint32_t *nhood_ptr(uint32_t slot)
    {
        return _nhood_buf + slot * (_max_degree + 1);
    }

  private:
    struct Slot
    {
        uint32_t node_id = INVALID_NODE;
        std::list<uint32_t>::iterator lru_it;
    };

    void touch_lru_unlocked(uint32_t slot);
    uint32_t lru_victim_unlocked() const;

    uint64_t _capacity = 0;
    uint64_t _aligned_dim = 0;
    uint64_t _max_degree = 0;

    T *_coords_buf = nullptr;
    uint32_t *_nhood_buf = nullptr;
    std::vector<Slot> _slots;
    std::vector<uint32_t> _free_slots;

    tsl::robin_map<uint32_t, uint32_t> _id_to_slot;
    tsl::robin_set<uint32_t> _pinned;
    std::list<uint32_t> _lru; // front = MRU, back = LRU

    uint64_t _eviction_count = 0;

    mutable std::mutex _mu;
};

} // namespace diskann
