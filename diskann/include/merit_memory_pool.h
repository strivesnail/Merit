#pragma once

#include "tsl/robin_map.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace diskann
{

// MERIT-managed DRAM node cache: fixed slots + hash lookup + min-importance eviction.
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
    bool lookup(uint32_t node_id, T *&coords, std::pair<uint32_t, uint32_t *> &nhood);

    void bump(uint32_t node_id, uint64_t delta = 1);

    // Copy node from disk layout buffer (full node record). Returns evicted node id if one was replaced.
    uint32_t try_admit(uint32_t node_id, const char *node_disk_buf, uint64_t disk_bytes_per_point,
                       uint64_t max_node_len_for_coords);

    // After bulk read into slot indices 0..node_ids.size()-1.
    void commit_initial_load(const std::vector<uint32_t> &node_ids, const std::vector<uint64_t> &importance);

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
        uint64_t importance = 0;
    };

    uint32_t find_min_slot_unlocked() const;
    void heap_push_unlocked(uint32_t slot);

    uint64_t _capacity = 0;
    uint64_t _aligned_dim = 0;
    uint64_t _max_degree = 0;

    T *_coords_buf = nullptr;
    uint32_t *_nhood_buf = nullptr;
    std::vector<Slot> _slots;

    tsl::robin_map<uint32_t, uint32_t> _id_to_slot;

    // Lazy min-heap (importance, slot).
    std::vector<std::pair<uint64_t, uint32_t>> _evict_heap;

    uint64_t _eviction_count = 0;

    mutable std::mutex _mu;
};

} // namespace diskann
