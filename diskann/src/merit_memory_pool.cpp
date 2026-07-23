#include "merit_memory_pool.h"

#include "common_includes.h"
#include "utils.h"

#include <algorithm>

namespace diskann
{

template <typename T> void MeritMemoryPool<T>::clear()
{
    std::lock_guard<std::mutex> lock(_mu);
    _id_to_slot.clear();
    _slots.clear();
    _evict_heap.clear();
    _capacity = 0;
    _aligned_dim = 0;
    _max_degree = 0;
    _eviction_count = 0;
    if (_coords_buf != nullptr)
    {
        aligned_free(_coords_buf);
        _coords_buf = nullptr;
    }
    if (_nhood_buf != nullptr)
    {
        delete[] _nhood_buf;
        _nhood_buf = nullptr;
    }
}

template <typename T> void MeritMemoryPool<T>::init(uint64_t capacity, uint64_t aligned_dim, uint64_t max_degree)
{
    clear();
    if (capacity == 0)
        return;

    _capacity = capacity;
    _aligned_dim = aligned_dim;
    _max_degree = max_degree;
    _slots.resize(static_cast<size_t>(capacity));

    const size_t coord_len = static_cast<size_t>(capacity * aligned_dim);
    alloc_aligned((void **)&_coords_buf, coord_len * sizeof(T), 8 * sizeof(T));
    memset(_coords_buf, 0, coord_len * sizeof(T));

    _nhood_buf = new uint32_t[capacity * (max_degree + 1)];
    memset(_nhood_buf, 0, capacity * (max_degree + 1) * sizeof(uint32_t));
}

template <typename T> bool MeritMemoryPool<T>::contains(uint32_t node_id) const
{
    std::lock_guard<std::mutex> lock(_mu);
    return _id_to_slot.find(node_id) != _id_to_slot.end();
}

template <typename T>
bool MeritMemoryPool<T>::lookup(uint32_t node_id, T *&coords, std::pair<uint32_t, uint32_t *> &nhood)
{
    std::lock_guard<std::mutex> lock(_mu);
    const auto it = _id_to_slot.find(node_id);
    if (it == _id_to_slot.end())
        return false;
    const uint32_t slot = it->second;
    coords = _coords_buf + static_cast<uint64_t>(slot) * _aligned_dim;
    uint32_t *nh = _nhood_buf + static_cast<uint64_t>(slot) * (_max_degree + 1);
    nhood = {nh[0], nh + 1};
    return true;
}

template <typename T> void MeritMemoryPool<T>::heap_push_unlocked(uint32_t slot)
{
    _evict_heap.emplace_back(_slots[slot].importance, slot);
    std::push_heap(_evict_heap.begin(), _evict_heap.end(),
                   [](const std::pair<uint64_t, uint32_t> &a, const std::pair<uint64_t, uint32_t> &b) {
                       return a.first > b.first;
                   });
}

template <typename T> uint32_t MeritMemoryPool<T>::find_min_slot_unlocked() const
{
    uint32_t best = 0;
    uint64_t best_imp = std::numeric_limits<uint64_t>::max();
    for (uint32_t s = 0; s < _slots.size(); s++)
    {
        if (_slots[s].node_id == INVALID_NODE)
            continue;
        if (_slots[s].importance < best_imp)
        {
            best_imp = _slots[s].importance;
            best = s;
        }
    }
    return best;
}

template <typename T> void MeritMemoryPool<T>::bump(uint32_t node_id, uint64_t delta)
{
    std::lock_guard<std::mutex> lock(_mu);
    const auto it = _id_to_slot.find(node_id);
    if (it == _id_to_slot.end())
        return;
    const uint32_t slot = it->second;
    _slots[slot].importance += delta;
    heap_push_unlocked(slot);
}

template <typename T>
void MeritMemoryPool<T>::commit_initial_load(const std::vector<uint32_t> &node_ids,
                                             const std::vector<uint64_t> &importance)
{
    std::lock_guard<std::mutex> lock(_mu);
    _id_to_slot.clear();
    _evict_heap.clear();
    for (size_t i = 0; i < node_ids.size() && i < _slots.size(); i++)
    {
        _slots[i].node_id = node_ids[i];
        const uint64_t imp =
            (node_ids[i] < importance.size()) ? importance[node_ids[i]] : 1ULL;
        _slots[i].importance = imp > 0 ? imp : 1ULL;
        _id_to_slot[node_ids[i]] = static_cast<uint32_t>(i);
        heap_push_unlocked(static_cast<uint32_t>(i));
    }
}

template <typename T>
uint32_t MeritMemoryPool<T>::try_admit(uint32_t node_id, const char *node_disk_buf, uint64_t disk_bytes_per_point,
                                       uint64_t max_node_len_for_coords)
{
    (void)max_node_len_for_coords;
    std::lock_guard<std::mutex> lock(_mu);
    if (!_coords_buf || node_disk_buf == nullptr)
        return INVALID_NODE;
    if (_id_to_slot.find(node_id) != _id_to_slot.end())
    {
        const uint32_t slot = _id_to_slot[node_id];
        _slots[slot].importance++;
        heap_push_unlocked(slot);
        return INVALID_NODE;
    }

    const uint32_t *nhood_base = reinterpret_cast<const uint32_t *>(node_disk_buf + disk_bytes_per_point);
    const uint32_t nnbrs = nhood_base[0];
    if (nnbrs > _max_degree)
        return INVALID_NODE;

    uint32_t slot = INVALID_NODE;
    uint32_t evicted = INVALID_NODE;

    if (_id_to_slot.size() < _capacity)
    {
        // Find a free slot index (initial load may not fill all capacity slots).
        for (uint32_t s = 0; s < static_cast<uint32_t>(_slots.size()); s++)
        {
            if (_slots[s].node_id == INVALID_NODE)
            {
                slot = s;
                break;
            }
        }
        if (slot == INVALID_NODE)
            slot = static_cast<uint32_t>(_id_to_slot.size());
    }
    else
    {
        slot = find_min_slot_unlocked();
        evicted = _slots[slot].node_id;
        if (evicted != INVALID_NODE)
            _id_to_slot.erase(evicted);
        _eviction_count++;
    }

    T *coord_dst = _coords_buf + static_cast<uint64_t>(slot) * _aligned_dim;
    uint32_t *nh_dst = _nhood_buf + static_cast<uint64_t>(slot) * (_max_degree + 1);

    memcpy(coord_dst, node_disk_buf, disk_bytes_per_point);
    nh_dst[0] = nnbrs;
    memcpy(nh_dst + 1, nhood_base + 1, nnbrs * sizeof(uint32_t));

    _slots[slot].node_id = node_id;
    _slots[slot].importance = 1;
    _id_to_slot[node_id] = slot;
    heap_push_unlocked(slot);

    return evicted;
}

template class MeritMemoryPool<float>;
template class MeritMemoryPool<int8_t>;
template class MeritMemoryPool<uint8_t>;

} // namespace diskann
