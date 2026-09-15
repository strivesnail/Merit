#include "merit_memory_pool.h"

#include "common_includes.h"
#include "utils.h"

namespace diskann
{

template <typename T> void MeritMemoryPool<T>::clear()
{
    std::lock_guard<std::mutex> lock(_mu);
    _id_to_slot.clear();
    _pinned.clear();
    _slots.clear();
    _free_slots.clear();
    _lru.clear();
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
    _free_slots.reserve(static_cast<size_t>(capacity));
    for (uint32_t s = static_cast<uint32_t>(capacity); s-- > 0;)
        _free_slots.push_back(s);

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

template <typename T> void MeritMemoryPool<T>::touch_lru_unlocked(uint32_t slot)
{
    _lru.splice(_lru.begin(), _lru, _slots[slot].lru_it);
    _slots[slot].lru_it = _lru.begin();
}

template <typename T>
bool MeritMemoryPool<T>::lookup(uint32_t node_id, T *&coords, std::pair<uint32_t, uint32_t *> &nhood)
{
    std::lock_guard<std::mutex> lock(_mu);
    const auto it = _id_to_slot.find(node_id);
    if (it == _id_to_slot.end())
        return false;
    const uint32_t slot = it->second;
    touch_lru_unlocked(slot);
    coords = _coords_buf + static_cast<uint64_t>(slot) * _aligned_dim;
    uint32_t *nh = _nhood_buf + static_cast<uint64_t>(slot) * (_max_degree + 1);
    nhood = {nh[0], nh + 1};
    return true;
}

template <typename T> void MeritMemoryPool<T>::bump(uint32_t node_id, uint64_t delta)
{
    (void)delta;
    std::lock_guard<std::mutex> lock(_mu);
    const auto it = _id_to_slot.find(node_id);
    if (it == _id_to_slot.end())
        return;
    touch_lru_unlocked(it->second);
}

template <typename T> uint32_t MeritMemoryPool<T>::lru_victim_unlocked() const
{
    for (auto it = _lru.rbegin(); it != _lru.rend(); ++it)
    {
        if (_pinned.find(*it) == _pinned.end())
            return *it;
    }
    return INVALID_NODE;
}

template <typename T> void MeritMemoryPool<T>::pin(uint32_t node_id)
{
    std::lock_guard<std::mutex> lock(_mu);
    _pinned.insert(node_id);
}

template <typename T> bool MeritMemoryPool<T>::is_pinned(uint32_t node_id) const
{
    std::lock_guard<std::mutex> lock(_mu);
    return _pinned.find(node_id) != _pinned.end();
}

template <typename T> uint32_t MeritMemoryPool<T>::peek_lru_victim() const
{
    std::lock_guard<std::mutex> lock(_mu);
    if (_id_to_slot.size() < _capacity)
        return INVALID_NODE;
    return lru_victim_unlocked();
}

template <typename T>
bool MeritMemoryPool<T>::copy_payload(uint32_t node_id, std::vector<T> &coords, std::vector<uint32_t> &nbrs) const
{
    std::lock_guard<std::mutex> lock(_mu);
    const auto it = _id_to_slot.find(node_id);
    if (it == _id_to_slot.end())
        return false;
    const uint32_t slot = it->second;
    const T *src = _coords_buf + static_cast<uint64_t>(slot) * _aligned_dim;
    coords.assign(src, src + _aligned_dim);
    const uint32_t *nh = _nhood_buf + static_cast<uint64_t>(slot) * (_max_degree + 1);
    const uint32_t nnbrs = nh[0];
    nbrs.assign(nh + 1, nh + 1 + nnbrs);
    return true;
}

template <typename T>
void MeritMemoryPool<T>::commit_initial_load(const std::vector<uint32_t> &node_ids,
                                             const std::vector<uint64_t> &importance)
{
    (void)importance;
    std::lock_guard<std::mutex> lock(_mu);
    _id_to_slot.clear();
    _lru.clear();
    _free_slots.clear();
    for (uint32_t s = 0; s < static_cast<uint32_t>(_slots.size()); s++)
        _slots[s].node_id = INVALID_NODE;

    // node_ids[0] is hottest: LRU front = MRU.
    for (size_t i = 0; i < node_ids.size() && i < _slots.size(); i++)
        _lru.push_back(node_ids[i]);
    auto lit = _lru.begin();
    for (size_t i = 0; i < node_ids.size() && i < _slots.size(); i++, ++lit)
    {
        const uint32_t slot = static_cast<uint32_t>(i);
        _slots[slot].node_id = node_ids[i];
        _slots[slot].lru_it = lit;
        _id_to_slot[node_ids[i]] = slot;
    }
    for (uint32_t s = static_cast<uint32_t>(std::min(node_ids.size(), _slots.size()));
         s < static_cast<uint32_t>(_slots.size()); s++)
        _free_slots.push_back(s);
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
        touch_lru_unlocked(_id_to_slot[node_id]);
        return INVALID_NODE;
    }

    const uint32_t *nhood_base = reinterpret_cast<const uint32_t *>(node_disk_buf + disk_bytes_per_point);
    const uint32_t nnbrs = nhood_base[0];
    if (nnbrs > _max_degree)
        return INVALID_NODE;

    uint32_t slot = INVALID_NODE;
    uint32_t evicted = INVALID_NODE;

    if (!_free_slots.empty())
    {
        slot = _free_slots.back();
        _free_slots.pop_back();
    }
    else
    {
        evicted = lru_victim_unlocked();
        if (evicted == INVALID_NODE)
            return INVALID_NODE;
        const auto eit = _id_to_slot.find(evicted);
        if (eit == _id_to_slot.end())
            return INVALID_NODE;
        slot = eit->second;
        _lru.erase(_slots[slot].lru_it);
        _id_to_slot.erase(eit);
        _eviction_count++;
    }

    if (slot == INVALID_NODE || slot >= _slots.size())
        return INVALID_NODE;

    T *coord_dst = _coords_buf + static_cast<uint64_t>(slot) * _aligned_dim;
    uint32_t *nh_dst = _nhood_buf + static_cast<uint64_t>(slot) * (_max_degree + 1);

    memcpy(coord_dst, node_disk_buf, disk_bytes_per_point);
    nh_dst[0] = nnbrs;
    memcpy(nh_dst + 1, nhood_base + 1, nnbrs * sizeof(uint32_t));

    _slots[slot].node_id = node_id;
    _id_to_slot[node_id] = slot;
    _lru.push_front(node_id);
    _slots[slot].lru_it = _lru.begin();

    return evicted;
}

template class MeritMemoryPool<float>;
template class MeritMemoryPool<int8_t>;
template class MeritMemoryPool<uint8_t>;

} // namespace diskann
