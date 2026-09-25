#include "merit_memory_pool.h"

#include "common_includes.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>

namespace diskann
{

template <typename T> void MeritMemoryPool<T>::clear()
{
    for (Shard &shard : _shards)
    {
        std::lock_guard<MeritNcacheMutex> lock(shard.mu);
        shard.id_to_slot.clear();
        shard.lru_head = INVALID_NODE;
        shard.lru_tail = INVALID_NODE;
        shard.ghost_tokens.clear();
        shard.ghost_ring.clear();
        shard.ghost_sequence = 0;
        shard.ghost_cursor = 0;
        shard.bip_counter = 0;
        shard.lookup_hits.store(0, std::memory_order_relaxed);
        shard.lookup_misses.store(0, std::memory_order_relaxed);
    }
    _pinned.clear();
    _slots.clear();
    _capacity = 0;
    _node_count = 0;
    _aligned_dim = 0;
    _max_degree = 0;
    _membership_words.reset();
    _recently_accessed.reset();
    _ghost_admitted_slot_words.reset();
    _ghost_admitted_hit_words.reset();
    _membership_word_count = 0;
    _ghost_slot_word_count = 0;
    _fast_miss_bypasses.store(0, std::memory_order_relaxed);
    _clock_second_chances.store(0, std::memory_order_relaxed);
    _next_free_slot.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> free_lock(_free_slots_mu);
        _free_slots.clear();
    }
    _size.store(0, std::memory_order_relaxed);
    _eviction_count.store(0, std::memory_order_relaxed);
    _admission_rejections.store(0, std::memory_order_relaxed);
    _admission_second_hits.store(0, std::memory_order_relaxed);
    _low_priority_admissions.store(0, std::memory_order_relaxed);
    _ghost_actual_admissions.store(0, std::memory_order_relaxed);
    _ghost_post_admission_hits.store(0, std::memory_order_relaxed);
    _ghost_reused_admissions.store(0, std::memory_order_relaxed);
    _ghost_admission_evictions.store(0, std::memory_order_relaxed);
    _ghost_zero_hit_evictions.store(0, std::memory_order_relaxed);
    _shadow_dropped_samples.store(0, std::memory_order_relaxed);
    _shadow_real_admit_samples.store(0, std::memory_order_relaxed);
    _shadow_real_admit_ns.store(0, std::memory_order_relaxed);
    _shadow_query_disk_reads.store(0, std::memory_order_relaxed);
    _shadow_query_io_ns.store(0, std::memory_order_relaxed);
    _shadow_last_stats = {};
    _shadow_last_real_admit_samples = 0;
    _shadow_last_real_admit_ns = 0;
    _shadow_last_query_disk_reads = 0;
    _shadow_last_query_io_ns = 0;
    _shadow_ewma_admit_ns = 0;
    _shadow_controller_ready = false;
    _shadow_candidate_mode = _shadow_models.size();
    _shadow_candidate_windows = 0;
    {
        std::lock_guard<std::mutex> shadow_lock(_shadow_mu);
        for (ShadowModel &model : _shadow_models)
        {
            model.slots.clear();
            model.id_to_slot.clear();
            model.ghost_ring.clear();
            model.ghost_tokens.clear();
            model.next_free_slot = 0;
            model.clock_hand = 0;
            model.ghost_cursor = 0;
            model.ghost_sequence = 0;
            model.accesses = 0;
            model.hits = 0;
            model.misses = 0;
            model.admissions = 0;
            model.rejections = 0;
            model.second_hits = 0;
            model.evictions = 0;
            model.clock_second_chances = 0;
        }
    }
    _adaptive_query_count.store(0, std::memory_order_relaxed);
    _adaptive_reject_enabled.store(false, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(_adaptive_mu);
        _adaptive_last_hits = 0;
        _adaptive_last_misses = 0;
        _adaptive_last_rejections = 0;
        _adaptive_last_second_hits = 0;
        _adaptive_last_evictions = 0;
        _adaptive_ewma_hit_rate = 0;
        _adaptive_ewma_reuse_rate = 0;
        _adaptive_ewma_evictions_per_query = 0;
        _adaptive_low_stages = 0;
        _adaptive_high_stages = 0;
        _adaptive_transitions = 0;
        _adaptive_ewma_initialized = false;
    }
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

template <typename T>
void MeritMemoryPool<T>::init(uint64_t capacity, uint64_t aligned_dim, uint64_t max_degree, uint64_t node_count)
{
    const char *clock = std::getenv("MERIT_NCACHE_CLOCK");
    _clock_enabled = clock != nullptr && std::strtoull(clock, nullptr, 10) != 0;
    const char *spin_lock = std::getenv("MERIT_NCACHE_SPIN");
    _spin_lock_enabled = spin_lock == nullptr || std::strtoull(spin_lock, nullptr, 10) != 0;
    const char *shard_count = std::getenv("MERIT_NCACHE_SHARDS");
    const uint64_t requested_shards =
        shard_count == nullptr ? kDefaultShardCount : std::strtoull(shard_count, nullptr, 10);
    _shard_count = kDefaultShardCount;
    for (size_t candidate = 1; candidate <= kMaxShardCount; candidate <<= 1)
    {
        if (requested_shards == candidate)
        {
            _shard_count = candidate;
            break;
        }
    }
    for (Shard &shard : _shards)
        shard.mu.set_spin(_spin_lock_enabled);
    const char *fast_miss = std::getenv("MERIT_NCACHE_FAST_MISS");
    _fast_miss_enabled = fast_miss != nullptr && std::strtoull(fast_miss, nullptr, 10) != 0;
    const char *ghost_stats = std::getenv("MERIT_NCACHE_GHOST_STATS");
    _ghost_stats_enabled =
        ghost_stats != nullptr && std::strtoull(ghost_stats, nullptr, 10) != 0;
    const char *shadow = std::getenv("MERIT_NCACHE_SHADOW");
    const char *shadow_control = std::getenv("MERIT_NCACHE_SHADOW_CONTROL");
    _shadow_control_enabled =
        shadow_control != nullptr && std::strtoull(shadow_control, nullptr, 10) != 0;
    const char *shadow_shift = std::getenv("MERIT_NCACHE_SHADOW_SAMPLE_SHIFT");
    _shadow_sample_shift =
        shadow_shift == nullptr
            ? 6u
            : std::min<uint32_t>(16u, static_cast<uint32_t>(std::strtoul(shadow_shift, nullptr, 10)));
    const char *shadow_min_gain = std::getenv("MERIT_NCACHE_SHADOW_MIN_GAIN");
    _shadow_min_gain =
        shadow_min_gain == nullptr
            ? 0.02
            : std::clamp(std::strtod(shadow_min_gain, nullptr), 0.0, 0.50);

    _admission_policy = AdmissionPolicy::Mru;
    const char *admission = std::getenv("MERIT_NCACHE_ADMISSION");
    if (admission != nullptr)
    {
        if (std::strcmp(admission, "lip") == 0)
            _admission_policy = AdmissionPolicy::Lip;
        else if (std::strcmp(admission, "bip") == 0)
            _admission_policy = AdmissionPolicy::Bip;
        else if (std::strcmp(admission, "hop_bip") == 0)
            _admission_policy = AdmissionPolicy::HopBip;
        else if (std::strcmp(admission, "reject_first") == 0)
            _admission_policy = AdmissionPolicy::RejectFirst;
        else if (std::strcmp(admission, "hop_reject") == 0)
            _admission_policy = AdmissionPolicy::HopReject;
        else if (std::strcmp(admission, "adaptive_hop_reject") == 0)
            _admission_policy = AdmissionPolicy::AdaptiveHopReject;
    }
    if (shadow_control == nullptr &&
        _admission_policy == AdmissionPolicy::AdaptiveHopReject)
        _shadow_control_enabled = true;
    _shadow_enabled =
        (shadow != nullptr && std::strtoull(shadow, nullptr, 10) != 0) ||
        _shadow_control_enabled;
    const char *hop_threshold = std::getenv("MERIT_NCACHE_HOP_THRESHOLD");
    const uint32_t default_hop_threshold =
        _admission_policy == AdmissionPolicy::AdaptiveHopReject ? 12u : 8u;
    _adaptive_initial_hop_threshold =
        hop_threshold == nullptr ? default_hop_threshold : static_cast<uint32_t>(std::strtoul(hop_threshold, nullptr, 10));
    _admission_hop_threshold.store(_adaptive_initial_hop_threshold, std::memory_order_relaxed);
    const char *minimum_hop_threshold = std::getenv("MERIT_NCACHE_ADAPT_MIN_HOP");
    _adaptive_min_hop_threshold = minimum_hop_threshold == nullptr
                                      ? 8u
                                      : static_cast<uint32_t>(std::strtoul(minimum_hop_threshold, nullptr, 10));
    _adaptive_min_hop_threshold =
        std::min(_adaptive_min_hop_threshold, _adaptive_initial_hop_threshold);
    const char *half_life_queries = std::getenv("MERIT_NCACHE_ADAPT_HALF_LIFE_QUERIES");
    _adaptive_half_life_queries =
        half_life_queries == nullptr ? 25000 : std::max<uint64_t>(1000, std::strtoull(half_life_queries, nullptr, 10));
    const char *bip_period = std::getenv("MERIT_NCACHE_BIP_PERIOD");
    _bip_period = bip_period == nullptr ? 32u : static_cast<uint32_t>(std::strtoul(bip_period, nullptr, 10));
    _bip_period = std::max<uint32_t>(_bip_period, 1);
    const char *ghost_percent = std::getenv("MERIT_NCACHE_GHOST_PERCENT");
    _ghost_percent =
        ghost_percent == nullptr ? 10u : static_cast<uint32_t>(std::strtoul(ghost_percent, nullptr, 10));
    _ghost_percent = std::min<uint32_t>(_ghost_percent, 100);

    clear();
    if (capacity == 0)
        return;

    _capacity = capacity;
    _node_count = node_count;
    _aligned_dim = aligned_dim;
    _max_degree = max_degree;
    _slots.resize(static_cast<size_t>(capacity));

    const size_t coord_len = static_cast<size_t>(capacity * aligned_dim);
    alloc_aligned((void **)&_coords_buf, coord_len * sizeof(T), 8 * sizeof(T));
    memset(_coords_buf, 0, coord_len * sizeof(T));

    _nhood_buf = new uint32_t[capacity * (max_degree + 1)];
    memset(_nhood_buf, 0, capacity * (max_degree + 1) * sizeof(uint32_t));

    if (_fast_miss_enabled && _node_count > 0)
    {
        _membership_word_count = (_node_count + 63) / 64;
        _membership_words = std::make_unique<std::atomic<uint64_t>[]>(_membership_word_count);
        for (uint64_t index = 0; index < _membership_word_count; ++index)
            _membership_words[index].store(0, std::memory_order_relaxed);
    }
    if (_clock_enabled)
    {
        _recently_accessed = std::make_unique<std::atomic<uint8_t>[]>(capacity);
        for (uint64_t index = 0; index < capacity; ++index)
            _recently_accessed[index].store(0, std::memory_order_relaxed);
    }

    if ((_admission_policy == AdmissionPolicy::RejectFirst ||
         _admission_policy == AdmissionPolicy::HopReject ||
         _admission_policy == AdmissionPolicy::AdaptiveHopReject) &&
        _ghost_percent > 0)
    {
        if (_ghost_stats_enabled)
        {
            _ghost_slot_word_count = (capacity + 63) / 64;
            _ghost_admitted_slot_words =
                std::make_unique<std::atomic<uint64_t>[]>(_ghost_slot_word_count);
            _ghost_admitted_hit_words =
                std::make_unique<std::atomic<uint64_t>[]>(_ghost_slot_word_count);
            for (uint64_t index = 0; index < _ghost_slot_word_count; ++index)
            {
                _ghost_admitted_slot_words[index].store(0, std::memory_order_relaxed);
                _ghost_admitted_hit_words[index].store(0, std::memory_order_relaxed);
            }
        }
        const uint64_t ghost_total = std::max<uint64_t>(1, capacity * _ghost_percent / 100);
        const size_t ghost_per_shard =
            static_cast<size_t>(std::max<uint64_t>(1, ghost_total / _shard_count));
        for (size_t index = 0; index < _shard_count; ++index)
        {
            _shards[index].ghost_ring.resize(ghost_per_shard);
            _shards[index].ghost_tokens.reserve(ghost_per_shard);
        }
    }

    if (_shadow_enabled)
    {
        const uint64_t sample_divisor = uint64_t{1} << _shadow_sample_shift;
        const size_t shadow_capacity =
            static_cast<size_t>(std::max<uint64_t>(1, capacity / sample_divisor));
        const size_t shadow_ghost_capacity =
            static_cast<size_t>(std::max<uint64_t>(
                1, capacity * _ghost_percent / 100 / sample_divisor));
        const std::array<const char *, 3> names = {"mru", "hop12", "hop8"};
        const std::array<uint32_t, 3> thresholds = {0, 12, 8};
        for (size_t index = 0; index < _shadow_models.size(); ++index)
        {
            ShadowModel &model = _shadow_models[index];
            model.name = names[index];
            model.hop_threshold = thresholds[index];
            model.reject_first = index != 0;
            model.slots.resize(shadow_capacity);
            model.id_to_slot.reserve(shadow_capacity);
            if (model.reject_first)
            {
                model.ghost_ring.resize(shadow_ghost_capacity);
                model.ghost_tokens.reserve(shadow_ghost_capacity);
            }
        }
    }
}

template <typename T> bool MeritMemoryPool<T>::membership_maybe_contains(uint32_t node_id) const
{
    if (!_fast_miss_enabled || _membership_words == nullptr || node_id >= _node_count)
        return true;
    const uint64_t word = _membership_words[node_id >> 6].load(std::memory_order_acquire);
    return (word & (uint64_t{1} << (node_id & 63))) != 0;
}

template <typename T> void MeritMemoryPool<T>::membership_set(uint32_t node_id)
{
    if (!_fast_miss_enabled || _membership_words == nullptr || node_id >= _node_count)
        return;
    _membership_words[node_id >> 6].fetch_or(uint64_t{1} << (node_id & 63), std::memory_order_release);
}

template <typename T> void MeritMemoryPool<T>::membership_clear(uint32_t node_id)
{
    if (!_fast_miss_enabled || _membership_words == nullptr || node_id >= _node_count)
        return;
    _membership_words[node_id >> 6].fetch_and(~(uint64_t{1} << (node_id & 63)), std::memory_order_release);
}

template <typename T> bool MeritMemoryPool<T>::ghost_admitted_slot(uint32_t slot) const
{
    if (!_ghost_admitted_slot_words || static_cast<uint64_t>(slot) >= _capacity)
        return false;
    const uint64_t word = static_cast<uint64_t>(slot) >> 6;
    const uint64_t bit = uint64_t{1} << (slot & 63U);
    return (_ghost_admitted_slot_words[word].load(std::memory_order_acquire) & bit) != 0;
}

template <typename T> bool MeritMemoryPool<T>::ghost_admitted_slot_was_hit(uint32_t slot) const
{
    if (!_ghost_admitted_hit_words || static_cast<uint64_t>(slot) >= _capacity)
        return false;
    const uint64_t word = static_cast<uint64_t>(slot) >> 6;
    const uint64_t bit = uint64_t{1} << (slot & 63U);
    return (_ghost_admitted_hit_words[word].load(std::memory_order_acquire) & bit) != 0;
}

template <typename T>
void MeritMemoryPool<T>::set_ghost_admitted_slot(uint32_t slot, bool admitted_after_rejection)
{
    if (!_ghost_admitted_slot_words || !_ghost_admitted_hit_words ||
        static_cast<uint64_t>(slot) >= _capacity)
        return;

    const uint64_t word = static_cast<uint64_t>(slot) >> 6;
    const uint64_t bit = uint64_t{1} << (slot & 63U);
    _ghost_admitted_hit_words[word].fetch_and(~bit, std::memory_order_release);
    if (admitted_after_rejection)
    {
        _ghost_admitted_slot_words[word].fetch_or(bit, std::memory_order_release);
        _ghost_actual_admissions.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        _ghost_admitted_slot_words[word].fetch_and(~bit, std::memory_order_release);
    }
}

template <typename T> void MeritMemoryPool<T>::note_ghost_admitted_slot_hit(uint32_t slot)
{
    if (!ghost_admitted_slot(slot))
        return;

    _ghost_post_admission_hits.fetch_add(1, std::memory_order_relaxed);
    const uint64_t word = static_cast<uint64_t>(slot) >> 6;
    const uint64_t bit = uint64_t{1} << (slot & 63U);
    if ((_ghost_admitted_hit_words[word].fetch_or(bit, std::memory_order_relaxed) & bit) == 0)
        _ghost_reused_admissions.fetch_add(1, std::memory_order_relaxed);
}

template <typename T> void MeritMemoryPool<T>::note_ghost_admitted_slot_eviction(uint32_t slot)
{
    if (!ghost_admitted_slot(slot))
        return;

    _ghost_admission_evictions.fetch_add(1, std::memory_order_relaxed);
    if (!ghost_admitted_slot_was_hit(slot))
        _ghost_zero_hit_evictions.fetch_add(1, std::memory_order_relaxed);
}

template <typename T> bool MeritMemoryPool<T>::contains(uint32_t node_id) const
{
    if (!membership_maybe_contains(node_id))
        return false;
    const Shard &shard = _shards[shard_index(node_id)];
    if (_clock_enabled)
    {
        std::shared_lock<MeritNcacheMutex> lock(shard.mu);
        return shard.id_to_slot.find(node_id) != shard.id_to_slot.end();
    }
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    return shard.id_to_slot.find(node_id) != shard.id_to_slot.end();
}

template <typename T> bool MeritMemoryPool<T>::contains_evictable(uint32_t node_id) const
{
    const Shard &shard = _shards[shard_index(node_id)];
    if (_clock_enabled)
    {
        std::shared_lock<MeritNcacheMutex> lock(shard.mu);
        return shard.id_to_slot.find(node_id) != shard.id_to_slot.end() && _pinned.find(node_id) == _pinned.end();
    }
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    return shard.id_to_slot.find(node_id) != shard.id_to_slot.end() && _pinned.find(node_id) == _pinned.end();
}

template <typename T> size_t MeritMemoryPool<T>::shard_index(uint32_t node_id) const
{
    constexpr uint64_t kHashMultiplier = 11400714819323198485ull;
    const uint32_t hash = static_cast<uint32_t>((static_cast<uint64_t>(node_id) * kHashMultiplier) >> 32);
    return static_cast<size_t>(hash) & (_shard_count - 1);
}

template <typename T> void MeritMemoryPool<T>::unlink_lru_unlocked(Shard &shard, uint32_t slot)
{
    Slot &s = _slots[slot];
    if (s.prev != INVALID_NODE)
        _slots[s.prev].next = s.next;
    else
        shard.lru_head = s.next;
    if (s.next != INVALID_NODE)
        _slots[s.next].prev = s.prev;
    else
        shard.lru_tail = s.prev;
    s.prev = INVALID_NODE;
    s.next = INVALID_NODE;
}

template <typename T> void MeritMemoryPool<T>::push_lru_front_unlocked(Shard &shard, uint32_t slot)
{
    Slot &s = _slots[slot];
    s.prev = INVALID_NODE;
    s.next = shard.lru_head;
    if (shard.lru_head != INVALID_NODE)
        _slots[shard.lru_head].prev = slot;
    else
        shard.lru_tail = slot;
    shard.lru_head = slot;
}

template <typename T> void MeritMemoryPool<T>::push_lru_back_unlocked(Shard &shard, uint32_t slot)
{
    Slot &s = _slots[slot];
    s.prev = shard.lru_tail;
    s.next = INVALID_NODE;
    if (shard.lru_tail != INVALID_NODE)
        _slots[shard.lru_tail].next = slot;
    else
        shard.lru_head = slot;
    shard.lru_tail = slot;
}

template <typename T> void MeritMemoryPool<T>::touch_lru_unlocked(Shard &shard, uint32_t slot)
{
    if (shard.lru_head == slot)
        return;
    unlink_lru_unlocked(shard, slot);
    push_lru_front_unlocked(shard, slot);
}

template <typename T> void MeritMemoryPool<T>::touch_or_mark_unlocked(Shard &shard, uint32_t slot)
{
    if (_clock_enabled)
    {
        _recently_accessed[slot].store(1, std::memory_order_relaxed);
        return;
    }
    touch_lru_unlocked(shard, slot);
}

template <typename T>
bool MeritMemoryPool<T>::low_priority_insert_unlocked(Shard &shard, uint32_t search_hop)
{
    if (_admission_policy == AdmissionPolicy::Lip)
        return true;
    const bool use_bip =
        _admission_policy == AdmissionPolicy::Bip ||
        (_admission_policy == AdmissionPolicy::HopBip &&
         search_hop >= _admission_hop_threshold.load(std::memory_order_relaxed));
    if (!use_bip)
        return false;
    shard.bip_counter++;
    return shard.bip_counter % _bip_period != 0;
}

template <typename T>
bool MeritMemoryPool<T>::lookup(uint32_t node_id, T *&coords, std::pair<uint32_t, uint32_t *> &nhood,
                                bool track_ghost_value)
{
    Shard &shard = _shards[shard_index(node_id)];
    if (!membership_maybe_contains(node_id))
    {
        shard.lookup_misses.fetch_add(1, std::memory_order_relaxed);
        _fast_miss_bypasses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (_clock_enabled)
    {
        std::shared_lock<MeritNcacheMutex> lock(shard.mu);
        const auto it = shard.id_to_slot.find(node_id);
        if (it == shard.id_to_slot.end())
        {
            shard.lookup_misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        shard.lookup_hits.fetch_add(1, std::memory_order_relaxed);
        const uint32_t slot = it->second;
        _recently_accessed[slot].store(1, std::memory_order_relaxed);
        if (track_ghost_value && _ghost_stats_enabled)
            note_ghost_admitted_slot_hit(slot);
        coords = _coords_buf + static_cast<uint64_t>(slot) * _aligned_dim;
        uint32_t *nh = _nhood_buf + static_cast<uint64_t>(slot) * (_max_degree + 1);
        nhood = {nh[0], nh + 1};
        return true;
    }
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    const auto it = shard.id_to_slot.find(node_id);
    if (it == shard.id_to_slot.end())
    {
        shard.lookup_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    shard.lookup_hits.fetch_add(1, std::memory_order_relaxed);
    const uint32_t slot = it->second;
    touch_lru_unlocked(shard, slot);
    if (track_ghost_value && _ghost_stats_enabled)
        note_ghost_admitted_slot_hit(slot);
    coords = _coords_buf + static_cast<uint64_t>(slot) * _aligned_dim;
    uint32_t *nh = _nhood_buf + static_cast<uint64_t>(slot) * (_max_degree + 1);
    nhood = {nh[0], nh + 1};
    return true;
}

template <typename T>
void MeritMemoryPool<T>::shadow_model_access(ShadowModel &model, uint32_t node_id,
                                             uint32_t search_hop)
{
    model.accesses++;
    const auto found = model.id_to_slot.find(node_id);
    if (found != model.id_to_slot.end())
    {
        model.hits++;
        model.slots[found->second].recently_accessed = 1;
        return;
    }
    model.misses++;

    if (model.reject_first && search_hop >= model.hop_threshold)
    {
        const auto seen = model.ghost_tokens.find(node_id);
        if (seen == model.ghost_tokens.end())
        {
            if (model.ghost_ring.empty())
                return;
            GhostEntry &entry = model.ghost_ring[model.ghost_cursor];
            if (entry.node_id != INVALID_NODE)
            {
                const auto old = model.ghost_tokens.find(entry.node_id);
                if (old != model.ghost_tokens.end() && old->second == entry.token)
                    model.ghost_tokens.erase(old);
            }
            model.ghost_sequence++;
            if (model.ghost_sequence == 0)
                model.ghost_sequence++;
            entry.node_id = node_id;
            entry.token = model.ghost_sequence;
            model.ghost_tokens[node_id] = entry.token;
            model.ghost_cursor = (model.ghost_cursor + 1) % model.ghost_ring.size();
            model.rejections++;
            return;
        }
        model.ghost_tokens.erase(seen);
        model.second_hits++;
    }

    if (model.slots.empty())
        return;
    uint32_t slot = INVALID_NODE;
    if (model.next_free_slot < model.slots.size())
    {
        slot = model.next_free_slot++;
    }
    else
    {
        const size_t scan_budget = model.slots.size() * 2;
        for (size_t attempt = 0; attempt < scan_budget; ++attempt)
        {
            const uint32_t candidate = model.clock_hand;
            model.clock_hand =
                static_cast<uint32_t>((model.clock_hand + 1) % model.slots.size());
            ShadowSlot &candidate_slot = model.slots[candidate];
            if (candidate_slot.recently_accessed != 0)
            {
                candidate_slot.recently_accessed = 0;
                model.clock_second_chances++;
                continue;
            }
            model.id_to_slot.erase(candidate_slot.node_id);
            model.evictions++;
            slot = candidate;
            break;
        }
    }
    if (slot == INVALID_NODE)
        return;

    model.slots[slot].node_id = node_id;
    model.slots[slot].recently_accessed = 1;
    model.id_to_slot[node_id] = slot;
    model.admissions++;
}

template <typename T> bool MeritMemoryPool<T>::shadow_sampled_node(uint32_t node_id) const
{
    if (!_shadow_enabled)
        return false;
    const uint32_t hash = static_cast<uint32_t>(
        (static_cast<uint64_t>(node_id) * 11400714819323198485ull) >> 32);
    const uint32_t sample_mask =
        _shadow_sample_shift == 0 ? 0u : (uint32_t{1} << _shadow_sample_shift) - 1;
    return (hash & sample_mask) == 0;
}

template <typename T> void MeritMemoryPool<T>::shadow_access(uint32_t node_id, uint32_t search_hop)
{
    if (!shadow_sampled_node(node_id))
        return;

    std::unique_lock<std::mutex> lock(_shadow_mu, std::try_to_lock);
    if (!lock.owns_lock())
    {
        _shadow_dropped_samples.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    for (ShadowModel &model : _shadow_models)
        shadow_model_access(model, node_id, search_hop);
}

template <typename T>
std::array<typename MeritMemoryPool<T>::ShadowStats, 3> MeritMemoryPool<T>::shadow_stats() const
{
    std::array<ShadowStats, 3> result;
    std::lock_guard<std::mutex> lock(_shadow_mu);
    for (size_t index = 0; index < _shadow_models.size(); ++index)
    {
        const ShadowModel &model = _shadow_models[index];
        ShadowStats &stats = result[index];
        stats.name = model.name;
        stats.resident = model.next_free_slot;
        stats.accesses = model.accesses;
        stats.hits = model.hits;
        stats.misses = model.misses;
        stats.admissions = model.admissions;
        stats.rejections = model.rejections;
        stats.second_hits = model.second_hits;
        stats.evictions = model.evictions;
        stats.clock_second_chances = model.clock_second_chances;
        stats.sampled_capacity = model.slots.size();
    }
    return result;
}

template <typename T> void MeritMemoryPool<T>::shadow_evaluate_controller(uint64_t query_count)
{
    const auto current_stats = shadow_stats();
    const uint64_t current_admit_samples =
        _shadow_real_admit_samples.load(std::memory_order_relaxed);
    const uint64_t current_admit_ns =
        _shadow_real_admit_ns.load(std::memory_order_relaxed);
    const uint64_t current_disk_reads =
        _shadow_query_disk_reads.load(std::memory_order_relaxed);
    const uint64_t current_io_ns =
        _shadow_query_io_ns.load(std::memory_order_relaxed);

    std::array<ShadowStats, 3> delta;
    for (size_t index = 0; index < delta.size(); ++index)
    {
        delta[index].name = current_stats[index].name;
        delta[index].accesses =
            current_stats[index].accesses - _shadow_last_stats[index].accesses;
        delta[index].hits = current_stats[index].hits - _shadow_last_stats[index].hits;
        delta[index].misses = current_stats[index].misses - _shadow_last_stats[index].misses;
        delta[index].admissions =
            current_stats[index].admissions - _shadow_last_stats[index].admissions;
        delta[index].rejections =
            current_stats[index].rejections - _shadow_last_stats[index].rejections;
        delta[index].second_hits =
            current_stats[index].second_hits - _shadow_last_stats[index].second_hits;
        delta[index].evictions =
            current_stats[index].evictions - _shadow_last_stats[index].evictions;
        delta[index].clock_second_chances =
            current_stats[index].clock_second_chances -
            _shadow_last_stats[index].clock_second_chances;
    }
    const uint64_t admit_sample_delta =
        current_admit_samples - _shadow_last_real_admit_samples;
    const uint64_t admit_ns_delta = current_admit_ns - _shadow_last_real_admit_ns;
    const uint64_t disk_read_delta =
        current_disk_reads - _shadow_last_query_disk_reads;
    const uint64_t io_ns_delta = current_io_ns - _shadow_last_query_io_ns;

    _shadow_last_stats = current_stats;
    _shadow_last_real_admit_samples = current_admit_samples;
    _shadow_last_real_admit_ns = current_admit_ns;
    _shadow_last_query_disk_reads = current_disk_reads;
    _shadow_last_query_io_ns = current_io_ns;

    if (admit_sample_delta != 0)
    {
        const double window_admit_ns =
            static_cast<double>(admit_ns_delta) / static_cast<double>(admit_sample_delta);
        _shadow_ewma_admit_ns =
            _shadow_ewma_admit_ns == 0
                ? window_admit_ns
                : 0.5 * _shadow_ewma_admit_ns + 0.5 * window_admit_ns;
    }

    const bool reject_enabled =
        _adaptive_reject_enabled.load(std::memory_order_relaxed);
    const uint32_t current_threshold =
        _admission_hop_threshold.load(std::memory_order_relaxed);
    const size_t current_mode =
        !reject_enabled ? 0 : (current_threshold <= _adaptive_min_hop_threshold ? 2 : 1);
    if (!_shadow_controller_ready)
    {
        if (current_stats[current_mode].resident < current_stats[current_mode].sampled_capacity)
        {
            diskann::cout << "MERIT ncache shadow controller: queries=" << query_count
                          << " current=" << current_stats[current_mode].name
                          << " ready=no resident=" << current_stats[current_mode].resident
                          << " capacity=" << current_stats[current_mode].sampled_capacity
                          << std::endl;
            return;
        }
        _shadow_controller_ready = true;
        diskann::cout << "MERIT ncache shadow controller: queries=" << query_count
                      << " current=" << current_stats[current_mode].name
                      << " ready=warming resident=" << current_stats[current_mode].resident
                      << " capacity=" << current_stats[current_mode].sampled_capacity
                      << std::endl;
        return;
    }
    if (delta[current_mode].misses == 0 || io_ns_delta == 0 ||
        _shadow_ewma_admit_ns == 0)
        return;

    const double represented_io_ns_per_miss =
        static_cast<double>(io_ns_delta) /
        static_cast<double>(delta[current_mode].misses);
    const double represented_admit_ns =
        _shadow_ewma_admit_ns * static_cast<double>(shadow_sample_divisor());
    std::array<double, 3> scores{};
    for (size_t index = 0; index < scores.size(); ++index)
    {
        scores[index] =
            static_cast<double>(delta[index].misses) * represented_io_ns_per_miss +
            static_cast<double>(delta[index].admissions) * represented_admit_ns;
    }

    size_t best_mode = 0;
    for (size_t index = 1; index < scores.size(); ++index)
    {
        if (scores[index] < scores[best_mode])
            best_mode = index;
    }
    const double relative_gain =
        best_mode == current_mode || scores[current_mode] == 0
            ? 0.0
            : 1.0 - scores[best_mode] / scores[current_mode];
    if (best_mode == current_mode || relative_gain < 0.005)
    {
        _shadow_candidate_mode = _shadow_models.size();
        _shadow_candidate_windows = 0;
    }
    else if (_shadow_candidate_mode == best_mode)
    {
        _shadow_candidate_windows++;
    }
    else
    {
        _shadow_candidate_mode = best_mode;
        _shadow_candidate_windows = 1;
    }
    const bool worthwhile =
        best_mode != current_mode &&
        (relative_gain >= _shadow_min_gain || _shadow_candidate_windows >= 3);
    diskann::cout << "MERIT ncache shadow controller: queries=" << query_count
                  << " current=" << current_stats[current_mode].name
                  << " best=" << current_stats[best_mode].name
                  << " score_mru=" << scores[0]
                  << " score_hop12=" << scores[1]
                  << " score_hop8=" << scores[2]
                  << " relative_gain=" << relative_gain
                  << " candidate_windows=" << _shadow_candidate_windows
                  << " represented_io_ns_per_miss=" << represented_io_ns_per_miss
                  << " represented_admit_ns=" << represented_admit_ns
                  << " disk_reads=" << disk_read_delta
                  << " switch=" << (worthwhile ? "yes" : "no") << std::endl;
    if (!worthwhile)
        return;

    if (best_mode == 0)
    {
        _adaptive_reject_enabled.store(false, std::memory_order_relaxed);
        _admission_hop_threshold.store(_adaptive_initial_hop_threshold,
                                       std::memory_order_relaxed);
    }
    else
    {
        _adaptive_reject_enabled.store(true, std::memory_order_relaxed);
        _admission_hop_threshold.store(
            best_mode == 1 ? _adaptive_initial_hop_threshold
                           : _adaptive_min_hop_threshold,
            std::memory_order_relaxed);
    }
    _adaptive_transitions++;
    _shadow_candidate_mode = _shadow_models.size();
    _shadow_candidate_windows = 0;
}

template <typename T> void MeritMemoryPool<T>::bump(uint32_t node_id, uint64_t delta)
{
    (void)delta;
    Shard &shard = _shards[shard_index(node_id)];
    if (_clock_enabled)
    {
        std::shared_lock<MeritNcacheMutex> lock(shard.mu);
        const auto it = shard.id_to_slot.find(node_id);
        if (it != shard.id_to_slot.end())
            _recently_accessed[it->second].store(1, std::memory_order_relaxed);
        return;
    }
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    const auto it = shard.id_to_slot.find(node_id);
    if (it == shard.id_to_slot.end())
        return;
    touch_lru_unlocked(shard, it->second);
}

template <typename T> uint32_t MeritMemoryPool<T>::lru_victim_unlocked(const Shard &shard) const
{
    for (uint32_t slot = shard.lru_tail; slot != INVALID_NODE; slot = _slots[slot].prev)
    {
        const uint32_t node_id = _slots[slot].node_id;
        if (_pinned.find(node_id) == _pinned.end())
            return node_id;
    }
    return INVALID_NODE;
}

template <typename T> uint32_t MeritMemoryPool<T>::clock_victim_unlocked(Shard &shard)
{
    const size_t budget = shard.id_to_slot.size() * 2;
    for (size_t attempt = 0; attempt < budget && shard.lru_tail != INVALID_NODE; ++attempt)
    {
        const uint32_t slot = shard.lru_tail;
        const uint32_t node_id = _slots[slot].node_id;
        if (_pinned.find(node_id) != _pinned.end())
        {
            unlink_lru_unlocked(shard, slot);
            push_lru_front_unlocked(shard, slot);
            continue;
        }
        if (_recently_accessed[slot].exchange(0, std::memory_order_relaxed) == 0)
            return node_id;
        _clock_second_chances.fetch_add(1, std::memory_order_relaxed);
        unlink_lru_unlocked(shard, slot);
        push_lru_front_unlocked(shard, slot);
    }
    return INVALID_NODE;
}

template <typename T> void MeritMemoryPool<T>::pin(uint32_t node_id)
{
    Shard &shard = _shards[shard_index(node_id)];
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    _pinned.insert(node_id);
}

template <typename T> bool MeritMemoryPool<T>::is_pinned(uint32_t node_id) const
{
    const Shard &shard = _shards[shard_index(node_id)];
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    return _pinned.find(node_id) != _pinned.end();
}

template <typename T> uint32_t MeritMemoryPool<T>::peek_lru_victim(uint32_t incoming_node_id) const
{
    if (_size.load(std::memory_order_relaxed) < _capacity)
        return INVALID_NODE;
    const Shard &shard = _shards[shard_index(incoming_node_id)];
    if (_clock_enabled)
    {
        std::shared_lock<MeritNcacheMutex> lock(shard.mu);
        return lru_victim_unlocked(shard);
    }
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    return lru_victim_unlocked(shard);
}

template <typename T>
bool MeritMemoryPool<T>::copy_payload(uint32_t node_id, std::vector<T> &coords, std::vector<uint32_t> &nbrs) const
{
    const Shard &shard = _shards[shard_index(node_id)];
    if (_clock_enabled)
    {
        std::shared_lock<MeritNcacheMutex> lock(shard.mu);
        const auto it = shard.id_to_slot.find(node_id);
        if (it == shard.id_to_slot.end())
            return false;
        const uint32_t slot = it->second;
        const T *src = _coords_buf + static_cast<uint64_t>(slot) * _aligned_dim;
        coords.assign(src, src + _aligned_dim);
        const uint32_t *nh = _nhood_buf + static_cast<uint64_t>(slot) * (_max_degree + 1);
        const uint32_t nnbrs = nh[0];
        nbrs.assign(nh + 1, nh + 1 + nnbrs);
        return true;
    }
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    const auto it = shard.id_to_slot.find(node_id);
    if (it == shard.id_to_slot.end())
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
bool MeritMemoryPool<T>::should_admit(uint32_t node_id, uint32_t search_hop,
                                     bool &admitted_after_rejection)
{
    admitted_after_rejection = false;
    const bool reject_first =
        _admission_policy == AdmissionPolicy::RejectFirst ||
        (_admission_policy == AdmissionPolicy::HopReject &&
         search_hop >= _admission_hop_threshold.load(std::memory_order_relaxed)) ||
        (_admission_policy == AdmissionPolicy::AdaptiveHopReject &&
         _adaptive_reject_enabled.load(std::memory_order_relaxed) &&
         search_hop >= _admission_hop_threshold.load(std::memory_order_relaxed));
    if (!reject_first)
        return true;

    Shard &shard = _shards[shard_index(node_id)];
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    if (shard.id_to_slot.find(node_id) != shard.id_to_slot.end() || shard.ghost_ring.empty())
        return true;

    const auto seen = shard.ghost_tokens.find(node_id);
    if (seen != shard.ghost_tokens.end())
    {
        shard.ghost_tokens.erase(seen);
        _admission_second_hits.fetch_add(1, std::memory_order_relaxed);
        admitted_after_rejection = true;
        return true;
    }

    GhostEntry &entry = shard.ghost_ring[shard.ghost_cursor];
    if (entry.node_id != INVALID_NODE)
    {
        const auto old = shard.ghost_tokens.find(entry.node_id);
        if (old != shard.ghost_tokens.end() && old->second == entry.token)
            shard.ghost_tokens.erase(old);
    }
    shard.ghost_sequence++;
    if (shard.ghost_sequence == 0)
        shard.ghost_sequence++;
    entry.node_id = node_id;
    entry.token = shard.ghost_sequence;
    shard.ghost_tokens[node_id] = entry.token;
    shard.ghost_cursor = (shard.ghost_cursor + 1) % shard.ghost_ring.size();
    _admission_rejections.fetch_add(1, std::memory_order_relaxed);
    return false;
}

template <typename T> void MeritMemoryPool<T>::on_query_end(uint32_t disk_reads, double io_us)
{
    if (_shadow_enabled)
    {
        _shadow_query_disk_reads.fetch_add(disk_reads, std::memory_order_relaxed);
        _shadow_query_io_ns.fetch_add(
            static_cast<uint64_t>(std::max(0.0, io_us) * 1000.0),
            std::memory_order_relaxed);
    }
    if (_admission_policy != AdmissionPolicy::AdaptiveHopReject)
        return;

    const uint64_t query_count = _adaptive_query_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (_shadow_control_enabled)
    {
        if (query_count % _adaptive_half_life_queries == 0)
            shadow_evaluate_controller(query_count);
        return;
    }
    if (query_count % _adaptive_stage_queries != 0)
        return;

    std::lock_guard<std::mutex> adaptive_lock(_adaptive_mu);
    uint64_t hits = 0;
    uint64_t misses = 0;
    for (size_t index = 0; index < _shard_count; ++index)
    {
        const Shard &shard = _shards[index];
        hits += shard.lookup_hits.load(std::memory_order_relaxed);
        misses += shard.lookup_misses.load(std::memory_order_relaxed);
    }

    const uint64_t rejections = _admission_rejections.load(std::memory_order_relaxed);
    const uint64_t second_hits = _admission_second_hits.load(std::memory_order_relaxed);
    const uint64_t evictions = _eviction_count.load(std::memory_order_relaxed);
    const uint64_t hit_delta = hits - _adaptive_last_hits;
    const uint64_t miss_delta = misses - _adaptive_last_misses;
    const uint64_t rejection_delta = rejections - _adaptive_last_rejections;
    const uint64_t second_hit_delta = second_hits - _adaptive_last_second_hits;
    const uint64_t eviction_delta = evictions - _adaptive_last_evictions;
    _adaptive_last_hits = hits;
    _adaptive_last_misses = misses;
    _adaptive_last_rejections = rejections;
    _adaptive_last_second_hits = second_hits;
    _adaptive_last_evictions = evictions;

    const uint64_t lookup_delta = hit_delta + miss_delta;
    const double hit_rate =
        lookup_delta == 0 ? 0.0 : static_cast<double>(hit_delta) / static_cast<double>(lookup_delta);
    const double reuse_rate = rejection_delta == 0
                                  ? 0.0
                                  : static_cast<double>(second_hit_delta) / static_cast<double>(rejection_delta);
    const double evictions_per_query =
        static_cast<double>(eviction_delta) / static_cast<double>(_adaptive_stage_queries);
    const double decay =
        std::pow(0.5, static_cast<double>(_adaptive_stage_queries) /
                          static_cast<double>(_adaptive_half_life_queries));
    if (!_adaptive_ewma_initialized)
    {
        _adaptive_ewma_hit_rate = hit_rate;
        _adaptive_ewma_reuse_rate = reuse_rate;
        _adaptive_ewma_evictions_per_query = evictions_per_query;
        _adaptive_ewma_initialized = true;
    }
    else
    {
        _adaptive_ewma_hit_rate =
            decay * _adaptive_ewma_hit_rate + (1.0 - decay) * hit_rate;
        _adaptive_ewma_reuse_rate =
            decay * _adaptive_ewma_reuse_rate + (1.0 - decay) * reuse_rate;
        _adaptive_ewma_evictions_per_query =
            decay * _adaptive_ewma_evictions_per_query + (1.0 - decay) * evictions_per_query;
    }

    const bool reject_enabled = _adaptive_reject_enabled.load(std::memory_order_relaxed);
    const bool enough_ghost_samples = rejection_delta >= 100;
    const bool low_locality =
        hit_rate < _adaptive_low_hit_rate &&
        ((!reject_enabled && evictions_per_query >= _adaptive_mru_min_evictions_per_query) ||
         (reject_enabled && enough_ghost_samples &&
          reuse_rate < _adaptive_low_reuse_rate &&
          evictions_per_query >= _adaptive_reject_min_evictions_per_query));
    const bool high_locality =
        reject_enabled &&
        (hit_rate > _adaptive_high_hit_rate ||
         (enough_ghost_samples && reuse_rate > _adaptive_high_reuse_rate));
    _adaptive_low_stages = low_locality ? _adaptive_low_stages + 1 : 0;
    _adaptive_high_stages = high_locality ? _adaptive_high_stages + 1 : 0;

    const uint32_t half_life_stages = static_cast<uint32_t>(
        std::max<uint64_t>(1, (_adaptive_half_life_queries + _adaptive_stage_queries - 1) /
                                  _adaptive_stage_queries));
    const uint32_t recovery_stages = std::max<uint32_t>(1, half_life_stages / 4);
    const uint32_t current_threshold = _admission_hop_threshold.load(std::memory_order_relaxed);
    bool next_reject_enabled = reject_enabled;
    uint32_t next_threshold = current_threshold;
    if (_adaptive_low_stages >= half_life_stages)
    {
        if (!reject_enabled)
        {
            next_reject_enabled = true;
            next_threshold = _adaptive_initial_hop_threshold;
        }
        else if (current_threshold > _adaptive_min_hop_threshold)
        {
            next_threshold = _adaptive_min_hop_threshold;
        }
        _adaptive_low_stages = 0;
        _adaptive_high_stages = 0;
    }
    else if (_adaptive_high_stages >= recovery_stages)
    {
        if (current_threshold < _adaptive_initial_hop_threshold)
            next_threshold = _adaptive_initial_hop_threshold;
        else
            next_reject_enabled = false;
        _adaptive_low_stages = 0;
        _adaptive_high_stages = 0;
    }

    if (next_reject_enabled != reject_enabled || next_threshold != current_threshold)
    {
        _admission_hop_threshold.store(next_threshold, std::memory_order_relaxed);
        _adaptive_reject_enabled.store(next_reject_enabled, std::memory_order_relaxed);
        _adaptive_transitions++;
        diskann::cout << "MERIT ncache adaptive: queries=" << query_count
                      << " mode="
                      << (reject_enabled ? (current_threshold <= 8 ? "hop8" : "hop12") : "mru")
                      << "->"
                      << (next_reject_enabled ? (next_threshold <= 8 ? "hop8" : "hop12") : "mru")
                      << " hit_rate=" << _adaptive_ewma_hit_rate
                      << " ghost_reuse=" << _adaptive_ewma_reuse_rate
                      << " evictions_per_query=" << _adaptive_ewma_evictions_per_query << std::endl;
    }
}

template <typename T>
void MeritMemoryPool<T>::commit_initial_load(const std::vector<uint32_t> &node_ids,
                                             const std::vector<uint64_t> &importance)
{
    (void)importance;
    for (uint64_t index = 0; index < _membership_word_count; ++index)
        _membership_words[index].store(0, std::memory_order_relaxed);
    for (Shard &shard : _shards)
    {
        std::lock_guard<MeritNcacheMutex> lock(shard.mu);
        shard.id_to_slot.clear();
        shard.lru_head = INVALID_NODE;
        shard.lru_tail = INVALID_NODE;
    }
    for (uint32_t s = 0; s < static_cast<uint32_t>(_slots.size()); s++)
    {
        _slots[s].node_id = INVALID_NODE;
        _slots[s].prev = INVALID_NODE;
        _slots[s].next = INVALID_NODE;
        if (_clock_enabled)
            _recently_accessed[s].store(0, std::memory_order_relaxed);
    }

    const size_t load_count = std::min(node_ids.size(), _slots.size());
    for (size_t i = 0; i < load_count; ++i)
    {
        const uint32_t node_id = node_ids[i];
        Shard &shard = _shards[shard_index(node_id)];
        const uint32_t slot = static_cast<uint32_t>(i);
        _slots[slot].node_id = node_id;
        shard.id_to_slot[node_id] = slot;
        _slots[slot].prev = shard.lru_tail;
        if (shard.lru_tail != INVALID_NODE)
            _slots[shard.lru_tail].next = slot;
        else
            shard.lru_head = slot;
        shard.lru_tail = slot;
        membership_set(node_id);
    }
    {
        std::lock_guard<std::mutex> free_lock(_free_slots_mu);
        _free_slots.clear();
    }
    _next_free_slot.store(static_cast<uint32_t>(load_count), std::memory_order_relaxed);
    _size.store(load_count, std::memory_order_relaxed);
    _eviction_count.store(0, std::memory_order_relaxed);
}

template <typename T>
uint32_t MeritMemoryPool<T>::try_admit(uint32_t node_id, const char *node_disk_buf, uint64_t disk_bytes_per_point,
                                       uint64_t max_node_len_for_coords, uint32_t search_hop,
                                       bool admitted_after_rejection)
{
    (void)max_node_len_for_coords;
    const bool measure_shadow_cost = shadow_sampled_node(node_id);
    const auto shadow_cost_start =
        measure_shadow_cost ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
    const auto finish = [&](uint32_t result, bool admitted) {
        if (measure_shadow_cost && admitted)
        {
            const uint64_t elapsed_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - shadow_cost_start)
                    .count());
            _shadow_real_admit_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
            _shadow_real_admit_samples.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    };
    Shard &shard = _shards[shard_index(node_id)];
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    if (!_coords_buf || node_disk_buf == nullptr)
        return finish(INVALID_NODE, false);
    const auto existing = shard.id_to_slot.find(node_id);
    if (existing != shard.id_to_slot.end())
    {
        touch_or_mark_unlocked(shard, existing->second);
        membership_set(node_id);
        return finish(INVALID_NODE, false);
    }

    const uint32_t *nhood_base = reinterpret_cast<const uint32_t *>(node_disk_buf + disk_bytes_per_point);
    const uint32_t nnbrs = nhood_base[0];
    if (nnbrs > _max_degree)
        return finish(INVALID_NODE, false);

    uint32_t slot = INVALID_NODE;
    uint32_t evicted = INVALID_NODE;

    {
        std::lock_guard<std::mutex> free_lock(_free_slots_mu);
        if (!_free_slots.empty())
        {
            slot = _free_slots.back();
            _free_slots.pop_back();
        }
    }
    uint32_t next_free = _next_free_slot.load(std::memory_order_relaxed);
    while (slot == INVALID_NODE && next_free < _slots.size() &&
           !_next_free_slot.compare_exchange_weak(next_free, next_free + 1, std::memory_order_relaxed))
    {
    }
    if (slot != INVALID_NODE)
    {
        _size.fetch_add(1, std::memory_order_relaxed);
    }
    else if (next_free < _slots.size())
    {
        slot = next_free;
        _size.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        evicted = _clock_enabled ? clock_victim_unlocked(shard) : lru_victim_unlocked(shard);
        if (evicted == INVALID_NODE)
            return finish(INVALID_NODE, false);
        const auto eit = shard.id_to_slot.find(evicted);
        if (eit == shard.id_to_slot.end())
            return finish(INVALID_NODE, false);
        slot = eit->second;
        membership_clear(evicted);
        unlink_lru_unlocked(shard, slot);
        shard.id_to_slot.erase(eit);
        _eviction_count.fetch_add(1, std::memory_order_relaxed);
    }

    if (slot == INVALID_NODE || slot >= _slots.size())
        return finish(INVALID_NODE, false);
    if (_ghost_stats_enabled && evicted != INVALID_NODE)
        note_ghost_admitted_slot_eviction(slot);

    T *coord_dst = _coords_buf + static_cast<uint64_t>(slot) * _aligned_dim;
    uint32_t *nh_dst = _nhood_buf + static_cast<uint64_t>(slot) * (_max_degree + 1);

    memcpy(coord_dst, node_disk_buf, disk_bytes_per_point);
    nh_dst[0] = nnbrs;
    memcpy(nh_dst + 1, nhood_base + 1, nnbrs * sizeof(uint32_t));

    _slots[slot].node_id = node_id;
    if (_ghost_stats_enabled)
        set_ghost_admitted_slot(slot, admitted_after_rejection);
    membership_set(node_id);
    shard.id_to_slot[node_id] = slot;
    const bool low_priority = low_priority_insert_unlocked(shard, search_hop);
    if (low_priority)
    {
        push_lru_back_unlocked(shard, slot);
        _low_priority_admissions.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        push_lru_front_unlocked(shard, slot);
    }
    if (_clock_enabled)
        _recently_accessed[slot].store(low_priority ? 0 : 1, std::memory_order_relaxed);

    return finish(evicted, true);
}

template <typename T> bool MeritMemoryPool<T>::erase(uint32_t node_id)
{
    if (!membership_maybe_contains(node_id))
        return false;
    Shard &shard = _shards[shard_index(node_id)];
    std::lock_guard<MeritNcacheMutex> lock(shard.mu);
    const auto it = shard.id_to_slot.find(node_id);
    if (it == shard.id_to_slot.end())
        return false;

    const uint32_t slot = it->second;
    membership_clear(node_id);
    unlink_lru_unlocked(shard, slot);
    shard.id_to_slot.erase(it);
    if (_ghost_stats_enabled)
    {
        note_ghost_admitted_slot_eviction(slot);
        set_ghost_admitted_slot(slot, false);
    }
    if (_clock_enabled)
        _recently_accessed[slot].store(0, std::memory_order_relaxed);
    _slots[slot].node_id = INVALID_NODE;
    _slots[slot].prev = INVALID_NODE;
    _slots[slot].next = INVALID_NODE;
    _size.fetch_sub(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> free_lock(_free_slots_mu);
        _free_slots.push_back(slot);
    }
    return true;
}

template <typename T> const char *MeritMemoryPool<T>::admission_policy_name() const
{
    switch (_admission_policy)
    {
    case AdmissionPolicy::Lip:
        return "lip";
    case AdmissionPolicy::Bip:
        return "bip";
    case AdmissionPolicy::HopBip:
        return "hop_bip";
    case AdmissionPolicy::RejectFirst:
        return "reject_first";
    case AdmissionPolicy::HopReject:
        return "hop_reject";
    case AdmissionPolicy::AdaptiveHopReject:
        return "adaptive_hop_reject";
    default:
        return "mru";
    }
}

template <typename T> uint64_t MeritMemoryPool<T>::adaptive_query_count() const
{
    return _adaptive_query_count.load(std::memory_order_relaxed);
}

template <typename T> uint64_t MeritMemoryPool<T>::adaptive_half_life_queries() const
{
    return _adaptive_half_life_queries;
}

template <typename T> bool MeritMemoryPool<T>::adaptive_reject_enabled() const
{
    return _adaptive_reject_enabled.load(std::memory_order_relaxed);
}

template <typename T> double MeritMemoryPool<T>::adaptive_hit_rate() const
{
    std::lock_guard<std::mutex> lock(_adaptive_mu);
    return _adaptive_ewma_hit_rate;
}

template <typename T> double MeritMemoryPool<T>::adaptive_ghost_reuse_rate() const
{
    std::lock_guard<std::mutex> lock(_adaptive_mu);
    return _adaptive_ewma_reuse_rate;
}

template <typename T> double MeritMemoryPool<T>::adaptive_evictions_per_query() const
{
    std::lock_guard<std::mutex> lock(_adaptive_mu);
    return _adaptive_ewma_evictions_per_query;
}

template <typename T> uint64_t MeritMemoryPool<T>::adaptive_transition_count() const
{
    std::lock_guard<std::mutex> lock(_adaptive_mu);
    return _adaptive_transitions;
}

template class MeritMemoryPool<float>;
template class MeritMemoryPool<int8_t>;
template class MeritMemoryPool<uint8_t>;

} // namespace diskann
