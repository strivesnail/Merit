#pragma once

#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace diskann
{

class MeritNcacheMutex
{
  public:
    void set_spin(bool spin)
    {
        _spin = spin;
    }

    void lock()
    {
        if (!_spin)
        {
            _mutex.lock();
            return;
        }
        uint32_t spins = 0;
        while (_flag.test_and_set(std::memory_order_acquire))
        {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
            if (++spins == 256)
            {
                std::this_thread::yield();
                spins = 0;
            }
        }
    }

    void unlock()
    {
        if (_spin)
            _flag.clear(std::memory_order_release);
        else
            _mutex.unlock();
    }

    void lock_shared()
    {
        if (_spin)
        {
            lock();
            return;
        }
        _mutex.lock_shared();
    }

    void unlock_shared()
    {
        if (_spin)
        {
            unlock();
            return;
        }
        _mutex.unlock_shared();
    }

  private:
    bool _spin = false;
    std::atomic_flag _flag = ATOMIC_FLAG_INIT;
    std::shared_mutex _mutex;
};

template <typename T> class MeritMemoryPool
{
  public:
    static constexpr uint32_t INVALID_NODE = std::numeric_limits<uint32_t>::max();

    struct ShadowStats
    {
        const char *name = "";
        uint64_t resident = 0;
        uint64_t accesses = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t admissions = 0;
        uint64_t rejections = 0;
        uint64_t second_hits = 0;
        uint64_t evictions = 0;
        uint64_t clock_second_chances = 0;
        uint64_t sampled_capacity = 0;
    };

    MeritMemoryPool() = default;

    void clear();

    void init(uint64_t capacity, uint64_t aligned_dim, uint64_t max_degree, uint64_t node_count);

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
        return _size.load(std::memory_order_relaxed);
    }

    uint64_t eviction_count() const
    {
        return _eviction_count.load(std::memory_order_relaxed);
    }

    bool spin_lock_enabled() const
    {
        return _spin_lock_enabled;
    }

    uint64_t payload_capacity_bytes() const
    {
        return _capacity * (_aligned_dim * sizeof(T) + (_max_degree + 1) * sizeof(uint32_t));
    }

    bool contains(uint32_t node_id) const;
    bool contains_evictable(uint32_t node_id) const;

    bool lookup(uint32_t node_id, T *&coords, std::pair<uint32_t, uint32_t *> &nhood,
                bool track_ghost_value = false);
    void shadow_access(uint32_t node_id, uint32_t search_hop);
    bool shadow_enabled() const
    {
        return _shadow_enabled;
    }
    bool shadow_control_enabled() const
    {
        return _shadow_control_enabled;
    }
    uint64_t shadow_dropped_samples() const
    {
        return _shadow_dropped_samples.load(std::memory_order_relaxed);
    }
    uint64_t shadow_real_admit_samples() const
    {
        return _shadow_real_admit_samples.load(std::memory_order_relaxed);
    }
    uint64_t shadow_real_admit_ns() const
    {
        return _shadow_real_admit_ns.load(std::memory_order_relaxed);
    }
    uint64_t shadow_query_disk_reads() const
    {
        return _shadow_query_disk_reads.load(std::memory_order_relaxed);
    }
    uint64_t shadow_query_io_ns() const
    {
        return _shadow_query_io_ns.load(std::memory_order_relaxed);
    }
    uint64_t shadow_sample_divisor() const
    {
        return uint64_t{1} << _shadow_sample_shift;
    }
    std::array<ShadowStats, 3> shadow_stats() const;

    void bump(uint32_t node_id, uint64_t delta = 1);

    uint32_t peek_lru_victim(uint32_t incoming_node_id) const;

    bool copy_payload(uint32_t node_id, std::vector<T> &coords, std::vector<uint32_t> &nbrs) const;

    bool should_admit(uint32_t node_id, uint32_t search_hop, bool &admitted_after_rejection);

    uint32_t try_admit(uint32_t node_id, const char *node_disk_buf, uint64_t disk_bytes_per_point,
                       uint64_t max_node_len_for_coords, uint32_t search_hop,
                       bool admitted_after_rejection = false);
    bool erase(uint32_t node_id);
    void on_query_end(uint32_t disk_reads = 0, double io_us = 0.0);

    void commit_initial_load(const std::vector<uint32_t> &node_ids, const std::vector<uint64_t> &importance);

    void pin(uint32_t node_id);
    bool is_pinned(uint32_t node_id) const;

    const char *admission_policy_name() const;
    uint32_t admission_hop_threshold() const
    {
        return _admission_hop_threshold.load(std::memory_order_relaxed);
    }
    uint64_t adaptive_query_count() const;
    uint64_t adaptive_half_life_queries() const;
    bool adaptive_reject_enabled() const;
    double adaptive_hit_rate() const;
    double adaptive_ghost_reuse_rate() const;
    double adaptive_evictions_per_query() const;
    uint64_t adaptive_transition_count() const;
    bool fast_miss_enabled() const
    {
        return _fast_miss_enabled;
    }
    uint64_t fast_miss_bitmap_bytes() const
    {
        return _membership_counter_count * sizeof(uint8_t);
    }
    uint64_t fast_miss_bypasses() const
    {
        return _fast_miss_bypasses.load(std::memory_order_relaxed);
    }
    bool clock_enabled() const
    {
        return _clock_enabled;
    }
    uint64_t clock_second_chances() const
    {
        return _clock_second_chances.load(std::memory_order_relaxed);
    }
    uint64_t admission_rejections() const
    {
        return _admission_rejections.load(std::memory_order_relaxed);
    }
    uint64_t admission_second_hits() const
    {
        return _admission_second_hits.load(std::memory_order_relaxed);
    }
    uint64_t low_priority_admissions() const
    {
        return _low_priority_admissions.load(std::memory_order_relaxed);
    }
    uint64_t ghost_actual_admissions() const
    {
        return _ghost_actual_admissions.load(std::memory_order_relaxed);
    }
    uint64_t ghost_post_admission_hits() const
    {
        return _ghost_post_admission_hits.load(std::memory_order_relaxed);
    }
    uint64_t ghost_reused_admissions() const
    {
        return _ghost_reused_admissions.load(std::memory_order_relaxed);
    }
    uint64_t ghost_admission_evictions() const
    {
        return _ghost_admission_evictions.load(std::memory_order_relaxed);
    }
    uint64_t ghost_zero_hit_evictions() const
    {
        return _ghost_zero_hit_evictions.load(std::memory_order_relaxed);
    }
    uint64_t ghost_tracking_bitmap_bytes() const
    {
        return _ghost_slot_word_count * sizeof(uint64_t) * 2;
    }
    bool ghost_stats_enabled() const
    {
        return _ghost_stats_enabled;
    }

    T *coord_ptr(uint32_t slot)
    {
        return _coords_buf + slot * _aligned_dim;
    }

    uint32_t *nhood_ptr(uint32_t slot)
    {
        return _nhood_buf + slot * (_max_degree + 1);
    }

  private:
    enum class AdmissionPolicy : uint8_t
    {
        Mru,
        Lip,
        Bip,
        HopBip,
        RejectFirst,
        HopReject,
        AdaptiveHopReject
    };

    struct Slot
    {
        uint32_t node_id = INVALID_NODE;
        uint32_t prev = INVALID_NODE;
        uint32_t next = INVALID_NODE;
    };

    struct GhostEntry
    {
        uint32_t node_id = INVALID_NODE;
        uint64_t token = 0;
    };

    struct ShadowSlot
    {
        uint32_t node_id = INVALID_NODE;
        uint8_t recently_accessed = 0;
    };

    struct ShadowModel
    {
        const char *name = "";
        uint32_t hop_threshold = 0;
        bool reject_first = false;
        std::vector<ShadowSlot> slots;
        tsl::robin_map<uint32_t, uint32_t> id_to_slot;
        std::vector<GhostEntry> ghost_ring;
        tsl::robin_map<uint32_t, uint64_t> ghost_tokens;
        uint32_t next_free_slot = 0;
        uint32_t clock_hand = 0;
        size_t ghost_cursor = 0;
        uint64_t ghost_sequence = 0;
        uint64_t accesses = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t admissions = 0;
        uint64_t rejections = 0;
        uint64_t second_hits = 0;
        uint64_t evictions = 0;
        uint64_t clock_second_chances = 0;
    };

    static constexpr size_t kDefaultShardCount = 128;
    static constexpr size_t kMaxShardCount = 128;

    struct alignas(64) Shard
    {
        mutable MeritNcacheMutex mu;
        tsl::robin_map<uint32_t, uint32_t> id_to_slot;
        uint32_t lru_head = INVALID_NODE;
        uint32_t lru_tail = INVALID_NODE;
        tsl::robin_map<uint32_t, uint64_t> ghost_tokens;
        std::vector<GhostEntry> ghost_ring;
        uint64_t ghost_sequence = 0;
        size_t ghost_cursor = 0;
        uint32_t bip_counter = 0;
        std::atomic<uint64_t> lookup_hits{0};
        std::atomic<uint64_t> lookup_misses{0};
    };

    size_t shard_index(uint32_t node_id) const;
    void unlink_lru_unlocked(Shard &shard, uint32_t slot);
    void push_lru_front_unlocked(Shard &shard, uint32_t slot);
    void push_lru_back_unlocked(Shard &shard, uint32_t slot);
    void touch_lru_unlocked(Shard &shard, uint32_t slot);
    void touch_or_mark_unlocked(Shard &shard, uint32_t slot);
    uint32_t lru_victim_unlocked(const Shard &shard) const;
    uint32_t clock_victim_unlocked(Shard &shard);
    bool low_priority_insert_unlocked(Shard &shard, uint32_t search_hop);
    bool membership_maybe_contains(uint32_t node_id) const;
    void membership_set(uint32_t node_id);
    void membership_clear(uint32_t node_id);
    bool ghost_admitted_slot(uint32_t slot) const;
    bool ghost_admitted_slot_was_hit(uint32_t slot) const;
    void set_ghost_admitted_slot(uint32_t slot, bool admitted_after_rejection);
    void note_ghost_admitted_slot_hit(uint32_t slot);
    void note_ghost_admitted_slot_eviction(uint32_t slot);
    bool shadow_sampled_node(uint32_t node_id) const;
    void shadow_model_access(ShadowModel &model, uint32_t node_id, uint32_t search_hop);
    void shadow_evaluate_controller(uint64_t query_count);

    uint64_t _capacity = 0;
    uint64_t _node_count = 0;
    uint64_t _aligned_dim = 0;
    uint64_t _max_degree = 0;

    T *_coords_buf = nullptr;
    uint32_t *_nhood_buf = nullptr;
    std::vector<Slot> _slots;

    std::array<Shard, kMaxShardCount> _shards;
    tsl::robin_set<uint32_t> _pinned;

    std::atomic<uint32_t> _next_free_slot{0};
    std::mutex _free_slots_mu;
    std::vector<uint32_t> _free_slots;
    std::atomic<size_t> _free_slots_count{0};
    std::atomic<uint64_t> _size{0};
    std::atomic<uint64_t> _eviction_count{0};
    std::atomic<uint64_t> _admission_rejections{0};
    std::atomic<uint64_t> _admission_second_hits{0};
    std::atomic<uint64_t> _low_priority_admissions{0};

    size_t _shard_count = kDefaultShardCount;
    bool _spin_lock_enabled = false;
    bool _fast_miss_enabled = false;
    bool _clock_enabled = false;
    bool _ghost_stats_enabled = false;
    bool _shadow_enabled = false;
    bool _shadow_control_enabled = false;
    uint32_t _shadow_sample_shift = 6;
    mutable std::mutex _shadow_mu;
    std::array<ShadowModel, 3> _shadow_models;
    std::atomic<uint64_t> _shadow_dropped_samples{0};
    std::atomic<uint64_t> _shadow_real_admit_samples{0};
    std::atomic<uint64_t> _shadow_real_admit_ns{0};
    std::atomic<uint64_t> _shadow_query_disk_reads{0};
    std::atomic<uint64_t> _shadow_query_io_ns{0};
    std::array<ShadowStats, 3> _shadow_last_stats;
    uint64_t _shadow_last_real_admit_samples = 0;
    uint64_t _shadow_last_real_admit_ns = 0;
    uint64_t _shadow_last_query_disk_reads = 0;
    uint64_t _shadow_last_query_io_ns = 0;
    double _shadow_ewma_admit_ns = 0;
    double _shadow_min_gain = 0.02;
    bool _shadow_controller_ready = false;
    size_t _shadow_candidate_mode = 3;
    uint32_t _shadow_candidate_windows = 0;
    // Counting Bloom filter over resident node ids, sized to the cache capacity rather than the index.
    std::unique_ptr<std::atomic<uint8_t>[]> _membership_counters;
    std::unique_ptr<std::atomic<uint8_t>[]> _recently_accessed;
    std::unique_ptr<std::atomic<uint64_t>[]> _ghost_admitted_slot_words;
    std::unique_ptr<std::atomic<uint64_t>[]> _ghost_admitted_hit_words;
    uint64_t _membership_counter_count = 0;
    uint64_t _ghost_slot_word_count = 0;
    std::atomic<uint64_t> _fast_miss_bypasses{0};
    std::atomic<uint64_t> _clock_second_chances{0};
    std::atomic<uint64_t> _ghost_actual_admissions{0};
    std::atomic<uint64_t> _ghost_post_admission_hits{0};
    std::atomic<uint64_t> _ghost_reused_admissions{0};
    std::atomic<uint64_t> _ghost_admission_evictions{0};
    std::atomic<uint64_t> _ghost_zero_hit_evictions{0};
    AdmissionPolicy _admission_policy = AdmissionPolicy::Mru;
    std::atomic<uint32_t> _admission_hop_threshold{8};
    uint32_t _bip_period = 32;
    uint32_t _ghost_percent = 10;
    uint32_t _adaptive_initial_hop_threshold = 12;
    uint32_t _adaptive_min_hop_threshold = 8;
    uint64_t _adaptive_half_life_queries = 25000;
    uint64_t _adaptive_stage_queries = 1000;
    double _adaptive_low_hit_rate = 0.40;
    double _adaptive_high_hit_rate = 0.60;
    double _adaptive_low_reuse_rate = 0.02;
    double _adaptive_high_reuse_rate = 0.10;
    double _adaptive_mru_min_evictions_per_query = 30.0;
    double _adaptive_reject_min_evictions_per_query = 10.0;
    std::atomic<bool> _adaptive_reject_enabled{false};
    std::atomic<uint64_t> _adaptive_query_count{0};
    mutable std::mutex _adaptive_mu;
    uint64_t _adaptive_last_hits = 0;
    uint64_t _adaptive_last_misses = 0;
    uint64_t _adaptive_last_rejections = 0;
    uint64_t _adaptive_last_second_hits = 0;
    uint64_t _adaptive_last_evictions = 0;
    double _adaptive_ewma_hit_rate = 0;
    double _adaptive_ewma_reuse_rate = 0;
    double _adaptive_ewma_evictions_per_query = 0;
    uint32_t _adaptive_low_stages = 0;
    uint32_t _adaptive_high_stages = 0;
    uint64_t _adaptive_transitions = 0;
    bool _adaptive_ewma_initialized = false;
};

} // namespace diskann
