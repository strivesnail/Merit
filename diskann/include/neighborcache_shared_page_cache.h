// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "defaults.h"
#include "neighborcache_thread_page_cache.h"

namespace diskann
{

// D-cache pages shared by all search threads, keyed by D-cache file sector.
// Fixed capacity, sharded, LRU eviction. Admitted pages are treated as
// immutable: callers must erase() a sector before it is rewritten.
//
// Readers never take a lock. A hit stores a per-shard recency stamp; eviction
// drops the live page with the oldest stamp. Each shard has a seqlock-style version
// for its open-addressing index and each slot has its own version for its page
// bytes; readers copy optimistically and retry if a writer intervened.
// Writers (put/erase) serialize on a per-shard spin lock.
//
// Zero-copy reads: get_ptr() returns a pointer into the cache that stays valid
// until the calling thread's ReadSection ends. A full cache still accepts a new
// page immediately into a spare slot (write first). The evicted page stays
// readable until every ReadSection that could hold its pointer has ended, and
// only then is its slot reused (delete later, epoch-based reclamation).
// Spare slots are one extra generation per shard; the live page count stays at
// the configured capacity.
class NeighborCacheSharedPageCache
{
  public:
    class ReadSection
    {
      public:
        explicit ReadSection(NeighborCacheSharedPageCache &c) : _c(c.enabled() ? &c : nullptr)
        {
            if (_c != nullptr)
                _c->enter();
        }
        ~ReadSection()
        {
            if (_c != nullptr)
                _c->exit();
        }
        ReadSection(const ReadSection &) = delete;
        ReadSection &operator=(const ReadSection &) = delete;

      private:
        NeighborCacheSharedPageCache *_c;
    };

    NeighborCacheSharedPageCache() = default;
    NeighborCacheSharedPageCache(const NeighborCacheSharedPageCache &) = delete;
    NeighborCacheSharedPageCache &operator=(const NeighborCacheSharedPageCache &) = delete;

    ~NeighborCacheSharedPageCache()
    {
        std::free(_data);
    }

    bool init(uint64_t capacity_pages, uint64_t query_threads = 1)
    {
        if (capacity_pages == 0)
            return false;
        const char *mode = std::getenv("NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE");
        if (mode && std::strcmp(mode, "1") == 0) {
            const char *frozen = std::getenv("NEIGHBORCACHE_FREEZE_MAINTENANCE");
            if (!frozen || std::strcmp(frozen, "1") != 0)
                throw std::runtime_error("Thread-private prefetch currently requires immutable D: FREEZE_MAINTENANCE=1");
            const char *reset = std::getenv("NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY");
            _private = std::make_unique<NeighborCacheThreadPageCache>();
            return _private->init(capacity_pages, query_threads, reset && std::strcmp(reset,"1")==0);
        }
        uint64_t nshards = 256;
        while (nshards > 1 && capacity_pages / nshards < 64)
            nshards >>= 1;
        const uint64_t per_shard = (capacity_pages + nshards - 1) / nshards;
        // One spare generation so a new page can be published before an evicted
        // page's slot is safe to reuse.
        const uint64_t phys = per_shard * 2;
        uint64_t index_size = 1;
        while (index_size < 2 * phys)
            index_size <<= 1;
        _data = static_cast<char *>(std::aligned_alloc(defaults::SECTOR_LEN, nshards * phys * defaults::SECTOR_LEN));
        if (_data == nullptr)
            return false;
        _shards = std::make_unique<Shard[]>(nshards);
        for (uint64_t s = 0; s < nshards; ++s)
        {
            Shard &sh = _shards[s];
            sh.cap = static_cast<uint32_t>(per_shard);
            sh.phys = static_cast<uint32_t>(phys);
            sh.data = _data + s * phys * defaults::SECTOR_LEN;
            sh.imask = static_cast<uint32_t>(index_size - 1);
            sh.ikeys = std::make_unique<std::atomic<uint32_t>[]>(index_size);
            sh.ivals = std::make_unique<std::atomic<uint32_t>[]>(index_size);
            for (uint64_t i = 0; i < index_size; ++i)
                sh.ikeys[i].store(kNoKey, std::memory_order_relaxed);
            sh.slot_key = std::make_unique<std::atomic<uint32_t>[]>(phys);
            sh.slot_ver = std::make_unique<std::atomic<uint32_t>[]>(phys);
            sh.stamp = std::make_unique<std::atomic<uint64_t>[]>(phys);
            sh.retired_flag = std::make_unique<uint8_t[]>(phys);
            std::memset(sh.retired_flag.get(), 0, phys);
            sh.retired_slot = std::make_unique<uint32_t[]>(phys);
            sh.retired_epoch = std::make_unique<uint64_t[]>(phys);
            sh.free_slot = std::make_unique<uint32_t[]>(phys);
            for (uint64_t i = 0; i < phys; ++i)
            {
                sh.slot_key[i].store(kNoKey, std::memory_order_relaxed);
                sh.slot_ver[i].store(0, std::memory_order_relaxed);
                sh.stamp[i].store(0, std::memory_order_relaxed);
            }
        }
        _nshards = static_cast<uint32_t>(nshards);
        _capacity = nshards * per_shard;
        return true;
    }

    bool enabled() const
    {
        return _private || _nshards > 0;
    }

    bool thread_private() const { return bool(_private); }

    uint64_t capacity_pages() const
    {
        return _private ? _private->capacity_pages() : _capacity;
    }

    bool contains_range(uint32_t base, uint16_t n) const
    {
        if (_private) return _private->contains_range(base,n);
        for (uint16_t i = 0; i < n; ++i)
        {
            if (find_slot(shard(base + i), base + i) == kNoKey)
                return false;
        }
        return true;
    }

    // All-or-nothing copy of sectors [base, base+n) into dst.
    bool get_range(uint32_t base, uint16_t n, char *dst)
    {
        if (_private) return _private->get_range(base,n,dst);
        for (uint16_t i = 0; i < n; ++i)
        {
            if (!copy_page(shard(base + i), base + i, dst + static_cast<size_t>(i) * defaults::SECTOR_LEN))
            {
                stripe().misses.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
        }
        stripe().hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Zero-copy lookup. Requires an open ReadSection on this thread; the pointer
    // stays valid until that section ends. Returns nullptr on miss.
    const char *get_ptr(uint32_t key)
    {
        if (_private) return _private->get_ptr(key);
        if (t_reader_depth == 0)
            return nullptr;
        Shard &sh = shard(key);
        for (int attempt = 0; attempt < kMaxRetries; ++attempt)
        {
            const uint32_t s = find_slot(sh, key);
            if (s == kNoKey)
                break;
            const uint32_t v1 = sh.slot_ver[s].load(std::memory_order_acquire);
            if (v1 & 1u)
            {
                cpu_relax();
                continue;
            }
            const bool same = sh.slot_key[s].load(std::memory_order_relaxed) == key;
            std::atomic_thread_fence(std::memory_order_acquire);
            if (sh.slot_ver[s].load(std::memory_order_relaxed) != v1)
                continue;
            if (!same)
                break;
            touch(sh, s);
            stripe().hits.fetch_add(1, std::memory_order_relaxed);
            return sh.data + static_cast<size_t>(s) * defaults::SECTOR_LEN;
        }
        stripe().misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    void erase(uint32_t key)
    {
        if (_private) { _private->erase(key); return; }
        Shard &sh = shard(key);
        // The owner may veto invalidation; the detached writer's hook never
        // does so. Its IO pins protect the physical sector independently.
        for (;;)
        {
            {
                std::lock_guard<SpinMutex> lk(sh.wmu);
                const uint32_t s = index_find_locked(sh, key);
                if (s == kNoKey)
                    return;
                if (fire_evict_locked(sh, s, true))
                {
                    retire_locked(sh, s);
                    return;
                }
            }
            std::this_thread::yield();
        }
    }

    using EvictHook = bool (*)(void *ctx, uint32_t key, const char *page, bool discard);
    // Called with the page bytes still published, before the LRU slot is dropped.
    // False keeps the victim resident (asynchronous writeback/backpressure).
    void set_evict_hook(EvictHook hook, void *ctx)
    {
        _evict = hook;
        _evict_ctx = ctx;
    }

    // overwrite rewrites a live page in place. Seed updates use it; a disk-read
    // admission leaves an existing page untouched. Returns false when a new page
    // cannot be stored.
    bool put(uint32_t key, const char *src, bool overwrite = false, std::atomic<uint64_t> *dirty = nullptr,
             const std::atomic<uint64_t> *disk_dirty = nullptr)
    {
        if (_private) {
            if (dirty || (disk_dirty && disk_dirty->load(std::memory_order_acquire)))
                throw std::runtime_error("Thread-private prefetch does not support dirty online pages");
            return _private->put(key,src,overwrite);
        }
        Shard &sh = shard(key);
        std::lock_guard<SpinMutex> lk(sh.wmu);
        // Recheck after acquiring the shard lock: an older disk read may race
        // with a dirty update followed by eviction of that updated page.
        if (disk_dirty && disk_dirty->load(std::memory_order_acquire) != 0)
            return false;
        const uint32_t existing = index_find_locked(sh, key);
        if (existing != kNoKey)
        {
            if (overwrite)
            {
                const uint32_t v = slot_write_begin(sh, existing);
                std::memcpy(sh.data + static_cast<size_t>(existing) * defaults::SECTOR_LEN, src, defaults::SECTOR_LEN);
                slot_write_end(sh, existing, v);
                if (dirty) dirty->fetch_add(1, std::memory_order_release);
            }
            touch(sh, existing);
            return true;
        }
        // The hook takes ownership of any dirty snapshot before A is removed.
        // Detached writeback always accepts eviction, including under overload.
        if (_evict != nullptr && sh.live >= sh.cap)
        {
            if (!retire_lru_locked(sh))
            {
                _skipped.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            _evictions.fetch_add(1, std::memory_order_relaxed);
        }
        uint32_t slot = kNoKey;
        if (!alloc_slot_locked(sh, slot))
        {
            _skipped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const uint32_t v = slot_write_begin(sh, slot);
        std::memcpy(sh.data + static_cast<size_t>(slot) * defaults::SECTOR_LEN, src, defaults::SECTOR_LEN);
        sh.slot_key[slot].store(key, std::memory_order_relaxed);
        slot_write_end(sh, slot, v);
        if (dirty) dirty->fetch_add(1, std::memory_order_release);
        index_insert_locked(sh, key, slot);
        touch(sh, slot);
        sh.live++;
        _inserts.fetch_add(1, std::memory_order_relaxed);
        while (sh.live > sh.cap)
        {
            if (!retire_lru_locked(sh))
                break;
            _evictions.fetch_add(1, std::memory_order_relaxed);
        }
        return true;
    }

    uint64_t hits() const
    {
        if (_private) return _private->hits();
        uint64_t t = 0;
        for (const auto &c : _stripes)
            t += c.hits.load(std::memory_order_relaxed);
        return t;
    }
    uint64_t misses() const
    {
        if (_private) return _private->misses();
        uint64_t t = 0;
        for (const auto &c : _stripes)
            t += c.misses.load(std::memory_order_relaxed);
        return t;
    }
    uint64_t inserts() const
    {
        if (_private) return _private->inserts();
        return _inserts.load(std::memory_order_relaxed);
    }
    uint64_t evictions() const
    {
        if (_private) return _private->evictions();
        return _evictions.load(std::memory_order_relaxed);
    }
    uint64_t skipped_puts() const
    {
        if (_private) return 0;
        return _skipped.load(std::memory_order_relaxed);
    }

  private:
    std::unique_ptr<NeighborCacheThreadPageCache> _private;
    struct Shard;
    static constexpr uint32_t kNoKey = 0xFFFFFFFFu;
    static constexpr int kMaxRetries = 1024;
    static constexpr uint32_t kStripes = 64;
    static constexpr uint32_t kMaxReaders = 512;

    struct alignas(64) ReaderSlot
    {
        std::atomic<uint64_t> epoch{0};
    };

    static inline thread_local uint32_t t_reader_depth = 0;

    static uint32_t reader_id()
    {
        static std::atomic<uint32_t> next{0};
        thread_local const uint32_t id = next.fetch_add(1, std::memory_order_relaxed);
        return id;
    }

    void enter()
    {
        if (_private) { _private->enter(); return; }
        if (t_reader_depth++ != 0)
            return;
        const uint32_t id = reader_id();
        if (id >= kMaxReaders)
        {
            t_reader_depth = 0;
            return;
        }
        uint32_t hw = _reader_hw.load(std::memory_order_relaxed);
        while (hw <= id && !_reader_hw.compare_exchange_weak(hw, id + 1, std::memory_order_relaxed))
        {
        }
        _readers[id].epoch.store(_epoch.load(std::memory_order_relaxed), std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    void exit()
    {
        if (_private) { _private->exit(); return; }
        if (t_reader_depth == 0 || --t_reader_depth != 0)
            return;
        _readers[reader_id()].epoch.store(0, std::memory_order_release);
    }

    uint64_t min_active_epoch() const
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        uint64_t m = UINT64_MAX;
        const uint32_t hw = _reader_hw.load(std::memory_order_relaxed);
        for (uint32_t i = 0; i < hw; ++i)
        {
            const uint64_t e = _readers[i].epoch.load(std::memory_order_acquire);
            if (e != 0 && e < m)
                m = e;
        }
        return m;
    }

    // Return retired slots whose readers have all left to the freelist.
    void reclaim_locked(Shard &sh)
    {
        if (sh.nretired == 0)
            return;
        const uint64_t min_epoch = min_active_epoch();
        while (sh.nretired > 0 && sh.retired_epoch[sh.retired_head] < min_epoch)
        {
            const uint32_t slot = sh.retired_slot[sh.retired_head];
            sh.retired_head = (sh.retired_head + 1) % sh.phys;
            sh.nretired--;
            sh.retired_flag[slot] = 0;
            sh.free_slot[sh.nfree++] = slot;
        }
    }

    // Spare slot first (never used, or already reclaimed). Does not overwrite a page a reader might hold.
    bool alloc_slot_locked(Shard &sh, uint32_t &slot)
    {
        reclaim_locked(sh);
        if (sh.nfree > 0)
        {
            slot = sh.free_slot[--sh.nfree];
            return true;
        }
        if (sh.used < sh.phys)
        {
            slot = sh.used++;
            return true;
        }
        reclaim_locked(sh);
        if (sh.nfree > 0)
        {
            slot = sh.free_slot[--sh.nfree];
            return true;
        }
        return false;
    }

    // A live slot stays the oldest until its stamp changes. Touches and new
    // inserts only move a stamp forward, so a recorded oldest slot is still
    // the victim while its stamp matches.
    bool cached_lru_live(const Shard &sh, uint32_t slot, uint64_t stamp) const
    {
        return slot < sh.phys && sh.retired_flag[slot] == 0 &&
               sh.slot_key[slot].load(std::memory_order_relaxed) != kNoKey &&
               sh.stamp[slot].load(std::memory_order_relaxed) == stamp;
    }

    // Unpublish the least-recently used live page. Its bytes stay until reclaim_locked.
    bool retire_lru_locked(Shard &sh)
    {
        if (cached_lru_live(sh, sh.cached_lru, sh.cached_lru_stamp))
        {
            const uint32_t victim = sh.cached_lru;
            if (!fire_evict_locked(sh, victim))
                return false;
            if (cached_lru_live(sh, sh.cached_lru2, sh.cached_lru2_stamp))
            {
                sh.cached_lru = sh.cached_lru2;
                sh.cached_lru_stamp = sh.cached_lru2_stamp;
            }
            else
                sh.cached_lru = kNoKey;
            sh.cached_lru2 = kNoKey;
            retire_locked(sh, victim);
            return true;
        }

        uint32_t best = kNoKey;
        uint32_t second = kNoKey;
        uint64_t best_stamp = UINT64_MAX;
        uint64_t second_stamp = UINT64_MAX;
        for (uint32_t i = 0; i < sh.phys; ++i)
        {
            if (sh.retired_flag[i] || sh.slot_key[i].load(std::memory_order_relaxed) == kNoKey)
                continue;
            const uint64_t stamp = sh.stamp[i].load(std::memory_order_relaxed);
            if (stamp < best_stamp)
            {
                second = best;
                second_stamp = best_stamp;
                best = i;
                best_stamp = stamp;
            }
            else if (stamp < second_stamp)
            {
                second = i;
                second_stamp = stamp;
            }
        }
        if (best == kNoKey)
        {
            sh.cached_lru = kNoKey;
            sh.cached_lru2 = kNoKey;
            return false;
        }
        if (!fire_evict_locked(sh, best))
        {
            sh.cached_lru = best;
            sh.cached_lru_stamp = best_stamp;
            sh.cached_lru2 = second;
            sh.cached_lru2_stamp = second_stamp;
            return false;
        }
        sh.cached_lru = second;
        sh.cached_lru_stamp = second_stamp;
        sh.cached_lru2 = kNoKey;
        retire_locked(sh, best);
        return true;
    }

    bool fire_evict_locked(Shard &sh, uint32_t slot, bool discard = false)
    {
        if (_evict == nullptr)
            return true;
        const uint32_t key = sh.slot_key[slot].load(std::memory_order_relaxed);
        if (key == kNoKey)
            return true;
        return _evict(_evict_ctx, key, sh.data + static_cast<size_t>(slot) * defaults::SECTOR_LEN, discard);
    }

    // Unpublish slot s; it is reusable once every open reader section started after the retire epoch.
    void retire_locked(Shard &sh, uint32_t s)
    {
        const uint32_t key = sh.slot_key[s].load(std::memory_order_relaxed);
        if (key != kNoKey)
        {
            index_remove_locked(sh, key);
            if (sh.live > 0)
                sh.live--;
        }
        const uint32_t v = slot_write_begin(sh, s);
        sh.slot_key[s].store(kNoKey, std::memory_order_relaxed);
        slot_write_end(sh, s, v);
        sh.stamp[s].store(0, std::memory_order_relaxed);
        if (sh.retired_flag[s])
            return;
        const uint64_t e = _epoch.fetch_add(1, std::memory_order_seq_cst);
        const uint32_t tail = (sh.retired_head + sh.nretired) % sh.phys;
        sh.retired_slot[tail] = s;
        sh.retired_epoch[tail] = e;
        sh.nretired++;
        sh.retired_flag[s] = 1;
    }

    struct SpinMutex
    {
        void lock()
        {
            uint32_t spins = 0;
            while (flag.test_and_set(std::memory_order_acquire))
            {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#endif
                if (++spins == 128)
                {
                    std::this_thread::yield();
                    spins = 0;
                }
            }
        }
        void unlock()
        {
            flag.clear(std::memory_order_release);
        }
        std::atomic_flag flag = ATOMIC_FLAG_INIT;
    };

    struct alignas(64) Shard
    {
        std::atomic<uint32_t> idx_ver{0};
        SpinMutex wmu;
        uint32_t imask = 0;
        std::unique_ptr<std::atomic<uint32_t>[]> ikeys;
        std::unique_ptr<std::atomic<uint32_t>[]> ivals;
        std::unique_ptr<std::atomic<uint32_t>[]> slot_key;
        std::unique_ptr<std::atomic<uint32_t>[]> slot_ver;
        std::unique_ptr<std::atomic<uint64_t>[]> stamp;
        std::atomic<uint64_t> tick{0};
        std::unique_ptr<uint8_t[]> retired_flag;
        std::unique_ptr<uint32_t[]> retired_slot;
        std::unique_ptr<uint64_t[]> retired_epoch;
        std::unique_ptr<uint32_t[]> free_slot;
        char *data = nullptr;
        uint32_t cap = 0;
        uint32_t phys = 0;
        uint32_t used = 0;
        uint32_t live = 0;
        uint32_t cached_lru = kNoKey;
        uint64_t cached_lru_stamp = 0;
        uint32_t cached_lru2 = kNoKey;
        uint64_t cached_lru2_stamp = 0;
        uint32_t retired_head = 0;
        uint32_t nretired = 0;
        uint32_t nfree = 0;
    };

    struct alignas(64) Stripe
    {
        std::atomic<uint64_t> hits{0};
        std::atomic<uint64_t> misses{0};
    };

    static void touch(Shard &sh, uint32_t s)
    {
        sh.stamp[s].store(sh.tick.fetch_add(1, std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }

    static void cpu_relax()
    {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }

    static uint32_t index_hash(uint32_t k)
    {
        k ^= k >> 16;
        k *= 0x85EBCA6Bu;
        k ^= k >> 13;
        k *= 0xC2B2AE35u;
        k ^= k >> 16;
        return k;
    }

    Stripe &stripe()
    {
        static std::atomic<uint32_t> next{0};
        thread_local const uint32_t id = next.fetch_add(1, std::memory_order_relaxed) % kStripes;
        return _stripes[id];
    }

    Shard &shard(uint32_t key) const
    {
        return _shards[(static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ull >> 32) % _nshards];
    }

    // Optimistic index lookup; returns the slot or kNoKey (miss, or persistent writer contention).
    uint32_t find_slot(const Shard &sh, uint32_t key) const
    {
        for (int attempt = 0; attempt < kMaxRetries; ++attempt)
        {
            const uint32_t v1 = sh.idx_ver.load(std::memory_order_acquire);
            if (v1 & 1u)
            {
                cpu_relax();
                continue;
            }
            uint32_t found = kNoKey;
            uint32_t pos = index_hash(key) & sh.imask;
            for (uint32_t probes = 0; probes <= sh.imask; ++probes)
            {
                const uint32_t k = sh.ikeys[pos].load(std::memory_order_relaxed);
                if (k == key)
                {
                    found = sh.ivals[pos].load(std::memory_order_relaxed);
                    break;
                }
                if (k == kNoKey)
                    break;
                pos = (pos + 1) & sh.imask;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (sh.idx_ver.load(std::memory_order_relaxed) == v1)
                return found < sh.phys ? found : kNoKey;
        }
        return kNoKey;
    }

    bool copy_page(Shard &sh, uint32_t key, char *dst)
    {
        // Dynamic pages can be overwritten. Copy under the page shard lock,
        // rather than racing memcpy against a writer and checking a version later.
        std::lock_guard<SpinMutex> lock(sh.wmu);
        const uint32_t slot = index_find_locked(sh, key);
        if (slot == kNoKey) return false;
        std::memcpy(dst, sh.data + static_cast<size_t>(slot) * defaults::SECTOR_LEN, defaults::SECTOR_LEN);
        touch(sh, slot);
        return true;
    }

    static uint32_t slot_write_begin(Shard &sh, uint32_t s)
    {
        const uint32_t v = sh.slot_ver[s].load(std::memory_order_relaxed);
        sh.slot_ver[s].store(v + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        return v;
    }

    static void slot_write_end(Shard &sh, uint32_t s, uint32_t v)
    {
        sh.slot_ver[s].store(v + 2, std::memory_order_release);
    }

    static void index_write_begin(Shard &sh)
    {
        sh.idx_ver.store(sh.idx_ver.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
    }

    static void index_write_end(Shard &sh)
    {
        sh.idx_ver.store(sh.idx_ver.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }

    static uint32_t index_find_locked(const Shard &sh, uint32_t key)
    {
        uint32_t pos = index_hash(key) & sh.imask;
        while (true)
        {
            const uint32_t k = sh.ikeys[pos].load(std::memory_order_relaxed);
            if (k == key)
                return sh.ivals[pos].load(std::memory_order_relaxed);
            if (k == kNoKey)
                return kNoKey;
            pos = (pos + 1) & sh.imask;
        }
    }

    static void index_insert_locked(Shard &sh, uint32_t key, uint32_t slot)
    {
        uint32_t pos = index_hash(key) & sh.imask;
        while (sh.ikeys[pos].load(std::memory_order_relaxed) != kNoKey)
            pos = (pos + 1) & sh.imask;
        index_write_begin(sh);
        sh.ivals[pos].store(slot, std::memory_order_relaxed);
        sh.ikeys[pos].store(key, std::memory_order_relaxed);
        index_write_end(sh);
    }

    // Linear-probing delete with backward shift (no tombstones).
    static void index_remove_locked(Shard &sh, uint32_t key)
    {
        uint32_t hole = index_hash(key) & sh.imask;
        while (true)
        {
            const uint32_t k = sh.ikeys[hole].load(std::memory_order_relaxed);
            if (k == key)
                break;
            if (k == kNoKey)
                return;
            hole = (hole + 1) & sh.imask;
        }
        index_write_begin(sh);
        uint32_t j = hole;
        while (true)
        {
            j = (j + 1) & sh.imask;
            const uint32_t k = sh.ikeys[j].load(std::memory_order_relaxed);
            if (k == kNoKey)
                break;
            const uint32_t home = index_hash(k) & sh.imask;
            const bool stays = hole <= j ? (home > hole && home <= j) : (home > hole || home <= j);
            if (stays)
                continue;
            sh.ivals[hole].store(sh.ivals[j].load(std::memory_order_relaxed), std::memory_order_relaxed);
            sh.ikeys[hole].store(k, std::memory_order_relaxed);
            hole = j;
        }
        sh.ikeys[hole].store(kNoKey, std::memory_order_relaxed);
        index_write_end(sh);
    }

    std::unique_ptr<Shard[]> _shards;
    char *_data = nullptr;
    uint32_t _nshards = 0;
    uint64_t _capacity = 0;
    Stripe _stripes[kStripes];
    ReaderSlot _readers[kMaxReaders];
    std::atomic<uint32_t> _reader_hw{0};
    alignas(64) std::atomic<uint64_t> _epoch{1};
    std::atomic<uint64_t> _inserts{0};
    std::atomic<uint64_t> _evictions{0};
    std::atomic<uint64_t> _skipped{0};
    EvictHook _evict = nullptr;
    void *_evict_ctx = nullptr;
};

} // namespace diskann
