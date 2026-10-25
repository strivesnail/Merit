#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>

namespace diskann
{

enum class NeighborCacheLockKind : size_t
{
    Score = 0,
    Heap,
    Pair,
    Metadata,
    DynamicShared,
    DynamicUnique,
    CommitIo,
    Count
};

struct NeighborCacheLockMetric
{
    std::atomic<uint64_t> samples{0};
    std::atomic<uint64_t> wait_ns{0};
    std::atomic<uint64_t> hold_ns{0};
};

inline std::array<NeighborCacheLockMetric, static_cast<size_t>(NeighborCacheLockKind::Count)> &neighborcache_lock_metrics()
{
    static std::array<NeighborCacheLockMetric, static_cast<size_t>(NeighborCacheLockKind::Count)> metrics;
    return metrics;
}

inline bool neighborcache_lock_should_sample(NeighborCacheLockKind kind)
{
    thread_local std::array<uint32_t, static_cast<size_t>(NeighborCacheLockKind::Count)> counters{};
    const size_t idx = static_cast<size_t>(kind);
    return (++counters[idx] & 1023u) == 0;
}

inline uint64_t neighborcache_lock_now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline void neighborcache_lock_record(NeighborCacheLockKind kind, uint64_t wait_ns, uint64_t hold_ns)
{
    NeighborCacheLockMetric &metric = neighborcache_lock_metrics()[static_cast<size_t>(kind)];
    metric.samples.fetch_add(1, std::memory_order_relaxed);
    metric.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
    metric.hold_ns.fetch_add(hold_ns, std::memory_order_relaxed);
}

class NeighborCacheTimedMutexGuard
{
  public:
    NeighborCacheTimedMutexGuard(std::mutex &mu, NeighborCacheLockKind kind) : _mu(mu), _kind(kind)
    {
        _sampled = neighborcache_lock_should_sample(kind);
        const uint64_t before = _sampled ? neighborcache_lock_now_ns() : 0;
        _mu.lock();
        if (_sampled)
        {
            _locked_at = neighborcache_lock_now_ns();
            _wait_ns = _locked_at - before;
        }
    }

    ~NeighborCacheTimedMutexGuard()
    {
        if (_sampled)
            neighborcache_lock_record(_kind, _wait_ns, neighborcache_lock_now_ns() - _locked_at);
        _mu.unlock();
    }

    NeighborCacheTimedMutexGuard(const NeighborCacheTimedMutexGuard &) = delete;
    NeighborCacheTimedMutexGuard &operator=(const NeighborCacheTimedMutexGuard &) = delete;

  private:
    std::mutex &_mu;
    NeighborCacheLockKind _kind;
    bool _sampled = false;
    uint64_t _wait_ns = 0;
    uint64_t _locked_at = 0;
};

template <typename Mutex> class NeighborCacheTimedSharedMutexGuard
{
  public:
    NeighborCacheTimedSharedMutexGuard(Mutex &mu, NeighborCacheLockKind kind, bool exclusive)
        : _mu(mu), _kind(kind), _exclusive(exclusive)
    {
        _sampled = neighborcache_lock_should_sample(kind);
        const uint64_t before = _sampled ? neighborcache_lock_now_ns() : 0;
        if (_exclusive)
            _mu.lock();
        else
            _mu.lock_shared();
        if (_sampled)
        {
            _locked_at = neighborcache_lock_now_ns();
            _wait_ns = _locked_at - before;
        }
    }

    ~NeighborCacheTimedSharedMutexGuard()
    {
        if (!_owns)
            return;
        if (_sampled)
            neighborcache_lock_record(_kind, _wait_ns, neighborcache_lock_now_ns() - _locked_at);
        if (_exclusive)
            _mu.unlock();
        else
            _mu.unlock_shared();
    }

    void unlock()
    {
        if (!_owns)
            return;
        if (_sampled)
            neighborcache_lock_record(_kind, _wait_ns, neighborcache_lock_now_ns() - _locked_at);
        if (_exclusive)
            _mu.unlock();
        else
            _mu.unlock_shared();
        _owns = false;
        _sampled = false;
    }

    NeighborCacheTimedSharedMutexGuard(const NeighborCacheTimedSharedMutexGuard &) = delete;
    NeighborCacheTimedSharedMutexGuard &operator=(const NeighborCacheTimedSharedMutexGuard &) = delete;

  private:
    Mutex &_mu;
    NeighborCacheLockKind _kind;
    bool _exclusive = false;
    bool _sampled = false;
    bool _owns = true;
    uint64_t _wait_ns = 0;
    uint64_t _locked_at = 0;
};

class NeighborCacheTimedRegion
{
  public:
    explicit NeighborCacheTimedRegion(NeighborCacheLockKind kind) : _kind(kind), _sampled(neighborcache_lock_should_sample(kind))
    {
        if (_sampled)
            _started_at = neighborcache_lock_now_ns();
    }

    ~NeighborCacheTimedRegion()
    {
        if (_sampled)
            neighborcache_lock_record(_kind, 0, neighborcache_lock_now_ns() - _started_at);
    }

  private:
    NeighborCacheLockKind _kind;
    bool _sampled = false;
    uint64_t _started_at = 0;
};

} // namespace diskann
