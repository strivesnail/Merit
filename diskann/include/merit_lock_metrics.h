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

enum class MeritLockKind : size_t
{
    Score = 0,
    Heap,
    Metadata,
    DynamicShared,
    DynamicUnique,
    DirectoryShared,
    DirectoryUnique,
    CommitIo,
    Count
};

struct MeritLockMetric
{
    std::atomic<uint64_t> samples{0};
    std::atomic<uint64_t> wait_ns{0};
    std::atomic<uint64_t> hold_ns{0};
};

inline std::array<MeritLockMetric, static_cast<size_t>(MeritLockKind::Count)> &merit_lock_metrics()
{
    static std::array<MeritLockMetric, static_cast<size_t>(MeritLockKind::Count)> metrics;
    return metrics;
}

inline bool merit_lock_should_sample(MeritLockKind kind)
{
    thread_local std::array<uint32_t, static_cast<size_t>(MeritLockKind::Count)> counters{};
    const size_t idx = static_cast<size_t>(kind);
    return (++counters[idx] & 1023u) == 0;
}

inline uint64_t merit_lock_now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline void merit_lock_record(MeritLockKind kind, uint64_t wait_ns, uint64_t hold_ns)
{
    MeritLockMetric &metric = merit_lock_metrics()[static_cast<size_t>(kind)];
    metric.samples.fetch_add(1, std::memory_order_relaxed);
    metric.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
    metric.hold_ns.fetch_add(hold_ns, std::memory_order_relaxed);
}

class MeritTimedMutexGuard
{
  public:
    MeritTimedMutexGuard(std::mutex &mu, MeritLockKind kind) : _mu(mu), _kind(kind)
    {
        _sampled = merit_lock_should_sample(kind);
        const uint64_t before = _sampled ? merit_lock_now_ns() : 0;
        _mu.lock();
        if (_sampled)
        {
            _locked_at = merit_lock_now_ns();
            _wait_ns = _locked_at - before;
        }
    }

    ~MeritTimedMutexGuard()
    {
        if (_sampled)
            merit_lock_record(_kind, _wait_ns, merit_lock_now_ns() - _locked_at);
        _mu.unlock();
    }

    MeritTimedMutexGuard(const MeritTimedMutexGuard &) = delete;
    MeritTimedMutexGuard &operator=(const MeritTimedMutexGuard &) = delete;

  private:
    std::mutex &_mu;
    MeritLockKind _kind;
    bool _sampled = false;
    uint64_t _wait_ns = 0;
    uint64_t _locked_at = 0;
};

class MeritTimedSharedMutexGuard
{
  public:
    MeritTimedSharedMutexGuard(std::shared_mutex &mu, MeritLockKind kind, bool exclusive)
        : _mu(mu), _kind(kind), _exclusive(exclusive)
    {
        _sampled = merit_lock_should_sample(kind);
        const uint64_t before = _sampled ? merit_lock_now_ns() : 0;
        if (_exclusive)
            _mu.lock();
        else
            _mu.lock_shared();
        if (_sampled)
        {
            _locked_at = merit_lock_now_ns();
            _wait_ns = _locked_at - before;
        }
    }

    ~MeritTimedSharedMutexGuard()
    {
        if (!_owns)
            return;
        if (_sampled)
            merit_lock_record(_kind, _wait_ns, merit_lock_now_ns() - _locked_at);
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
            merit_lock_record(_kind, _wait_ns, merit_lock_now_ns() - _locked_at);
        if (_exclusive)
            _mu.unlock();
        else
            _mu.unlock_shared();
        _owns = false;
        _sampled = false;
    }

    MeritTimedSharedMutexGuard(const MeritTimedSharedMutexGuard &) = delete;
    MeritTimedSharedMutexGuard &operator=(const MeritTimedSharedMutexGuard &) = delete;

  private:
    std::shared_mutex &_mu;
    MeritLockKind _kind;
    bool _exclusive = false;
    bool _sampled = false;
    bool _owns = true;
    uint64_t _wait_ns = 0;
    uint64_t _locked_at = 0;
};

class MeritTimedRegion
{
  public:
    explicit MeritTimedRegion(MeritLockKind kind) : _kind(kind), _sampled(merit_lock_should_sample(kind))
    {
        if (_sampled)
            _started_at = merit_lock_now_ns();
    }

    ~MeritTimedRegion()
    {
        if (_sampled)
            merit_lock_record(_kind, 0, merit_lock_now_ns() - _started_at);
    }

  private:
    MeritLockKind _kind;
    bool _sampled = false;
    uint64_t _started_at = 0;
};

} // namespace diskann
