#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace diskann
{
class NeighborCacheNodes
{
    struct Shard
    {
        using Order = std::list<uint32_t>;
        struct Entry
        {
            size_t slot;
            Order::iterator position;
        };
        std::mutex mutex;
        Order order;
        std::unordered_map<uint32_t, Entry> entries;
        std::vector<char> records;
        size_t capacity = 0;
    };
    std::array<Shard, 64> shards;
    const size_t width;
    size_t mask;
    std::unique_ptr<std::atomic<uint8_t>[]> counters;

    Shard &shard(uint32_t id)
    {
        return shards[((uint64_t(id) * 0x9e3779b97f4a7c15ULL) >> 32) & 63];
    }
    std::array<size_t, 2> buckets(uint32_t id) const
    {
        uint64_t h = uint64_t(id) * 0x9e3779b97f4a7c15ULL;
        h ^= h >> 29;
        size_t a = h & mask, b = (h >> 32) & mask;
        return {a, a == b ? ((a + 1) & mask) : b};
    }
    void update_filter(uint32_t id, int change)
    {
        for (auto pos : buckets(id))
        {
            auto &counter = counters[pos];
            auto old = counter.load(std::memory_order_relaxed);
            // Saturated buckets stay positive to avoid false negatives.
            while (old != UINT8_MAX &&
                   !counter.compare_exchange_weak(old, uint8_t(old + change), std::memory_order_release,
                                                  std::memory_order_relaxed))
            {
            }
        }
    }
    bool maybe_contains(uint32_t id) const
    {
        auto b = buckets(id);
        return counters[b[0]].load(std::memory_order_acquire) && counters[b[1]].load(std::memory_order_acquire);
    }

  public:
    NeighborCacheNodes(size_t capacity, size_t record_bytes) : width(record_bytes)
    {
        if (!capacity || !width || capacity > SIZE_MAX / width || capacity > SIZE_MAX / 16)
            throw std::invalid_argument("invalid node cache dimensions");
        size_t n = 2;
        while (n < capacity * 8)
            n *= 2;
        mask = n - 1;
        counters = std::make_unique<std::atomic<uint8_t>[]>(n);
        for (size_t i = 0; i < n; ++i)
            counters[i].store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < shards.size(); ++i)
        {
            auto &s = shards[i];
            s.capacity = capacity / 64 + (i < capacity % 64);
            s.records.resize(width * s.capacity);
            s.entries.reserve(s.capacity);
        }
    }
    bool contains(uint32_t id)
    {
        if (!maybe_contains(id))
            return false;
        auto &s = shard(id);
        std::lock_guard lock(s.mutex);
        return s.entries.count(id) != 0;
    }
    bool lookup(uint32_t id, char *dst)
    {
        if (!maybe_contains(id))
            return false;
        auto &s = shard(id);
        std::lock_guard lock(s.mutex);
        auto it = s.entries.find(id);
        if (it == s.entries.end())
            return false;
        s.order.splice(s.order.begin(), s.order, it->second.position);
        std::memcpy(dst, s.records.data() + width * it->second.slot, width);
        return true;
    }
    void admit(uint32_t id, const char *src)
    {
        auto &s = shard(id);
        if (!s.capacity)
            return;
        std::lock_guard lock(s.mutex);
        auto it = s.entries.find(id);
        if (it != s.entries.end())
        {
            s.order.splice(s.order.begin(), s.order, it->second.position);
            return;
        }
        size_t slot = s.entries.size();
        if (slot == s.capacity)
        {
            const uint32_t victim = s.order.back();
            auto entry = s.entries.extract(victim);
            slot = entry.mapped().slot;
            auto position = entry.mapped().position;
            *position = id;
            s.order.splice(s.order.begin(), s.order, position);
            entry.key() = id;
            s.entries.insert(std::move(entry));
            update_filter(victim, -1);
        }
        else
        {
            s.order.push_front(id);
            try
            {
                s.entries.emplace(id, Shard::Entry{slot, s.order.begin()});
            }
            catch (...)
            {
                s.order.pop_front();
                throw;
            }
        }
        std::memcpy(s.records.data() + slot * width, src, width);
        update_filter(id, 1);
    }
};
} // namespace diskann
