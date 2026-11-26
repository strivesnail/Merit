#pragma once

#include "neighborcache_lazy_array.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>

namespace diskann
{
template <size_t MaxMembers> class NeighborCacheSeedPageDirectory
{
  public:
    static constexpr uint32_t kInvalid = UINT32_MAX;

    struct Page
    {
        uint32_t number = kInvalid;
        uint16_t count = 0;
        std::array<uint32_t, MaxMembers> members{};

      private:
        friend class NeighborCacheSeedPageDirectory;
        Page *retired_next = nullptr;
    };

    ~NeighborCacheSeedPageDirectory() { reset(0); }

    // Reset only after query and maintenance threads have stopped using the directory.
    void reset(size_t node_count)
    {
        _entries.for_each_allocated([&](const std::atomic<Page *> &entry) {
            Page *page = entry.load(std::memory_order_relaxed);
            if (page != &_pending)
                delete page;
        });
        for (auto &group : _groups)
            destroy(group.retired.exchange(nullptr, std::memory_order_relaxed));
        _entries.assign(node_count, nullptr, 256);
    }

    bool try_reserve(uint32_t seed)
    {
        if (seed >= _entries.size())
            return false;
        Page *empty = nullptr;
        return _entries[seed].compare_exchange_strong(empty, &_pending);
    }

    void cancel(uint32_t seed)
    {
        Page *pending = &_pending;
        _entries[seed].compare_exchange_strong(pending, nullptr);
    }

    static std::unique_ptr<Page> make_page(uint32_t number, const uint32_t *members, size_t count)
    {
        assert(count > 0 && count <= MaxMembers);
        auto result = std::make_unique<Page>();
        result->number = number;
        result->count = static_cast<uint16_t>(count);
        std::copy_n(members, count, result->members.begin());
        return result;
    }

    // Callers serialize updates to this seed. The pointer is the only
    // publication step visible to directory readers.
    void publish(uint32_t seed, std::unique_ptr<Page> page)
    {
        Page *old = _entries[seed].exchange(page.release());
        if (old == nullptr || old == &_pending)
            return;
        auto &group = group_for(seed);
        queue_retired(group, old, old);
        reclaim(group);
    }

    uint32_t page_number(uint32_t seed) const
    {
        uint32_t number = kInvalid;
        read(seed, [&](const Page &page) { number = page.number; });
        return number;
    }

    bool member_slot(uint32_t seed, uint32_t number, uint32_t node, uint16_t &slot) const
    {
        bool found = false;
        read(seed, [&](const Page &page) {
            if (page.number != number)
                return;
            const auto end = page.members.begin() + page.count;
            const auto member = std::find(page.members.begin(), end, node);
            if (member != end)
            {
                slot = static_cast<uint16_t>(member - page.members.begin());
                found = true;
            }
        });
        return found;
    }

  private:
    struct alignas(64) Group
    {
        std::atomic<uint32_t> readers{0};
        std::atomic<Page *> retired{nullptr};
    };

    template <typename Read> void read(uint32_t seed, Read &&copy) const
    {
        const auto *entry = _entries.find(seed);
        if (!entry || !entry->load(std::memory_order_relaxed))
            return;
        auto &group = group_for(seed);
        // Order entering the read section before loading anything that may
        // be retired. The descriptor never escapes this read section.
        group.readers.fetch_add(1);
        const Page *page = entry->load();
        if (page && page != &_pending)
            copy(*page);
        group.readers.fetch_sub(1);
    }

    Group &group_for(uint32_t seed) const { return _groups[seed % _groups.size()]; }

    static void queue_retired(Group &group, Page *first, Page *last)
    {
        Page *head = group.retired.load(std::memory_order_relaxed);
        do
        {
            last->retired_next = head;
        } while (!group.retired.compare_exchange_weak(head, first, std::memory_order_release,
                                                      std::memory_order_relaxed));
    }

    static void reclaim(Group &group)
    {
        if (group.readers.load() != 0)
            return;
        Page *list = group.retired.exchange(nullptr, std::memory_order_acquire);
        if (!list)
            return;
        // Detach before the second check so a concurrent retirement cannot
        // add a newer descriptor after the reader check.
        if (group.readers.load() == 0)
            destroy(list);
        else
        {
            Page *last = list;
            while (last->retired_next)
                last = last->retired_next;
            queue_retired(group, list, last);
        }
    }

    static void destroy(Page *list)
    {
        while (list)
        {
            Page *next = list->retired_next;
            delete list;
            list = next;
        }
    }

    NeighborCacheLazyArray<std::atomic<Page *>> _entries;
    mutable std::array<Group, 256> _groups;
    Page _pending;
};
} // namespace diskann
