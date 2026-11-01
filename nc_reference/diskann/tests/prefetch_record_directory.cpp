#include "neighborcache_thread_page_cache.h"
#include <cassert>
#include <cstring>
#include <list>
#include <random>
#include <unordered_map>

// A reference page cache tracks registration order and LRU independently of
// the implementation's record directory. Exercise duplicate location lists,
// collapse to single locations, recycled overflow groups, and page-slot reuse.
int main()
{
    diskann::NeighborCacheThreadPageCache cache;
    assert(cache.init(8, 1, false, false));
    cache.enter();
    constexpr size_t record_bytes = 64;
    struct Page {
        std::vector<uint32_t> ids;
        uint32_t epoch;
        std::list<uint32_t>::iterator lru;
    };
    std::unordered_map<uint32_t, Page> pages;
    std::list<uint32_t> lru;
    std::mt19937 rng(719031);
    auto find = [&](uint32_t id, bool seed_only, uint32_t &key, uint32_t &slot) {
        bool found = false;
        uint32_t newest = 0;
        for (const auto &entry : pages)
            for (uint32_t j = 0; j < entry.second.ids.size(); ++j)
                if (entry.second.ids[j] == id && (!seed_only || j == 0) &&
                    (!found || entry.second.epoch > newest || (entry.second.epoch == newest && j > slot)))
                {
                    found = true; newest = entry.second.epoch; key = entry.first; slot = j;
                }
        return found;
    };
    for (uint32_t epoch = 1; epoch <= 6000; ++epoch)
    {
        const uint32_t key = rng() % 24;
        if (epoch % 7 == 0)
        {
            cache.erase(key);
            auto old = pages.find(key);
            if (old != pages.end()) { lru.erase(old->second.lru); pages.erase(old); }
        }
        else
        {
            auto old = pages.find(key);
            if (old != pages.end()) { lru.erase(old->second.lru); pages.erase(old); }
            else if (pages.size() == 8) { pages.erase(lru.back()); lru.pop_back(); }
            std::vector<uint32_t> ids;
            // Frequent duplicates include multiple occurrences within one page.
            const uint32_t count = 1 + rng() % 6;
            std::array<char, diskann::defaults::SECTOR_LEN> bytes{};
            for (uint32_t j = 0; j < count; ++j)
            {
                const uint32_t id = rng() % 48;
                ids.push_back(id);
                const uint64_t value = (uint64_t(epoch) << 32) | id;
                std::memcpy(bytes.data() + j * record_bytes, &value, sizeof(value));
            }
            assert(cache.put(key, bytes.data(), true));
            cache.register_records(key, ids.data(), ids.size());
            lru.push_front(key);
            pages.emplace(key, Page{ids, epoch, lru.begin()});
        }
        for (uint32_t id = 0; id < 48; ++id)
            for (bool seed_only : {false, true})
            {
                uint32_t want_key = 0, want_slot = 0, got_key = 0, got_slot = 0;
                const bool expected = find(id, seed_only, want_key, want_slot);
                assert(cache.record_location(id, got_key, got_slot, seed_only) == expected);
                if (expected) assert(got_key == want_key && got_slot == want_slot);
            }
        const uint32_t id = rng() % 64;
        uint32_t want_key = 0, want_slot = 0;
        const bool expected = find(id, false, want_key, want_slot);
        std::array<char, record_bytes> record{};
        assert(cache.copy_record(id, record.size(), record.data()) == expected);
        if (expected)
        {
            uint64_t value = 0;
            std::memcpy(&value, record.data(), sizeof(value));
            assert(value == ((uint64_t(pages.at(want_key).epoch) << 32) | id));
            lru.splice(lru.begin(), lru, pages.at(want_key).lru);
        }
    }
    cache.exit();
}
