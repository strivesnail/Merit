#include "pq_flash_index.h"
#include <cassert>
#include <random>
#include <thread>
#include <unordered_map>

namespace diskann {
class NeighborCacheAdapter
{
    using Index = PQFlashIndex<uint8_t>;
    static Index::NeighborCacheDiskLoc location(unsigned sequence)
    {
        Index::NeighborCacheDiskLoc out;
        out.sector = Index::NEIGHBORCACHE_DYN_SECTOR_BASE + 1 + sequence % 127;
        out.slot = sequence % 10; out.nsectors = 1;
        return out;
    }
    static void reset(Index &idx, size_t slots)
    {
        idx._neighborcache_canon_slots.reset(new std::atomic<uint64_t>[slots]);
        idx._neighborcache_canon_mask = slots - 1;
        idx._neighborcache_canon_authoritative = 1;
        for (size_t i = 0; i < slots; ++i) idx._neighborcache_canon_slots[i] = 0;
    }
    static bool same(const Index::NeighborCacheDiskLoc &a, const Index::NeighborCacheDiskLoc &b)
    {
        return a.sector == b.sector && a.slot == b.slot && a.nsectors == b.nsectors;
    }
  public:
    static void run()
    {
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader, Metric::L2);
        reset(idx, 8);
        // All three keys share a home slot. Updating key 9 must find its old
        // slot beyond key 1's tombstone, not leave a second stale key 9 behind.
        idx.neighborcache_canon_publish(1, location(1));
        idx.neighborcache_canon_publish(9, location(2));
        idx.neighborcache_canon_clear_node(1);
        idx.neighborcache_canon_publish(9, location(3));
        Index::NeighborCacheDiskLoc got;
        assert(idx.neighborcache_canon_lookup(9, got) == 1 && same(got, location(3)));
        idx.neighborcache_canon_clear_node(9);
        idx.neighborcache_canon_publish(17, location(4));
        assert(idx.neighborcache_canon_lookup(9, got) == 0);
        std::mt19937 rng(314197);
        for (unsigned cycle = 0; cycle < 200; ++cycle)
        {
            reset(idx, 256);
            std::unordered_map<uint32_t, Index::NeighborCacheDiskLoc> expected;
            for (unsigned step = 0; step < 150; ++step)
            {
                const uint32_t id = rng() % 1024;
                if (rng() % 4 == 0) { idx.neighborcache_canon_clear_node(id); expected.erase(id); }
                else { const auto loc = location(step); idx.neighborcache_canon_publish(id, loc); expected[id] = loc; }
                for (unsigned probe = 0; probe < 5; ++probe)
                {
                    const uint32_t query = probe == 0 ? id : rng() % 1024;
                    const int result = idx.neighborcache_canon_lookup(query, got);
                    auto want = expected.find(query);
                    if (result >= 0)
                    {
                        assert((result == 1) == (want != expected.end()));
                        if (result == 1) assert(same(got, want->second));
                    }
                }
            }
        }
        reset(idx, 8);
        for (uint32_t id = 1; id <= 9; ++id) idx.neighborcache_canon_publish(id, location(id));
        assert(idx._neighborcache_canon_authoritative == 0);
        assert(idx.neighborcache_canon_lookup(1, got) == -1);
        // Independent publishers collide in the accelerator without a shared
        // allocator or a second copy of any key. Their final values must agree.
        reset(idx, 1024);
        std::vector<std::thread> threads;
        for (unsigned thread = 0; thread < 4; ++thread)
            threads.emplace_back([&, thread] {
                for (unsigned round = 0; round < 20; ++round)
                    for (uint32_t i = 0; i < 96; ++i)
                    {
                        const uint32_t id = i * 4 + thread;
                        if (round % 3 == 0) idx.neighborcache_canon_clear_node(id);
                        idx.neighborcache_canon_publish(id, location(id + round));
                    }
            });
        for (auto &thread : threads) thread.join();
        assert(idx._neighborcache_canon_authoritative == 1);
        for (uint32_t id = 0; id < 384; ++id)
            assert(idx.neighborcache_canon_lookup(id, got) == 1 && same(got, location(id + 19)));
        setenv("NEIGHBORCACHE_CANON_SLOTS", "1048576", 1);
        idx._neighborcache_dyn_page_cap = 65536;
        idx._neighborcache_dyn_members_per_page = 10;
        idx.neighborcache_canon_reset();
        assert(idx._neighborcache_canon_mask + 1 == 1048576);
        setenv("NEIGHBORCACHE_CANON_SLOTS", "2049", 1);
        bool rejected = false;
        try { idx.neighborcache_canon_reset(); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
        unsetenv("NEIGHBORCACHE_CANON_SLOTS");
    }
};
}
int main() { diskann::NeighborCacheAdapter::run(); }
