#include "neighborcache_metadata_cache.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <random>
#include <thread>
#include <unordered_map>

void reference_membership(bool bitmap)
{
    setenv("NEIGHBORCACHE_MCACHE_LOOKUP_BITMAP", bitmap ? "1" : "0", 1);
    diskann::NeighborCacheMetadataCache cache;
    cache.init(128, 2, 2048);
    assert(cache.membership_bitmap_bytes() == (bitmap ? 256 : 0));
    std::unordered_map<uint32_t, float> resident;
    std::atomic<float> unit{1.0f};
    std::mt19937 rng(231731);
    for (unsigned i = 0; i < 15000; ++i)
    {
        // Include IDs outside the supplied bitmap range: those use the map.
        const uint32_t node = rng() % 4096;
        auto tr = cache.on_expand(node, unit);
        assert(tr.present);
        if (tr.evicted_id != cache.kInvalid)
            assert(resident.erase(tr.evicted_id) == 1);
        resident[node] += 1.0f;
        assert(tr.score == resident.at(node));
        for (unsigned j = 0; j < 5; ++j)
        {
            const uint32_t id = j == 0 ? node : rng() % 4096;
            const bool present = resident.count(id) != 0;
            assert(cache.contains(id) == present);
            const uint32_t slot = cache.slot_of(id);
            assert((slot != cache.kInvalid) == present);
            assert(cache.score(id) == (present ? resident.at(id) : 0.0f));
            diskann::NeighborCacheMetadataCache::Snapshot snap;
            assert(cache.snapshot(id, snap) == present);
            if (present)
            {
                assert(cache.node_at(slot) == id && snap.slot_id == slot);
                assert(snap.score == resident.at(id));
            }
            assert(cache.set_evictable(id, false) == present);
            assert(cache.set_evictable(id, true) == present);
        }
    }
}
void concurrent_bits()
{
    setenv("NEIGHBORCACHE_MCACHE_LOOKUP_BITMAP", "1", 1);
    diskann::NeighborCacheMetadataCache cache;
    cache.init(128, 2, 4096);
    std::atomic<float> unit{1.0f};
    for (uint32_t id = 1; id <= 32; ++id)
    {
        assert(cache.on_expand(id, unit).present);
        assert(cache.set_evictable(id, false));
    }
    // Different M-cache shards update distinct bits of the same bitmap words.
    std::thread writer([&] {
        std::mt19937 rng(419);
        for (unsigned i = 0; i < 30000; ++i)
            assert(cache.on_expand(33 + rng() % 31 + 64 * (rng() % 64), unit).present);
    });
    auto read = [&] {
        for (unsigned i = 0; i < 30000; ++i)
        {
            const uint32_t id = 1 + i % 32;
            assert(cache.contains(id));
            assert(cache.node_at(cache.slot_of(id)) == id);
            assert(cache.score(id) == 1.0f);
        }
    };
    std::thread first(read), second(read);
    writer.join(); first.join(); second.join();
}
void snapshot_sorting()
{
    diskann::NeighborCacheMetadataCache cache;
    cache.init(128, 2, 4096);
    std::atomic<float> unit{1.0f};
    for (unsigned i = 0; i < 1000; ++i)
        cache.on_edge(1, 2 + i % 13, unit);
    diskann::NeighborCacheMetadataCache::Snapshot snap;
    assert(cache.snapshot(1, snap));
    assert(!snap.significant.empty());
    assert(std::is_sorted(snap.significant.begin(), snap.significant.end(), [](auto a, auto b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    }));
    const auto saved = snap.significant;
    cache.on_edge(1, 99, unit);
    assert(snap.significant == saved);
}
void lazy_edge_slots()
{
    diskann::NeighborCacheMetadataCache cache;
    std::atomic<float> unit{1.0f};
    setenv("NEIGHBORCACHE_MCACHE_EAGER_EDGE_SLOTS", "1", 1);
    cache.init(128,2,4096);
    const auto full_bytes=cache.edge_descriptor_capacity_bytes();
    assert(full_bytes>0);
    setenv("NEIGHBORCACHE_MCACHE_EAGER_EDGE_SLOTS", "0", 1);
    cache.init(65536,2,1000000);
    assert(cache.capacity()==65536 && cache.edge_descriptor_capacity_bytes()==0);
    for(uint32_t id=0;id<1000;++id)assert(cache.on_expand(id,unit).present);
    assert(cache.edge_descriptor_capacity_bytes()>0);
    // Full occupancy retains capacity and eviction behavior, without allocating
    // beyond the eager representation. Reinitialization releases descriptors.
    cache.init(128,2,4096);
    assert(cache.edge_descriptor_capacity_bytes()==0);
    for(uint32_t id=0;id<128;++id)assert(cache.on_expand(id,unit).present);
    assert(cache.size()==128 && cache.edge_descriptor_capacity_bytes()==full_bytes);
    cache.on_real_io_edge(0,1);cache.on_real_io_edge(0,1);
    diskann::NeighborCacheMetadataCache::Snapshot before;
    assert(cache.snapshot(0,before) && !before.significant.empty());
    assert(cache.on_expand(64,unit).present); // Make the parent the cold slot.
    auto replaced=cache.on_expand(128,unit);
    assert(replaced.present && replaced.evicted_id==0);
    assert(cache.edge_descriptor_capacity_bytes()==full_bytes);
    diskann::NeighborCacheMetadataCache::Snapshot fresh;
    assert(cache.snapshot(128,fresh) && fresh.significant.empty());
    cache.clear();assert(cache.edge_descriptor_capacity_bytes()==0);
}
void concurrent_edge_slot_growth()
{
    diskann::NeighborCacheMetadataCache cache;
    cache.init(524288,2,1000000);
    std::atomic<float> unit{1.0f};
    assert(cache.on_expand(0,unit).present);
    cache.on_real_io_edge(0,1);cache.on_real_io_edge(0,1);
    diskann::NeighborCacheMetadataCache::Snapshot expected;
    assert(cache.snapshot(0,expected) && !expected.significant.empty());
    std::thread writer([&] {
        // Force repeated descriptor growth in the same shard as the snapshot.
        for(uint32_t i=1;i<=5000;++i)assert(cache.on_expand(64*i,unit).present);
    });
    for(unsigned i=0;i<20000;++i) {
        diskann::NeighborCacheMetadataCache::Snapshot snap;
        assert(cache.snapshot(0,snap));
        assert(snap.slot_id==expected.slot_id && snap.score==expected.score);
        assert(snap.significant==expected.significant);
    }
    writer.join();assert(cache.size()==5001 && cache.capacity()==524288);
}
int main()
{
    setenv("NEIGHBORCACHE_MCACHE_ADAPTIVE_UPDATE", "0", 1);
    lazy_edge_slots();
    concurrent_edge_slot_growth();
    reference_membership(false);
    reference_membership(true);
    concurrent_bits();
    snapshot_sorting();
}
