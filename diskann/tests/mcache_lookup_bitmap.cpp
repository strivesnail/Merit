#include "neighborcache_metadata_cache.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <limits>
#include <random>
#include <thread>
#include <unordered_map>

using Heat = diskann::NeighborCacheMetadataCache::Heat;

void reference_membership(bool bitmap)
{
    setenv("NEIGHBORCACHE_MCACHE_LOOKUP_BITMAP", bitmap ? "1" : "0", 1);
    diskann::NeighborCacheMetadataCache cache;
    cache.init(128, 2, 2048);
    assert(cache.membership_bitmap_bytes() == (bitmap ? 256 : 0));
    std::unordered_map<uint32_t, Heat> resident;
    std::atomic<Heat> unit{1};
    std::mt19937 rng(231731);
    for (unsigned i = 0; i < 15000; ++i)
    {
        // Include IDs outside the supplied bitmap range: those use the map.
        const uint32_t node = rng() % 4096;
        auto tr = cache.on_expand(node, unit);
        assert(tr.present);
        if (tr.evicted_id != cache.kInvalid)
            assert(resident.erase(tr.evicted_id) == 1);
        resident[node] += 1u;
        assert(tr.score == resident.at(node));
        for (unsigned j = 0; j < 5; ++j)
        {
            const uint32_t id = j == 0 ? node : rng() % 4096;
            const bool present = resident.count(id) != 0;
            assert(cache.contains(id) == present);
            const uint32_t slot = cache.slot_of(id);
            assert((slot != cache.kInvalid) == present);
            assert(cache.score(id) == (present ? resident.at(id) : 0u));
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
    std::atomic<Heat> unit{1};
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
            assert(cache.score(id) == 1u);
        }
    };
    std::thread first(read), second(read);
    writer.join(); first.join(); second.join();
}
void snapshot_sorting()
{
    diskann::NeighborCacheMetadataCache cache;
    cache.init(128, 2, 4096);
    std::atomic<Heat> unit{1};
    cache.on_edge(1, 99, unit);
    cache.on_edge(1, 99, unit);
    diskann::NeighborCacheMetadataCache::Snapshot registered;
    assert(cache.snapshot(1, registered));
    assert(registered.significant.size() == 1);
    assert(registered.significant.front().first == 99);
    assert(registered.significant.front().second == 2);
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
    std::atomic<Heat> unit{1};
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
    std::atomic<Heat> unit{1};
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
void lazy_slot_blocks()
{
    using Slot = std::atomic<uint32_t>;
    diskann::NeighborCacheLazyArray<Slot> slots;
    slots.assign(1'000'000, UINT32_MAX);
    assert(slots.capacity() == 0 && slots.find(500'000) == nullptr);

    auto touch = [&](size_t index) {
        slots[index].store(static_cast<uint32_t>(index), std::memory_order_release);
    };
    std::thread first(touch, 500'000);
    std::thread second(touch, 500'001);
    first.join();
    second.join();
    assert(slots.capacity() == slots.kBlockSize);
    assert(slots[500'000].load() == 500'000 && slots[500'001].load() == 500'001);
    assert(slots[500'002].load() == UINT32_MAX);
    const auto *stable = slots.find(500'000);
    slots[999'999].store(7);
    assert(slots.find(500'000) == stable && stable->load() == 500'000);
    assert(slots.find(0) == nullptr);
    slots.assign(10, 0);
    assert(slots.capacity() == 0 && slots[0].load() == 0);

    slots.assign(15, UINT32_MAX, 6);
    slots[5].store(5);
    slots[6].store(6);
    assert(slots.capacity() == 12 && slots[5].load() == 5 && slots[6].load() == 6);
    assert(slots.find(14) == nullptr);
    slots[14].store(14);
    assert(slots.capacity() == 15 && slots.find(14)->load() == 14 && slots.find(15) == nullptr);
}

void stable_index_slots(diskann::NeighborCacheMetadataCache::Access access, size_t shards)
{
    diskann::NeighborCacheMetadataCache cache;
    std::atomic<Heat> unit{1};
    constexpr uint32_t nodes = 129;
    cache.init(nodes, 2, nodes, /*index_ids_only=*/true, access);
    assert(cache.has_stable_slots());
    assert(cache.shard_count() == shards);
    assert(cache.slot_capacity() >= nodes);
    for (uint32_t id = 0; id < nodes; ++id)
    {
        const auto touch = cache.on_expand(id, unit);
        assert(touch.inserted && touch.evicted_id == cache.kInvalid);
        assert(cache.node_at(touch.slot_id) == id);
    }
    assert(!cache.on_expand(nodes, unit).present);
    assert(cache.set_evictable(0, false));
    assert(cache.evictable_size() == nodes - 1);
    assert(cache.on_expand(0, unit).score == 2u);
    assert(cache.set_evictable(0, true));
    assert(cache.evictable_size() == nodes);

    cache.init(shards, 2, nodes, /*index_ids_only=*/true, access);
    assert(!cache.has_stable_slots());
    const auto first = cache.on_expand(0, unit);
    assert(first.inserted);
    cache.on_real_io_edge(0, 17);
    cache.on_real_io_edge(0, 17);
    cache.on_real_io_edge(0, 3);
    cache.on_real_io_edge(0, 9);
    diskann::NeighborCacheMetadataCache::Snapshot by_id, by_slot;
    for (const bool include_all_edges : {false, true})
    {
        assert(cache.snapshot(0, by_id, include_all_edges));
        assert(cache.snapshot_from_slot(0, first.slot_id, by_slot, include_all_edges));
        assert(by_slot.slot_id == by_id.slot_id && by_slot.score == by_id.score);
        assert(by_slot.significant == by_id.significant);
    }
    const auto replacement = cache.on_expand(64, unit);
    assert(replacement.evicted_id == 0 && replacement.slot_id == first.slot_id);
    assert(!cache.snapshot_from_slot(0, first.slot_id, by_slot, true));
    assert(cache.snapshot_from_slot(64, replacement.slot_id, by_slot, true));
    assert(by_slot.significant.empty());
}

void integer_heat()
{
    using Cache = diskann::NeighborCacheMetadataCache;
    Cache cache;
    cache.init(4, 2, 4, true, Cache::Access::SingleOwner);
    std::atomic<Heat> unit{1u << 24};
    const auto first = cache.on_expand(0, unit);
    assert(first.score == (1u << 24));
    unit.store(1);
    assert(cache.on_expand(0, unit).score == (1u << 24) + 1);
    const auto second = cache.on_expand(1, unit);
    unit.store(std::numeric_limits<Heat>::max());
    const auto max_heat = std::numeric_limits<Heat>::max();
    assert(cache.on_expand(0, unit).score == max_heat);
    unit.store(1);
    assert(cache.on_expand(0, unit).score == max_heat);

    cache.scale_scores(Cache::kDecayScale / 2, unit);
    uint32_t node = Cache::kInvalid;
    Heat score = 0;
    assert(cache.node_score_at(first.slot_id, node, score));
    assert(node == 0 && score == max_heat / 2);
    assert(cache.score_at(second.slot_id) == 0 && unit.load() == 1);
}

int main()
{
    using Access = diskann::NeighborCacheMetadataCache::Access;
    stable_index_slots(Access::Sharded, 64);
    stable_index_slots(Access::SingleShard, 1);
    stable_index_slots(Access::SingleOwner, 1);
    lazy_slot_blocks();
    lazy_edge_slots();
    concurrent_edge_slot_growth();
    reference_membership(false);
    reference_membership(true);
    concurrent_bits();
    snapshot_sorting();
    integer_heat();
}
