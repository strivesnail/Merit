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
    std::mt19937 rng(231731);
    for (unsigned i = 0; i < 15000; ++i)
    {
        // Include IDs outside the supplied bitmap range: those use the map.
        const uint32_t node = rng() % 4096;
        auto tr = cache.on_expand(node);
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
            diskann::NeighborCacheMetadataCache::Snapshot hinted;
            assert(cache.snapshot_from_slot(id, tr.slot_id, hinted, false) == present);
            if (present)
            {
                assert(cache.node_at(slot) == id && snap.slot_id == slot);
                assert(snap.score == resident.at(id));
                assert(hinted.slot_id == slot && hinted.score == snap.score);
            }
            const auto previous = cache.evictable_size();
            assert(cache.set_evictable_from_slot(id, tr.slot_id, false) == present);
            assert(cache.evictable_size() == previous - present);
            assert(cache.set_evictable(id, true) == present);
            assert(cache.evictable_size() == previous);
        }
    }
}
void concurrent_bits()
{
    setenv("NEIGHBORCACHE_MCACHE_LOOKUP_BITMAP", "1", 1);
    diskann::NeighborCacheMetadataCache cache;
    cache.init(128, 2, 4096);
    for (uint32_t id = 1; id <= 32; ++id)
    {
        assert(cache.on_expand(id).present);
        assert(cache.set_evictable(id, false));
    }
    // Different M-cache shards update distinct bits of the same bitmap words.
    std::thread writer([&] {
        std::mt19937 rng(419);
        for (unsigned i = 0; i < 30000; ++i)
            assert(cache.on_expand(33 + rng() % 31 + 64 * (rng() % 64)).present);
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
    cache.on_edge(1, 99);
    cache.on_edge(1, 99);
    diskann::NeighborCacheMetadataCache::Snapshot registered;
    assert(cache.snapshot(1, registered));
    assert(registered.significant.size() == 1);
    assert(registered.significant.front().first == 99);
    assert(registered.significant.front().second == 2);
    for (unsigned i = 0; i < 1000; ++i)
        cache.on_edge(1, 2 + i % 13);
    diskann::NeighborCacheMetadataCache::Snapshot snap;
    assert(cache.snapshot(1, snap));
    assert(!snap.significant.empty());
    assert(std::is_sorted(snap.significant.begin(), snap.significant.end(), [](auto a, auto b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    }));
    const auto saved = snap.significant;
    cache.on_edge(1, 99);
    assert(snap.significant == saved);
}
void lazy_edge_slots()
{
    diskann::NeighborCacheMetadataCache cache;
    setenv("NEIGHBORCACHE_MCACHE_EAGER_EDGE_SLOTS", "1", 1);
    cache.init(128,2,4096);
    const auto full_bytes=cache.edge_descriptor_capacity_bytes();
    assert(full_bytes>0);
    setenv("NEIGHBORCACHE_MCACHE_EAGER_EDGE_SLOTS", "0", 1);
    cache.init(65536,2,1000000);
    assert(cache.capacity()==65536 && cache.edge_descriptor_capacity_bytes()==0);
    for(uint32_t id=0;id<1000;++id)assert(cache.on_expand(id).present);
    assert(cache.edge_descriptor_capacity_bytes()>0);
    // Full occupancy retains capacity and eviction behavior, without allocating
    // beyond the eager representation. Reinitialization releases descriptors.
    cache.init(128,2,4096);
    assert(cache.edge_descriptor_capacity_bytes()==0);
    for(uint32_t id=0;id<128;++id)assert(cache.on_expand(id).present);
    assert(cache.size()==128 && cache.edge_descriptor_capacity_bytes()==full_bytes);
    cache.on_real_io_edge(0,1);cache.on_real_io_edge(0,1);
    diskann::NeighborCacheMetadataCache::Snapshot before;
    assert(cache.snapshot(0,before) && !before.significant.empty());
    assert(cache.on_expand(64).present); // Make the parent the cold slot.
    auto replaced=cache.on_expand(128);
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
    assert(cache.on_expand(0).present);
    cache.on_real_io_edge(0,1);cache.on_real_io_edge(0,1);
    diskann::NeighborCacheMetadataCache::Snapshot expected;
    assert(cache.snapshot(0,expected) && !expected.significant.empty());
    std::thread writer([&] {
        // Force repeated descriptor growth in the same shard as the snapshot.
        for(uint32_t i=1;i<=5000;++i)assert(cache.on_expand(64*i).present);
    });
    for(unsigned i=0;i<20000;++i) {
        diskann::NeighborCacheMetadataCache::Snapshot snap;
        assert(cache.snapshot(0,snap));
        assert(snap.slot_id==expected.slot_id && snap.score==expected.score);
        assert(snap.significant==expected.significant);
    }
    writer.join();assert(cache.size()==5001 && cache.capacity()==524288);
}
void edge_hits_preserve_victim_order(diskann::NeighborCacheMetadataCache::Access mode)
{
    using Cache = diskann::NeighborCacheMetadataCache;
    Cache cache;
    cache.init(mode == Cache::Access::Sharded ? 128 : 2, 2, 1024, true, mode);
    assert(cache.on_expand(42).inserted);
    assert(cache.on_expand(106).inserted);
    auto edge = cache.on_edge(42, 57);
    assert(edge.present && !edge.inserted && edge.score == 1);
    auto inserted = cache.on_edge(170, 57);
    assert(inserted.inserted && inserted.score == 0 && inserted.evicted_id == 42);
    assert(cache.set_evictable(106, false) && cache.set_evictable(170, false));
    assert(!cache.on_expand(234).present);
    assert(cache.set_evictable(170, true));
    assert(cache.on_expand(234).evicted_id == 170 && cache.contains(106));
}

void concurrent_parent_edges()
{
    using Cache = diskann::NeighborCacheMetadataCache;
    Cache cache;
    cache.init(128, 2, 128, true);
    constexpr unsigned writers = 6, updates = 160;
    std::atomic<bool> start{false};
    std::atomic<unsigned> inserted{0}, done{0};
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < writers; ++i)
        threads.emplace_back([&, i] {
            while (!start.load()) std::this_thread::yield();
            for (unsigned j = 0; j < updates; ++j)
                for (const auto parent : {0u,64u})
                {
                    auto r = cache.on_edge(parent, i + 1);
                    assert(r.present && r.score == 0);
                    if (r.inserted) ++inserted;
                }
            ++done;
        });
    start.store(true);
    uint32_t members[]{0,1};
    while (done.load() != writers)
    {
        Cache::Snapshot snapshot;
        if (cache.snapshot(0, snapshot, true))
        {
            cache.set_seed_members(0, 7, members, 2);
            assert(cache.snapshot_from_slot(0, snapshot.slot_id, snapshot, true));
            assert(snapshot.significant.size() <= writers);
        }
    }
    for (auto &t : threads) t.join();
    assert(inserted.load() == 2 && cache.edge_count() == 2 * writers);
    for (const auto parent : {0u,64u})
    {
        Cache::Snapshot snapshot;
        assert(cache.snapshot(parent, snapshot, true) && snapshot.significant.size() == writers);
        for (unsigned i = 0; i < writers; ++i)
            assert(snapshot.significant[i].first == i + 1 && snapshot.significant[i].second == updates);
    }
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
    constexpr uint32_t nodes = 129;
    cache.init(nodes, 2, nodes, /*index_ids_only=*/true, access);
    assert(cache.has_stable_slots());
    assert(cache.shard_count() == shards);
    assert(cache.slot_capacity() >= nodes);
    for (uint32_t id = 0; id < nodes; ++id)
    {
        const auto touch = cache.on_expand(id);
        assert(touch.inserted && touch.evicted_id == cache.kInvalid);
        assert(cache.node_at(touch.slot_id) == id);
    }
    assert(!cache.on_expand(nodes).present);
    assert(cache.set_evictable(0, false));
    assert(cache.evictable_size() == nodes - 1);
    assert(cache.on_expand(0).score == 2u);
    assert(cache.set_evictable(0, true));
    assert(cache.evictable_size() == nodes);

    cache.init(shards, 2, nodes, /*index_ids_only=*/true, access);
    assert(!cache.has_stable_slots());
    const auto first = cache.on_expand(0);
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
    const auto replacement = cache.on_expand(64);
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
    const auto first = cache.on_expand(0);
    const auto second = cache.on_expand(1);
    cache.on_edge(0, 1);
    cache.on_edge(0, 1);
    cache.on_edge(0, 2);
    for (uint32_t count = 2; count < Cache::kMaxHeat; ++count)
        assert(cache.on_expand(0).score == count);
    assert(!cache.heat_decay_pending());
    const auto saturated = cache.on_expand(0);
    assert(saturated.score == Cache::kMaxHeat && !saturated.heat_update_deferred);
    assert(cache.heat_decay_pending());
    const auto deferred = cache.on_expand(0);
    assert(deferred.score == Cache::kMaxHeat && deferred.heat_update_deferred);

    cache.halve_heat();
    assert(!cache.heat_decay_pending());
    uint32_t node = Cache::kInvalid;
    Heat score = 0;
    assert(cache.node_score_at(first.slot_id, node, score));
    assert(node == 0 && score == Cache::kMaxHeat / 2);
    assert(cache.score_at(second.slot_id) == 0);
    assert(cache.on_expand(0).score == Cache::kMaxHeat / 2 + 1);
    Cache::Snapshot snapshot;
    assert(cache.snapshot(0, snapshot, true));
    assert(snapshot.significant.size() == 1);
    assert(snapshot.significant[0].first == 1 && snapshot.significant[0].second == 1);
    assert(cache.edge_count() == 1);
}

void seed_membership_preserves_heat()
{
    using Cache = diskann::NeighborCacheMetadataCache;
    setenv("NEIGHBORCACHE_EDGE_HEAT_DECAY", "1", 1);
    Cache cache;
    cache.init(128, 2, 64, /*index_ids_only=*/true);
    cache.on_real_io_edge(1, 2);
    cache.on_real_io_edge(1, 2);
    cache.on_real_io_edge(1, 3);
    const uint32_t first[] = {1, 2, 4};
    cache.set_seed_members(1, 0, first, 3);

    uint16_t slot = 0;
    assert(cache.seed_member_slot(1, 0, 2, slot) && slot == 1);
    assert(cache.seed_member_slot(1, 0, 4, slot) && slot == 2);
    assert(!cache.seed_member_slot(1, 0, 3, slot));
    assert(!cache.seed_member_slot(1, 1, 2, slot));
    assert(cache.edge_count() == 2);
    cache.on_expand(5);
    const uint32_t another_seed[] = {5, 2};
    cache.set_seed_members(5, 1, another_seed, 2);

    cache.on_real_io_edge(1, 2);
    cache.on_real_io_edge(1, 4);
    const uint32_t replacement[] = {1, 4, 3};
    cache.set_seed_members(1, 0, replacement, 3);
    assert(!cache.seed_member_slot(1, 0, 2, slot));
    assert(cache.seed_member_slot(5, 1, 2, slot) && slot == 1);
    assert(cache.seed_member_slot(1, 0, 3, slot) && slot == 2);
    assert(cache.seed_member_slot(1, 0, 4, slot) && slot == 1);
    Cache::Snapshot snapshot;
    assert(cache.snapshot(1, snapshot, true));
    const std::vector<std::pair<uint32_t, uint8_t>> expected = {{2, 3}, {3, 1}, {4, 1}};
    assert(snapshot.significant == expected && cache.edge_count() == expected.size());

    cache.halve_heat();
    assert(cache.seed_member_slot(1, 0, 4, slot));
    assert(cache.edge_count() == 1);
    cache.on_real_io_edge(1, 4);
    cache.set_seed_members(1, 0, nullptr, 0);
    assert(!cache.seed_member_slot(1, 0, 4, slot));
    const std::vector<std::pair<uint32_t, uint8_t>> after_retirement = {{2, 1}, {4, 1}};
    assert(cache.snapshot(1, snapshot, true));
    assert(snapshot.significant == after_retirement && cache.edge_count() == after_retirement.size());
}

void slot_boundaries()
{
    using Cache = diskann::NeighborCacheMetadataCache;
    Cache cache;
    for (const uint32_t capacity : {0u, 1u, 63u, 64u, 65u, 385u, 4097u, 64001u})
    {
        cache.init(capacity);
        const uint32_t stride = (capacity + 63) / 64;
        std::vector<uint32_t> expected(64 * stride, Cache::kInvalid);
        for (uint32_t id = 0; id < capacity; ++id)
        {
            const uint32_t slot = (id % 64) * stride + id / 64;
            assert(cache.on_expand(id).slot_id == slot);
            expected[slot] = id;
        }
        auto check = [&](uint32_t slot, uint32_t expected_id) {
            uint32_t node;
            Heat score;
            const bool present = expected_id != Cache::kInvalid;
            assert(cache.node_at(slot) == expected_id);
            assert(cache.node_score_at(slot, node, score) == present);
            assert(node == expected_id && score == (present ? 1 : 0));
            assert(cache.score_at(slot) == score);
        };
        for (uint32_t slot = 0; slot < expected.size(); ++slot)
            check(slot, expected[slot]);
        check(expected.size(), Cache::kInvalid);
        check(UINT32_MAX - 1, Cache::kInvalid);
        check(Cache::kInvalid, Cache::kInvalid);
    }
}

int main()
{
    slot_boundaries();
    using Access = diskann::NeighborCacheMetadataCache::Access;
    for (auto mode : {Access::Sharded, Access::SingleShard, Access::SingleOwner})
        edge_hits_preserve_victim_order(mode);
    concurrent_parent_edges();
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
    seed_membership_preserves_heat();
}
