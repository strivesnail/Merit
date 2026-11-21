#include "neighborcache_memory_pool.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

// Deterministic slot reuse after a hit: the old raw pointers now refer to
// another node, while lock-protected copies retain the acquired record.
template <typename T> void eviction_after_hit(bool single_owner)
{
    constexpr size_t dim = 128;
    diskann::NeighborCacheMemoryPool<T> pool;
    pool.init(1, dim, 2, 16, 1);
    assert(pool.shard_count() == 1);
    pool.set_single_owner(single_owner);
    assert(pool.single_owner() == single_owner);
    assert(!pool.can_borrow_lru_records(2));
    pool.commit_initial_load({}, {});
    auto record = [](T value, uint32_t neighbor) {
        std::vector<uint32_t> bytes((dim * sizeof(T) + 12) / 4);
        std::vector<T> coords(dim, value);
        std::memcpy(bytes.data(), coords.data(), dim * sizeof(T));
        auto *nbrs = bytes.data() + dim * sizeof(T) / 4;
        nbrs[0] = 1; nbrs[1] = neighbor;
        return bytes;
    };
    auto a = record(T{3}, 11), b = record(T{7}, 12);
    pool.try_admit(1, reinterpret_cast<char *>(a.data()), dim * sizeof(T), 0, 0);
    T *raw_coords = nullptr;
    std::pair<uint32_t, uint32_t *> raw_nhood;
    std::vector<T> owned_coords;
    std::vector<uint32_t> owned_nbrs;
    assert(pool.lookup(1, raw_coords, raw_nhood, true, &owned_coords, &owned_nbrs));
    auto replace = [&] {
        T *missing_coords = nullptr;
        std::pair<uint32_t, uint32_t *> missing_nhood;
        assert(!pool.lookup(2, missing_coords, missing_nhood));
        const auto admitted = pool.admit_record(2, reinterpret_cast<char *>(b.data()), dim * sizeof(T),
                                                0, false, true);
        assert(admitted.evicted == 1 && admitted.resident);
        const auto already_resident = pool.admit_record(2, reinterpret_cast<char *>(b.data()), dim * sizeof(T),
                                                        0, false, !single_owner);
        assert(already_resident.evicted == pool.INVALID_NODE && already_resident.resident && pool.size() == 1);
    };
    if (single_owner)
        replace();
    else
        std::thread(replace).join();
    assert(!pool.contains(1));
    T *rechecked_coords = nullptr;
    std::pair<uint32_t, uint32_t *> rechecked_nhood;
    assert(!pool.lookup(1, rechecked_coords, rechecked_nhood));
    assert(raw_coords[0] == T{7} && raw_nhood.second[0] == 12);
    assert(owned_coords.size() == dim);
    assert(std::all_of(owned_coords.begin(), owned_coords.end(), [](T x) { return x == T{3}; }));
    assert(owned_nbrs == std::vector<uint32_t>{11});
}

static void borrow_from_single_owner_lru()
{
    setenv("NEIGHBORCACHE_NCACHE_CLOCK", "0", 1);
    diskann::NeighborCacheMemoryPool<uint8_t> pool;
    pool.init(3, 128, 2, 16, 1);
    pool.set_single_owner(true);
    pool.nhood_ptr(0)[0] = pool.nhood_ptr(1)[0] = pool.nhood_ptr(2)[0] = 0;
    std::fill_n(pool.coord_ptr(0), 128, uint8_t{3});
    pool.commit_initial_load({1, 2, 3}, {});
    pool.pin(2);
    assert(pool.can_borrow_lru_records(2));
    assert(!pool.can_borrow_lru_records(3));

    uint8_t *coords = nullptr;
    std::pair<uint32_t, uint32_t *> nhood;
    assert(pool.lookup(1, coords, nhood));
    alignas(uint32_t) std::array<char, 128 + sizeof(uint32_t)> record{};
    const auto admission = pool.admit_record(4, record.data(), 128, 0, false, true);
    assert(admission.resident && admission.evicted == 3);
    assert(coords[0] == 3 && nhood.first == 0);

    pool.enable_seed_hold_limit(true);
    assert(!pool.can_borrow_lru_records(2));
    pool.enable_seed_hold_limit(false);
    pool.set_single_owner(false);
    assert(!pool.can_borrow_lru_records(2));
}

int main()
{
    unsetenv("NEIGHBORCACHE_NCACHE_SHARDS");
    for (const char *clock : {"0", "1"}) {
        setenv("NEIGHBORCACHE_NCACHE_CLOCK", clock, 1);
        for (bool single_owner : {false, true}) {
            eviction_after_hit<uint8_t>(single_owner);
            eviction_after_hit<float>(single_owner);
        }
    }
    borrow_from_single_owner_lru();
    diskann::NeighborCacheMemoryPool<uint8_t> shared;
    shared.init(0, 128, 2, 16, 2);
    assert(shared.shard_count() == 128 && !shared.single_owner());
    std::cout << "N-cache slot reuse: raw pointers replaced; acquired snapshots preserved (LRU/clock, uint8/float)\n";
}
