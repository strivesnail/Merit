#include "neighborcache_memory_pool.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

// Deterministic slot reuse after a hit: the old raw pointers now refer to
// another node, while lock-protected copies retain the acquired record.
template <typename T> void eviction_after_hit()
{
    constexpr size_t dim = 128;
    diskann::NeighborCacheMemoryPool<T> pool;
    pool.init(1, dim, 2, 16);
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
    std::thread writer([&] {
        assert(pool.try_admit(2, reinterpret_cast<char *>(b.data()), dim * sizeof(T), 0, 0) == 1);
    });
    writer.join();
    assert(!pool.contains(1));
    T *rechecked_coords = nullptr;
    std::pair<uint32_t, uint32_t *> rechecked_nhood;
    assert(!pool.lookup(1, rechecked_coords, rechecked_nhood));
    assert(raw_coords[0] == T{7} && raw_nhood.second[0] == 12);
    assert(owned_coords.size() == dim);
    assert(std::all_of(owned_coords.begin(), owned_coords.end(), [](T x) { return x == T{3}; }));
    assert(owned_nbrs == std::vector<uint32_t>{11});
}
int main()
{
    setenv("NEIGHBORCACHE_NCACHE_SHARDS", "1", 1);
    for (const char *clock : {"0", "1"}) {
        setenv("NEIGHBORCACHE_NCACHE_CLOCK", clock, 1);
        eviction_after_hit<uint8_t>();
        eviction_after_hit<float>();
    }
    std::cout << "N-cache slot reuse: raw pointers replaced; acquired snapshots preserved (LRU/clock, uint8/float)\n";
}
