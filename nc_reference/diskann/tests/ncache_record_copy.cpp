#include "neighborcache_memory_pool.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <thread>

// Full-record copies must match the existing payload API, respect destination
// bounds, and remain owned by the reader when another thread reuses the slot.
template <typename T> void check_record()
{
    constexpr size_t dim = 32, coords_bytes = dim * sizeof(T), bytes = coords_bytes + 12;
    diskann::NeighborCacheMemoryPool<T> pool;
    pool.init(1, dim, 2, 16);
    pool.commit_initial_load({}, {});
    auto make_record = [](T value, uint32_t neighbor) {
        std::vector<uint32_t> data(bytes / 4);
        std::vector<T> coords(dim, value);
        std::memcpy(data.data(), coords.data(), coords_bytes);
        data[coords_bytes / 4] = 1;
        data[coords_bytes / 4 + 1] = neighbor;
        return data;
    };
    auto a = make_record(T{3}, 7), b = make_record(T{9}, 8);
    pool.try_admit(1, reinterpret_cast<char *>(a.data()), coords_bytes, 0, 0);
    std::array<char, bytes + 16> guarded{};
    guarded.fill(0x2a);
    char *dst = guarded.data() + 8;
    std::fill(dst, dst + bytes, 0);
    assert(pool.copy_record(1, dst, bytes, coords_bytes));
    assert(std::memcmp(dst, a.data(), bytes) == 0);
    for (size_t i = 0; i < 8; ++i)
        assert(guarded[i] == 0x2a && guarded[bytes + 8 + i] == 0x2a);
    std::vector<T> coords;
    std::vector<uint32_t> nbrs;
    assert(pool.copy_payload(1, coords, nbrs));
    assert(std::memcmp(dst, coords.data(), coords_bytes) == 0);
    uint32_t degree = 0;
    std::memcpy(&degree, dst + coords_bytes, sizeof(degree));
    assert(degree == nbrs.size());
    assert(std::memcmp(dst + coords_bytes + 4, nbrs.data(), degree * 4) == 0);
    const auto snapshot = guarded;
    assert(!pool.copy_record(1, dst, coords_bytes + 7, coords_bytes));
    assert(!pool.copy_record(1, dst, bytes, coords_bytes + sizeof(T)));
    assert(!pool.copy_record(1, nullptr, bytes, coords_bytes));
    assert(!pool.copy_record(2, dst, bytes, coords_bytes));
    assert(guarded == snapshot);
    std::thread writer([&] {
        assert(pool.try_admit(2, reinterpret_cast<char *>(b.data()), coords_bytes, 0, 0) == 1);
    });
    writer.join();
    assert(guarded == snapshot);
    assert(!pool.copy_record(1, dst, bytes, coords_bytes));
    assert(pool.copy_record(2, dst, bytes, coords_bytes));
    assert(std::memcmp(dst, b.data(), bytes) == 0);
}
int main()
{
    setenv("NEIGHBORCACHE_NCACHE_SHARDS", "1", 1);
    for (const char *clock : {"0", "1"})
        for (const char *spin : {"0", "1"})
        {
            setenv("NEIGHBORCACHE_NCACHE_CLOCK", clock, 1);
            setenv("NEIGHBORCACHE_NCACHE_SPIN", spin, 1);
            check_record<uint8_t>();
            check_record<float>();
        }
}
