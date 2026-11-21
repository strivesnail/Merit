#include "pq_flash_index.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <unistd.h>

namespace diskann {
// Exercise the actual record lookup, including reused outputs, not a substitute
// vector-copy helper. The old implementation returns dim instead of aligned_dim.
class NeighborCacheAdapter {
    template<class T> static void check(const std::vector<T> &coords,
                                        const std::vector<T> &expected, size_t aligned)
    {
        assert(coords.size() == aligned);
        assert(std::equal(expected.begin(), expected.end(), coords.begin()));
        for (size_t i = expected.size(); i < aligned; ++i) assert(coords[i] == T{});
        // Model both consumers: aligned-length copy and full-vector L2.
        std::vector<T> scratch(aligned, T{9});
        std::memcpy(scratch.data(), coords.data(), aligned * sizeof(T));
        double actual = 0, reference = 0;
        for (size_t i = 0; i < aligned; ++i) actual += double(scratch[i]) * double(scratch[i]);
        for (T value : expected) reference += double(value) * double(value);
        assert(actual == reference);
    }

    template<class T> static void run(size_t dim)
    {
        using Index = PQFlashIndex<T>;
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader, Metric::L2);
        const size_t aligned = (dim + 7) & ~size_t(7);
        idx._data_dim = dim; idx._aligned_dim = aligned;
        idx._disk_bytes_per_point = dim * sizeof(T);
        idx._max_degree = 3;
        idx._max_node_len = idx._disk_bytes_per_point + 16;
        idx._num_points = 64;
        idx._neighborcache_seed_prefetch = true;
        assert(idx._neighborcache_shared_cache.init(2, 1));
        assert(idx._neighborcache_shared_cache.thread_private());
        NeighborCacheSharedPageCache::ReadSection section(idx._neighborcache_shared_cache);
        const uint32_t member = 7;
        for (int pass = 0; pass < 3; ++pass)
        {
            // Nonzero graph bytes immediately follow vector bytes on disk;
            // copying those as vector padding is also a failure.
            std::array<char, defaults::SECTOR_LEN> page{};
            std::vector<T> expected(dim);
            for (size_t i = 0; i < dim; ++i) expected[i] = T((i + pass) % 11);
            std::memcpy(page.data(), expected.data(), dim * sizeof(T));
            const uint32_t hood[] = {3, 11, 12, 13};
            std::memcpy(page.data() + dim * sizeof(T), hood, sizeof(hood));
            assert(idx._neighborcache_shared_cache.put(1, page.data(), true));
            idx._neighborcache_shared_cache.register_private_records(1, &member, 1);
            // Start with dirty padding to detect stale data on reuse.
            std::vector<T> coords(aligned + 8, T{17});
            std::vector<uint32_t> nbrs{99};
            assert(idx.neighborcache_seed_page_lookup(member, coords, nbrs));
            check(coords, expected, aligned);
            assert((nbrs == std::vector<uint32_t>{11, 12, 13}));
        }

        // The legacy LRU is a separate producer; raw-length payloads must be
        // converted at its public lookup boundary too.
        idx._neighborcache_seed_prefetch = false;
        idx._neighborcache_seed_page_lru = true;
        for (size_t i = 0; i < Index::kNeighborCacheSeedPageShards; ++i)
        {
            idx._neighborcache_seed_page_shards.emplace_back(new typename Index::NeighborCacheSeedPageShard);
            idx._neighborcache_seed_page_owners.emplace_back(new typename Index::NeighborCacheSeedPageOwnerShard);
        }
        auto &shard = *idx._neighborcache_seed_page_shards[idx.neighborcache_seed_page_shard_index(member)];
        auto &owners = *idx._neighborcache_seed_page_owners[idx.neighborcache_seed_page_shard_index(member)];
        owners.by_node[member] = {member};
        shard.lru.push_front(member);
        typename Index::NeighborCacheSeedPage page;
        page.seed_id = member; page.count = 1; page.ids[0] = member;
        page.coords[0].assign(dim, T{3}); page.nbrs[0] = {11}; page.lru_it = shard.lru.begin();
        shard.pages.insert({member, std::move(page)});
        std::vector<T> coords(aligned, T{17});
        std::vector<uint32_t> nbrs;
        assert(idx.neighborcache_seed_page_lookup(member, coords, nbrs));
        check(coords, std::vector<T>(dim, T{3}), aligned);
        assert(nbrs == std::vector<uint32_t>{11});
        assert(!idx.neighborcache_seed_page_lookup(63, coords, nbrs));
        assert(coords.empty() && nbrs.empty());
    }

    template<class T> static void shared_and_disk()
    {
        using Index = PQFlashIndex<T>;
        setenv("NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE", "0", 1);
        setenv("NEIGHBORCACHE_DCACHE_CAP", "8", 1);
        setenv("NEIGHBORCACHE_MCACHE_CAP", "128", 1);
        setenv("NEIGHBORCACHE_DCACHE_UNBOUNDED", "1", 1);
        setenv("NEIGHBORCACHE_DCACHE_SECOND_PAGE", "0", 1);
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader, Metric::L2);
        idx._num_points = 128; idx._max_nthreads = 1;
        idx._data_dim = 100; idx._aligned_dim = 104;
        idx._disk_bytes_per_point = 100 * sizeof(T);
        idx._max_degree = 3; idx._max_node_len = idx._disk_bytes_per_point + 16;
        idx._nnodes_per_sector = defaults::SECTOR_LEN / idx._max_node_len;
        idx._neighborcache_mem_pool = std::make_unique<NeighborCacheMemoryPool<T>>();
        idx._neighborcache_mem_pool->init(32, 104, 3, 128);
        idx._neighborcache_mem_pool->commit_initial_load({}, {});
        idx._query_sector_cache_enabled = true;
        assert(idx._neighborcache_shared_cache.init(2, 1));
        idx._neighborcache_shared_cache_dyn = true;
        char prefix[] = "/tmp/nc_padding_XXXXXX";
        int fd = mkstemp(prefix); assert(fd >= 0); close(fd); unlink(prefix);
        idx.enable_neighborcache_dynamic_3cache(true, prefix);
        const std::vector<T> expected(100, T{3});
        std::array<char, defaults::SECTOR_LEN> page{};
        std::memcpy(page.data(), expected.data(), idx._disk_bytes_per_point);
        const uint32_t hood[] = {3, 11, 12, 13};
        std::memcpy(page.data() + idx._disk_bytes_per_point, hood, sizeof(hood));
        assert(idx.neighborcache_seed_prefetch_publish_page(7, {7}, page.data()));
        std::vector<T> coords(120, T{17});
        std::vector<uint32_t> nbrs;
        assert(idx.neighborcache_seed_page_lookup(7, coords, nbrs));
        check(coords, expected, 104);
        assert((nbrs == std::vector<uint32_t>{11, 12, 13}));
        assert(idx.neighborcache_dyn_extract_overlay_payload(7, coords, nbrs));
        check(coords, expected, 104);
        // Force a dirty eviction, drain the real asynchronous writer, and read
        // the same record back from D rather than a resident page.
        assert(idx.neighborcache_seed_prefetch_publish_page(8, {8}, page.data()));
        assert(idx.neighborcache_seed_prefetch_publish_page(9, {9}, page.data()));
        idx.stop_neighborcache_dyn_writer();
        assert(idx._neighborcache_dyn_writer_completed >= 1);
        assert(!idx.neighborcache_seed_page_lookup(7, coords, nbrs));
        coords.assign(120, T{17});
        assert(idx.neighborcache_dyn_extract_overlay_payload(7, coords, nbrs));
        check(coords, expected, 104);
        assert((nbrs == std::vector<uint32_t>{11, 12, 13}));
        idx.enable_neighborcache_dynamic_3cache(false);
        unlink((std::string(prefix) + "_neighborcache_dc.dyn.data").c_str());
        unsetenv("NEIGHBORCACHE_DCACHE_UNBOUNDED");
    }
public:
    static void test()
    {
        setenv("NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE", "1", 1);
        setenv("NEIGHBORCACHE_FREEZE_MAINTENANCE", "0", 1);
        for (size_t dim : {size_t(4), size_t(96), size_t(100), size_t(104), size_t(128)})
        {
            run<int8_t>(dim); run<uint8_t>(dim); run<float>(dim);
        }
        shared_and_disk<int8_t>(); shared_and_disk<uint8_t>(); shared_and_disk<float>();
    }
};
}
int main()
{
    diskann::NeighborCacheAdapter::test();
    std::cout << "Prefetch and legacy LRU vector padding: int8/uint8/float passed\n";
}
