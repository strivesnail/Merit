#include "pq_flash_index.h"
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

namespace diskann {
// Existing friend adapter gives this test access to the real cache/writer;
// no substitute writeback implementation or timing-dependent disk delay.
class NeighborCacheAdapter {
    using Index = PQFlashIndex<uint8_t>;
    static constexpr size_t bytes = 4096;
    static void wait_idle(Index &idx, uint32_t p) {
        auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (idx._neighborcache_prefetch_writeback[p].load()) {
            assert(std::chrono::steady_clock::now() < until);
            std::this_thread::yield();
        }
    }
public:
    static void run_query_prefetch_page_members()
    {
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader, Metric::L2);
        idx._neighborcache_dyn_physical_cap = 2;
        idx._neighborcache_dyn_members_per_page = 4;
        idx._neighborcache_dyn_page_member_count.assign(2, 0);
        idx._neighborcache_dyn_page_members.assign(8, Index::NEIGHBORCACHE_DYN_INVALID_PAGE);
        idx._neighborcache_dyn_page_gen.assign(2);
        idx._neighborcache_dyn_page_member_count[1] = 3;
        idx._neighborcache_dyn_page_members[4] = 11;
        idx._neighborcache_dyn_page_members[6] = 33;
        idx._neighborcache_dyn_page_gen[1].store(2);
        idx._neighborcache_dcache_query_patch = false;
        const uint32_t sector = Index::NEIGHBORCACHE_DYN_SECTOR_BASE + 1;
        const uint64_t packed = (uint64_t{sector} << 32) | (uint64_t{1} << 16);

        SSDQueryScratch<uint8_t> scratch(128, 4096);
        idx.neighborcache_pack_dyn_page_into_query_prefetch(sector, &scratch);
        assert(scratch.neighborcache_dcache_member_packed_loc.size() == 2);
        assert(scratch.neighborcache_dcache_member_packed_loc.at(11) == packed);
        assert(scratch.neighborcache_dcache_member_packed_loc.at(33) == (packed | 2));
        assert(scratch.neighborcache_prefetched_page_gen.at(1) == 2);
        assert(scratch.neighborcache_dcache_seed_members.empty());

        idx._neighborcache_dyn_page_members[5] = 22;
        idx._neighborcache_dyn_page_gen[1].store(4);
        idx.neighborcache_pack_dyn_page_into_query_prefetch(sector, &scratch);
        assert(scratch.neighborcache_dcache_member_packed_loc.at(22) == (packed | 1));
        assert(scratch.neighborcache_prefetched_page_gen.at(1) == 4);

        SSDQueryScratch<uint8_t> patch_scratch(128, 4096);
        idx._neighborcache_dcache_query_patch = true;
        idx.neighborcache_pack_dyn_page_into_query_prefetch(sector, &patch_scratch);
        const std::vector<uint32_t> expected_members{11, 22, 33};
        assert(patch_scratch.neighborcache_dcache_seed_members.at(11) == expected_members);
        assert(patch_scratch.neighborcache_dcache_member_to_seed.at(33) == 11);
    }

    static void run_heat_saturation(bool single_owner)
    {
        using Metadata = NeighborCacheMetadataCache;
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader, Metric::L2);
        idx._neighborcache_query_owns_metadata = single_owner;
        idx._neighborcache_heap_shard_count = 1;
        idx._neighborcache_heap_position.assign(4, Metadata::kInvalid);
        idx._neighborcache_heap_kind.assign(4, static_cast<uint8_t>(Index::NeighborCacheHeapKind::None));
        idx._neighborcache_mcache.init(4, 2, 4, true,
            single_owner ? Metadata::Access::SingleOwner : Metadata::Access::SingleShard);
        const auto first = idx.neighborcache_count_expansion(0);
        const auto second = idx.neighborcache_count_expansion(1);
        for (uint32_t n = 1; n < Metadata::kMaxHeat - 1u; ++n)
            idx.neighborcache_count_expansion(0);
        assert(idx._neighborcache_mcache.score(0) == Metadata::kMaxHeat - 1);
        assert(idx._neighborcache_decay_cycles.load() == 0);

        // Scores 2 and 3 tie after rounding. The larger slot then leads the max heap.
        using Kind = Index::NeighborCacheHeapKind;
        idx.neighborcache_dyn_heap_insert_unlocked({3, first.slot_id}, Kind::Max);
        idx.neighborcache_dyn_heap_insert_unlocked({2, second.slot_id}, Kind::Max);
        if (single_owner)
        {
            assert(idx.neighborcache_count_expansion(0).score == Metadata::kMaxHeat / 2);
            idx.neighborcache_count_expansion(0);
        }
        else
        {
            std::thread other([&] { idx.neighborcache_count_expansion(0); });
            idx.neighborcache_count_expansion(0);
            other.join();
        }
        assert(idx._neighborcache_mcache.score(0) == Metadata::kMaxHeat / 2 + 1);
        assert(idx._neighborcache_mcache.score(1) == 0);
        assert(idx._neighborcache_decay_cycles.load() == 1);
        const auto &heap = idx._neighborcache_heap_shards[0].max_heap;
        assert(heap.front().slot_id == second.slot_id && heap.front().score == 1);
        assert(idx._neighborcache_heap_position[first.slot_id] == 1);
        assert(idx._neighborcache_heap_position[second.slot_id] == 0);

        idx._neighborcache_node_state_capacity = 4;
        idx._neighborcache_node_state.assign(4, static_cast<uint8_t>(Index::NeighborCacheNodeState::NonSeed));
        idx.neighborcache_dyn_set_state_unlocked(0, first.slot_id, Index::NeighborCacheNodeState::Seed);
        SSDQueryScratch<uint8_t> scratch(128, 4096);
        scratch.mark_neighborcache_heap_dirty(first.slot_id);
        scratch.mark_neighborcache_heap_dirty(second.slot_id);
        idx.neighborcache_dyn_flush_dirty_heap(&scratch);
        assert(scratch.neighborcache_heap_dirty_slots.empty() && heap.empty());
        const auto &min_heap = idx._neighborcache_heap_shards[0].min_heap;
        assert(min_heap.size() == 1 && min_heap.front().slot_id == first.slot_id);
        assert(min_heap.front().score == idx._neighborcache_mcache.score(0));

        const uint32_t fast_node = 2;
        for (uint32_t n = 0; n < Metadata::kMaxHeat / 2u; ++n)
            idx.neighborcache_count_expansion(fast_node, 2);
        assert(idx._neighborcache_mcache.score(fast_node) == Metadata::kMaxHeat - 1);
        assert(idx.neighborcache_count_expansion(fast_node, 2).score == Metadata::kMaxHeat / 2u + 2);
        assert(idx._neighborcache_decay_cycles.load() == 2);
    }

    static void run_atomic_seed_update()
    {
        setenv("NEIGHBORCACHE_DCACHE_UNBOUNDED", "1", 1);
        setenv("NEIGHBORCACHE_ANCESTOR_SEED_LOOKUP", "1", 1);
        setenv("NEIGHBORCACHE_REFRESH_THREAD", "0", 1);
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader, Metric::L2);
        idx._num_points = 128;
        idx._max_nthreads = 2;
        idx._data_dim = idx._aligned_dim = idx._disk_bytes_per_point = 128;
        idx._max_degree = 3;
        idx._max_node_len = 144;
        idx._nnodes_per_sector = 4;
        idx._neighborcache_shared_cache_dyn = true;
        idx._query_sector_cache_enabled = true;
        assert(idx._neighborcache_shared_cache.init(8, 2, /*force_private=*/true));
        char prefix[] = "/tmp/nc_atomic_seed_XXXXXX";
        const int temporary = mkstemp(prefix);
        assert(temporary >= 0);
        close(temporary);
        unlink(prefix);
        idx.enable_neighborcache_dynamic_3cache(true, prefix);
        unlink((std::string(prefix) + "_neighborcache_dc.dyn.data").c_str());
        assert(idx._neighborcache_ancestor_seed_lookup);
        assert(idx._neighborcache_mcache.on_expand(1).present);

        std::atomic<unsigned> entered{0}, publications{0};
        const auto worker = [&] {
            NeighborCacheSharedPageCache::ReadSection query(idx._neighborcache_shared_cache);
            entered.fetch_add(1);
            while (entered.load() != 2)
                std::this_thread::yield();

            alignas(4096) std::array<char, bytes> page{};
            if (!idx.neighborcache_seed_prefetch_publish_page(1, {1, 2}, page.data()))
                return;
            publications.fetch_add(1);
            uint32_t page_id = UINT32_MAX;
            assert(idx.neighborcache_dyn_seed_page_unlocked(1, page_id));
            assert(idx._neighborcache_seed_directory.page_number(1) == page_id);
            uint32_t sector = Index::NEIGHBORCACHE_DYN_SECTOR_BASE + page_id;
            std::vector<uint32_t> members;
            uint64_t version = 0;
            assert(idx._neighborcache_shared_cache.private_seed_snapshot(
                1, sector, members, page.data(), UINT32_MAX, 0, sector, &version));

            idx._neighborcache_dyn_page_readers[page_id].fetch_add(1);
            assert(!idx.neighborcache_seed_prefetch_admit_existing(1, page_id, {1, 3}, page.data(), version));
            idx.neighborcache_dyn_release_page_pin(sector);
            assert(idx.neighborcache_seed_prefetch_admit_existing(1, page_id, {1, 3}, page.data(), version));
            assert(!idx.neighborcache_seed_prefetch_admit_existing(1, page_id, {1, 2}, page.data(), version));
            uint16_t slot = 0;
            assert(idx._neighborcache_mcache.seed_member_slot(1, page_id, 3, slot) && slot == 1);
            assert(!idx._neighborcache_mcache.seed_member_slot(1, page_id, 2, slot));
            assert(idx._neighborcache_seed_directory.member_slot(1, page_id, 3, slot) && slot == 1);
            assert(!idx._neighborcache_seed_directory.member_slot(1, page_id, 2, slot));
        };
        std::thread first(worker), second(worker);
        first.join();
        second.join();
        assert(publications.load() == 1 && idx._neighborcache_dyn_committed_pages.load() == 1);
        unsetenv("NEIGHBORCACHE_ANCESTOR_SEED_LOOKUP");
    }

    static void run_seed_member_order(bool bounded, uint32_t threads = 1)
    {
        setenv("NEIGHBORCACHE_DCACHE_UNBOUNDED", bounded ? "0" : "1", 1);
        if (bounded)
            setenv("NEIGHBORCACHE_DCACHE_CAP", "1", 1);
        setenv("NEIGHBORCACHE_MCACHE_CAP", "128", 1);
        setenv("NEIGHBORCACHE_SEED_T", "5", 1);
        setenv("NEIGHBORCACHE_REFRESH_THREAD", "0", 1);
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader, Metric::L2);
        idx._num_points = 128;
        idx._max_nthreads = threads;
        idx._data_dim = idx._aligned_dim = idx._disk_bytes_per_point = 128;
        idx._max_degree = 64;
        idx._max_node_len = 388;
        idx._nnodes_per_sector = 3;
        idx._neighborcache_mem_pool = std::make_unique<NeighborCacheMemoryPool<uint8_t>>();
        auto &pool = *idx._neighborcache_mem_pool;
        pool.init(32, 128, 64, 128);
        pool.nhood_ptr(0)[0] = 3;
        pool.nhood_ptr(0)[1] = 2;
        pool.nhood_ptr(0)[2] = 3;
        pool.nhood_ptr(0)[3] = 4;
        pool.nhood_ptr(1)[0] = pool.nhood_ptr(2)[0] = 0;
        pool.commit_initial_load({1, 2, 3}, {});
        idx._query_sector_cache_enabled = true;
        assert(idx._neighborcache_shared_cache.init(4, threads));
        idx._neighborcache_shared_cache_dyn = true;
        char prefix[] = "/tmp/nc_ranked_readback_XXXXXX";
        const int temporary = mkstemp(prefix);
        assert(temporary >= 0);
        close(temporary);
        unlink(prefix);
        idx.enable_neighborcache_dynamic_3cache(true, prefix);
        unlink((std::string(prefix) + "_neighborcache_dc.dyn.data").c_str());
        assert(idx._neighborcache_ranked_readback_edges != bounded);
        assert(!idx._neighborcache_ancestor_seed_lookup);
        assert(idx._neighborcache_d_metadata_shared == (threads > 1));
        if (!bounded)
        {
            assert(idx._neighborcache_parent_fetched_by_slot.size() == 0);
            assert(idx._neighborcache_parent_fetched_owner.size() == 0);
        }

        NeighborCacheSharedPageCache::ReadSection query(idx._neighborcache_shared_cache);
        SSDQueryScratch<uint8_t> scratch(128, 64);
        idx.neighborcache_dyn_prepare_query(&scratch);
        const auto seed = idx.neighborcache_count_expansion(1);
        assert(seed.present);
        assert(!idx._neighborcache_real_io_coaccess && !scratch.neighborcache_record_real_io);
        // Simulate completed expansions along the graph edges. Node 4 ranks
        // highest, but its payload is unavailable when the seed page opens.
        for (const uint32_t child : {2u, 3u, 3u, 4u, 4u, 4u})
            idx.neighborcache_dyn_note_touch(idx._neighborcache_mcache.on_real_io_edge(1, child, idx._max_degree), &scratch);
        alignas(uint32_t) std::array<char, 388> record{};
        idx.neighborcache_dyn_note_base_load(2, 1, record.data(), &scratch);
        NeighborCacheMetadataCache::Snapshot ranked;
        assert(idx._neighborcache_mcache.snapshot(1, ranked, true));
        const std::vector<std::pair<uint32_t, uint8_t>> expected{{4, 3}, {3, 2}, {2, 1}};
        assert(ranked.significant == expected && ranked.score == seed.score);
        assert(scratch.neighborcache_storage_node_order.empty());
        assert(idx._neighborcache_payload_stash.empty());
        idx.neighborcache_dyn_flush_dirty_heap(&scratch);
        {
            std::unique_lock<std::shared_mutex> legacy_lock(idx._neighborcache_dyn_mu, std::defer_lock);
            if (threads > 1)
            {
                assert(idx._neighborcache_dyn_free_shards);
                legacy_lock.lock();
            }
            idx.neighborcache_dyn_maybe_refresh_pair();
        }
        uint32_t key = 0;
        std::vector<uint32_t> members;
        alignas(4096) std::array<char, bytes> page{};
        assert(idx._neighborcache_shared_cache.private_seed_snapshot(1, key, members, page.data()));
        assert((members == std::vector<uint32_t>{1, 3, 2}));
        if (!bounded)
        {
            // The page can be live even when a saved slot belongs to a different node.
            alignas(4096) std::array<char, bytes> buffer{};
            std::vector<Index::NeighborCacheReadPending> pending;
            std::vector<AlignedRead> reads;
            std::unordered_map<uint32_t, std::vector<size_t>> groups;
            QueryStats stats{};
            uint64_t offset = 0;
            uint32_t ios = 0;
            const uint64_t location = (uint64_t{key} << 32) | (uint64_t{1} << 16);
            scratch.neighborcache_dcache_loc_cache[2] = location | 1;
            idx.prepare_neighborcache_disk_cache_io({2}, &scratch, buffer.data(), offset, 1, pending, reads, groups,
                                                    &stats, ios);
            assert(pending.empty() && scratch.neighborcache_dyn_pinned_sectors.empty());

            scratch.neighborcache_dcache_loc_cache[2] = location | 2;
            idx.prepare_neighborcache_disk_cache_io({2}, &scratch, buffer.data(), offset, 1, pending, reads, groups,
                                                    &stats, ios);
            assert(pending.size() == 1 && pending[0].id == 2);
            assert(pending[0].node_ptr != nullptr && reads.empty());
            std::promise<void> rejected_while_pinned;
            std::promise<void> pin_released;
            auto released = pin_released.get_future();
            std::thread worker;
            if (threads > 1)
            {
                assert(pending[0].dynamic_page_pinned);
                worker = std::thread([&] {
                    NeighborCacheSharedPageCache::ReadSection read(idx._neighborcache_shared_cache);
                    const auto page_idx = key - Index::NEIGHBORCACHE_DYN_SECTOR_BASE;
                    auto replacement = page;
                    replacement[2 * idx._max_node_len] = 42;
                    assert(!idx.neighborcache_seed_prefetch_admit_existing(
                        1, page_idx, {1, 3, 4}, replacement.data()));
                    rejected_while_pinned.set_value();
                    released.wait();
                    assert(idx.neighborcache_seed_prefetch_admit_existing(
                        1, page_idx, {1, 3, 4}, replacement.data()));
                    std::array<char, 388> value{};
                    assert(idx._neighborcache_shared_cache.copy_private_record(4, value.size(), value.data()));
                    assert(value.front() == 42);
                });
                rejected_while_pinned.get_future().wait();
            }
            for (uint32_t pinned : scratch.neighborcache_dyn_pinned_sectors)
                idx.neighborcache_dyn_release_page_pin(pinned);
            scratch.neighborcache_dyn_pinned_sectors.clear();
            if (threads > 1)
            {
                // Leave this worker's page bytes resident but delay registration
                // of their matching member IDs until after the new version is live.
                idx._neighborcache_shared_cache.register_private_records(key, nullptr, 0);
                {
                    // Updating a live page must not acquire the legacy lock.
                    std::unique_lock<std::shared_mutex> legacy_lock(idx._neighborcache_dyn_mu);
                    pin_released.set_value();
                    worker.join();
                }
                idx._neighborcache_shared_cache.register_private_records(
                    key, members.data(), members.size(), /*require_current_version=*/false);
                std::array<char, 388> old_value{};
                assert(idx._neighborcache_shared_cache.copy_private_record(2, old_value.size(), old_value.data()));
                assert(old_value.front() == page[2 * idx._max_node_len]);
                idx.neighborcache_pack_dyn_page_into_query_prefetch(key, &scratch);
                assert(scratch.neighborcache_dcache_member_packed_loc.at(4) == (location | 2));
                idx.prepare_neighborcache_disk_cache_io({2}, &scratch, buffer.data(), offset, 1, pending, reads, groups,
                                                        &stats, ios);
                assert(pending.empty() && scratch.neighborcache_dyn_pinned_sectors.empty());
            }
        }
        assert(idx.neighborcache_dyn_state_unlocked(seed.slot_id) == Index::NeighborCacheNodeState::Seed);
        assert(idx._neighborcache_ready_pairs.empty() && idx._neighborcache_pending_pair_count == 0);
        if (!bounded && threads == 1)
        {
            // Publish a newer member replica of seed 1. Vary disk visibility independently
            // of the resident copies to exercise demand routing without actual disk I/O.
            assert(idx.neighborcache_seed_prefetch_publish_page(18, {18, 1}, page.data()));
            const uint32_t other = idx.neighborcache_path_seed_sector(18);
            assert(other != UINT32_MAX && other != key);
            auto &own_dirty = idx._neighborcache_prefetch_dirty[key - Index::NEIGHBORCACHE_DYN_SECTOR_BASE];
            auto &other_dirty = idx._neighborcache_prefetch_dirty[other - Index::NEIGHBORCACHE_DYN_SECTOR_BASE];
            const auto saved_own = own_dirty.exchange(0);
            const auto saved_other = other_dirty.exchange(0);
            Index::NeighborCacheDiskLoc resolved{};
            scratch.neighborcache_dcache_member_packed_loc[1] = (uint64_t{other} << 32) | (uint64_t{1} << 16) | 1;
            assert(!idx.neighborcache_dyn_query_local_dcache_hit(1, &scratch));
            assert(idx.neighborcache_resolve_disk_cache_loc(1, &scratch, resolved));
            assert(resolved.sector == key && resolved.slot == 0);
            own_dirty.store(1);
            assert(idx.neighborcache_dyn_query_local_dcache_hit(1, &scratch));
            assert(idx.neighborcache_resolve_disk_cache_loc(1, &scratch, resolved));
            assert(resolved.sector == other && resolved.slot == 1);
            other_dirty.store(1);
            assert(!idx.neighborcache_dyn_query_local_dcache_hit(1, &scratch));
            assert(!idx.neighborcache_resolve_disk_cache_loc(1, &scratch, resolved));
            own_dirty.store(saved_own);
            other_dirty.store(saved_other);
        }
        if (threads > 1)
        {
            std::unique_lock<std::shared_mutex> legacy_lock(idx._neighborcache_dyn_mu);
            auto seed_lock = idx.neighborcache_lock_seed_write(1);
            const uint32_t page_idx = key - Index::NEIGHBORCACHE_DYN_SECTOR_BASE;
            idx._neighborcache_dyn_page_readers[page_idx].fetch_add(1);
            idx.neighborcache_dyn_invalidate_seed_page(1);
            assert(!idx.neighborcache_dyn_page_live(page_idx));
            auto &free = idx._neighborcache_dyn_free_shards[page_idx % Index::kNeighborCachePageShards].pages;
            assert(free.empty());
            idx.neighborcache_dyn_release_page_pin(key);
            assert(free.size() == 1 && free.back() == page_idx);
            assert(idx.neighborcache_dyn_alloc_partitioned_page(page_idx) == page_idx);
        }
        if (bounded)
            unsetenv("NEIGHBORCACHE_DCACHE_CAP");
        unsetenv("NEIGHBORCACHE_DCACHE_UNBOUNDED");
        unsetenv("NEIGHBORCACHE_SEED_T");
    }

    static void run_shared_ncache_eviction()
    {
        setenv("NEIGHBORCACHE_NCACHE_SHARDS", "1", 1);
        setenv("NEIGHBORCACHE_NCACHE_CLOCK", "1", 1);
        setenv("NEIGHBORCACHE_NCACHE_ADMISSION", "mru", 1);
        setenv("NEIGHBORCACHE_REFRESH_THREAD", "0", 1);
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader, Metric::L2);
        idx._num_points = 128;
        idx._max_nthreads = 2;
        idx._data_dim = idx._aligned_dim = idx._disk_bytes_per_point = 128;
        idx._max_degree = 1;
        idx._max_node_len = 136;
        idx._nnodes_per_sector = 30;
        idx._neighborcache_mem_pool = std::make_unique<NeighborCacheMemoryPool<uint8_t>>();
        auto &pool = *idx._neighborcache_mem_pool;
        pool.init(2, 128, 1, 128, 2);
        for (uint32_t slot = 0; slot < 2; ++slot)
            pool.nhood_ptr(slot)[0] = 0;
        pool.commit_initial_load({1, 2}, {});
        idx._query_sector_cache_enabled = true;
        assert(idx._neighborcache_shared_cache.init(4, 2));
        assert(idx._neighborcache_shared_cache.thread_private());
        idx._neighborcache_shared_cache_dyn = true;
        char prefix[] = "/tmp/nc_actual_victim_XXXXXX";
        const int temporary = mkstemp(prefix);
        assert(temporary >= 0);
        close(temporary);
        unlink(prefix);
        idx.enable_neighborcache_dynamic_3cache(true, prefix);
        unlink((std::string(prefix) + "_neighborcache_dc.dyn.data").c_str());
        assert(!pool.single_owner() && !idx.neighborcache_tracks_pending_members());
        // Keep candidate refresh from building pages while inspecting the heap.
        idx._neighborcache_refresh_running.store(true);
        const auto first = idx.neighborcache_count_expansion(1);
        const auto second = idx.neighborcache_count_expansion(2);
        idx.neighborcache_dyn_heap_upsert_unlocked(first.slot_id, first.score);
        idx.neighborcache_dyn_heap_upsert_unlocked(second.slot_id, second.score);
        assert(idx.neighborcache_dyn_heap_kind_unlocked(first.slot_id) == Index::NeighborCacheHeapKind::Max);
        assert(idx.neighborcache_dyn_heap_kind_unlocked(second.slot_id) == Index::NeighborCacheHeapKind::Max);

        // The preflight LRU tail is node 2. CLOCK gives it a second chance
        // after a hit on another worker and picks node 1 at commit.
        bool present = false;
        assert(pool.inspect_admission(3, present) == 2 && !present);
        std::thread([&] {
            uint8_t *coords = nullptr;
            std::pair<uint32_t, uint32_t *> neighbors;
            assert(pool.lookup(2, coords, neighbors));
        }).join();
        SSDQueryScratch<uint8_t> scratch(128, 4096);
        idx.neighborcache_dyn_prepare_query(&scratch);
        const std::array<uint32_t, 34> record{};
        assert(idx.neighborcache_dyn_admit_node(3, UINT32_MAX, reinterpret_cast<const char *>(record.data()), nullptr,
                                               &scratch, 0, true) == 1);
        assert(!pool.contains(1) && pool.contains(2));
        assert(idx.neighborcache_dyn_heap_kind_unlocked(first.slot_id) == Index::NeighborCacheHeapKind::None);
        assert(idx.neighborcache_dyn_heap_kind_unlocked(second.slot_id) == Index::NeighborCacheHeapKind::Max);
        idx.neighborcache_dyn_on_ncache_evict(2, nullptr, /*eviction_completed=*/true);
        assert(idx.neighborcache_dyn_heap_kind_unlocked(second.slot_id) == Index::NeighborCacheHeapKind::Max);
        idx._neighborcache_refresh_running.store(false);
        idx.enable_neighborcache_dynamic_3cache(false);
    }

    static void run_storage_budget(bool unlimited) {
        setenv("NEIGHBORCACHE_DCACHE_CAP","1",1);
        setenv("NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES","1",1);
        setenv("NEIGHBORCACHE_DCACHE_UNBOUNDED",unlimited ? "1" : "0",1);
        setenv("NEIGHBORCACHE_MCACHE_CAP","128",1);
        setenv("NEIGHBORCACHE_DCACHE_SECOND_PAGE","0",1);
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader,Metric::L2);
        idx._num_points=128;
        idx._max_nthreads = !unlimited && std::getenv("NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE") ? 2 : 1;
        idx._data_dim=idx._aligned_dim=idx._disk_bytes_per_point=128;
        idx._max_degree=64;idx._max_node_len=388;idx._nnodes_per_sector=10;
        idx._neighborcache_mem_pool=std::make_unique<NeighborCacheMemoryPool<uint8_t>>();
        idx._neighborcache_mem_pool->init(32,128,64,128);
        idx._neighborcache_mem_pool->commit_initial_load({},{});
        idx._query_sector_cache_enabled=true;
        assert(idx._neighborcache_shared_cache.init(2,1));
        idx._neighborcache_shared_cache_dyn=true;
        char prefix[]="/tmp/nc_storage_budget_XXXXXX";
        int temporary=mkstemp(prefix);assert(temporary>=0);close(temporary);unlink(prefix);
        idx.enable_neighborcache_dynamic_3cache(true,prefix);
        struct stat st{};
        assert(fstat(idx._neighborcache_dyn_write_fd,&st)==0);
        if(unlimited) {
            assert(idx._neighborcache_dyn_page_cap==std::numeric_limits<uint64_t>::max());
            assert(idx._neighborcache_dyn_physical_cap==130);
            assert(st.st_size==0 && st.st_blocks==0);
        } else {
            assert(idx._neighborcache_dyn_page_cap==1);
            assert(st.st_size == static_cast<off_t>((idx._max_nthreads + 2) * bytes));
        }
        std::unique_ptr<NeighborCacheSharedPageCache::ReadSection> section;
        if(idx._neighborcache_shared_cache.thread_private())
            section=std::make_unique<NeighborCacheSharedPageCache::ReadSection>(idx._neighborcache_shared_cache);
        alignas(4096) std::array<char,bytes> page{},copied{};
        for(uint32_t seed=1;seed<=8;++seed) {
            page.fill(0);std::fill(page.begin(),page.begin()+128,char(seed));
            const bool published=idx.neighborcache_seed_prefetch_publish_page(seed,{seed},page.data());
            assert(published==(unlimited || seed==1));
        }
        assert(idx._neighborcache_dyn_committed_pages==(unlimited ? 8 : 1));
        if (!unlimited && idx._neighborcache_shared_cache.thread_private())
        {
            const auto cold = idx.neighborcache_count_expansion(1);
            idx.neighborcache_seed_prefetch_finish_candidate(1, cold.slot_id);
            const auto record = [](uint32_t neighbor) {
                alignas(4) std::array<char, 388> data{};
                const uint32_t degree = 1;
                std::memcpy(data.data() + 128, &degree, sizeof(degree));
                std::memcpy(data.data() + 132, &neighbor, sizeof(neighbor));
                return data;
            };
            const auto hot_record = record(3);
            const auto neighbor_record = record(2);
            idx._neighborcache_mem_pool->try_admit(2, hot_record.data(), 128, 388, 0);
            idx._neighborcache_mem_pool->try_admit(3, neighbor_record.data(), 128, 388, 0);
            auto hot = idx.neighborcache_count_expansion(2);
            for (unsigned i = 0; i < 4; ++i)
                idx._neighborcache_mcache.on_edge(2, 3);
            SSDQueryScratch<uint8_t> scratch(128, 4096);
            idx.neighborcache_dyn_note_touch(hot, &scratch);
            idx.neighborcache_dyn_flush_dirty_heap(&scratch);
            idx.neighborcache_dyn_maybe_refresh_pair();
            uint32_t seed_page = Index::NEIGHBORCACHE_DYN_INVALID_PAGE;
            assert(idx.neighborcache_dyn_seed_page_unlocked(1, seed_page));
            assert(!idx.neighborcache_dyn_seed_page_unlocked(2, seed_page));

            hot = idx.neighborcache_count_expansion(2);
            idx.neighborcache_dyn_note_touch(hot, &scratch);
            idx.neighborcache_dyn_flush_dirty_heap(&scratch);
            idx.neighborcache_dyn_maybe_refresh_pair();
            assert(!idx.neighborcache_dyn_seed_page_unlocked(1, seed_page));
            assert(idx.neighborcache_dyn_seed_page_unlocked(2, seed_page));
            assert(idx._neighborcache_dyn_committed_pages == 1);
            assert(idx._neighborcache_heap_deletes == 1);
            assert(idx.neighborcache_dyn_state_unlocked(cold.slot_id) == Index::NeighborCacheNodeState::NonSeed);
            assert(idx.neighborcache_dyn_state_unlocked(hot.slot_id) == Index::NeighborCacheNodeState::Seed);
        }
        if (unlimited)
        {
            idx._neighborcache_heat_step_after_expansions = 28000;
            SSDQueryScratch<uint8_t> scratch(128, 4096);
            scratch.neighborcache_dcache_maintenance_enabled = true;
            scratch.neighborcache_expand_count = 27999;
            assert(idx.neighborcache_heat_increment() == 1);
            idx.neighborcache_dyn_on_query_end(nullptr, &scratch);
            assert(idx.neighborcache_heat_increment() == 1);
            scratch.neighborcache_expand_count = 1;
            idx.neighborcache_dyn_on_query_end(nullptr, &scratch);
            assert(idx.neighborcache_heat_increment() == 2);
            assert(idx._neighborcache_dyn_committed_pages == 8);
        }
        idx.stop_neighborcache_dyn_writer();
        if(unlimited) {
            assert(idx._neighborcache_dyn_writer_completed>=1);
            assert(idx._neighborcache_dyn_writer_failed==0);
            assert(idx.neighborcache_dyn_disk_readable(Index::NEIGHBORCACHE_DYN_SECTOR_BASE));
            assert(pread(idx._neighborcache_dyn_write_fd,copied.data(),bytes,0)==ssize_t(bytes));
            for(unsigned i=0;i<128;++i)assert(copied[i]==1);
            assert(fstat(idx._neighborcache_dyn_write_fd,&st)==0 && st.st_size>0);
            assert(st.st_size<=8*bytes); // Written pages only; no dataset-sized sparse file.
        }
        idx.enable_neighborcache_dynamic_3cache(false);
        unlink((std::string(prefix)+"_neighborcache_dc.dyn.data").c_str());
        unsetenv("NEIGHBORCACHE_DCACHE_UNBOUNDED");
        std::cout<<(unlimited ? "unlimited" : "bounded")<<" D storage budget passed\n";
    }
    static void run_private_read_pins() {
        setenv("NEIGHBORCACHE_DCACHE_CAP","32",1);
        setenv("NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES","32",1);
        setenv("NEIGHBORCACHE_MCACHE_CAP","128",1);
        setenv("NEIGHBORCACHE_DCACHE_SECOND_PAGE","0",1);
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader,Metric::L2);
        idx._num_points=100;idx._max_nthreads=1;
        idx._data_dim=idx._aligned_dim=idx._disk_bytes_per_point=128;
        idx._max_degree=64;idx._max_node_len=388;idx._nnodes_per_sector=10;
        idx._neighborcache_mem_pool=std::make_unique<NeighborCacheMemoryPool<uint8_t>>();
        idx._neighborcache_mem_pool->init(32,128,64,100);
        idx._neighborcache_mem_pool->commit_initial_load({},{});
        idx._query_sector_cache_enabled=true;
        assert(idx._neighborcache_shared_cache.init(4,1));
        idx._neighborcache_shared_cache_dyn=true;
        char prefix[]="/tmp/nc_read_pin_XXXXXX";
        int temporary=mkstemp(prefix);assert(temporary>=0);close(temporary);unlink(prefix);
        setenv("NEIGHBORCACHE_DCACHE_QUERY_PATCH","1",1);
        setenv("NEIGHBORCACHE_DCACHE_REPACK_ON_READ","1",1);
        idx.enable_neighborcache_dynamic_3cache(true,prefix);
        unsetenv("NEIGHBORCACHE_DCACHE_QUERY_PATCH");
        unsetenv("NEIGHBORCACHE_DCACHE_REPACK_ON_READ");
        assert(!idx._neighborcache_dcache_query_patch && !idx._neighborcache_repack_on_read);
        assert(!idx._neighborcache_patch_started);
        unlink((std::string(prefix)+"_neighborcache_dc.dyn.data").c_str());
        NeighborCacheSharedPageCache::ReadSection section(idx._neighborcache_shared_cache);
        auto record=[](uint32_t id,std::initializer_list<uint32_t> neighbors) {
            std::array<char,388> r{};std::fill(r.begin(),r.begin()+128,char(id));
            uint32_t n=neighbors.size();memcpy(r.data()+128,&n,4);
            size_t at=132;for(uint32_t v:neighbors){memcpy(r.data()+at,&v,4);at+=4;}
            return r;
        };
        auto a=record(1,{2,3}),b=record(2,{}),c=record(3,{});
        for(auto *r:{&a,&b})idx._neighborcache_mem_pool->try_admit(uint8_t((*r)[0]),r->data(),128,388,0,false);
        SSDQueryScratch<uint8_t> lifecycle(128,4096);
        lifecycle.neighborcache_dcache_maintenance_enabled=true;
        const auto first=idx.neighborcache_count_expansion(1);
        assert(first.present);
        idx.neighborcache_dyn_note_touch(first,&lifecycle);
        idx.neighborcache_dyn_note_base_load(2,1,b.data(),&lifecycle);
        // Immediate P publication has no N-eviction dependency. Waiting modes
        // still need their reverse index, including a full-page candidate.
        assert(!idx.neighborcache_tracks_pending_members());
        idx.neighborcache_pending_member_add(2,first.slot_id);
        assert(idx.neighborcache_pending_member_slots(2).empty());
        assert(!idx.neighborcache_pending_member_contains(2));
        idx._neighborcache_pending_require_full_page=true;
        assert(idx.neighborcache_tracks_pending_members());
        idx.neighborcache_pending_member_add(2,first.slot_id);
        assert(idx.neighborcache_pending_member_slots(2)==std::vector<uint32_t>{first.slot_id});
        assert(idx.neighborcache_pending_member_contains(2));
        idx.neighborcache_pending_member_remove(2,first.slot_id);
        assert(!idx.neighborcache_pending_member_contains(2));
        idx._neighborcache_pending_require_full_page=false;
        idx._neighborcache_deferred_ncache_page_write=true;
        assert(idx.neighborcache_tracks_pending_members());
        idx._neighborcache_deferred_ncache_page_write=false;
        idx._neighborcache_seed_pin_neighbors=true;
        assert(idx.neighborcache_tracks_pending_members());
        idx._neighborcache_seed_pin_neighbors=false;
        idx._neighborcache_pending_buffer_cap_bytes=4096;
        assert(idx.neighborcache_tracks_pending_members());
        idx._neighborcache_pending_buffer_cap_bytes=0;
        Index::NeighborCachePendingFlush preflight;
        NeighborCacheMetadataCache::Snapshot preflight_snapshot;
        assert(idx._neighborcache_mcache.snapshot(1,preflight_snapshot));
        assert(idx.neighborcache_dyn_build_pending_flush(1,preflight_snapshot,preflight));
        assert(preflight.member_ids.size()==2 && preflight.member_ids[0]==1 && preflight.member_ids[1]==2);
        const char *id_only=std::getenv("NEIGHBORCACHE_PREFETCH_ID_ONLY_PREFLIGHT");
        if(id_only && std::strcmp(id_only,"0")!=0) {
            assert(preflight.member_coords.empty() && preflight.member_nbrs.empty());
        } else {
            assert(preflight.member_coords.size()==2 && preflight.member_nbrs.size()==2);
        }
        // Interleave another N admission while this pair is preparing its P
        // page. Evicting a member must not enter the obsolete direct-D path
        // and cancel the in-progress publication.
        Index::NeighborCacheReadyPair preparing;
        preparing.insertion_id=1;preparing.insertion_slot=first.slot_id;
        preparing.snap=std::move(preflight);
        idx._neighborcache_ready_pairs[first.slot_id]=std::move(preparing);
        idx._neighborcache_pending_pair_count=1;idx._neighborcache_reserved_free_pages=1;
        idx.neighborcache_dyn_set_state_unlocked(1,first.slot_id,Index::NeighborCacheNodeState::ReadyToInsertion);
        idx.neighborcache_pending_member_add(2,first.slot_id);
        idx._neighborcache_refresh_running=true;
        idx.neighborcache_dyn_on_ncache_evict(2,nullptr);
        assert(idx._neighborcache_ready_pairs.count(first.slot_id)==1);
        assert(idx.neighborcache_dyn_state_unlocked(first.slot_id)==Index::NeighborCacheNodeState::ReadyToInsertion);
        idx._neighborcache_refresh_running=false;
        idx.neighborcache_dyn_clear_ready_pair_unlocked(first.slot_id);
        assert(idx._neighborcache_pending_pair_count==0 && idx._neighborcache_reserved_free_pages==0);
        // Legacy waiting mode keeps bucket and reverse-index cleanup even
        // without a payload buffer; do not leave stale bucket entries behind.
        idx._neighborcache_pending_require_full_page=true;
        Index::NeighborCacheReadyPair waiting;
        waiting.insertion_id=1;waiting.insertion_slot=first.slot_id;
        waiting.snap.member_ids={1,2};
        idx._neighborcache_ready_pairs[first.slot_id]=std::move(waiting);
        idx._neighborcache_pending_pair_count=1;idx._neighborcache_reserved_free_pages=1;
        idx._neighborcache_pending_buffer_buckets[0].insert(first.slot_id);
        idx.neighborcache_pending_member_add(2,first.slot_id);
        idx.neighborcache_dyn_clear_ready_pair_unlocked(first.slot_id);
        assert(idx._neighborcache_pending_buffer_buckets[0].empty());
        assert(!idx.neighborcache_pending_member_contains(2));
        idx._neighborcache_pending_require_full_page=false;
        idx.neighborcache_dyn_flush_dirty_heap(&lifecycle);
        idx.neighborcache_dyn_maybe_refresh_pair();
        assert(idx._neighborcache_seed_page_opens==1);
        assert(idx.neighborcache_dyn_state_unlocked(first.slot_id)==Index::NeighborCacheNodeState::Seed);
        assert(idx.neighborcache_dyn_heap_kind_unlocked(first.slot_id)==Index::NeighborCacheHeapKind::Min);
        assert(idx._neighborcache_ready_pairs.empty() && idx._neighborcache_pending_pair_count==0);
        assert(idx._neighborcache_reserved_free_pages==0);
        for(const auto &bucket:idx._neighborcache_pending_buffer_buckets)assert(bucket.empty());
        assert(!idx.neighborcache_pending_member_contains(1) && !idx.neighborcache_pending_member_contains(2));
        // Further heat must retain Seed eligibility and cannot create another
        // insertion or leave a pair waiting for the asynchronous SSD writer.
        idx.neighborcache_dyn_note_touch(idx.neighborcache_count_expansion(1),&lifecycle);
        idx.neighborcache_dyn_flush_dirty_heap(&lifecycle);
        idx.neighborcache_dyn_maybe_refresh_pair();
        assert(idx._neighborcache_seed_page_opens==1 && idx._neighborcache_ready_pairs.empty());
        assert(!idx.neighborcache_dyn_enqueue_ready_pair(first.slot_id,Index::NeighborCacheCommitSource::Evict));
        assert(idx.neighborcache_dyn_state_unlocked(first.slot_id)==Index::NeighborCacheNodeState::Seed);
        idx.neighborcache_seed_prefetch_finish_candidate(99,first.slot_id);
        assert(idx.neighborcache_dyn_state_unlocked(first.slot_id)==Index::NeighborCacheNodeState::Seed);
        // With all spare physical pages reserved, publication must leave the
        // victim live and release the pair so the candidate can retry.
        auto retry_record=record(50,{2});
        idx._neighborcache_mem_pool->try_admit(50,retry_record.data(),128,388,0,false);
        auto retry=idx.neighborcache_count_expansion(50);
        for(unsigned i=0;i<16;++i)retry=idx.neighborcache_count_expansion(50);
        idx.neighborcache_dyn_note_touch(retry,&lifecycle);
        idx.neighborcache_dyn_note_base_load(2,50,b.data(),&lifecycle);
        idx.neighborcache_dyn_flush_dirty_heap(&lifecycle);
        idx._neighborcache_dyn_page_cap=1;
        assert(idx._neighborcache_dyn_free.empty());
        const auto next_page = idx._neighborcache_dyn_next_page.load();
        idx._neighborcache_dyn_next_page = idx._neighborcache_dyn_physical_cap;
        idx.neighborcache_dyn_maybe_refresh_pair();
        assert(idx._neighborcache_seed_page_opens==1);
        assert(idx.neighborcache_dyn_state_unlocked(retry.slot_id)==Index::NeighborCacheNodeState::NonSeed);
        assert(idx.neighborcache_dyn_state_unlocked(first.slot_id)==Index::NeighborCacheNodeState::Seed);
        assert(idx._neighborcache_ready_pairs.empty() && idx._neighborcache_pending_pair_count==0);
        assert(idx._neighborcache_deletion_to_pending.empty() && idx._neighborcache_deletion_pair_count==0);
        assert(!idx.neighborcache_pending_member_contains(50));
        idx._neighborcache_dyn_next_page = next_page;
        idx._neighborcache_dyn_page_cap=32;
        idx.neighborcache_dyn_maybe_refresh_pair();
        assert(idx._neighborcache_seed_page_opens==2);
        assert(idx.neighborcache_dyn_state_unlocked(retry.slot_id)==Index::NeighborCacheNodeState::Seed);
        assert(idx._neighborcache_ready_pairs.empty() && idx._neighborcache_pending_pair_count==0);
        std::cout<<"seed publication: Seed state, no duplicate selection, immediate cleanup and failed-publication retry passed\n";
        uint32_t sector=0,slot=0;
        assert(idx._neighborcache_shared_cache.private_record_location(1,sector,slot,true,false));
        const uint32_t page=sector-Index::NEIGHBORCACHE_DYN_SECTOR_BASE;
        alignas(4096) std::array<char,bytes> owned{};
        assert(idx._neighborcache_shared_cache.get_range(sector,1,owned.data()));
        const auto original_page=owned; // seed slot includes its on-page tail metadata
        SSDQueryScratch<uint8_t> scratch(128,4096);
        scratch.neighborcache_dyn_pinned_sectors={sector,sector};
        idx._neighborcache_dyn_page_readers[page]=3; // two query tickets + one writer
        idx._neighborcache_prefetch_writeback[page]=1;
        scratch.neighborcache_dcache_loc_cache[2]=123;
        scratch.neighborcache_dcache_member_packed_loc[2]=456;
        scratch.neighborcache_prefetched_page_gen[page]=2;
        auto finish=[&](uint32_t id,uint16_t record_slot) {
            Index::NeighborCacheReadPending p{};p.id=id;p.sec_buf=owned.data();
            p.loc.sector=sector;p.loc.slot=record_slot;p.loc.nsectors=1;
            p.io_base_sector=sector;p.dynamic_page_pinned=true;
            std::vector<std::pair<uint32_t,char *>> ready;uint64_t used=0;
            idx.finalize_neighborcache_pending_nodes({p},{id},&scratch,owned.data(),used,ready,nullptr);
            assert(ready.size()==1);
            assert(memcmp(idx.offset_to_node(ready.front().second,id),
                          id==1?a.data():b.data(),id==1?140:132)==0);
        };
        finish(2,1);
        assert(idx._neighborcache_dyn_page_readers[page]==2);
        assert(scratch.neighborcache_dyn_pinned_sectors.size()==1);
        assert(scratch.neighborcache_dcache_loc_cache.empty());
        assert(scratch.neighborcache_dcache_member_packed_loc.empty());
        assert(scratch.neighborcache_prefetched_page_gen.empty());
        idx.neighborcache_seed_prefetch_note(3,1,c.data());
        assert(idx._neighborcache_seed_page_appends==0); // another ticket still reading
        finish(1,0);
        assert(scratch.neighborcache_dyn_pinned_sectors.empty());
        assert(idx._neighborcache_dyn_page_readers[page]==1); // writer is untouched
        idx.neighborcache_seed_prefetch_note(3,1,c.data());
        assert(idx._neighborcache_seed_page_appends==0);
        idx._neighborcache_prefetch_writeback[page].fetch_sub(1);
        idx.neighborcache_dyn_release_page_pin(sector);
        idx.neighborcache_seed_prefetch_note(3,1,c.data());
        assert(idx._neighborcache_seed_page_appends==1); // no query-end wait
        std::array<char,388> copied{};
        assert(idx._neighborcache_shared_cache.copy_private_record(3,388,copied.data()));
        assert(copied==c);
        auto patch_hit=[&](uint32_t id,uint32_t parent) {
            auto r=record(id,{2,3});
            std::array<uint8_t,128> coords{};
            memcpy(coords.data(),r.data(),coords.size());
            uint32_t neighbors[]={2,3};
            idx.neighborcache_seed_prefetch_note_ncache_hit(id,parent,coords.data(),neighbors,2);
            return r;
        };
        const auto before_append=idx._neighborcache_seed_page_appends.load();
        patch_hit(4,99); // absent parent: no page creation and no append
        assert(idx._neighborcache_seed_page_appends==before_append);
        const auto fourth=patch_hit(4,1);
        assert(idx._neighborcache_seed_page_appends==before_append+1);
        assert(idx._neighborcache_shared_cache.copy_private_record(4,388,copied.data()));
        assert(copied==fourth);
        patch_hit(4,1); // existing member: no duplicate
        assert(idx._neighborcache_seed_page_appends==before_append+1);
        for(uint32_t id=5;id<=10;++id)patch_hit(id,1);
        alignas(4096) std::array<char,bytes> full_page{},after_full{};
        assert(idx._neighborcache_shared_cache.get_range(sector,1,full_page.data()));
        const auto full_appends=idx._neighborcache_seed_page_appends.load();
        patch_hit(11,1); // full page: preserve every original member
        assert(idx._neighborcache_seed_page_appends==full_appends);
        assert(idx._neighborcache_shared_cache.get_range(sector,1,after_full.data()));
        assert(full_page==after_full);
        assert(!idx._neighborcache_shared_cache.private_record_location(11,sector,slot));
        std::cout<<"N-hit append: absent parent, owned payload, duplicate and full-page preservation passed\n";
        {
            std::unique_lock<std::shared_mutex> lock(idx._neighborcache_dyn_mu);
            idx.neighborcache_dyn_invalidate_seed_page(1);
            assert(idx.neighborcache_dyn_alloc_page_unlocked()==page);
        }
        alignas(4096) std::array<char,bytes> replacement;replacement.fill('Z');
        assert(pwrite(idx._neighborcache_dyn_write_fd,replacement.data(),bytes,page*bytes)==ssize_t(bytes));
        assert(owned==original_page);
        std::cout<<"private read pins: batch completion, independent tickets, writer lifetime, neighbor append and slot reuse passed\n";
    }
    static void run() {
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader, Metric::L2);
        char path[] = "/tmp/nc_async_writeback_XXXXXX";
        int fd = mkstemp(path);assert(fd >= 0);unlink(path);
        assert(ftruncate(fd, 128 * bytes) == 0);
        idx._neighborcache_dyn_write_fd = fd;
        idx._neighborcache_dyn_physical_cap = 128;
        idx._neighborcache_prefetch_dirty.assign(128);
        idx._neighborcache_prefetch_writeback.assign(128);
        idx._neighborcache_dyn_page_readers.assign(128);
        idx._neighborcache_dyn_page_retired.assign(128);
        for (unsigned p=0;p<128;++p) {
            idx._neighborcache_prefetch_dirty[p]=0;idx._neighborcache_prefetch_writeback[p]=0;
            idx._neighborcache_dyn_page_readers[p]=0;idx._neighborcache_dyn_page_retired[p]=0;
        }
        auto &cache=idx._neighborcache_shared_cache;
        assert(cache.init(2));assert(cache.capacity_pages()==2);
        cache.set_evict_hook(&Index::neighborcache_prefetch_evict_hook,&idx);
        idx._neighborcache_dyn_write_q_cap=1;
        idx._neighborcache_dyn_writer_thread_count=4;
        // Accept submissions but deliberately do not run the writer yet.
        idx._neighborcache_dyn_writer_started=true;idx._neighborcache_dyn_writer_stop=false;
        auto put=[&](unsigned p,char value) {
            std::array<char,bytes> data;data.fill(value);
            return cache.put(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+p,data.data(),true,&idx._neighborcache_prefetch_dirty[p]);
        };
        auto cached=[&](unsigned p,char expected) {
            std::array<char,bytes> data{};
            assert(cache.get_range(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+p,1,data.data()));
            for(char v:data)assert(v==expected);
        };
        auto disk=[&](unsigned p,char expected) {
            std::array<char,bytes> data{};
            assert(pread(fd,data.data(),bytes,p*bytes)==ssize_t(bytes));
            for(char v:data)assert(v==expected);
        };
        auto start=[&]() {idx._neighborcache_dyn_writer_started=false;idx.start_neighborcache_dyn_writer();};
        auto missing=[&](unsigned p) {
            std::array<char,bytes> data{};
            assert(!cache.get_range(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+p,1,data.data()));
        };
        auto readable=[&](unsigned p) {
            return idx.neighborcache_dyn_disk_readable(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+p);
        };
        auto clear=[&]() {for(unsigned p=0;p<128;++p)cache.erase(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+p);};
        auto pause=[&]() {idx._neighborcache_dyn_writer_started=true;idx._neighborcache_dyn_writer_stop=false;};
        idx._query_sector_cache_enabled=true;idx._neighborcache_shared_cache_dyn=true;
        idx._nnodes_per_sector=10;idx._max_node_len=388;
        idx._neighborcache_dyn_page_member_count.assign(128, 0);
        for (unsigned page = 0; page < 128; ++page)
            idx._neighborcache_dyn_page_member_count[page] = 1;
        Index::NeighborCacheDiskLoc location{};
        location.sector=Index::NEIGHBORCACHE_DYN_SECTOR_BASE+1;location.slot=0;location.nsectors=1;
        idx._neighborcache_dc_map[42]={location};
        auto plan_read=[&](bool ready) {
            SSDQueryScratch<uint8_t> scratch(128,4096);
            NeighborCacheSharedPageCache::ReadSection section(cache);
            alignas(4096) std::array<char,bytes> buffer{};
            uint64_t offset=0;uint32_t ios=0;QueryStats stats{};
            std::vector<Index::NeighborCacheReadPending> pending;
            std::vector<AlignedRead> reads;
            std::unordered_map<uint32_t,std::vector<size_t>> groups;
            idx.prepare_neighborcache_disk_cache_io({42},&scratch,buffer.data(),offset,1,pending,reads,groups,&stats,ios);
            assert(ios==unsigned(ready));assert(pending.size()==unsigned(ready));assert(reads.size()==unsigned(ready));
            assert(stats.n_neighborcache_dyn_hits==unsigned(ready));
            for(auto sector:scratch.neighborcache_dyn_pinned_sectors)idx.neighborcache_dyn_release_page_pin(sector);
            scratch.neighborcache_dyn_pinned_sectors.clear();
        };
        assert(put(1,'A'));assert(put(2,'B'));
        assert(put(3,'C')); // A is gone and C admitted, with NO writer running.
        assert(idx._neighborcache_prefetch_writeback[1]==1);
        assert(idx._neighborcache_dyn_page_readers[1]==1);
        missing(1);cached(3,'C');disk(1,0);assert(!readable(1));
        plan_read(false);assert(idx._neighborcache_writeback_read_bypasses==1);
        std::array<char,bytes> old_disk{};
        assert(!cache.put(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+1,old_disk.data(),false,nullptr,&idx._neighborcache_prefetch_dirty[1]));
        missing(1);
        assert(put(4,'D')); // Full queue cannot retain B or block D admission.
        missing(2);cached(4,'D');assert(!readable(2));
        assert(idx._neighborcache_writeback_queue_drops==1);
        assert(idx._neighborcache_dyn_write_q.size()==1);
        start();idx.stop_neighborcache_dyn_writer();
        disk(1,'A');assert(readable(1));assert(!readable(2));
        plan_read(true);

        // Queue-lock contention also returns immediately, without waiting for
        // the thread holding the queue. Admission remains successful.
        clear();pause();assert(put(30,'L'));assert(put(31,'M'));
        {
            std::lock_guard<std::mutex> lock(idx._neighborcache_dyn_write_q_mu);
            auto admit=std::async(std::launch::async,[&]{return put(32,'N');});
            assert(admit.wait_for(std::chrono::seconds(1))==std::future_status::ready);
            assert(admit.get());
        }
        missing(30);cached(32,'N');assert(!readable(30));
        assert(idx._neighborcache_writeback_queue_drops==2);
        start();idx.stop_neighborcache_dyn_writer();

        // Two versions of one sector queued before any writer starts. Four
        // writers must preserve order, and the older completion cannot publish
        // the newer version before its write completes.
        clear();idx._neighborcache_dyn_write_q_cap=8;pause();
        assert(put(1,'X'));assert(put(2,'B'));assert(put(3,'C'));
        assert(put(1,'Z'));assert(put(4,'D'));assert(put(5,'E'));
        assert(idx._neighborcache_prefetch_writeback[1]==2);
        assert(idx._neighborcache_dyn_write_q.size()==4);
        assert(idx._neighborcache_dyn_write_q.back()->predecessor==idx._neighborcache_dyn_write_q.front());
        assert(!readable(1));missing(1);
        start();idx.stop_neighborcache_dyn_writer();
        disk(1,'Z');assert(readable(1));assert(idx._neighborcache_prefetch_dirty[1]==0);

        // An actual IO error leaves D unreadable, without keeping the evicted
        // entry in prefetch. Demand lookups must use the base index instead.
        clear();idx._neighborcache_dyn_write_fd=-1;pause();
        assert(put(6,'F'));assert(put(7,'G'));assert(put(8,'H'));
        missing(6);cached(8,'H');start();idx.stop_neighborcache_dyn_writer();
        assert(idx._neighborcache_dyn_writer_failed==1);
        assert(!readable(6));disk(6,0);
        idx._neighborcache_dyn_write_fd=fd;

        // Explicit invalidation is immediate. Only physical sector reuse waits
        // for background-write and query pins; it never holds up prefetch.
        clear();pause();
        assert(put(9,'I'));assert(put(10,'J'));assert(put(11,'K'));
        assert(idx._neighborcache_prefetch_writeback[9]==1);
        auto erase=std::async(std::launch::async,[&]{cache.erase(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+9);});
        assert(erase.wait_for(std::chrono::seconds(1))==std::future_status::ready);erase.get();
        idx._neighborcache_dyn_page_readers[9].fetch_add(1); // outstanding query
        idx.neighborcache_dyn_retire_page_unlocked(9);
        assert(idx._neighborcache_dyn_free.empty());assert(idx._neighborcache_dyn_reclaim.empty());
        start();idx.stop_neighborcache_dyn_writer();disk(9,'I');
        assert(idx._neighborcache_dyn_page_readers[9]==1);
        assert(idx._neighborcache_dyn_reclaim.empty());
        idx.neighborcache_dyn_release_page_pin(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+9);
        assert(idx._neighborcache_dyn_reclaim.size()==1 && idx._neighborcache_dyn_reclaim[0]==9);
        assert(idx._neighborcache_writeback_tail.empty());

        // Multiple writers, tiny cache and bounded queue: read-after-write
        // snapshots must never expose a partially written or stale record.
        clear();idx._neighborcache_dyn_write_q_cap=1;
        idx.start_neighborcache_dyn_writer();
        std::vector<std::thread> workers;
        for(unsigned t=0;t<4;++t)workers.emplace_back([&,t]{
            for(unsigned i=0;i<300;++i){
                const unsigned page=20+t;const char value=char(32+i%90);
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
                while(!put(page,value)) {assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
                std::array<char,bytes> data{};
                if(cache.get_range(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+page,1,data.data()))
                    for(char v:data)assert(v==value);
            }
        });
        for(auto &w:workers)w.join();
        idx.stop_neighborcache_dyn_writer();
        for(unsigned p=0;p<128;++p){assert(idx._neighborcache_prefetch_writeback[p]==0);assert(idx._neighborcache_dyn_page_readers[p]==0);}
    }
    static void run_private() {
        run_private_read_pins();
        std::shared_ptr<AlignedFileReader> reader;
        Index idx(reader, Metric::L2);
        char path[] = "/tmp/nc_async_writeback_XXXXXX";
        int fd = mkstemp(path);assert(fd >= 0);unlink(path);
        assert(ftruncate(fd, 128 * bytes) == 0);
        idx._neighborcache_dyn_write_fd = fd;
        idx._neighborcache_dyn_physical_cap = 128;
        idx._neighborcache_prefetch_dirty.assign(128);
        idx._neighborcache_prefetch_writeback.assign(128);
        idx._neighborcache_dyn_page_readers.assign(128);
        idx._neighborcache_dyn_page_retired.assign(128);
        for (unsigned p=0;p<128;++p) {
            idx._neighborcache_prefetch_dirty[p]=0;idx._neighborcache_prefetch_writeback[p]=0;
            idx._neighborcache_dyn_page_readers[p]=0;idx._neighborcache_dyn_page_retired[p]=0;
        }
        auto &cache=idx._neighborcache_shared_cache;
        assert(cache.init(2,1));
        cache.configure_private_dynamic(Index::NEIGHBORCACHE_DYN_SECTOR_BASE,128);
        cache.set_evict_hook(&Index::neighborcache_prefetch_evict_hook,&idx);
        NeighborCacheSharedPageCache::ReadSection query(cache);
        idx._neighborcache_dyn_write_q_cap=1;
        idx._neighborcache_dyn_writer_thread_count=4;
        idx._neighborcache_dyn_writer_started=true;
        idx._neighborcache_dyn_writer_stop=false;
        auto put=[&](unsigned p,char value) {
            std::array<char,bytes> data;data.fill(value);
            assert(cache.put(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+p,data.data(),true,&idx._neighborcache_prefetch_dirty[p]));
        };
        auto readable=[&](unsigned p){return idx.neighborcache_dyn_disk_readable(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+p);};
        auto clear=[&](){for(unsigned p=0;p<128;p++)cache.erase(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+p);};
        auto start=[&](){idx._neighborcache_dyn_writer_started=false;idx.start_neighborcache_dyn_writer();};
        auto disk=[&](unsigned p,char expected){std::array<char,bytes> data{};assert(pread(fd,data.data(),bytes,p*bytes)==ssize_t(bytes));for(char v:data)assert(v==expected);};
        put(1,'A');put(2,'B');put(3,'C'); // A detached, worker is deliberately stopped
        assert(idx._neighborcache_prefetch_writeback[1]==1 && !readable(1));
        assert(!cache.contains_range(Index::NEIGHBORCACHE_DYN_SECTOR_BASE+1,1));
        put(4,'D');assert(idx._neighborcache_writeback_queue_drops==1 && !readable(2));
        start();idx.stop_neighborcache_dyn_writer();disk(1,'A');assert(readable(1));
        // Repeated writes to a sector have distinct versions and ordered tasks.
        clear();idx._neighborcache_dyn_write_q_cap=8;
        idx._neighborcache_dyn_writer_started=true;idx._neighborcache_dyn_writer_stop=false;
        put(1,'X');put(2,'B');put(3,'C');put(1,'Z');put(4,'D');put(5,'E');
        assert(idx._neighborcache_prefetch_writeback[1]==2 && !readable(1));
        start();idx.stop_neighborcache_dyn_writer();disk(1,'Z');assert(readable(1));
        // Failed disk writes do not restore the evicted prefetch entry.
        clear();idx._neighborcache_dyn_write_fd=-1;
        idx._neighborcache_dyn_writer_started=true;idx._neighborcache_dyn_writer_stop=false;
        put(6,'F');put(7,'G');put(8,'H');
        start();idx.stop_neighborcache_dyn_writer();assert(!readable(6));disk(6,0);
        idx._neighborcache_dyn_write_fd=fd;
    }
};
}
int main()
{
    // Exercise the default node-to-page directory in both capacity modes.
    unsetenv("NEIGHBORCACHE_ANCESTOR_SEED_LOOKUP");
    diskann::NeighborCacheAdapter::run_query_prefetch_page_members();
    diskann::NeighborCacheAdapter::run_heat_saturation(true);
    diskann::NeighborCacheAdapter::run_heat_saturation(false);
    diskann::NeighborCacheAdapter::run_storage_budget(false);
    diskann::NeighborCacheAdapter::run_storage_budget(true);
    if (std::getenv("NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE"))
    {
        diskann::NeighborCacheAdapter::run_seed_member_order(false);
        diskann::NeighborCacheAdapter::run_seed_member_order(false, 2);
        diskann::NeighborCacheAdapter::run_seed_member_order(true);
        diskann::NeighborCacheAdapter::run_private();
        diskann::NeighborCacheAdapter::run_shared_ncache_eviction();
    }
    else
        diskann::NeighborCacheAdapter::run();
    diskann::NeighborCacheAdapter::run_atomic_seed_update();
    std::cout << "prefetch async writeback tests passed\n";
}
