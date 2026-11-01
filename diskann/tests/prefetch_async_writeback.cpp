#include "pq_flash_index.h"
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
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
        const auto first=idx._neighborcache_mcache.on_expand(1,idx._neighborcache_score_unit);
        assert(first.present);
        idx.neighborcache_dyn_note_touch(first,&lifecycle);
        idx.neighborcache_dyn_note_base_load(2,1,b.data(),&lifecycle);
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
        idx.neighborcache_dyn_flush_dirty_heap(&lifecycle);
        idx.neighborcache_dyn_maybe_refresh_pair();
        assert(idx._neighborcache_seed_page_opens==1);
        assert(idx.neighborcache_dyn_state_unlocked(first.slot_id)==Index::NeighborCacheNodeState::Seed);
        assert(idx.neighborcache_dyn_heap_kind_unlocked(first.slot_id)==Index::NeighborCacheHeapKind::Min);
        assert(idx._neighborcache_ready_pairs.empty() && idx._neighborcache_pending_pair_count==0);
        assert(idx._neighborcache_reserved_free_pages==0);
        assert(!idx.neighborcache_pending_member_contains(1) && !idx.neighborcache_pending_member_contains(2));
        // Further heat must retain Seed eligibility and cannot create another
        // insertion or leave a pair waiting for the asynchronous SSD writer.
        idx.neighborcache_dyn_note_touch(idx._neighborcache_mcache.on_expand(1,idx._neighborcache_score_unit),&lifecycle);
        idx.neighborcache_dyn_flush_dirty_heap(&lifecycle);
        idx.neighborcache_dyn_maybe_refresh_pair();
        assert(idx._neighborcache_seed_page_opens==1 && idx._neighborcache_ready_pairs.empty());
        assert(!idx.neighborcache_dyn_enqueue_ready_pair(first.slot_id,Index::NeighborCacheCommitSource::Evict));
        assert(idx.neighborcache_dyn_state_unlocked(first.slot_id)==Index::NeighborCacheNodeState::Seed);
        idx.neighborcache_seed_prefetch_finish_candidate(99,first.slot_id);
        assert(idx.neighborcache_dyn_state_unlocked(first.slot_id)==Index::NeighborCacheNodeState::Seed);
        // Failed publication releases the pair and restores both candidates.
        // Raising the D cap then allows the same non-seed candidate to retry.
        auto retry_record=record(50,{2});
        idx._neighborcache_mem_pool->try_admit(50,retry_record.data(),128,388,0,false);
        auto retry=idx._neighborcache_mcache.on_expand(50,idx._neighborcache_score_unit);
        for(unsigned i=0;i<16;++i)retry=idx._neighborcache_mcache.on_expand(50,idx._neighborcache_score_unit);
        idx.neighborcache_dyn_note_touch(retry,&lifecycle);
        idx.neighborcache_dyn_note_base_load(2,50,b.data(),&lifecycle);
        idx.neighborcache_dyn_flush_dirty_heap(&lifecycle);
        idx._neighborcache_dyn_page_cap=1;
        idx.neighborcache_dyn_maybe_refresh_pair();
        assert(idx._neighborcache_seed_page_opens==1);
        assert(idx.neighborcache_dyn_state_unlocked(retry.slot_id)==Index::NeighborCacheNodeState::NonSeed);
        assert(idx.neighborcache_dyn_state_unlocked(first.slot_id)==Index::NeighborCacheNodeState::Seed);
        assert(idx._neighborcache_ready_pairs.empty() && idx._neighborcache_pending_pair_count==0);
        assert(idx._neighborcache_deletion_to_pending.empty() && idx._neighborcache_deletion_pair_count==0);
        assert(!idx.neighborcache_pending_member_contains(50));
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
        idx._neighborcache_prefetch_dirty.reset(new std::atomic<uint64_t>[128]);
        idx._neighborcache_prefetch_writeback.reset(new std::atomic<uint32_t>[128]);
        idx._neighborcache_dyn_page_readers.reset(new std::atomic<uint32_t>[128]);
        idx._neighborcache_dyn_page_retired.reset(new std::atomic<uint8_t>[128]);
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
        idx._neighborcache_dyn_page_member_count.resize(128,1);
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
        idx._neighborcache_prefetch_dirty.reset(new std::atomic<uint64_t>[128]);
        idx._neighborcache_prefetch_writeback.reset(new std::atomic<uint32_t>[128]);
        idx._neighborcache_dyn_page_readers.reset(new std::atomic<uint32_t>[128]);
        idx._neighborcache_dyn_page_retired.reset(new std::atomic<uint8_t>[128]);
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
int main(){if(std::getenv("NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE"))diskann::NeighborCacheAdapter::run_private();else diskann::NeighborCacheAdapter::run();std::cout<<"prefetch async writeback tests passed\n";}
