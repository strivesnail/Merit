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
};
}
int main(){diskann::NeighborCacheAdapter::run();std::cout<<"prefetch async writeback tests passed\n";}
