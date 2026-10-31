#include "neighborcache_thread_page_cache.h"
#include <atomic>
#include <cassert>
#include <thread>

namespace {
class Rendezvous {
    std::atomic<unsigned> arrived{0};
  public:
    void wait() {
        const unsigned round=arrived.fetch_add(1,std::memory_order_acq_rel)/2;
        while(arrived.load(std::memory_order_acquire)<2*(round+1)) std::this_thread::yield();
    }
};
}

int main() {
    diskann::NeighborCacheThreadPageCache cache;
    assert(cache.init(4,2,false,false));
    Rendezvous ready;
    auto worker=[&](uint32_t worker_id) {
        cache.enter();
        std::array<char,diskann::defaults::SECTOR_LEN> page{};
        page[0]=static_cast<char>(worker_id+1);
        const uint32_t first=10+worker_id*10;
        const uint32_t members[]={100,200+worker_id};
        assert(cache.put(first,page.data()));
        cache.register_records(first,members,2);
        ready.wait();
        uint32_t key=0,slot=0;
        assert(cache.record_location(100,key,slot,true) && key==first && slot==0);
        assert(cache.record_location(200+worker_id,key,slot) && key==first && slot==1);
        assert(!cache.record_location(200+(1-worker_id),key,slot));
        std::array<char,diskann::defaults::SECTOR_LEN> copy{};
        assert(cache.get_range(first,1,copy.data()) && copy[0]==page[0]);
        // Updating an existing member must preserve recency without allocating
        // or copying a page snapshot. Missing members still get the whole page.
        copy.fill(99);
        std::vector<uint32_t> snapshot_ids{999};
        assert(!cache.seed_snapshot(100,key,snapshot_ids,copy.data(),200+worker_id));
        assert(copy[0]==99 && snapshot_ids==std::vector<uint32_t>{999});
        assert(cache.seed_snapshot(100,key,snapshot_ids,copy.data(),900));
        assert(copy[0]==page[0] && snapshot_ids==std::vector<uint32_t>(members,members+2));
        const uint32_t second_members[]={300+worker_id,100};
        assert(cache.put(first+1,page.data()));
        cache.register_records(first+1,second_members,2);
        assert(cache.record_location(100,key,slot) && key==first+1 && slot==1);
        assert(cache.record_location(100,key,slot,true) && key==first && slot==0);
        // Evict the first page. Its seed and child must leave this worker's index.
        const uint32_t third_members[]={400+worker_id};
        assert(cache.put(first+2,page.data()));
        cache.register_records(first+2,third_members,1);
        assert(!cache.record_location(200+worker_id,key,slot));
        assert(!cache.record_location(100,key,slot,true));
        assert(cache.record_location(100,key,slot) && key==first+1);
        // Replacing page membership must remove the old record.
        const uint32_t replacement[]={500+worker_id};
        assert(cache.put(first+1,page.data(),true));
        cache.register_records(first+1,replacement,1);
        assert(!cache.record_location(100,key,slot));
        assert(cache.record_location(500+worker_id,key,slot));
        cache.erase(first+1);
        assert(!cache.record_location(500+worker_id,key,slot));
        ready.wait();
        cache.exit();
    };
    std::thread a(worker,0),b(worker,1);
    a.join();b.join();

    // Skipping an unnecessary snapshot must retain the original LRU touch.
    diskann::NeighborCacheThreadPageCache recency;
    assert(recency.init(2,1));
    recency.enter();
    std::array<char,diskann::defaults::SECTOR_LEN> page{},unused{};
    const uint32_t first_ids[]={10,11},second_ids[]={20},third_ids[]={30};
    assert(recency.put(1,page.data())); recency.register_records(1,first_ids,2);
    assert(recency.put(2,page.data())); recency.register_records(2,second_ids,1);
    uint32_t key=0,slot=0;
    std::vector<uint32_t> ids;
    assert(!recency.seed_snapshot(10,key,ids,unused.data(),11));
    assert(recency.put(3,page.data())); recency.register_records(3,third_ids,1);
    assert(recency.record_location(10,key,slot));
    assert(!recency.record_location(20,key,slot));
    recency.exit();

    // D-page version changes must still reject sector reads/writeback while
    // immutable node snapshots remain useful in the other worker's prefetch.
    diskann::NeighborCacheThreadPageCache snapshots;
    assert(snapshots.init(4,2,false,false));
    snapshots.configure_dynamic(1000,4);
    Rendezvous copied,invalidated;
    auto snapshot_worker=[&](unsigned id) {
        snapshots.enter();
        std::array<char,diskann::defaults::SECTOR_LEN> page{};
        page[0]=42;
        const uint32_t members[]={700};
        assert(snapshots.put(1000,page.data()));
        snapshots.register_records(1000,members,1);
        copied.wait();
        if(id==0) snapshots.erase(1000);
        invalidated.wait();
        if(id==1) {
            uint32_t key=0,slot=0;
            assert(!snapshots.record_location(700,key,slot));
            std::array<char,diskann::defaults::SECTOR_LEN> copy{};
            assert(!snapshots.get_range(1000,1,copy.data()));
            assert(snapshots.copy_record(700,128,copy.data()) && copy[0]==42);
            std::vector<uint32_t> local_members;
            assert(snapshots.seed_snapshot(700,key,local_members,copy.data()));
            assert(key==1000 && local_members==std::vector<uint32_t>{700});
            // Reusing the slot really evicts its old record and its index.
            page[0]=43;
            const uint32_t replacement[]={701};
            assert(snapshots.put(1000,page.data(),true));
            snapshots.register_records(1000,replacement,1);
            assert(!snapshots.copy_record(700,128,copy.data()));
            assert(snapshots.copy_record(701,128,copy.data()) && copy[0]==43);
        }
        snapshots.exit();
    };
    std::thread c(snapshot_worker,0),d(snapshot_worker,1);
    c.join();d.join();

    // The same backing seed can have different private neighbor layouts in
    // different workers. Publishing the second must not discard the first.
    diskann::NeighborCacheThreadPageCache replicas;
    assert(replicas.init(4,2,false,false));
    replicas.configure_dynamic(1000,4);
    Rendezvous published;
    std::mutex publication;
    auto replica_worker=[&](unsigned id) {
        replicas.enter();
        std::array<char,diskann::defaults::SECTOR_LEN> page{};
        page[0]=42;
        page[128]=static_cast<char>(id+1);
        const uint32_t ids[]={800,810+id};
        {
            std::lock_guard<std::mutex> lock(publication);
            assert(replicas.put(1000,page.data(),true));
            replicas.register_records(1000,ids,2);
        }
        published.wait();
        std::array<char,128> record{};
        assert(replicas.copy_record(800,record.size(),record.data()) && record[0]==42);
        assert(replicas.copy_record(810+id,record.size(),record.data()) && record[0]==char(id+1));
        assert(!replicas.copy_record(810+(1-id),record.size(),record.data()));
        replicas.exit();
    };
    std::thread e(replica_worker,0),f(replica_worker,1);
    e.join();f.join();
}
