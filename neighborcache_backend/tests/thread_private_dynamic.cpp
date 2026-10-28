#include "neighborcache_shared_page_cache.h"
#include <cassert>
#include <thread>
#include <array>
#include <vector>
#include <mutex>
#include <random>
#include <list>
#include <unordered_map>
using Cache=diskann::NeighborCacheSharedPageCache;
struct Writes {
    std::mutex mu;
    std::vector<std::pair<uint32_t,char>> tasks;
    static bool submit(void *ctx,uint32_t key,const char *bytes,bool discard) {
        auto &w=*static_cast<Writes*>(ctx);
        if(!discard){std::lock_guard<std::mutex> lock(w.mu);w.tasks.emplace_back(key,bytes[0]);}
        return true; // owned task accepted; no disk completion in this test
    }
};
int main() {
    setenv("NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE","1",1);
    setenv("NEIGHBORCACHE_FREEZE_MAINTENANCE","0",1);
    unsetenv("NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY");
    Cache c;assert(c.init(2,2));c.configure_private_dynamic(1000,16);
    Writes writes;c.set_evict_hook(Writes::submit,&writes);
    std::atomic<uint64_t> dirty{0};std::atomic<int> phase{0};
    auto wait=[&](int n){while(phase.load()!=n)std::this_thread::yield();};
    std::thread a([&]{
        Cache::ReadSection q(c);std::array<char,4096> bytes{},out{};bytes.fill('A');
        assert(c.put(1000,bytes.data(),false,nullptr,&dirty));phase=1;wait(2);
        bytes.fill('B');assert(c.put(1000,bytes.data(),true,&dirty));assert(dirty.load()!=0);
        phase=3;wait(4);
        // Own dirty B is evicted and handed off without waiting for disk.
        bytes.fill('C');assert(c.put(1001,bytes.data()));assert(dirty.load()!=0);
        phase=5;wait(6);
        assert(!c.get_range(1001,1,out.data())); // background invalidate
    });
    std::thread b([&]{
        wait(1);Cache::ReadSection q(c);std::array<char,4096> bytes{},out{};bytes.fill('A');
        assert(c.put(1000,bytes.data(),false,nullptr,&dirty));phase=2;wait(3);
        assert(!c.get_range(1000,1,out.data())); // old replica no longer usable
        assert(!c.put(1000,bytes.data(),false,nullptr,&dirty)); // dirty D is unreadable
        bytes.fill('D');assert(c.put(1002,bytes.data()));phase=4;wait(5);
        assert(writes.tasks.size()==1 && writes.tasks[0].first==1000 && writes.tasks[0].second=='B');
        // No shared prefetch: this worker cannot use the other worker's 1001.
        assert(!c.contains_range(1001,1));c.erase(1001);phase=6;
    });
    a.join();b.join();
    // Background thread has no query slot: invalidate all replicas and directly
    // hand a dirty maintenance snapshot to the writer, not to a query cache.
    std::array<char,4096> bytes{};bytes.fill('E');std::atomic<uint64_t> next_dirty{0};
    assert(c.put(1003,bytes.data(),true,&next_dirty));assert(next_dirty.load()!=0);
    assert(writes.tasks.size()==2 && writes.tasks.back().second=='E');

    setenv("NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY","1",1);
    {Cache reset;assert(reset.init(1,1));reset.configure_private_dynamic(1000,16);Writes w;reset.set_evict_hook(Writes::submit,&w);
     std::atomic<uint64_t> d{0};
     {Cache::ReadSection q(reset);assert(reset.put(1000,bytes.data(),false,&d));assert(d.load()!=0);}
     assert(w.tasks.empty());
     {Cache::ReadSection q(reset);assert(w.tasks.size()==1);assert(!reset.contains_range(1000,1));}
    }
    unsetenv("NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY");
    // Back-shift hash deletion, exact LRU and preallocated slot reuse under churn.
    setenv("NEIGHBORCACHE_FREEZE_MAINTENANCE","1",1);
    Cache stress;assert(stress.init(7,1));Cache::ReadSection q(stress);
    std::list<uint32_t> lru;std::unordered_map<uint32_t,std::list<uint32_t>::iterator> ref;
    std::mt19937 random(137);std::array<char,4096> out{};
    for(unsigned n=0;n<100000;n++) {
        uint32_t key=(random()%31)*1024;const bool present=ref.count(key);const unsigned op=random()%3;
        if(op==0){assert(stress.get_range(key,1,out.data())==present);if(present){assert(out[0]==char(key/1024));lru.splice(lru.end(),lru,ref[key]);}}
        else if(op==1){stress.erase(key);if(present){lru.erase(ref[key]);ref.erase(key);}}
        else {bytes.fill(char(key/1024));assert(stress.put(key,bytes.data()));if(present)lru.splice(lru.end(),lru,ref[key]);else{if(ref.size()==7){ref.erase(lru.front());lru.pop_front();}lru.push_back(key);ref[key]=std::prev(lru.end());}}
        for(unsigned k=0;k<31;k++)assert(stress.contains_range(k*1024,1)==bool(ref.count(k*1024)));
    }
}
