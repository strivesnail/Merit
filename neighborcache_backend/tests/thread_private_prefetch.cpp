#include "neighborcache_shared_page_cache.h"
#include <cassert>
#include <thread>
#include <array>
#include <new>
using Cache=diskann::NeighborCacheSharedPageCache;
int main(){
 setenv("NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE","1",1);unsetenv("NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY");
 unsetenv("NEIGHBORCACHE_FREEZE_MAINTENANCE");
 {Cache c;bool rejected=false;try{c.init(4,2);}catch(const std::runtime_error&){rejected=true;}assert(rejected);}
 setenv("NEIGHBORCACHE_FREEZE_MAINTENANCE","1",1);
 {
  Cache c;assert(c.init(4,2));assert(c.thread_private()&&c.capacity_pages()==4);
  std::atomic<unsigned> ready{0};std::array<std::thread,2> workers;
  for(unsigned t=0;t<2;t++)workers[t]=std::thread([&,t]{
   std::array<char,4096> a{},b{},out{};a.fill(char(t+1));b.fill(char(t+10));
   {Cache::ReadSection read(c);assert(c.put(1,a.data()));assert(c.put(10+t,b.data()));ready.fetch_add(1);while(ready.load()!=2)std::this_thread::yield();
    assert(!c.contains_range(10+(1-t),1));assert(c.get_range(1,1,out.data()));assert(out==a);
    const char *held=c.get_ptr(1);assert(held);assert(c.get_range(10+t,1,out.data()));assert(c.put(100,b.data()));
    assert(!c.contains_range(1,1));assert(std::memcmp(held,a.data(),4096)==0); // pinned through eviction
    assert(c.get_range(100,1,out.data()));c.erase(100);assert(!c.get_range(100,1,out.data()));
    assert(c.put(200,a.data()));bool rejected=false;try{c.put(200,b.data(),true);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
    std::atomic<uint64_t> dirty{0};rejected=false;try{c.put(300,a.data(),false,&dirty);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
   }
   {Cache::ReadSection next(c);assert(c.get_range(200,1,out.data()));assert(out==a);}
  });
  for(auto &w:workers)w.join();assert(c.evictions()==2);assert(c.inserts()==8);
 }
 {
  setenv("NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY","1",1);Cache c;assert(c.init(1,1));std::array<char,4096> a{};
  {Cache::ReadSection q(c);assert(c.put(1,a.data()));{Cache::ReadSection nested(c);assert(c.contains_range(1,1));}}
  {Cache::ReadSection q(c);assert(!c.contains_range(1,1));}
  unsetenv("NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY");
 }
 {
  alignas(Cache) char storage[sizeof(Cache)];std::array<char,4096> a{};
  auto *c=new(storage) Cache;assert(c->init(1,1));assert(c->put(1,a.data()));c->~Cache();
  c=new(storage) Cache;assert(c->init(1,1));assert(!c->contains_range(1,1));assert(c->put(2,a.data()));c->~Cache();
 }
 {
  Cache c;bool rejected=false;try{c.init(1,2);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 }
}
