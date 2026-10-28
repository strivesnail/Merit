#include <bits/stdc++.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#define private public
#include NMD_HEADER
#undef private
#ifdef TEST_PIPEANN
using Cache=pipeann::NMDCache;
#define PREFIX "PIPEANN_NMD"
#else
using Cache=aisaq_nmd::NMDCache;
#define PREFIX "AISAQ_NMD"
#endif
void require(bool b) { if(!b) throw std::runtime_error("LRU regression"); }
int main() {
 setenv(PREFIX,"1",1);setenv(PREFIX "_NMB","1",1);setenv(PREFIX "_PMB","1",1);
 setenv(PREFIX "_PREFILL","0",1);setenv(PREFIX "_PATH","/tmp/nmd-lru-test.bin",1);
 Cache c;c.init(10000,772,5,128,"/tmp/nmd-lru-test.bin","/dev/null",0);
 std::array<char,772> bytes{},got{};
 auto &sh=c.page_shards_[0];const size_t cap=sh.free.size();require(cap>=3);
 auto put=[&](uint32_t id){c.rebuild(id,bytes.data(),nullptr,0);};
 for(size_t i=0;i<cap;++i) put(i*64);
 require(sh.seeds.size()==cap);
 auto first=sh.seeds.at(0);auto second=sh.seeds.at(64);
 c.pages_[second].seed_heat=UINT32_MAX; // heat must not protect an LRU victim
 Cache::Loc loc;loc.page=first;loc.slot=0;
 require(c.page_copy(loc,0,got.data())); // oldest becomes MRU
 put(cap*64);require(sh.seeds.count(0)==1 && sh.seeds.count(64)==0);
 loc.page=sh.seeds.at(128);require(c.page_copy(loc,128,got.data())); // read promotes; existing seed creation is a no-op

 put((cap+1)*64);require(sh.seeds.count(128)==1 && sh.seeds.count(192)==0);
 const auto tail=sh.tail;put(1);require(sh.tail==tail); // other shard is independent
 // Every live page appears exactly once in the LRU list.
 for(size_t si=0;si<c.page_shards_count_;++si){auto &s=c.page_shards_[si];
  std::set<uint32_t> seen;uint32_t prev=UINT32_MAX;
  for(auto pi=s.head;pi!=UINT32_MAX;pi=c.pages_[pi].next){
   require(seen.insert(pi).second);require(c.pages_[pi].prev==prev);prev=pi;
  }
  require(prev==s.tail && seen.size()==s.seeds.size());
 }
 unlink("/tmp/nmd-lru-test.bin");puts("LRU read promotion, heat independence, shard isolation, list integrity: PASS");
}
