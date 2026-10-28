#include "full_seed_prefetch.h"
#include <thread>
#include <cassert>
#include <cstdlib>
static size_t h(uint32_t x,size_t n){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x%n;}
static std::array<char,4096> page(uint32_t seed,uint32_t member){std::array<char,4096> p{};uint32_t n=2;memcpy(p.data(),&seed,4);memcpy(p.data()+388,&member,4);memcpy(p.data()+4052,&n,4);memcpy(p.data()+4056,&seed,4);memcpy(p.data()+4060,&member,4);return p;}
int main(int argc,char**argv){
 assert(FullSeedPrefetch::shards_for_threads(1)==1);assert(FullSeedPrefetch::shards_for_threads(8)==32);assert(FullSeedPrefetch::shards_for_threads(16)==64);assert(FullSeedPrefetch::shards_for_threads(20)==128);
 auto &c=FullSeedPrefetch::instance();char out[388];int mode=argc>1?std::atoi(argv[1]):0;
 if(mode==0){
  bool threw=false;try{c.init(0,"",1);}catch(const std::runtime_error&){threw=true;}assert(threw);
  threw=false;try{c.init(2,"",0);}catch(const std::runtime_error&){threw=true;}assert(threw);
  c.init(2,"",1);auto a=page(1,101),b=page(2,102),d=page(3,103);
  c.put(1,a.data());c.put(2,b.data());assert(c.lookup(101,out));c.put(3,d.data());
  assert(c.lookup(1,out));assert(c.lookup(101,out));assert(!c.lookup(2,out));assert(!c.lookup(102,out));assert(c.lookup(103,out));
 }else if(mode==1||mode==2){
  size_t pages=mode==1?23:3,threads=mode==1?3:20,n=std::min(pages,FullSeedPrefetch::shards_for_threads(threads));c.init(pages,"",threads);
  std::vector<uint32_t> residents;uint32_t next=1;
  for(size_t shard=0;shard<n;shard++){
   size_t cap=pages/n+(shard<pages%n);std::vector<uint32_t> ids;
   for(size_t i=0;i<cap+1;i++){while(h(next,n)!=shard)next++;ids.push_back(next++);auto p=page(ids.back(),ids.back()+1000000);c.put(ids.back(),p.data());}
   assert(!c.lookup(ids.front(),out));for(size_t i=1;i<ids.size();i++)residents.push_back(ids[i]);
  }
  assert(residents.size()==pages);for(auto id:residents){assert(c.lookup(id,out));uint32_t got;memcpy(&got,out,4);assert(got==id);}
 }else{
  c.init(137,"",20);std::vector<std::thread> threads;
  for(uint32_t t=0;t<20;t++)threads.emplace_back([t,&c](){char out[388];for(uint32_t i=0;i<2000;i++){uint32_t id=1+t*2000+i;auto p=page(id,id+1000000);c.put(id,p.data());for(auto n:{id,id+1000000})if(c.lookup(n,out)){uint32_t got;memcpy(&got,out,4);assert(got==n);}}});
  for(auto &t:threads)t.join();
 }
 std::cout<<"THREAD SHARD TEST PASS "<<mode<<std::endl;
}
