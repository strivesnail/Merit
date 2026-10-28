#include "full_seed_prefetch.h"
#include <cassert>
#include <cstdlib>
static std::array<char,4096> page(uint32_t seed,uint32_t neighbor,uint32_t marker){
 std::array<char,4096> p{};uint32_t n=2;
 memcpy(p.data(),&seed,4);memcpy(p.data()+388,&neighbor,4);memcpy(p.data()+392,&marker,4);
 memcpy(p.data()+4052,&n,4);memcpy(p.data()+4056,&seed,4);memcpy(p.data()+4060,&neighbor,4);return p;
}
int main(int argc,char**argv){
 omp_set_dynamic(0);auto &c=FullSeedPrefetch::instance();int mode=argc>1?std::atoi(argv[1]):0;
 if(mode==0){
  c.init(4,"",2,true);
#pragma omp parallel num_threads(2)
  {
   int t=omp_get_thread_num();char out[388];auto a=page(1,100,t+1);c.put(1,a.data());
#pragma omp barrier
   assert(c.lookup(100,out));uint32_t marker;memcpy(&marker,out+4,4);assert(marker==uint32_t(t+1));
   if(t==0){auto b=page(2,200,10);c.put(2,b.data());}
#pragma omp barrier
   if(t==1)assert(!c.lookup(2,out));
   if(t==0)assert(c.clear_between_queries()==2);
#pragma omp barrier
   if(t==0){assert(!c.lookup(1,out));assert(!c.lookup(100,out));}
   else{assert(c.lookup(100,out));assert(c.clear_between_queries()==1);}
#pragma omp barrier
   auto b=page(2,100,t+1),d=page(3,300,t+1);
   c.put(1,a.data());c.put(2,b.data());assert(c.lookup(1,out));c.put(3,d.data());
   assert(!c.lookup(2,out));assert(c.lookup(100,out));assert(c.lookup(300,out));
   assert(c.clear_between_queries()==2);assert(!c.lookup(100,out));assert(!c.lookup(300,out));
  }
 }else if(mode==1){
  c.init(2061,"",20,true);size_t capacities=0;
#pragma omp parallel num_threads(20) reduction(+:capacities)
  {
   int t=omp_get_thread_num();char out[388];size_t cap=c.local_capacity();assert(cap==size_t(t==0?104:103));capacities+=cap;
   for(uint32_t i=0;i<1000;i++){auto a=page(i,10000+i,t);c.put(i,a.data());assert(c.lookup(10000+i,out));uint32_t marker;memcpy(&marker,out+4,4);assert(marker==uint32_t(t));}
   assert(!c.lookup(0,out));assert(c.clear_between_queries()==cap);
  }
  assert(capacities==2061);
 }else{
  bool rejected=false;try{c.init(1,"",2,true);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 }
 std::cout<<"PRIVATE CACHE TEST PASS "<<mode<<std::endl;
}
