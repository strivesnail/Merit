#include "nc_client.h"
#include <cassert>
#include <thread>
#include <atomic>
#include <iostream>
#include <filesystem>
using neighborcache::Client;
std::vector<char> record(uint32_t id, unsigned bytes, unsigned count_bytes, unsigned vector_bytes) {
 std::vector<char> r(bytes,0);
 for(unsigned i=0;i<vector_bytes;++i) r[i]=(id*17+i)%251;
 uint32_t n=32; std::memcpy(r.data()+vector_bytes,&n,count_bytes);
 for(unsigned i=0;i<n;++i) {uint32_t v=(id+i+1)%2048;std::memcpy(r.data()+vector_bytes+4+i*4,&v,4);}
 for(unsigned i=vector_bytes+132;i<bytes;++i)r[i]=(id*19+i)%253;
 return r;
}
int main(int argc,char **argv){
 unsigned bytes=argc>1?std::stoul(argv[1]):260, count=argc>2?std::stoul(argv[2]):4;
 unsigned vector_bytes=argc>3?std::stoul(argv[3]):128;
 assert(bytes>=vector_bytes+132);
 setenv("NEIGHBORCACHE_DCACHE_CAP","32",1);
 setenv("NEIGHBORCACHE_MCACHE_CAP","512",1);
 setenv("NEIGHBORCACHE_REAL_IO_SAMPLE_QUERIES","1",1);
 setenv("NEIGHBORCACHE_DCACHE_COMMIT_INLINE","1",1);
 std::string path=(std::filesystem::temp_directory_path() / ("nc_trace_"+std::to_string(getpid())+"_"+std::to_string(bytes)+"_"+std::to_string(count))).string();
 const bool n_only = neighborcache::env("NEIGHBORCACHE_NCACHE_ONLY", 0) != 0;
 if (n_only) setenv("NEIGHBORCACHE_NCACHE_SHARDS", "4", 1);
 Client c;NCConfig cfg{NC_BACKEND_ABI,count,2048,bytes,vector_bytes,32,4096/bytes,128,n_only ? 0u : 4u,4,path.c_str()};c.init(cfg);
 std::atomic<unsigned> checked{0},extras_checked{0};
 auto work=[&](unsigned t){
  for(unsigned q=0;q<100;++q){Client::QueryGuard guard(c);
   for(unsigned k=0;k<96;++k){
    uint32_t id=(q%4*64+k+t*128)%2048;
    auto want=record(id,bytes,count,vector_bytes), got=std::vector<char>(bytes);
    NCRead read;int kind=c.resolve(id,got.data(),read);
    if(kind==3){auto n=pread(read.fd,read.buffer,read.bytes,read.offset);if(!c.complete(read,n,got.data()))kind=0;}
    if(kind){if(want!=got){std::cerr<<"bad record id="<<id<<" kind="<<kind<<"\n";std::abort();}++checked;}
    std::vector<std::pair<uint32_t,std::vector<char>>> extras;
    c.append_prefetch_members(id,extras);
    if(kind!=2) assert(extras.empty());
    for(const auto &extra:extras) {
     assert(extra.first!=id && extra.second==record(extra.first,bytes,count,vector_bytes));
     ++extras_checked;
    }
    c.expand(id,want.data(),0,k ? (id+2047)%2048 : UINT32_MAX);
   }
  }
 };
 std::vector<std::thread> threads;for(unsigned t=0;t<4;++t)threads.emplace_back(work,t);for(auto&t:threads)t.join();
 auto s=c.stats();c.print();assert(checked>0);assert(s.n_hits>0);
 if (n_only) {
  assert(s.n_hits == checked && s.n_hits + s.base_misses == 4u*100u*96u);
  assert(s.m_entries == 0 && s.seeds == 0 && s.opens == 0 && s.d_attempts == 0);
  assert(s.d_reads == 0 && s.prefetch_hits == 0 && s.writes == 0);
  assert(!std::filesystem::exists(path + "_neighborcache_dc.dyn.data"));
 } else {
  assert(s.m_entries>0);assert(s.seeds<=s.capacity);assert(s.opens>0);assert(s.d_reads>0);assert(s.prefetch_hits>0);assert(s.writes>0);
 }
 std::cout<<"PASS checked="<<checked<<" bytes="<<bytes<<" count="<<count<<"\n";
 if(c.expands_prefetch_page()) assert(extras_checked>0);
}
