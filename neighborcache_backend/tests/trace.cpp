#include "nc_client.h"
#include <cassert>
#include <thread>
#include <atomic>
#include <iostream>
#include <filesystem>
using neighborcache::Client;
std::vector<char> record(uint32_t id, unsigned bytes, unsigned count_bytes) {
 std::vector<char> r(bytes,0);
 for(unsigned i=0;i<128;++i) r[i]=(id*17+i)%251;
 uint32_t n=32; std::memcpy(r.data()+128,&n,count_bytes);
 for(unsigned i=0;i<n;++i) {uint32_t v=(id+i+1)%2048;std::memcpy(r.data()+132+i*4,&v,4);}
 for(unsigned i=260;i<bytes;++i)r[i]=(id*19+i)%253;
 return r;
}
int main(int argc,char **argv){
 unsigned bytes=argc>1?std::stoul(argv[1]):260, count=argc>2?std::stoul(argv[2]):4;
 setenv("NEIGHBORCACHE_DCACHE_CAP","32",1);
 setenv("NEIGHBORCACHE_MCACHE_CAP","512",1);
 setenv("NEIGHBORCACHE_REAL_IO_SAMPLE_QUERIES","1",1);
 setenv("NEIGHBORCACHE_DCACHE_COMMIT_INLINE","1",1);
 std::string path=(std::filesystem::temp_directory_path() / ("nc_trace_"+std::to_string(getpid())+"_"+std::to_string(bytes)+"_"+std::to_string(count))).string();
 Client c;NCConfig cfg{NC_BACKEND_ABI,count,2048,bytes,128,32,4096/bytes,128,4,4,path.c_str()};c.init(cfg);
 std::atomic<unsigned> checked{0};
 auto work=[&](unsigned t){
  for(unsigned q=0;q<100;++q){Client::QueryGuard guard(c);
   for(unsigned k=0;k<96;++k){
    uint32_t id=(q%4*64+k+t*128)%2048;
    auto want=record(id,bytes,count), got=std::vector<char>(bytes);
    NCRead read;int kind=c.resolve(id,got.data(),read);
    if(kind==3){auto n=pread(read.fd,read.buffer,read.bytes,read.offset);if(!c.complete(read,n,got.data()))kind=0;}
    if(kind){if(want!=got){std::cerr<<"bad record id="<<id<<" kind="<<kind<<"\n";std::abort();}++checked;}
    c.expand(id,want.data(),0,k ? (id+2047)%2048 : UINT32_MAX);
   }
  }
 };
 std::vector<std::thread> threads;for(unsigned t=0;t<4;++t)threads.emplace_back(work,t);for(auto&t:threads)t.join();
 auto s=c.stats();c.print();assert(checked>0);assert(s.n_hits>0);assert(s.m_entries>0);assert(s.seeds<=s.capacity);assert(s.opens>0);assert(s.d_reads>0);assert(s.prefetch_hits>0);assert(s.writes>0);
 std::cout<<"PASS checked="<<checked<<" bytes="<<bytes<<" count="<<count<<"\n";
}
