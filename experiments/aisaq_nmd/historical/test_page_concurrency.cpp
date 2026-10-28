#ifndef NMD_HEADER
#define NMD_HEADER "nmd_cache.h"
#endif
#include NMD_HEADER
#include <thread>
#include <array>
#include <stdexcept>
// Deterministic immutable node records; exercise shared seeds, eviction and lookup.
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  const unsigned threads = std::stoul(argv[1]);
  constexpr unsigned nodes=5000, len=772, vec=128, degree=32;
  setenv("AISAQ_NMD","1",1);setenv("AISAQ_NMD_NMB","1",1);
  setenv("AISAQ_NMD_PMB","1",1);setenv("AISAQ_NMD_PREFILL","0",1);
  setenv("AISAQ_NMD_PATH",argv[2],1);
  std::vector<std::array<char,len>> records(nodes);
  std::vector<std::array<uint32_t,degree>> neighbors(nodes);
  for (unsigned id=0;id<nodes;++id) {
    for (unsigned j=0;j<len;++j) records[id][j]=char((id*7+j)%251);
    const uint32_t n=degree;std::memcpy(records[id].data()+vec,&n,4);
    for (unsigned j=0;j<degree;++j) neighbors[id][j]=(id+j+1)%nodes;
    std::memcpy(records[id].data()+vec+4,neighbors[id].data(),degree*4);
  }
  aisaq_nmd::NMDCache cache;
  cache.init(nodes,len,5,vec,argv[2],"/dev/null",0);
  std::atomic<unsigned> hits{0};
  std::vector<std::thread> workers;
  for (unsigned t=0;t<threads;++t) workers.emplace_back([&,t] {
    std::array<char,len> got;
    for (unsigned i=0;i<12000;++i) {
      // Repeat each seed to admit/rebuild; mix shared hot and broad working sets.
      unsigned id=(i/3*37+t*11)%nodes;
      if (i%5==0) id=(i/5)%64;
      cache.on_expand(id,records[id].data(),neighbors[id].data(),degree);
      int kind=0;
      if (cache.lookup(id,got.data(),kind)) {
        if (std::memcmp(got.data(),records[id].data(),len)) std::abort();
        ++hits;
      }
      unsigned neighbor=neighbors[id][i%degree];
      if (cache.lookup(neighbor,got.data(),kind)) {
        if (std::memcmp(got.data(),records[neighbor].data(),len)) std::abort();
        ++hits;
      }
    }
  });
  for (auto &w:workers) w.join();
  if (!hits) return 3;
  cache.print();
  printf("record integrity PASS threads=%u hits=%u\n",threads,hits.load());
}
