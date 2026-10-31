#pragma once
#include "neighborcache_thread_page_cache.h"
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

// Same-workload oracle experiment. Uses the production private page cache,
// compact record directory, zero-tail representation and LRU without M/updates.
class FullVisitedCache {
    diskann::NeighborCacheThreadPageCache cache;
    std::vector<uint32_t> seeds;
    bool on=false;
    static uint32_t u32(const char *p) {uint32_t n;std::memcpy(&n,p,4);return n;}
  public:
    static FullVisitedCache &instance() {static FullVisitedCache c;return c;}
    bool active() const {return on;}
    uint64_t page_count() const {return seeds.size();}
    uint64_t page_for(uint32_t id) const {
        auto it=std::lower_bound(seeds.begin(),seeds.end(),id);
        if(it==seeds.end() || *it!=id) throw std::runtime_error("Full-vector target absent from profiled visited set");
        return it-seeds.begin();
    }
    void init(const char *path,uint64_t pages,uint64_t threads) {
        if(on || threads!=1 || !pages) throw std::runtime_error("Fullvisited experiment requires one query worker");
        std::ifstream in(path,std::ios::binary|std::ios::ate);
        const auto bytes=in.tellg();
        if(!in || bytes<=0 || bytes%4)throw std::runtime_error("Invalid fullvisited IDs");
        seeds.resize(bytes/4);in.seekg(0);in.read(reinterpret_cast<char*>(seeds.data()),bytes);
        if(!in || seeds.back()>=100000000 || !std::is_sorted(seeds.begin(),seeds.end()) ||
           std::adjacent_find(seeds.begin(),seeds.end())!=seeds.end())throw std::runtime_error("Seed IDs must be sorted unique");
        if(!cache.init(pages,1,true,false))throw std::runtime_error("Cannot initialize fullvisited prefetch");
        cache.enter();on=true;
        std::cout<<"FULLVISITED seeds="<<seeds.size()<<" D_bytes="<<seeds.size()*4096ULL
                 <<" P_pages="<<pages<<" private_LRUs=1 M=0 writes=0 sorted_ID_bytes="<<seeds.size()*4ULL<<std::endl;
    }
    bool lookup(uint32_t id,char *record) {return cache.copy_record(id,388,record,false);}
    void put(uint32_t seed,const char *page) {
        const uint32_t count=u32(page+4052);
        if(!count || count>10 || u32(page+4056)!=seed)throw std::runtime_error("Invalid immutable seed page footer");
        uint32_t ids[10];std::memcpy(ids,page+4056,count*4);
        for(uint32_t i=0;i<count;++i)if(ids[i]>=100000000)throw std::runtime_error("Invalid member ID");
        const auto key=static_cast<uint32_t>(page_for(seed));
        if(!cache.put(key,page))throw std::runtime_error("Immutable prefetch admission failed");
        cache.register_records(key,ids,count);
    }
};
