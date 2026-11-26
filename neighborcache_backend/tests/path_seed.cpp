#include "nc_client.h"
#include <array>
#include <cassert>
#include <filesystem>

using Record = std::array<char, 260>;

static Record payload(uint32_t id) {
    Record raw;
    raw.fill(static_cast<char>(id));
    memcpy(raw.data(), &id, sizeof(id));
    uint16_t count=1, dense_count=2;
    memcpy(raw.data()+128,&count,2);
    memcpy(raw.data()+130,&dense_count,2);
    return raw;
}

int main() {
    const std::string prefix = (std::filesystem::temp_directory_path() /
                                ("path-seed-"+std::to_string(getpid()))).string();
    setenv("MERIT_PATH_SEED","shared",1);
    setenv("NEIGHBORCACHE_NCACHE_ADMISSION","mru",1);
    NCConfig cfg{NC_BACKEND_ABI,2,20000,260,128,32,15,256,64,1,prefix.c_str()};
    {
        neighborcache::Client cache;
        cache.init(cfg);
        auto expand=[&](uint32_t id,uint32_t parent) {
            auto raw=payload(id);
            cache.expand(id,raw.data(),0,parent);
        };
        auto group=[&](uint32_t first) {
            neighborcache::Client::QueryGuard q(cache);
            expand(first,UINT32_MAX);
            expand(first+1,first);
            expand(first+2,first+1);
            for(uint32_t i=3;i<6;++i) expand(first+i,first+2);
            for(uint32_t i=6;i<9;++i) expand(first+i,first+i-3);
        };
        group(0);
        assert(cache.stats().seeds==1);
        Record out{};
        NCRead primary{}, other{};
        {
            neighborcache::Client::QueryGuard q(cache);
            assert(cache.resolve(3,out.data(),primary,2)==1 && out==payload(3));
        }
        // A different query extends seed 3 inside the shared page, then spills.
        {
            neighborcache::Client::QueryGuard q(cache);
            expand(0,UINT32_MAX); expand(1,0); expand(2,1); expand(3,2);
            for(uint32_t child : {109u,119u,129u,139u,149u}) expand(child,3);
        }
        assert(cache.stats().seeds==2);
        // Replace both the original N records and its prefetch image.
        for(uint32_t i=1;i<=1024;++i) group(i*10);
        {
            neighborcache::Client::QueryGuard q(cache);
            assert(cache.resolve(6,out.data(),primary,3)==3);
            assert(cache.resolve(8,out.data(),other,5)==3);
            assert(cache.share_path_read(primary,other));
            assert(cache.stats().d_reads==1 && primary.buffer==other.buffer);
            auto bytes=pread(primary.fd,primary.buffer,primary.bytes,primary.offset);
            assert(cache.complete(primary,bytes,out.data()) && out==payload(6));
            assert(cache.complete(other,bytes,out.data()) && out==payload(8));
            assert(cache.resolve(7,out.data(),other,4)==2 && out==payload(7));
            assert(cache.resolve(109,out.data(),other,3)==2 && out==payload(109));
            assert(cache.resolve(149,out.data(),other,3)==3);
            bytes=pread(other.fd,other.buffer,other.bytes,other.offset);
            assert(cache.complete(other,bytes,out.data()) && out==payload(149));
        }
    }
    std::filesystem::remove(prefix+"_pathseed.data");
}
