#include "nc_client.h"
#include <atomic>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unordered_set>

int main(int argc, char **) {
    const bool two = argc > 1;
    namespace fs=std::filesystem;
    const auto root=fs::temp_directory_path()/("nc_expand_concurrent_"+std::to_string(getpid()));
    fs::create_directory(root);
    const auto prefix=(root/"prefill").string(),runtime=(root/"runtime").string();
    auto record=[](uint32_t id) {
        std::array<char,388> r{};
        for (size_t j=0;j<128;++j) r[j]=static_cast<char>((id*17+j)%251);
        uint32_t degree=1,neighbor=(id+1)%2048;
        memcpy(r.data()+128,&degree,4);memcpy(r.data()+132,&neighbor,4);
        return r;
    };
    {
        std::ofstream slots(prefix+".slots",std::ios::binary),pages(prefix+".pages",std::ios::binary);
        uint64_t header[]{32768,388,128,64,10,uint64_t(two ? 512 : 256)};
        slots.write(two ? "NCHOT2P\0" : "NCHOT20\0",8);slots.write(reinterpret_cast<char*>(header),sizeof(header));
        for(uint32_t p=0;p<256;++p) {
            const uint32_t seed=1+(two ? 20 : 3)*p;
            for(uint32_t part=0;part<(two ? 2u : 1u);++part) {
                uint32_t ids[10];std::fill(ids,ids+10,seed);
                if(two)for(uint32_t j=1;j<10;++j)ids[j]=seed+part*9+j;
                else {ids[1]=seed+1;ids[2]=seed+2;}
                slots.write(reinterpret_cast<char*>(ids),sizeof(ids));
                std::array<char,4096> page{};
                for(size_t j=0;j<10;++j) memcpy(page.data()+j*388,record(ids[j]).data(),388);
                pages.write(page.data(),page.size());
            }
        }
    }
    setenv("NEIGHBORCACHE_HOT_PREFILL_PREFIX",prefix.c_str(),1);
    setenv("NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC","1",1);
    setenv("NEIGHBORCACHE_FREEZE_MAINTENANCE","1",1);
    setenv("NEIGHBORCACHE_DCACHE_SECOND_PAGE",two ? "1" : "0",1);
    setenv("NEIGHBORCACHE_DCACHE_MAX_PAGES",two ? "2" : "1",1);
    setenv("NEIGHBORCACHE_DCACHE_CAP",two ? "512" : "256",1);
    setenv("NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES",two ? "512" : "256",1);
    setenv("NEIGHBORCACHE_EXPAND_PREFETCH_PAGE","1",1);
    {
        neighborcache::Client c;
        // Working set exceeds resident page capacity; concurrent readers evict pages.
        NCConfig cfg{NC_BACKEND_ABI,4,32768,388,128,64,10,128,128,4,runtime.c_str()};
        c.init(cfg);
        c.set_verifier([&](uint32_t id,const char *raw){assert(memcmp(raw,record(id).data(),388)==0);});
        std::atomic<uint64_t> extras_seen{0};
        std::vector<std::thread> threads;
        for(uint32_t t=0;t<4;++t) threads.emplace_back([&,t] {
            for(uint32_t i=0;i<600;++i) {
                neighborcache::Client::QueryGuard query(c);
                uint32_t seed=1+(two ? 20 : 3)*((i*37+t*19)%256);
                for(uint32_t target:{seed+1,seed+(two ? 18u : 2u)}) {
                    char raw[388]{};NCRead request{};
                    int kind=c.resolve(target,raw,request);assert(kind==2 || kind==3);
                    if(kind==3) {
                        assert(request.bytes==(two ? 8192u : 4096u));
                        assert(c.complete(request,pread(request.fd,request.buffer,request.bytes,request.offset),raw));
                    }
                    std::vector<std::pair<uint32_t,std::vector<char>>> extras;
                    c.append_prefetch_members(target,extras);
                    if(kind==3) assert(extras.empty());
                    for(const auto &x:extras) {
                        assert(x.first!=target && x.first>=seed && x.first<=seed+(two ? 18u : 2u));
                        assert(memcmp(x.second.data(),record(x.first).data(),388)==0);
                    }
                    assert(two ? (extras.size()==0 || extras.size()==9 || extras.size()==18) : (extras.size()==0 || extras.size()==2));
                    std::unordered_set<uint32_t> unique;
                    for(const auto &x:extras)assert(unique.insert(x.first).second);
                    extras_seen+=extras.size();
                }
            }
        });
        for(auto &t:threads)t.join();
        assert(extras_seen>0);assert(c.stats().d_reads>128);assert(c.stats().m_entries==0);
    }
    fs::remove_all(root);
}
