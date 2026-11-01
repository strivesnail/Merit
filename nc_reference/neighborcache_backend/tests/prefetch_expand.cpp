#include "nc_client.h"
#include <cassert>
#include <filesystem>
#include <fstream>

int main() {
    namespace fs=std::filesystem;
    const auto root=fs::temp_directory_path()/("nc_expand_"+std::to_string(getpid()));
    fs::create_directory(root);
    const auto prefix=(root/"prefill").string(),runtime=(root/"runtime").string();
    auto record=[](uint32_t id) {
        std::array<char,388> r{};
        memcpy(r.data(),&id,4);
        uint32_t degree=1,neighbor=id+1;
        memcpy(r.data()+128,&degree,4);memcpy(r.data()+132,&neighbor,4);
        return r;
    };
    {
        std::ofstream slots(prefix+".slots",std::ios::binary),pages(prefix+".pages",std::ios::binary);
        uint64_t header[]{2048,388,128,64,10,1};uint32_t ids[]{1,2,3,1,1,1,1,1,1,1};
        slots.write("NCHOT20\0",8);slots.write(reinterpret_cast<char*>(header),sizeof(header));
        slots.write(reinterpret_cast<char*>(ids),sizeof(ids));
        std::array<char,4096> page{};
        for(size_t j=0;j<10;++j) memcpy(page.data()+j*388,record(ids[j]).data(),388);
        pages.write(page.data(),page.size());
    }
    setenv("NEIGHBORCACHE_HOT_PREFILL_PREFIX",prefix.c_str(),1);
    setenv("NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC","1",1);
    setenv("NEIGHBORCACHE_FREEZE_MAINTENANCE","1",1);
    setenv("NEIGHBORCACHE_DCACHE_SECOND_PAGE","0",1);
    setenv("NEIGHBORCACHE_DCACHE_CAP","1",1);
    setenv("NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES","1",1);
    setenv("NEIGHBORCACHE_EXPAND_PREFETCH_PAGE","1",1);
    {
        neighborcache::Client c;
        NCConfig cfg{NC_BACKEND_ABI,4,2048,388,128,64,10,128,128,1,runtime.c_str()};
        c.init(cfg);assert(c.expands_prefetch_page());
        neighborcache::Client::QueryGuard query(c);
        char raw[388]{};NCRead request{};
        std::vector<std::pair<uint32_t,std::vector<char>>> extras;
        assert(c.resolve(2,raw,request)==3);assert(request.bytes==4096);
        assert(c.complete(request,pread(request.fd,request.buffer,request.bytes,request.offset),raw));
        c.append_prefetch_members(2,extras);assert(extras.empty()); // Fresh D IO is not a prefetch hit.
        assert(c.resolve(3,raw,request)==2);
        c.append_prefetch_members(3,extras);
        assert(extras.size()==2 && extras[0].first==1 && extras[1].first==2);
        for(const auto &x:extras) assert(memcmp(x.second.data(),record(x.first).data(),388)==0);
        c.append_prefetch_members(3,extras);assert(extras.size()==2); // Snapshot consumed once.
        assert(c.stats().d_reads==1); // Expanding the page issued no additional disk IO.
        auto r4=record(4);c.expand(4,r4.data());
        assert(c.resolve(4,raw,request)==1);
        extras.clear();c.append_prefetch_members(4,extras);assert(extras.empty()); // N is target-only.
        assert(c.resolve(100,raw,request)==0);
        c.append_prefetch_members(100,extras);assert(extras.empty());
        assert(c.stats().m_entries==0);
    }
    fs::remove_all(root);
}
