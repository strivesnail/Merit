#include "nc_client.h"
#include <cassert>
#include <filesystem>

int main(int argc, char **) {
    const bool frozen = argc > 1;
    setenv("NEIGHBORCACHE_FREEZE_MAINTENANCE", frozen ? "1" : "0", 1);
    setenv("NEIGHBORCACHE_DCACHE_CAP", "32", 1);
    setenv("NEIGHBORCACHE_MCACHE_CAP", "128", 1);
    std::string path = (std::filesystem::temp_directory_path() /
        ("nc_acquired_parent_" + std::to_string(getpid()))).string();
    neighborcache::Client c;
    NCConfig cfg{NC_BACKEND_ABI,4,2048,260,128,32,15,128,4,1,path.c_str()};
    c.init(cfg);
    {
        neighborcache::Client::QueryGuard query(c);
        // Source metadata and a miss are not successful record acquisition.
        for (uint32_t child=1; child<1000; ++child) c.prepare_parent(0,child);
        char raw[260]{}; NCRead request{};
        assert(c.resolve(1001,raw,request,7)==0);
    }
    assert(c.stats().recorded_parents==0);
    assert(c.stats().acquired_records==0);
    char record[260]{};
    uint32_t degree=1, neighbor=10;
    memcpy(record+128,&degree,4); memcpy(record+132,&neighbor,4);
    {
        neighborcache::Client::QueryGuard query(c);
        c.expand(9,record,0,7);
        c.expand(9,record,0,8); // The first actual acquisition owns the parent.
    }
    assert(c.stats().recorded_parents==(frozen ? 0 : 1));
    assert(c.stats().acquired_records==2);
    {
        neighborcache::Client::QueryGuard query(c);
        char raw[260]{}; NCRead request{};
        assert(c.resolve(9,raw,request,6)==1);
        c.expand(9,raw,0,6); // A new query starts a fresh acquired-parent map.
    }
    assert(c.stats().recorded_parents==(frozen ? 0 : 2));
    assert(c.stats().acquired_records==3);
    if (frozen) assert(c.stats().m_entries==0);
}
