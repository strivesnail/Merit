#include "nc_client.h"
#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("nc_groups_" + std::to_string(getpid()));
    fs::create_directory(root);
    const auto prefix = (root / "prefill").string();
    const auto runtime = (root / "runtime").string();
    auto record = [](uint32_t id) {
        std::array<char, 388> r{};
        std::memcpy(r.data(), &id, 4);
        const uint32_t degree = 1, neighbor = id + 1;
        std::memcpy(r.data() + 128, &degree, 4);
        std::memcpy(r.data() + 132, &neighbor, 4);
        return r;
    };
    {
        std::ofstream slots(prefix + ".slots", std::ios::binary), pages(prefix + ".pages", std::ios::binary);
        const uint64_t header[]{2048, 388, 128, 64, 10, 3};
        slots.write("NCHOT2P\0", 8);
        slots.write(reinterpret_cast<const char *>(header), sizeof(header));
        const uint32_t ids[3][10]{{1,2,3,4,5,6,7,8,9,10}, {1,11,12,1,1,1,1,1,1,1}, {30,31,30,30,30,30,30,30,30,30}};
        for (const auto &row : ids) {
            std::array<char,4096> page{};
            for (size_t j=0; j<10; ++j) {
                const auto raw=record(row[j]);
                std::memcpy(page.data()+388*j,raw.data(),388);
            }
            slots.write(reinterpret_cast<const char *>(row), sizeof(row));
            pages.write(page.data(),page.size());
        }
    }
    setenv("NEIGHBORCACHE_HOT_PREFILL_PREFIX", prefix.c_str(), 1);
    setenv("NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC", "1", 1);
    setenv("NEIGHBORCACHE_FREEZE_MAINTENANCE", "1", 1);
    setenv("NEIGHBORCACHE_DCACHE_SECOND_PAGE", "1", 1);
    setenv("NEIGHBORCACHE_DCACHE_MAX_PAGES", "2", 1);
    setenv("NEIGHBORCACHE_DCACHE_CAP", "3", 1);
    setenv("NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES", "3", 1);
    {
        neighborcache::Client c;
        NCConfig cfg{NC_BACKEND_ABI,4,2048,388,128,64,10,128,128,1,runtime.c_str()};
        c.init(cfg);
        {
            neighborcache::Client::QueryGuard query(c);
            char raw[388]{}; NCRead request{};
            assert(c.resolve(11,raw,request,1)==3);
            assert(request.bytes==8192 && request.offset==0);
            const auto got=pread(request.fd,request.buffer,request.bytes,request.offset);
            assert(c.complete(request,got,raw)==1);
            assert(std::memcmp(raw,record(11).data(),388)==0);
            // Both pages were installed into prefetch, including the first page.
            assert(c.resolve(2,raw,request,1)==2);
            assert(std::memcmp(raw,record(2).data(),388)==0);
            assert(c.resolve(12,raw,request,1)==2);
            assert(std::memcmp(raw,record(12).data(),388)==0);
        }
        {
            neighborcache::Client::QueryGuard query(c);
            char raw[388]{}; NCRead request{};
            assert(c.resolve(31,raw,request,30)==3);
            assert(request.bytes==4096 && request.offset==8192);
            const auto got=pread(request.fd,request.buffer,request.bytes,request.offset);
            assert(c.complete(request,got,raw)==1);
            assert(std::memcmp(raw,record(31).data(),388)==0);
        }
        assert(c.stats().m_entries==0);
    }
    fs::remove_all(root);
}
