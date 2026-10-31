#include "neighborcache_memory_pool.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// Preflight reports a locked snapshot, not a pin. Verify failure, duplicate
// admission, victim selection and subsequent concurrent slot reuse.
template<typename T> void check_admission() {
    constexpr size_t dim=128;
    diskann::NeighborCacheMemoryPool<T> pool;
    pool.init(1,dim,2,16);
    pool.commit_initial_load({},{});
    auto make_record=[](T value, uint32_t degree) {
        std::vector<uint32_t> data((dim*sizeof(T)+12)/4);
        std::vector<T> coords(dim,value);
        std::memcpy(data.data(),coords.data(),dim*sizeof(T));
        data[dim*sizeof(T)/4]=degree;
        data[dim*sizeof(T)/4+1]=9;
        return data;
    };
    auto a=make_record(T{3},1),b=make_record(T{7},1),bad=make_record(T{1},3);
    bool present=true;
    assert(pool.inspect_admission(1,present)==pool.INVALID_NODE && !present);
    assert(pool.try_admit(1,reinterpret_cast<char*>(a.data()),dim*sizeof(T),0,0)==pool.INVALID_NODE);
    assert(pool.inspect_admission(1,present)==pool.INVALID_NODE && present);
    assert(pool.inspect_admission(2,present)==1 && !present);
    assert(pool.try_admit(1,reinterpret_cast<char*>(a.data()),dim*sizeof(T),0,0)==pool.INVALID_NODE);
    assert(pool.try_admit(2,nullptr,dim*sizeof(T),0,0)==pool.INVALID_NODE);
    assert(pool.try_admit(2,reinterpret_cast<char*>(bad.data()),dim*sizeof(T),0,0)==pool.INVALID_NODE);
    assert(pool.contains(1));
    // The earlier snapshot is not permission to dereference a cache slot later.
    std::thread writer([&] {
        assert(pool.try_admit(2,reinterpret_cast<char*>(b.data()),dim*sizeof(T),0,0)==1);
    });
    writer.join();
    assert(!pool.contains(1));
    assert(pool.inspect_admission(2,present)==pool.INVALID_NODE && present);
    std::vector<T> owned;
    std::vector<uint32_t> nbrs;
    assert(pool.copy_payload(2,owned,nbrs) && owned[0]==T{7});
}
int main() {
    setenv("NEIGHBORCACHE_NCACHE_SHARDS","1",1);
    for(const char *spin:{"0","1"}) for(const char *clock:{"0","1"}) {
        setenv("NEIGHBORCACHE_NCACHE_SPIN",spin,1);
        setenv("NEIGHBORCACHE_NCACHE_CLOCK",clock,1);
        check_admission<uint8_t>();
        check_admission<float>();
    }
}
