#include "neighborcache_heat_sample.h"
#include <cassert>
#include <cstdlib>
#include <thread>

int main()
{
    setenv("NEIGHBORCACHE_HEAT_SAMPLE_SEED", "20261031", 1);
    // No acquired node may be dropped by the heat gate, including the first
    // acquisition on a fresh worker. The old RNG seed must have no effect.
    auto check = [] {
        for (unsigned i = 0; i < 10000; ++i)
            assert(diskann::neighborcache_sample_node_heat());
    };
    std::thread a(check), b(check);
    a.join(); b.join();
    unsetenv("NEIGHBORCACHE_HEAT_SAMPLE_SEED");
    check();
}
