#include "neighborcache_heat_sample.h"
#include <cassert>
#include <thread>

int main()
{
    setenv("NEIGHBORCACHE_HEAT_SAMPLE_SEED", "20261031", 1);
    // Fresh workers reproduce the sequence, with the draw performed at every
    // full-record acquisition rather than only after taking an M-cache lock.
    auto check = [] {
        std::mt19937 expected{20261031u + 0x9e3779b9u};
        unsigned sampled = 0;
        for (unsigned i = 0; i < 10000; ++i)
        {
            const bool want = expected() % 10u == 0u;
            const bool got = diskann::neighborcache_sample_node_heat();
            assert(want == got);
            sampled += got;
        }
        assert(sampled > 800 && sampled < 1200);
    };
    std::thread a(check), b(check);
    a.join(); b.join();
}
