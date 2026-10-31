#pragma once
#include <cstdint>
#include <cstdlib>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace diskann
{
// Called only after a full record is acquired, before M-cache decisions/locks.
// Native search and every host adapter use the same 1-in-10 heat policy.
inline bool neighborcache_sample_node_heat()
{
    thread_local std::mt19937 rng{[]() -> uint32_t {
        const char *seed = std::getenv("NEIGHBORCACHE_HEAT_SAMPLE_SEED");
        if (seed == nullptr)
            return std::random_device{}();
        uint32_t worker = 0;
#ifdef _OPENMP
        worker = static_cast<uint32_t>(omp_get_thread_num());
#endif
        return static_cast<uint32_t>(std::strtoul(seed, nullptr, 10)) + 0x9e3779b9u * (worker + 1);
    }()};
    return rng() % 10u == 0u;
}
}
