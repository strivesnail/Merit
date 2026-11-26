#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

namespace diskann
{

struct NeighborCacheSpinLock
{
    void lock()
    {
        uint32_t spins = 0;
        while (_flag.test_and_set(std::memory_order_acquire))
            pause(spins);
    }
    void unlock()
    {
        _flag.clear(std::memory_order_release);
    }
    std::atomic_flag _flag = ATOMIC_FLAG_INIT;

    static void pause(uint32_t &spins)
    {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
        if (++spins == 128)
        {
            std::this_thread::yield();
            spins = 0;
        }
    }
};

} // namespace diskann
