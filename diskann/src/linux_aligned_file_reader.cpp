// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <mutex>
#include <vector>
#include "tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 128

namespace
{
typedef struct io_event io_event_t;
typedef struct iocb iocb_t;

enum MeritLatClass : int
{
    kLatReq4k = 0,
    kLatReq8k,
    kLatReqOther,
    kLatBatchNo8k,
    kLatBatchWith8k,
    kLatClassCount
};
constexpr int kLatBins = 4096;
struct MeritLatAcc
{
    uint64_t bins[kLatClassCount][kLatBins] = {};
    uint64_t sum_ns[kLatClassCount] = {};
};
bool merit_lat_on()
{
    static const bool on = [] {
        const char *v = std::getenv("MERIT_IO_LAT_PROFILE");
        return v != nullptr && std::strcmp(v, "0") != 0;
    }();
    return on;
}
std::mutex &merit_lat_mu()
{
    static std::mutex mu;
    return mu;
}
std::vector<MeritLatAcc *> &merit_lat_registry()
{
    static std::vector<MeritLatAcc *> reg;
    return reg;
}
MeritLatAcc &merit_lat_acc()
{
    thread_local MeritLatAcc *acc = [] {
        auto *a = new MeritLatAcc();
        std::lock_guard<std::mutex> lock(merit_lat_mu());
        merit_lat_registry().push_back(a);
        return a;
    }();
    return *acc;
}
uint64_t merit_lat_now_ns()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}
void merit_lat_record(int cls, uint64_t ns)
{
    MeritLatAcc &acc = merit_lat_acc();
    uint64_t us = ns / 1000;
    if (us >= static_cast<uint64_t>(kLatBins))
        us = kLatBins - 1;
    acc.bins[cls][us]++;
    acc.sum_ns[cls] += ns;
}

// Reaps completions one at a time so each request gets its own completion timestamp.
bool merit_lat_wait(io_context_t ctx, struct iocb *cb, uint64_t n_ops, std::vector<io_event_t> &evts, uint64_t t0)
{
    uint64_t done = 0;
    uint64_t t_last = t0;
    bool has8k = false;
    while (done < n_ops)
    {
        const int64_t ret = io_getevents(ctx, 1, (int64_t)(n_ops - done), evts.data(), nullptr);
        if (ret <= 0)
            return false;
        const uint64_t t = merit_lat_now_ns();
        for (int64_t e = 0; e < ret; ++e)
        {
            const struct iocb *obj = evts[e].obj;
            const size_t len = obj->u.c.nbytes;
            const int cls = len == 4096 ? kLatReq4k : (len == 8192 ? kLatReq8k : kLatReqOther);
            if (len > 4096)
                has8k = true;
            merit_lat_record(cls, t - t0);
        }
        t_last = t;
        done += static_cast<uint64_t>(ret);
    }
    (void)cb;
    merit_lat_record(has8k ? kLatBatchWith8k : kLatBatchNo8k, t_last - t0);
    return true;
}

void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs, uint64_t n_retries = 0)
{
#ifdef DEBUG
    for (auto &req : read_reqs)
    {
        assert(IS_ALIGNED(req.len, 512));
        // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
        assert(IS_ALIGNED(req.offset, 512));
        assert(IS_ALIGNED(req.buf, 512));
        // assert(malloc_usable_size(req.buf) >= req.len);
    }
#endif

    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++)
    {
        uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * MAX_EVENTS), (uint64_t)MAX_EVENTS);
        std::vector<iocb_t *> cbs(n_ops, nullptr);
        std::vector<io_event_t> evts(n_ops);
        std::vector<struct iocb> cb(n_ops);
        for (uint64_t j = 0; j < n_ops; j++)
        {
            io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf, read_reqs[j + iter * MAX_EVENTS].len,
                          read_reqs[j + iter * MAX_EVENTS].offset);
        }

        // initialize `cbs` using `cb` array
        //

        for (uint64_t i = 0; i < n_ops; i++)
        {
            cbs[i] = cb.data() + i;
        }

        uint64_t n_tries = 0;
        while (n_tries <= n_retries)
        {
            // issue reads
            const bool lat = merit_lat_on();
            const uint64_t t0 = lat ? merit_lat_now_ns() : 0;
            int64_t ret = io_submit(ctx, (int64_t)n_ops, cbs.data());
            // if requests didn't get accepted
            if (ret != (int64_t)n_ops)
            {
                std::cerr << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                          << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                std::cout << "ctx: " << ctx << "\n";
                exit(-1);
            }
            else
            {
                // wait on io_getevents
                if (lat)
                    ret = merit_lat_wait(ctx, cb.data(), n_ops, evts, t0) ? (int64_t)n_ops : -1;
                else
                    ret = io_getevents(ctx, (int64_t)n_ops, (int64_t)n_ops, evts.data(), nullptr);
                // if requests didn't complete
                if (ret != (int64_t)n_ops)
                {
                    std::cerr << "io_getevents() failed; returned " << ret << ", expected=" << n_ops
                              << ", ernno=" << errno << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                    exit(-1);
                }
                else
                {
                    break;
                }
            }
        }
        // disabled since req.buf could be an offset into another buf
        /*
        for (auto &req : read_reqs) {
          // corruption check
          assert(malloc_usable_size(req.buf) >= req.len);
        }
        */
    }
}

void execute_io_multi(io_context_t ctx, std::vector<FdAlignedRead> &read_reqs, uint64_t n_retries = 0)
{
    if (read_reqs.empty())
        return;
    uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++)
    {
        uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * MAX_EVENTS), (uint64_t)MAX_EVENTS);
        std::vector<iocb_t *> cbs(n_ops, nullptr);
        std::vector<io_event_t> evts(n_ops);
        std::vector<struct iocb> cb(n_ops);
        for (uint64_t j = 0; j < n_ops; j++)
        {
            const FdAlignedRead &fr = read_reqs[j + iter * MAX_EVENTS];
            io_prep_pread(cb.data() + j, fr.fd, fr.read.buf, fr.read.len, fr.read.offset);
        }
        for (uint64_t i = 0; i < n_ops; i++)
            cbs[i] = cb.data() + i;

        uint64_t n_tries = 0;
        while (n_tries <= n_retries)
        {
            const bool lat = merit_lat_on();
            const uint64_t t0 = lat ? merit_lat_now_ns() : 0;
            int64_t ret = io_submit(ctx, (int64_t)n_ops, cbs.data());
            if (ret != (int64_t)n_ops)
            {
                std::cerr << "io_submit(multi) failed; returned " << ret << ", expected=" << n_ops << std::endl;
                exit(-1);
            }
            if (lat)
                ret = merit_lat_wait(ctx, cb.data(), n_ops, evts, t0) ? (int64_t)n_ops : -1;
            else
                ret = io_getevents(ctx, (int64_t)n_ops, (int64_t)n_ops, evts.data(), nullptr);
            if (ret != (int64_t)n_ops)
            {
                std::cerr << "io_getevents(multi) failed; returned " << ret << ", expected=" << n_ops << std::endl;
                exit(-1);
            }
            break;
        }
    }
}
} // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader()
{
    this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader()
{
    int64_t ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(this->file_desc, F_GETFD);
    if (ret == -1)
    {
        if (errno != EBADF)
        {
            std::cerr << "close() not called" << std::endl;
            // close file desc
            ret = ::close(this->file_desc);
            // error checks
            if (ret == -1)
            {
                std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno)
                          << std::endl;
            }
        }
    }
}

io_context_t &LinuxAlignedFileReader::get_ctx()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    // perform checks only in DEBUG mode
    if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end())
    {
        std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
        return this->bad_ctx;
    }
    else
    {
        return ctx_map[std::this_thread::get_id()];
    }
}

void LinuxAlignedFileReader::register_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    if (ctx_map.find(my_id) != ctx_map.end())
    {
        std::cerr << "multiple calls to register_thread from the same thread" << std::endl;
        return;
    }
    io_context_t ctx = 0;
    int ret = io_setup(MAX_EVENTS, &ctx);
    if (ret != 0)
    {
        lk.unlock();
        if (ret == -EAGAIN)
        {
            std::cerr << "io_setup() failed with EAGAIN: Consider increasing /proc/sys/fs/aio-max-nr" << std::endl;
        }
        else
        {
            std::cerr << "io_setup() failed; returned " << ret << ": " << ::strerror(-ret) << std::endl;
        }
    }
    else
    {
        diskann::cout << "allocating ctx: " << ctx << " to thread-id:" << my_id << std::endl;
        ctx_map[my_id] = ctx;
    }
    lk.unlock();
}

void LinuxAlignedFileReader::deregister_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    assert(ctx_map.find(my_id) != ctx_map.end());

    lk.unlock();
    io_context_t ctx = this->get_ctx();
    io_destroy(ctx);
    //  assert(ret == 0);
    lk.lock();
    ctx_map.erase(my_id);
    std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
    lk.unlock();
}

void LinuxAlignedFileReader::deregister_all_threads()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    for (auto x = ctx_map.begin(); x != ctx_map.end(); x++)
    {
        io_context_t ctx = x.value();
        io_destroy(ctx);
        //  assert(ret == 0);
        //  lk.lock();
        //  ctx_map.erase(my_id);
        //  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
    }
    ctx_map.clear();
    //  lk.unlock();
}

void LinuxAlignedFileReader::open(const std::string &fname)
{
    int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
    this->file_desc = ::open(fname.c_str(), flags);
    // error checks
    assert(this->file_desc != -1);
    std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close()
{
    //  int64_t ret;

    // check to make sure file_desc is closed
    ::fcntl(this->file_desc, F_GETFD);
    //  assert(ret != -1);

    ::close(this->file_desc);
    //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs, io_context_t &ctx, bool async)
{
    if (async == true)
    {
        diskann::cout << "Async currently not supported in linux." << std::endl;
    }
    assert(this->file_desc != -1);
    execute_io(ctx, this->file_desc, read_reqs);
}

void LinuxAlignedFileReader::read_multi(std::vector<FdAlignedRead> &read_reqs, io_context_t &ctx)
{
    if (read_reqs.empty())
        return;
    execute_io_multi(ctx, read_reqs);
}

void merit_io_latency_report()
{
    if (!merit_lat_on())
        return;
    static std::vector<uint64_t> bins[kLatClassCount];
    uint64_t sum_ns[kLatClassCount] = {};
    for (int c = 0; c < kLatClassCount; ++c)
        bins[c].assign(kLatBins, 0);
    {
        std::lock_guard<std::mutex> lock(merit_lat_mu());
        for (MeritLatAcc *a : merit_lat_registry())
            for (int c = 0; c < kLatClassCount; ++c)
            {
                sum_ns[c] += a->sum_ns[c];
                for (int b = 0; b < kLatBins; ++b)
                    bins[c][b] += a->bins[c][b];
            }
    }
    static const char *names[kLatClassCount] = {"req_4k", "req_8k", "req_other", "batch_no8k", "batch_with8k"};
    for (int c = 0; c < kLatClassCount; ++c)
    {
        uint64_t n = 0;
        for (uint64_t v : bins[c])
            n += v;
        if (n == 0)
            continue;
        auto pct = [&](double p) {
            const uint64_t target = static_cast<uint64_t>(p * static_cast<double>(n));
            uint64_t acc = 0;
            for (int b = 0; b < kLatBins; ++b)
            {
                acc += bins[c][b];
                if (acc > target)
                    return b;
            }
            return kLatBins - 1;
        };
        std::cout << "MERIT io_lat " << names[c] << ": n=" << n << " mean_us=" << (sum_ns[c] / 1000.0 / n)
                  << " p50=" << pct(0.50) << " p90=" << pct(0.90) << " p99=" << pct(0.99)
                  << " p999=" << pct(0.999) << std::endl;
    }
}
