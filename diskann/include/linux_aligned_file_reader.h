// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#ifndef _WINDOWS

#include "aligned_file_reader.h"

class LinuxAlignedFileReader : public AlignedFileReader
{
  private:
    uint64_t file_sz;
    FileHandle file_desc;
    io_context_t bad_ctx = (io_context_t)-1;

  public:
    int get_file_desc() const
    {
        return static_cast<int>(file_desc);
    }
    LinuxAlignedFileReader();
    ~LinuxAlignedFileReader();

    IOContext &get_ctx();

    // register thread-id for a context
    void register_thread();

    // de-register thread-id for a context
    void deregister_thread();
    void deregister_all_threads();

    // Open & close ops
    // Blocking calls
    void open(const std::string &fname);
    void close();

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false) override;

    void read_multi(std::vector<FdAlignedRead> &read_reqs, IOContext &ctx) override;
};

// Per-request / per-batch read latency histogram, enabled by MERIT_IO_LAT_PROFILE=1.
void merit_io_latency_report();

#endif
