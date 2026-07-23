// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <memory>

#include "common_includes.h"
#include "windows_customizations.h"

namespace diskann
{

inline uint64_t directed_edge_key(uint32_t u, uint32_t v)
{
    return (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);
}

// Tracks node expansion and directed edge (parent -> child) access during beam search.
class HotnessProfiler
{
  public:
    DISKANN_DLLEXPORT void init(uint64_t num_points);
    DISKANN_DLLEXPORT void set_enabled(bool enabled);
    DISKANN_DLLEXPORT bool enabled() const;

    DISKANN_DLLEXPORT void on_node_expand(uint32_t node_id);
    DISKANN_DLLEXPORT void on_directed_edge(uint32_t parent, uint32_t child);

    DISKANN_DLLEXPORT int save(const std::string &output_prefix) const;
    DISKANN_DLLEXPORT void print_cdf_summary() const;

    DISKANN_DLLEXPORT static int load(const std::string &profile_prefix, std::vector<uint64_t> &node_expand,
                                      std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges);

  private:
    bool _enabled = false;
    uint64_t _num_points = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> _node_expand;
    mutable std::mutex _edge_mutex;
    std::unordered_map<uint64_t, uint64_t> _directed_edges;
};

} // namespace diskann
