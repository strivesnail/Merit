// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <map>
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
    DISKANN_DLLEXPORT void on_merit_hop_frontier(const std::vector<uint32_t> &merit_nodes);

    DISKANN_DLLEXPORT int save(const std::string &output_prefix) const;
    DISKANN_DLLEXPORT void print_cdf_summary() const;

    DISKANN_DLLEXPORT static int load(const std::string &profile_prefix, std::vector<uint64_t> &node_expand,
                                      std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges);

    // Layout D: sorted merit frontier -> weight (hop co-occurrence from profiling).
    DISKANN_DLLEXPORT static int load_frontier_templates(
        const std::string &profile_prefix, std::vector<std::pair<uint64_t, std::vector<uint32_t>>> &templates);

  private:
    bool _enabled = false;
    uint64_t _num_points = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> _node_expand;
    mutable std::mutex _edge_mutex;
    std::unordered_map<uint64_t, uint64_t> _directed_edges;
    mutable std::mutex _frontier_mutex;
    std::map<std::vector<uint32_t>, uint64_t> _frontier_templates;
};

} // namespace diskann
