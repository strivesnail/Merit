// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common_includes.h"
#include "windows_customizations.h"

#include <unordered_set>

namespace diskann
{

struct VamanaGraph
{
    uint32_t width = 0;
    uint64_t medoid = 0;
    uint64_t num_points = 0;
    std::vector<std::vector<uint32_t>> adjacency;
};

DISKANN_DLLEXPORT int load_vamana_graph(const std::string &mem_index_file, VamanaGraph &graph);

// order[new_id] = old_id. Jiang edge-importance + k-hop paths per page.
DISKANN_DLLEXPORT int compute_jiang_relayout_order(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t k_hops, std::vector<uint32_t> &order);

// order[new_id] = old_id. Hot-node seed + k-hop BFS on graph; page fill by node expand count H(v).
DISKANN_DLLEXPORT int compute_hot_node_relayout_order(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                                      uint64_t nnodes_per_sector, uint32_t k_hops,
                                                      std::vector<uint32_t> &order);

// MERIT disk sidecar: same hot-node + k-hop page packing as relayout, but only first max_nodes (after exclude_ids).
DISKANN_DLLEXPORT int compute_hot_node_disk_cache_list(const VamanaGraph &graph,
                                                       const std::vector<uint64_t> &node_expand,
                                                       uint64_t nnodes_per_sector, uint32_t k_hops, uint64_t max_nodes,
                                                       const std::unordered_set<uint32_t> &exclude_ids,
                                                       std::vector<uint32_t> &node_list);

DISKANN_DLLEXPORT int save_relayout_order(const std::string &path, const std::vector<uint32_t> &order,
                                          uint64_t nnodes_per_sector);

DISKANN_DLLEXPORT int load_relayout_order(const std::string &path, std::vector<uint32_t> &order,
                                          uint64_t &nnodes_per_sector);

DISKANN_DLLEXPORT int read_disk_index_nnodes_per_sector(const std::string &disk_index_file, uint64_t &nnodes_per_sector);

} // namespace diskann
