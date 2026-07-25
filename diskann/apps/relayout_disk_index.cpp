// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>

#include "defaults.h"
#include "logger.h"
#include "hotness_profiler.h"
#include "relayout_utils.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    std::string mem_index, profile_prefix, output_order, disk_index, layout_mode = "hotnode";
    uint32_t k_hops = 2;
    uint64_t nnodes_per_sector = 0;

    po::options_description desc("relayout_disk_index: offline page relayout from access profile");
    desc.add_options()("help,h", "Print help")("mem_index", po::value<std::string>(&mem_index)->required(),
                                               "Path to _mem.index file")(
        "profile_prefix", po::value<std::string>(&profile_prefix)->required(),
        "Prefix for access profile files from Run2")("output_order", po::value<std::string>(&output_order)->required(),
                                                     "Output order file prefix")(
        "layout", po::value<std::string>(&layout_mode)->default_value("hotnode"),
        "Layout algorithm: hotnode (node expand) or edge (edge-importance packing)")(
        "disk_index", po::value<std::string>(&disk_index)->default_value(std::string("")),
        "Path to _disk.index (used to read nnodes_per_sector when auto-detecting)")(
        "k_hops", po::value<uint32_t>(&k_hops)->default_value(2), "Undirected k-hop neighborhood radius")(
        "nnodes_per_sector", po::value<uint64_t>(&nnodes_per_sector)->default_value(0),
        "Nodes per 4KB sector (0 = read from disk_index, else auto from graph width)");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    diskann::VamanaGraph graph;
    if (diskann::load_vamana_graph(mem_index, graph) != 0)
    {
        std::cerr << "Failed to load vamana graph from " << mem_index << std::endl;
        return -1;
    }

    std::vector<uint64_t> node_expand;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> unused_edges;
    if (diskann::HotnessProfiler::load(profile_prefix, node_expand, unused_edges) != 0)
    {
        std::cerr << "Failed to load access profile from prefix " << profile_prefix << std::endl;
        return -1;
    }

    if (node_expand.size() != graph.num_points)
    {
        std::cerr << "Node expand count size mismatch: " << node_expand.size() << " vs " << graph.num_points << std::endl;
        return -1;
    }

    if (nnodes_per_sector == 0)
    {
        if (disk_index.empty())
        {
            disk_index = mem_index;
            const std::string suffix = "_mem.index";
            if (disk_index.size() > suffix.size() &&
                disk_index.compare(disk_index.size() - suffix.size(), suffix.size(), suffix) == 0)
                disk_index.replace(disk_index.size() - suffix.size(), suffix.size(), "_disk.index");
        }
        if (diskann::read_disk_index_nnodes_per_sector(disk_index, nnodes_per_sector) != 0)
        {
            const uint64_t max_node_len = (static_cast<uint64_t>(graph.width) + 1) * sizeof(uint32_t);
            nnodes_per_sector = diskann::defaults::SECTOR_LEN / max_node_len;
            if (nnodes_per_sector == 0)
                nnodes_per_sector = 1;
            diskann::cout << "Warning: failed to read nnodes_per_sector from " << disk_index
                          << ", falling back to " << nnodes_per_sector << std::endl;
        }
    }

    std::vector<uint32_t> order;
    const bool use_edge_layout = (layout_mode == "edge" || layout_mode == "jiang");
    if (use_edge_layout)
    {
        if (diskann::compute_edge_relayout_order(graph, node_expand, unused_edges, nnodes_per_sector, k_hops,
                                                  order) != 0)
        {
            std::cerr << "Failed to compute edge-importance relayout order." << std::endl;
            return -1;
        }
    }
    else if (diskann::compute_hot_node_relayout_order(graph, node_expand, unused_edges, nnodes_per_sector, k_hops,
                                                    order) != 0)
    {
        std::cerr << "Failed to compute hot-node relayout order." << std::endl;
        return -1;
    }

    if (diskann::save_relayout_order(output_order, order, nnodes_per_sector) != 0)
    {
        std::cerr << "Failed to save relayout order." << std::endl;
        return -1;
    }

    diskann::cout << "Relayout order written to " << output_order << " (" << order.size()
                  << " nodes, nnodes_per_sector=" << nnodes_per_sector << ", k_hops=" << k_hops
                  << ", layout=" << layout_mode << ")" << std::endl;
    return 0;
}
