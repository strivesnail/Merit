// Dump MERIT disk-cache disk cache pages for layout verification.
#include "common_includes.h"
#include <boost/program_options.hpp>
#include <fstream>

#include "hotness_profiler.h"
#include "logger.h"
#include "relayout_utils.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    std::string mem_index, profile_prefix, output_json, layout_mode = "node";
    uint32_t k_hops = 1;
    uint64_t nnodes_per_sector = 5;
    uint64_t max_nodes = 100000;
    std::string disk_index;

    po::options_description desc("dump_disk_cache_pages");
    desc.add_options()("help,h", "Print help")("mem_index", po::value<std::string>(&mem_index)->required(),
                                               "Path to _mem.index")("profile_prefix",
                                                                     po::value<std::string>(&profile_prefix)->required(),
                                                                     "Access profile prefix")(
        "output", po::value<std::string>(&output_json)->required(), "Output JSON path")(
        "layout", po::value<std::string>(&layout_mode)->default_value("node"),
        "Layout: node(B) | edge(C) | directed_beam(E) | directed_beam_pct80(E_pct80) | parent(P)")(
        "k_hops", po::value<uint32_t>(&k_hops)->default_value(1), "k_hops / beam_width for E")(
        "nnodes_per_sector", po::value<uint64_t>(&nnodes_per_sector)->default_value(5), "Page capacity (nodes)")(
        "max_nodes", po::value<uint64_t>(&max_nodes)->default_value(100000), "Disk cache node cap")(
        "disk_index", po::value<std::string>(&disk_index)->default_value(""), "Optional _disk.index for nps");

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
        std::cerr << "Failed to load graph: " << mem_index << std::endl;
        return -1;
    }

    std::vector<uint64_t> node_expand;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> directed_edges;
    if (diskann::HotnessProfiler::load(profile_prefix, node_expand, directed_edges) != 0)
    {
        std::cerr << "Failed to load profile: " << profile_prefix << std::endl;
        return -1;
    }

    if (nnodes_per_sector == 0 && !disk_index.empty())
        diskann::read_disk_index_nnodes_per_sector(disk_index, nnodes_per_sector);

    std::unordered_set<uint32_t> exclude_ids;
    std::vector<std::vector<uint32_t>> pages;
    const std::string layout = diskann::normalize_disk_cache_layout(layout_mode);
    if (diskann::merit_dump_disk_cache_pages(graph, node_expand, directed_edges, layout, nnodes_per_sector, k_hops,
                                             max_nodes, exclude_ids, pages) != 0)
    {
        std::cerr << "merit_dump_disk_cache_pages failed for layout=" << layout << std::endl;
        return -1;
    }

    std::ofstream out(output_json);
    if (!out)
    {
        std::cerr << "Failed to open " << output_json << std::endl;
        return -1;
    }
    out << "{\n  \"layout\": \"" << layout << "\",\n  \"k_hops\": " << k_hops
        << ",\n  \"nnodes_per_sector\": " << nnodes_per_sector << ",\n  \"num_pages\": " << pages.size()
        << ",\n  \"pages\": [\n";
    for (size_t i = 0; i < pages.size(); ++i)
    {
        out << "    [";
        for (size_t j = 0; j < pages[i].size(); ++j)
        {
            if (j)
                out << ", ";
            out << pages[i][j];
        }
        out << "]";
        if (i + 1 < pages.size())
            out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
    diskann::cout << "Wrote " << pages.size() << " pages to " << output_json << " (layout=" << layout
                  << ", k_hops=" << k_hops << ")" << std::endl;
    return 0;
}
