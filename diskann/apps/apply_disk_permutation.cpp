// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>
#include <fstream>

#include "disk_utils.h"
#include "logger.h"
#include "relayout_utils.h"

namespace po = boost::program_options;

static void copy_file(const std::string &src, const std::string &dst)
{
    std::ifstream in(src, std::ios::binary);
    std::ofstream out(dst, std::ios::binary);
    out << in.rdbuf();
}

template <typename T>
int apply_permutation(const std::string &base_file, const std::string &mem_index_file, const std::string &index_prefix,
                      const std::string &order_file, const std::string &output_prefix)
{
    std::vector<uint32_t> order;
    uint64_t nnodes_per_sector = 0;
    if (diskann::load_relayout_order(order_file, order, nnodes_per_sector) != 0)
    {
        std::cerr << "Failed to load relayout order from " << order_file << std::endl;
        return -1;
    }

    const std::string output_disk = output_prefix + "_disk.index";
    diskann::create_disk_layout_with_order<T>(base_file, mem_index_file, order, output_disk);

    const std::string input_pq = index_prefix + "_pq_compressed.bin";
    const std::string output_pq = output_prefix + "_pq_compressed.bin";
    diskann::permute_pq_compressed(input_pq, output_pq, order);

    copy_file(index_prefix + "_pq_pivots.bin", output_prefix + "_pq_pivots.bin");

    diskann::cout << "Applied permutation: " << output_disk << ", " << output_pq << std::endl;
    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, base_file, mem_index, index_prefix, order_file, output_prefix;

    po::options_description desc("apply_disk_permutation: rewrite disk index and PQ with relayout order");
    desc.add_options()("help,h", "Print help")("data_type", po::value<std::string>(&data_type)->required(),
                                                 "float | int8 | uint8")(
        "base_file", po::value<std::string>(&base_file)->required(), "Original base data .bin")(
        "mem_index", po::value<std::string>(&mem_index)->required(), "Original _mem.index path")(
        "index_prefix", po::value<std::string>(&index_prefix)->required(), "Original built index prefix")(
        "order_file", po::value<std::string>(&order_file)->required(), "Relayout order file prefix")(
        "output_prefix", po::value<std::string>(&output_prefix)->required(), "Output index prefix");

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

    try
    {
        if (data_type == "float")
            return apply_permutation<float>(base_file, mem_index, index_prefix, order_file, output_prefix);
        if (data_type == "int8")
            return apply_permutation<int8_t>(base_file, mem_index, index_prefix, order_file, output_prefix);
        if (data_type == "uint8")
            return apply_permutation<uint8_t>(base_file, mem_index, index_prefix, order_file, output_prefix);
        std::cerr << "Unsupported data_type: " << data_type << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
