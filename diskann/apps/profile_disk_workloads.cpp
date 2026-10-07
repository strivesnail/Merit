// Load a disk index once, then profile every query file with vanilla DiskANN (no MERIT cache).

#include "common_includes.h"
#include "linux_aligned_file_reader.h"
#include "pq_flash_index.h"
#include "program_options_utils.hpp"
#include "utils.h"

#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <sstream>

namespace po = boost::program_options;

struct Workload
{
    std::string name;
    std::string query_path;
};

static std::vector<Workload> parse_manifest(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
    {
        diskann::cerr << "cannot open query manifest: " << path << std::endl;
        return {};
    }
    std::vector<Workload> jobs;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        Workload w;
        if (!(iss >> w.name >> w.query_path))
            continue;
        jobs.push_back(std::move(w));
    }
    return jobs;
}

template <typename T>
int run(const std::string &index_prefix, const std::string &manifest_path, const std::string &profile_dir,
        uint32_t num_threads, uint32_t L, uint32_t W, uint32_t K)
{
    auto jobs = parse_manifest(manifest_path);
    if (jobs.empty())
    {
        diskann::cerr << "empty query manifest\n";
        return -1;
    }

    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    diskann::PQFlashIndex<T> index(reader, diskann::Metric::L2);
    if (index.load(num_threads, index_prefix.c_str()) != 0)
        return -1;

    index.enable_access_profile(true);
    omp_set_num_threads(num_threads);

    for (size_t ji = 0; ji < jobs.size(); ji++)
    {
        const auto &job = jobs[ji];
        diskann::cout << "=== workload " << (ji + 1) << "/" << jobs.size() << " " << job.name << " ===" << std::endl;
        if (ji > 0)
            index.reset_access_profile();

        T *query = nullptr;
        size_t query_num = 0, query_dim = 0, query_aligned_dim = 0;
        diskann::load_aligned_bin<T>(job.query_path, query, query_num, query_dim, query_aligned_dim);

        std::vector<uint64_t> ids(query_num * K);
        std::vector<float> dists(query_num * K);
        auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            index.cached_beam_search(query + i * query_aligned_dim, K, L, ids.data() + i * K, dists.data() + i * K, W);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        diskann::cout << job.name << " nq=" << query_num << " qps=" << (query_num / sec) << std::endl;

        const std::string prefix = profile_dir + "/" + job.name;
        index.print_access_profile_cdf();
        if (index.save_access_profile(prefix) != 0)
        {
            diskann::cerr << "failed to save profile " << prefix << std::endl;
            diskann::aligned_free(query);
            return -1;
        }
        diskann::aligned_free(query);
    }
    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, index_prefix, manifest, profile_dir;
    uint32_t threads = 8, L = 50, W = 4, K = 1;
    po::options_description desc("profile_disk_workloads");
    desc.add_options()("help,h", "help")("data_type", po::value<std::string>(&data_type)->required())(
        "index_path_prefix", po::value<std::string>(&index_prefix)->required())(
        "query_manifest", po::value<std::string>(&manifest)->required(), "lines: name query_path")(
        "profile_dir", po::value<std::string>(&profile_dir)->required())(
        "num_threads,T", po::value<uint32_t>(&threads)->default_value(8))(
        "search_list,L", po::value<uint32_t>(&L)->default_value(50))("beamwidth,W",
                                                                    po::value<uint32_t>(&W)->default_value(4))(
        "recall_at,K", po::value<uint32_t>(&K)->default_value(1));
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    if (data_type == "uint8")
        return run<uint8_t>(index_prefix, manifest, profile_dir, threads, L, W, K);
    if (data_type == "float")
        return run<float>(index_prefix, manifest, profile_dir, threads, L, W, K);
    if (data_type == "int8")
        return run<int8_t>(index_prefix, manifest, profile_dir, threads, L, W, K);
    std::cerr << "unsupported data_type\n";
    return -1;
}
