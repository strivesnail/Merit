// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "hotness_profiler.h"
#include "utils.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace diskann
{

void HotnessProfiler::init(uint64_t num_points)
{
    _num_points = num_points;
    _node_expand = std::make_unique<std::atomic<uint64_t>[]>(num_points);
    for (uint64_t i = 0; i < num_points; i++)
        _node_expand[i].store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(_edge_mutex);
    _directed_edges.clear();
}

void HotnessProfiler::set_enabled(bool enabled)
{
    _enabled = enabled;
}

bool HotnessProfiler::enabled() const
{
    return _enabled;
}

void HotnessProfiler::on_node_expand(uint32_t node_id)
{
    if (!_enabled || node_id >= _num_points)
        return;
    _node_expand[node_id].fetch_add(1, std::memory_order_relaxed);
}

void HotnessProfiler::on_directed_edge(uint32_t parent, uint32_t child)
{
    if (!_enabled || parent >= _num_points || child >= _num_points)
        return;
    const uint64_t key = directed_edge_key(parent, child);
    std::lock_guard<std::mutex> lock(_edge_mutex);
    _directed_edges[key]++;
}

static void print_count_cdf(const std::string &name, std::vector<uint64_t> counts)
{
    if (counts.empty())
    {
        diskann::cout << name << " CDF: no data" << std::endl;
        return;
    }
    std::sort(counts.begin(), counts.end(), std::greater<uint64_t>());
    const uint64_t total = std::accumulate(counts.begin(), counts.end(), uint64_t(0));
    diskann::cout << name << " CDF (total accesses=" << total << ", unique=" << counts.size() << "):" << std::endl;
    const std::vector<double> percentiles = {0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99};
    for (double p : percentiles)
    {
        const size_t idx = static_cast<size_t>(p * static_cast<double>(counts.size() - 1));
        diskann::cout << "  p" << std::setw(5) << static_cast<int>(p * 100) << ": count>=" << counts[idx] << std::endl;
    }
}

void HotnessProfiler::print_cdf_summary() const
{
    std::vector<uint64_t> node_counts;
    node_counts.reserve(_num_points);
    for (uint64_t i = 0; i < _num_points; i++)
    {
        const uint64_t c = _node_expand[i].load(std::memory_order_relaxed);
        if (c > 0)
            node_counts.push_back(c);
    }
    print_count_cdf("Node expand", node_counts);

    std::vector<uint64_t> edge_counts;
    {
        std::lock_guard<std::mutex> lock(_edge_mutex);
        edge_counts.reserve(_directed_edges.size());
        for (const auto &kv : _directed_edges)
        {
            if (kv.second > 0)
                edge_counts.push_back(kv.second);
        }
    }
    print_count_cdf("Directed edge", edge_counts);
}

int HotnessProfiler::save(const std::string &output_prefix) const
{
    std::vector<uint64_t> node_counts(_num_points, 0);
    for (uint64_t i = 0; i < _num_points; i++)
    {
        node_counts[i] = _node_expand[i].load(std::memory_order_relaxed);
    }
    diskann::save_bin<uint64_t>(output_prefix + "_node_expand.bin", node_counts.data(), _num_points, 1);

    std::vector<uint32_t> edge_u;
    std::vector<uint32_t> edge_v;
    std::vector<uint64_t> edge_c;
    {
        std::lock_guard<std::mutex> lock(_edge_mutex);
        edge_u.reserve(_directed_edges.size());
        edge_v.reserve(_directed_edges.size());
        edge_c.reserve(_directed_edges.size());
        for (const auto &kv : _directed_edges)
        {
            edge_u.push_back(static_cast<uint32_t>(kv.first >> 32));
            edge_v.push_back(static_cast<uint32_t>(kv.first & 0xFFFFFFFFu));
            edge_c.push_back(kv.second);
        }
    }
    const size_t nedges = edge_u.size();
    diskann::save_bin<uint32_t>(output_prefix + "_edge_u.bin", edge_u.data(), nedges, 1);
    diskann::save_bin<uint32_t>(output_prefix + "_edge_v.bin", edge_v.data(), nedges, 1);
    diskann::save_bin<uint64_t>(output_prefix + "_edge_count.bin", edge_c.data(), nedges, 1);

    diskann::cout << "Saved access profile to prefix " << output_prefix << " (" << nedges << " directed edges)"
                  << std::endl;
    return 0;
}

int HotnessProfiler::load(const std::string &profile_prefix, std::vector<uint64_t> &node_expand,
                          std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges)
{
    if (!file_exists(profile_prefix + "_node_expand.bin"))
        return -1;

    size_t npts = 0, ndim = 0;
    uint64_t *node_buf = nullptr;
    diskann::load_bin<uint64_t>(profile_prefix + "_node_expand.bin", node_buf, npts, ndim);
    node_expand.assign(node_buf, node_buf + npts);
    delete[] node_buf;

    directed_edges.clear();
    if (!file_exists(profile_prefix + "_edge_u.bin"))
        return 0;

    uint32_t *u_buf = nullptr;
    uint32_t *v_buf = nullptr;
    uint64_t *c_buf = nullptr;
    size_t nu = 0, nv = 0, nc = 0;
    size_t du = 0, dv = 0, dc = 0;
    diskann::load_bin<uint32_t>(profile_prefix + "_edge_u.bin", u_buf, nu, du);
    diskann::load_bin<uint32_t>(profile_prefix + "_edge_v.bin", v_buf, nv, dv);
    diskann::load_bin<uint64_t>(profile_prefix + "_edge_count.bin", c_buf, nc, dc);
    if (nu != nv || nu != nc)
    {
        delete[] u_buf;
        delete[] v_buf;
        delete[] c_buf;
        return -1;
    }
    directed_edges.reserve(nu);
    for (size_t i = 0; i < nu; i++)
        directed_edges.emplace_back(u_buf[i], v_buf[i], c_buf[i]);
    delete[] u_buf;
    delete[] v_buf;
    delete[] c_buf;
    return 0;
}

} // namespace diskann
