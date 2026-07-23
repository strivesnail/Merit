// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "relayout_utils.h"
#include "utils.h"

#include <algorithm>
#include <numeric>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace diskann
{

int load_vamana_graph(const std::string &mem_index_file, VamanaGraph &graph)
{
    const size_t actual_file_size = get_file_size(mem_index_file);
    std::ifstream reader(mem_index_file, std::ios::binary);
    if (!reader.is_open())
        return -1;

    size_t index_file_size = 0;
    reader.read((char *)&index_file_size, sizeof(uint64_t));
    if (index_file_size != actual_file_size)
        return -1;

    uint32_t width_u32 = 0, medoid_u32 = 0;
    uint64_t frozen_num = 0;
    reader.read((char *)&width_u32, sizeof(uint32_t));
    reader.read((char *)&medoid_u32, sizeof(uint32_t));
    reader.read((char *)&frozen_num, sizeof(uint64_t));

    graph.width = width_u32;
    graph.medoid = medoid_u32;
    graph.adjacency.clear();

    while (reader.peek() != EOF)
    {
        uint32_t nnbrs = 0;
        reader.read((char *)&nnbrs, sizeof(uint32_t));
        if (reader.eof())
            break;
        std::vector<uint32_t> nbrs(nnbrs);
        if (nnbrs > 0)
            reader.read((char *)nbrs.data(), nnbrs * sizeof(uint32_t));
        if (nnbrs > width_u32)
            reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), std::ios::cur);
        graph.adjacency.push_back(std::move(nbrs));
    }

    graph.num_points = graph.adjacency.size();
    return 0;
}
static inline uint64_t undirected_edge_key(uint32_t a, uint32_t b)
{
    if (a > b)
        std::swap(a, b);
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

static std::unordered_map<uint64_t, uint64_t> build_undirected_weights(
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges)
{
    std::unordered_map<uint64_t, uint64_t> weights;
    for (const auto &edge : directed_edges)
    {
        const uint32_t u = std::get<0>(edge);
        const uint32_t v = std::get<1>(edge);
        const uint64_t c = std::get<2>(edge);
        weights[undirected_edge_key(u, v)] += c;
    }
    return weights;
}

static std::vector<std::pair<uint32_t, uint32_t>> sorted_undirected_edges(
    const std::unordered_map<uint64_t, uint64_t> &weights)
{
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    edges.reserve(weights.size());
    for (const auto &kv : weights)
    {
        edges.emplace_back(static_cast<uint32_t>(kv.first >> 32), static_cast<uint32_t>(kv.first & 0xFFFFFFFFu));
    }
    std::sort(edges.begin(), edges.end(), [&](const auto &a, const auto &b) {
        return weights.at(undirected_edge_key(a.first, a.second)) > weights.at(undirected_edge_key(b.first, b.second));
    });
    return edges;
}

static std::vector<uint32_t> k_hop_neighborhood(const VamanaGraph &graph, const std::vector<uint32_t> &seeds,
                                                uint32_t k_hops)
{
    std::vector<int32_t> dist(graph.num_points, -1);
    std::queue<uint32_t> q;
    for (uint32_t s : seeds)
    {
        if (s >= graph.num_points)
            continue;
        dist[s] = 0;
        q.push(s);
    }
    while (!q.empty())
    {
        const uint32_t u = q.front();
        q.pop();
        if (static_cast<uint32_t>(dist[u]) >= k_hops)
            continue;
        if (u >= graph.adjacency.size())
            continue;
        for (uint32_t v : graph.adjacency[u])
        {
            if (v >= graph.num_points)
                continue;
            if (dist[v] == -1)
            {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    std::vector<uint32_t> nodes;
    for (uint64_t i = 0; i < graph.num_points; i++)
    {
        if (dist[i] != -1)
            nodes.push_back(static_cast<uint32_t>(i));
    }
    return nodes;
}

static std::vector<uint32_t> shortest_path(const VamanaGraph &graph, uint32_t src, uint32_t dst, uint32_t max_hops)
{
    if (src == dst)
        return {src};
    std::vector<int32_t> dist(graph.num_points, -1);
    std::vector<uint32_t> parent(graph.num_points, graph.num_points);
    std::queue<uint32_t> q;
    dist[src] = 0;
    q.push(src);
    while (!q.empty())
    {
        const uint32_t u = q.front();
        q.pop();
        if (static_cast<uint32_t>(dist[u]) >= max_hops)
            continue;
        if (u >= graph.adjacency.size())
            continue;
        for (uint32_t v : graph.adjacency[u])
        {
            if (v >= graph.num_points || dist[v] != -1)
                continue;
            dist[v] = dist[u] + 1;
            parent[v] = u;
            if (v == dst)
            {
                std::vector<uint32_t> path;
                for (uint32_t cur = dst; cur != graph.num_points; cur = parent[cur])
                    path.push_back(cur);
                std::reverse(path.begin(), path.end());
                return path;
            }
            q.push(v);
        }
    }
    return {};
}

static double path_average_weight(const std::vector<uint32_t> &path,
                                  const std::unordered_map<uint64_t, uint64_t> &weights)
{
    if (path.size() < 2)
        return 0.0;
    uint64_t sum = 0;
    for (size_t i = 1; i < path.size(); i++)
    {
        auto it = weights.find(undirected_edge_key(path[i - 1], path[i]));
        sum += (it == weights.end()) ? 0 : it->second;
    }
    return static_cast<double>(sum) / static_cast<double>(path.size() - 1);
}

static bool path_edges_disjoint(const std::vector<uint32_t> &path, const std::unordered_set<uint64_t> &used_edges)
{
    for (size_t i = 1; i < path.size(); i++)
    {
        if (used_edges.count(undirected_edge_key(path[i - 1], path[i])) > 0)
            return false;
    }
    return true;
}

static void mark_path_edges(const std::vector<uint32_t> &path, std::unordered_set<uint64_t> &used_edges)
{
    for (size_t i = 1; i < path.size(); i++)
        used_edges.insert(undirected_edge_key(path[i - 1], path[i]));
}

int compute_jiang_relayout_order(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                 const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                 uint64_t nnodes_per_sector, uint32_t k_hops, std::vector<uint32_t> &order)
{
    if (graph.num_points == 0 || nnodes_per_sector == 0)
        return -1;

    const auto undirected_weights = build_undirected_weights(directed_edges);
    auto sorted_edges = sorted_undirected_edges(undirected_weights);

    std::vector<bool> placed(graph.num_points, false);
    order.clear();
    order.reserve(graph.num_points);

    std::vector<uint32_t> nodes_by_expand(graph.num_points);
    std::iota(nodes_by_expand.begin(), nodes_by_expand.end(), 0);
    std::sort(nodes_by_expand.begin(), nodes_by_expand.end(), [&](uint32_t a, uint32_t b) {
        const uint64_t ca = (a < node_expand.size()) ? node_expand[a] : 0;
        const uint64_t cb = (b < node_expand.size()) ? node_expand[b] : 0;
        return ca > cb;
    });

    size_t edge_cursor = 0;
    while (order.size() < graph.num_points)
    {
        const uint64_t page_cap = nnodes_per_sector;
        std::vector<uint32_t> page_nodes;
        page_nodes.reserve(page_cap);

        uint32_t seed_u = graph.num_points, seed_v = graph.num_points;
        while (edge_cursor < sorted_edges.size())
        {
            const auto [u, v] = sorted_edges[edge_cursor++];
            if (!placed[u] && !placed[v])
            {
                seed_u = u;
                seed_v = v;
                break;
            }
        }

        if (seed_u == graph.num_points)
        {
            for (uint32_t node : nodes_by_expand)
            {
                if (!placed[node])
                {
                    page_nodes.push_back(node);
                    placed[node] = true;
                    if (page_nodes.size() >= page_cap)
                        break;
                }
            }
        }
        else
        {
            page_nodes.push_back(seed_u);
            page_nodes.push_back(seed_v);
            placed[seed_u] = true;
            placed[seed_v] = true;

            const auto neighborhood = k_hop_neighborhood(graph, {seed_u, seed_v}, k_hops);
            std::unordered_set<uint64_t> used_page_edges;
            used_page_edges.insert(undirected_edge_key(seed_u, seed_v));

            while (page_nodes.size() < page_cap)
            {
                std::vector<uint32_t> best_path;
                double best_avg = -1.0;
                for (uint32_t target : neighborhood)
                {
                    if (placed[target])
                        continue;
                    for (uint32_t src : page_nodes)
                    {
                        const auto path = shortest_path(graph, src, target, k_hops);
                        if (path.size() < 2)
                            continue;
                        if (!path_edges_disjoint(path, used_page_edges))
                            continue;
                        const double avg = path_average_weight(path, undirected_weights);
                        if (avg > best_avg)
                        {
                            best_avg = avg;
                            best_path = path;
                        }
                    }
                }
                if (best_path.empty())
                    break;
                mark_path_edges(best_path, used_page_edges);
                for (uint32_t node : best_path)
                {
                    if (placed[node])
                        continue;
                    page_nodes.push_back(node);
                    placed[node] = true;
                    if (page_nodes.size() >= page_cap)
                        break;
                }
            }

            for (uint32_t node : nodes_by_expand)
            {
                if (page_nodes.size() >= page_cap)
                    break;
                if (!placed[node])
                {
                    page_nodes.push_back(node);
                    placed[node] = true;
                }
            }
        }

        for (uint32_t node : page_nodes)
            order.push_back(node);
    }

    if (order.size() != graph.num_points)
        return -1;
    return 0;
}
static std::vector<std::vector<uint32_t>> build_undirected_adjacency(const VamanaGraph &graph)
{
    std::vector<std::vector<uint32_t>> adj(graph.num_points);
    for (uint64_t u = 0; u < graph.num_points; u++)
    {
        for (uint32_t v : graph.adjacency[u])
        {
            if (v < graph.num_points)
            {
                adj[u].push_back(v);
                adj[v].push_back(static_cast<uint32_t>(u));
            }
        }
    }
    for (auto &nbrs : adj)
    {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
    return adj;
}

static uint64_t node_heat(const std::vector<uint64_t> &node_expand, uint32_t id)
{
    return (id < node_expand.size()) ? node_expand[id] : 0;
}

int compute_hot_node_relayout_order(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                    uint64_t nnodes_per_sector, uint32_t k_hops, std::vector<uint32_t> &order)
{
    if (graph.num_points == 0 || nnodes_per_sector == 0)
        return -1;

    const auto undirected_adj = build_undirected_adjacency(graph);

    std::vector<bool> placed(graph.num_points, false);
    order.clear();
    order.reserve(graph.num_points);

    std::vector<uint32_t> nodes_by_expand(static_cast<size_t>(graph.num_points));
    std::iota(nodes_by_expand.begin(), nodes_by_expand.end(), 0);
    std::sort(nodes_by_expand.begin(), nodes_by_expand.end(), [&](uint32_t a, uint32_t b) {
        const uint64_t ca = node_heat(node_expand, a);
        const uint64_t cb = node_heat(node_expand, b);
        if (ca != cb)
            return ca > cb;
        return a < b;
    });

    const uint32_t invalid = static_cast<uint32_t>(graph.num_points);

    while (order.size() < graph.num_points)
    {
        const uint64_t page_cap = nnodes_per_sector;
        std::vector<uint32_t> page_nodes;
        page_nodes.reserve(page_cap);

        uint32_t seed = invalid;
        for (uint32_t node : nodes_by_expand)
        {
            if (!placed[node])
            {
                seed = node;
                break;
            }
        }
        if (seed == invalid)
            break;

        page_nodes.push_back(seed);
        placed[seed] = true;

        while (page_nodes.size() < page_cap)
        {
            std::vector<int32_t> dist(graph.num_points, -1);
            std::queue<uint32_t> q;
            for (uint32_t src : page_nodes)
            {
                if (dist[src] != -1)
                    continue;
                dist[src] = 0;
                q.push(src);
            }
            while (!q.empty())
            {
                const uint32_t u = q.front();
                q.pop();
                if (static_cast<uint32_t>(dist[u]) >= k_hops)
                    continue;
                for (uint32_t v : undirected_adj[u])
                {
                    if (dist[v] != -1)
                        continue;
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }

            uint32_t best = invalid;
            uint64_t best_h = 0;
            for (uint64_t i = 0; i < graph.num_points; i++)
            {
                if (placed[i] || dist[i] == -1)
                    continue;
                const uint64_t h = node_heat(node_expand, static_cast<uint32_t>(i));
                if (h > best_h || (h == best_h && i < best))
                {
                    best_h = h;
                    best = static_cast<uint32_t>(i);
                }
            }
            if (best == invalid)
                break;

            page_nodes.push_back(best);
            placed[best] = true;
        }

        for (uint32_t node : page_nodes)
            order.push_back(node);
    }

    for (uint32_t node : nodes_by_expand)
    {
        if (!placed[node])
        {
            order.push_back(node);
            placed[node] = true;
        }
    }

    if (order.size() != graph.num_points)
        return -1;
    return 0;
}

int compute_hot_node_disk_cache_list(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                     uint64_t nnodes_per_sector, uint32_t k_hops, uint64_t max_nodes,
                                     const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    const auto undirected_adj = build_undirected_adjacency(graph);

    std::vector<bool> placed(graph.num_points, false);
    for (uint32_t ex : exclude_ids)
    {
        if (ex < graph.num_points)
            placed[ex] = true;
    }

    std::vector<uint32_t> nodes_by_expand(static_cast<size_t>(graph.num_points));
    std::iota(nodes_by_expand.begin(), nodes_by_expand.end(), 0);
    std::sort(nodes_by_expand.begin(), nodes_by_expand.end(), [&](uint32_t a, uint32_t b) {
        const uint64_t ca = node_heat(node_expand, a);
        const uint64_t cb = node_heat(node_expand, b);
        if (ca != cb)
            return ca > cb;
        return a < b;
    });

    const uint32_t invalid = static_cast<uint32_t>(graph.num_points);
    node_list.reserve(static_cast<size_t>(max_nodes));

    while (node_list.size() < max_nodes)
    {
        const uint64_t page_cap = nnodes_per_sector;
        std::vector<uint32_t> page_nodes;
        page_nodes.reserve(page_cap);

        uint32_t seed = invalid;
        for (uint32_t node : nodes_by_expand)
        {
            if (!placed[node] && node_heat(node_expand, node) > 0)
            {
                seed = node;
                break;
            }
        }
        if (seed == invalid)
            break;

        page_nodes.push_back(seed);
        placed[seed] = true;

        while (page_nodes.size() < page_cap && node_list.size() + page_nodes.size() < max_nodes)
        {
            std::vector<int32_t> dist(graph.num_points, -1);
            std::queue<uint32_t> q;
            for (uint32_t src : page_nodes)
            {
                if (dist[src] != -1)
                    continue;
                dist[src] = 0;
                q.push(src);
            }
            while (!q.empty())
            {
                const uint32_t u = q.front();
                q.pop();
                if (static_cast<uint32_t>(dist[u]) >= k_hops)
                    continue;
                for (uint32_t v : undirected_adj[u])
                {
                    if (dist[v] != -1)
                        continue;
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }

            uint32_t best = invalid;
            uint64_t best_h = 0;
            for (uint64_t i = 0; i < graph.num_points; i++)
            {
                if (placed[i] || dist[i] == -1)
                    continue;
                const uint64_t h = node_heat(node_expand, static_cast<uint32_t>(i));
                if (h == 0)
                    continue;
                if (h > best_h || (h == best_h && i < best))
                {
                    best_h = h;
                    best = static_cast<uint32_t>(i);
                }
            }
            if (best == invalid)
                break;

            page_nodes.push_back(best);
            placed[best] = true;
        }

        for (uint32_t node : page_nodes)
        {
            if (exclude_ids.find(node) != exclude_ids.end())
                continue;
            if (node_heat(node_expand, node) == 0)
                continue;
            node_list.push_back(node);
            if (node_list.size() >= max_nodes)
                break;
        }
    }

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: hot-node k-hop list empty (max_nodes=" << max_nodes << ", k_hops="
                      << k_hops << ")." << std::endl;
        return -1;
    }
    return 0;
}

int save_relayout_order(const std::string &path, const std::vector<uint32_t> &order, uint64_t nnodes_per_sector)
{
    std::vector<uint64_t> meta = {order.size(), nnodes_per_sector};
    diskann::save_bin<uint64_t>(path, meta.data(), meta.size(), 1);
    diskann::save_bin<uint32_t>(path + ".nodes", const_cast<uint32_t *>(order.data()), order.size(), 1);
    return 0;
}

int load_relayout_order(const std::string &path, std::vector<uint32_t> &order, uint64_t &nnodes_per_sector)
{
    size_t nmeta = 0, nd = 0;
    uint64_t *meta = nullptr;
    diskann::load_bin<uint64_t>(path, meta, nmeta, nd);
    if (nmeta < 2)
    {
        delete[] meta;
        return -1;
    }
    const uint64_t num_points = meta[0];
    nnodes_per_sector = meta[1];
    delete[] meta;

    uint32_t *nodes = nullptr;
    size_t npts = 0, dim = 0;
    diskann::load_bin<uint32_t>(path + ".nodes", nodes, npts, dim);
    if (npts != num_points)
    {
        delete[] nodes;
        return -1;
    }
    order.assign(nodes, nodes + npts);
    delete[] nodes;
    return 0;
}

int read_disk_index_nnodes_per_sector(const std::string &disk_index_file, uint64_t &nnodes_per_sector)
{
    std::ifstream in(disk_index_file, std::ios::binary);
    if (!in.is_open())
        return -1;
    uint32_t nr = 0, nc = 0;
    uint64_t npts = 0, ndims = 0, medoid = 0, max_node_len = 0;
    in.read((char *)&nr, sizeof(uint32_t));
    in.read((char *)&nc, sizeof(uint32_t));
    in.read((char *)&npts, sizeof(uint64_t));
    in.read((char *)&ndims, sizeof(uint64_t));
    in.read((char *)&medoid, sizeof(uint64_t));
    in.read((char *)&max_node_len, sizeof(uint64_t));
    in.read((char *)&nnodes_per_sector, sizeof(uint64_t));
    if (!in)
        return -1;
    return 0;
}

} // namespace diskann
