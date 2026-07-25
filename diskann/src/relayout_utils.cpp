// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "relayout_utils.h"
#include "utils.h"

#include <algorithm>
#include <deque>
#include <functional>
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

static std::vector<int32_t> bfs_distances_limited(const std::vector<std::vector<uint32_t>> &adj, uint32_t src,
                                                  uint32_t max_hops)
{
    std::vector<int32_t> dist(adj.size(), -1);
    if (src >= adj.size())
        return dist;
    std::queue<uint32_t> q;
    dist[src] = 0;
    q.push(src);
    while (!q.empty())
    {
        const uint32_t u = q.front();
        q.pop();
        if (static_cast<uint32_t>(dist[u]) >= max_hops)
            continue;
        for (uint32_t v : adj[u])
        {
            if (v >= adj.size() || dist[v] != -1)
                continue;
            dist[v] = dist[u] + 1;
            q.push(v);
        }
    }
    return dist;
}

static inline uint64_t edge_weight(const std::unordered_map<uint64_t, uint64_t> &weights, uint32_t u, uint32_t v)
{
    const auto it = weights.find(undirected_edge_key(u, v));
    return (it == weights.end()) ? 0 : it->second;
}

static bool graph_has_edge(const std::vector<std::vector<uint32_t>> &adj, uint32_t u, uint32_t v)
{
    if (u >= adj.size())
        return false;
    for (uint32_t nbr : adj[u])
    {
        if (nbr == v)
            return true;
    }
    return false;
}

std::string normalize_disk_cache_layout(std::string layout)
{
    for (char &c : layout)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (layout == "khop" || layout == "hotnode" || layout == "hot-node")
        layout = "node";
    if (layout == "c")
        layout = "edge";
    return layout;
}

static double path_average_weight(const std::vector<uint32_t> &path,
                                  const std::unordered_map<uint64_t, uint64_t> &weights)
{
    if (path.size() < 2)
        return 0.0;
    uint64_t sum = 0;
    size_t counted_edges = 0;
    for (size_t i = 1; i < path.size(); i++)
    {
        const uint64_t w = edge_weight(weights, path[i - 1], path[i]);
        if (w == 0)
            continue;
        sum += w;
        counted_edges++;
    }
    if (counted_edges == 0)
        return 0.0;
    return static_cast<double>(sum) / static_cast<double>(counted_edges);
}

static bool path_has_target_edge(const std::vector<uint32_t> &path, uint32_t ta, uint32_t tb)
{
    for (size_t i = 1; i < path.size(); i++)
    {
        if ((path[i - 1] == ta && path[i] == tb) || (path[i - 1] == tb && path[i] == ta))
            return true;
    }
    return false;
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

static void try_candidate_path(const std::vector<uint32_t> &path, uint32_t ta, uint32_t tb,
                               const std::unordered_set<uint64_t> &used_page_edges,
                               const std::unordered_map<uint64_t, uint64_t> &weights, double &best_avg,
                               std::vector<uint32_t> &best_path)
{
    if (path.size() < 2)
        return;
    if (!path_has_target_edge(path, ta, tb))
        return;
    if (!path_edges_disjoint(path, used_page_edges))
        return;
    const double avg = path_average_weight(path, weights);
    if (avg <= 0.0)
        return;
    if (avg > best_avg)
    {
        best_avg = avg;
        best_path = path;
    }
}

static void profile_bfs_distances_limited(
    const std::vector<std::vector<std::pair<uint32_t, uint64_t>>> &profile_edges_by_node, uint32_t src,
    uint32_t max_hops, std::unordered_map<uint32_t, uint32_t> &dist)
{
    dist.clear();
    if (src >= profile_edges_by_node.size())
        return;
    std::queue<uint32_t> q;
    dist[src] = 0;
    q.push(src);
    while (!q.empty())
    {
        const uint32_t u = q.front();
        q.pop();
        if (dist[u] >= max_hops)
            continue;
        for (const auto &nbr_edge : profile_edges_by_node[u])
        {
            const uint32_t v = nbr_edge.first;
            if (dist.find(v) != dist.end())
                continue;
            dist[v] = dist[u] + 1;
            q.push(v);
        }
    }
}

static bool edge_in_khop_neighborhood(uint32_t a, uint32_t b, const std::vector<int32_t> &dist_u,
                                      const std::vector<int32_t> &dist_v, uint32_t k_hops)
{
    auto endpoint_dist = [&](uint32_t x) -> int32_t {
        const int32_t du = (x < dist_u.size()) ? dist_u[x] : -1;
        const int32_t dv = (x < dist_v.size()) ? dist_v[x] : -1;
        if (du < 0)
            return dv;
        if (dv < 0)
            return du;
        return std::min(du, dv);
    };
    const int32_t da = endpoint_dist(a);
    const int32_t db = endpoint_dist(b);
    if (da < 0 && db < 0)
        return false;
    int32_t best = -1;
    if (da >= 0)
        best = da;
    if (db >= 0 && (best < 0 || db < best))
        best = db;
    return static_cast<uint32_t>(best) <= k_hops;
}

static bool edge_in_khop_neighborhood(uint32_t a, uint32_t b, const std::unordered_map<uint32_t, uint32_t> &dist_u,
                                      const std::unordered_map<uint32_t, uint32_t> &dist_v, uint32_t k_hops)
{
    auto endpoint_dist = [&](uint32_t x) -> uint32_t {
        const auto iu = dist_u.find(x);
        const auto iv = dist_v.find(x);
        const bool hu = iu != dist_u.end();
        const bool hv = iv != dist_v.end();
        if (!hu)
            return hv ? iv->second : static_cast<uint32_t>(k_hops + 1);
        if (!hv)
            return iu->second;
        return std::min(iu->second, iv->second);
    };
    return std::min(endpoint_dist(a), endpoint_dist(b)) <= k_hops;
}

static std::unordered_set<uint64_t> collect_khop_graph_edges(
    uint32_t k_hops, const std::vector<int32_t> &dist_u, const std::vector<int32_t> &dist_v,
    const std::vector<std::vector<uint32_t>> &adj)
{
    std::unordered_set<uint32_t> near_nodes;
    for (uint32_t i = 0; i < dist_u.size(); i++)
    {
        if (dist_u[i] >= 0 && static_cast<uint32_t>(dist_u[i]) <= k_hops)
            near_nodes.insert(i);
        if (dist_v[i] >= 0 && static_cast<uint32_t>(dist_v[i]) <= k_hops)
            near_nodes.insert(i);
    }

    std::unordered_set<uint64_t> neighborhood;
    for (uint32_t node : near_nodes)
    {
        if (node >= adj.size())
            continue;
        for (uint32_t nbr : adj[node])
        {
            if (!edge_in_khop_neighborhood(node, nbr, dist_u, dist_v, k_hops))
                continue;
            neighborhood.insert(undirected_edge_key(node, nbr));
        }
    }
    return neighborhood;
}

static std::unordered_set<uint64_t> collect_khop_profile_edges(
    uint32_t seed_u, uint32_t seed_v, uint32_t k_hops, const std::unordered_set<uint64_t> &edge_in_list,
    const std::unordered_map<uint32_t, uint32_t> &dist_u, const std::unordered_map<uint32_t, uint32_t> &dist_v,
    const std::vector<std::vector<std::pair<uint32_t, uint64_t>>> &profile_edges_by_node)
{
    std::unordered_set<uint32_t> near_nodes;
    for (const auto &kv : dist_u)
    {
        if (kv.second <= k_hops)
            near_nodes.insert(kv.first);
    }
    for (const auto &kv : dist_v)
    {
        if (kv.second <= k_hops)
            near_nodes.insert(kv.first);
    }

    std::unordered_set<uint64_t> neighborhood;
    for (uint32_t node : near_nodes)
    {
        if (node >= profile_edges_by_node.size())
            continue;
        for (const auto &nbr_edge : profile_edges_by_node[node])
        {
            const uint64_t key = nbr_edge.second;
            if (edge_in_list.count(key) == 0)
                continue;
            const uint32_t nbr = nbr_edge.first;
            if (edge_in_khop_neighborhood(node, nbr, dist_u, dist_v, k_hops))
                neighborhood.insert(key);
        }
    }
    return neighborhood;
}

static void find_best_path_to_edge_k2(uint32_t start, uint32_t ta, uint32_t tb,
                                      const std::vector<std::vector<uint32_t>> &adj,
                                      const std::unordered_map<uint64_t, uint64_t> &weights,
                                      const std::unordered_set<uint64_t> &used_page_edges, double &best_avg,
                                      std::vector<uint32_t> &best_path)
{
    if (start >= adj.size())
        return;

    const auto edge_used = [&](uint32_t u, uint32_t v) -> bool {
        return used_page_edges.count(undirected_edge_key(u, v)) > 0;
    };

    if (start == ta && graph_has_edge(adj, ta, tb) && !edge_used(ta, tb))
        try_candidate_path({ta, tb}, ta, tb, used_page_edges, weights, best_avg, best_path);
    if (start == tb && graph_has_edge(adj, tb, ta) && !edge_used(ta, tb))
        try_candidate_path({tb, ta}, ta, tb, used_page_edges, weights, best_avg, best_path);

    for (uint32_t n1 : adj[start])
    {
        if (edge_used(start, n1))
            continue;

        if (n1 == ta && graph_has_edge(adj, ta, tb) && !edge_used(ta, tb))
            try_candidate_path({start, ta, tb}, ta, tb, used_page_edges, weights, best_avg, best_path);
        if (n1 == tb && graph_has_edge(adj, ta, tb) && !edge_used(ta, tb))
            try_candidate_path({start, tb, ta}, ta, tb, used_page_edges, weights, best_avg, best_path);

        if (n1 >= adj.size())
            continue;
        for (uint32_t n2 : adj[n1])
        {
            if (n2 == start || edge_used(n1, n2))
                continue;

            if (n1 == ta && n2 == tb)
                try_candidate_path({start, ta, tb}, ta, tb, used_page_edges, weights, best_avg, best_path);
            if (n1 == tb && n2 == ta)
                try_candidate_path({start, tb, ta}, ta, tb, used_page_edges, weights, best_avg, best_path);

            if (n2 == ta && graph_has_edge(adj, ta, tb) && !edge_used(ta, tb))
                try_candidate_path({start, n1, ta, tb}, ta, tb, used_page_edges, weights, best_avg, best_path);
            if (n2 == tb && graph_has_edge(adj, ta, tb) && !edge_used(ta, tb))
                try_candidate_path({start, n1, tb, ta}, ta, tb, used_page_edges, weights, best_avg, best_path);
        }
    }
}

static void find_best_path_to_edge(uint32_t start, uint32_t ta, uint32_t tb, const std::vector<std::vector<uint32_t>> &adj,
                                   const std::unordered_map<uint64_t, uint64_t> &weights,
                                   const std::unordered_set<uint64_t> &used_page_edges, uint32_t max_hops,
                                   double &best_avg, std::vector<uint32_t> &best_path)
{
    if (start >= adj.size() || max_hops == 0)
        return;

    if (max_hops <= 2)
    {
        find_best_path_to_edge_k2(start, ta, tb, adj, weights, used_page_edges, best_avg, best_path);
        return;
    }

    struct PathState
    {
        uint32_t node;
        std::vector<uint32_t> nodes;
        size_t num_edges;
    };
    std::deque<PathState> q;
    q.push_back({start, {start}, 0});

    while (!q.empty())
    {
        PathState cur = std::move(q.front());
        q.pop_front();
        if (cur.num_edges >= max_hops)
            continue;

        try_candidate_path(cur.nodes, ta, tb, used_page_edges, weights, best_avg, best_path);

        if (cur.node >= adj.size())
            continue;
        for (uint32_t nbr : adj[cur.node])
        {
            const uint64_t ek = undirected_edge_key(cur.node, nbr);
            if (used_page_edges.count(ek) > 0)
                continue;

            bool on_path = false;
            for (uint32_t n : cur.nodes)
            {
                if (n == nbr)
                {
                    on_path = true;
                    break;
                }
            }
            if (on_path)
                continue;

            PathState next;
            next.node = nbr;
            next.nodes = cur.nodes;
            next.nodes.push_back(nbr);
            next.num_edges = cur.num_edges + 1;
            try_candidate_path(next.nodes, ta, tb, used_page_edges, weights, best_avg, best_path);
            q.push_back(std::move(next));
        }
    }
}

static bool path_has_target_node(const std::vector<uint32_t> &path, uint32_t target)
{
    for (uint32_t n : path)
    {
        if (n == target)
            return true;
    }
    return false;
}

static void try_candidate_path_to_node(const std::vector<uint32_t> &path, uint32_t target,
                                       const std::unordered_set<uint64_t> &used_page_edges,
                                       const std::unordered_map<uint64_t, uint64_t> &weights, double &best_avg,
                                       std::vector<uint32_t> &best_path)
{
    if (path.empty())
        return;
    if (!path_has_target_node(path, target))
        return;
    if (path.size() >= 2 && !path_edges_disjoint(path, used_page_edges))
        return;
    const double avg = path_average_weight(path, weights);
    if (avg <= 0.0)
        return;
    if (avg > best_avg)
    {
        best_avg = avg;
        best_path = path;
    }
}

static void find_best_path_to_node_k2(uint32_t start, uint32_t target, const std::vector<std::vector<uint32_t>> &adj,
                                      const std::unordered_map<uint64_t, uint64_t> &weights,
                                      const std::unordered_set<uint64_t> &used_page_edges, double &best_avg,
                                      std::vector<uint32_t> &best_path)
{
    if (start >= adj.size() || target >= adj.size())
        return;

    const auto edge_ok = [&](uint32_t u, uint32_t v) -> bool {
        return graph_has_edge(adj, u, v) && used_page_edges.count(undirected_edge_key(u, v)) == 0;
    };

    if (start == target)
        try_candidate_path_to_node({target}, target, used_page_edges, weights, best_avg, best_path);

    if (edge_ok(start, target))
        try_candidate_path_to_node({start, target}, target, used_page_edges, weights, best_avg, best_path);

    for (uint32_t n1 : adj[start])
    {
        if (used_page_edges.count(undirected_edge_key(start, n1)) > 0)
            continue;
        if (n1 == target)
            continue;
        if (edge_ok(n1, target))
            try_candidate_path_to_node({start, n1, target}, target, used_page_edges, weights, best_avg, best_path);
    }
}

static void find_best_path_from_seed_limited(uint32_t seed, const std::vector<std::vector<uint32_t>> &adj,
                                             const std::unordered_map<uint64_t, uint64_t> &weights,
                                             const std::unordered_set<uint64_t> &used_page_edges, uint32_t max_hops,
                                             const std::function<bool(uint32_t)> &target_eligible, double &best_avg,
                                             std::vector<uint32_t> &best_path, uint32_t &best_target)
{
    best_path.clear();
    best_target = static_cast<uint32_t>(adj.size());
    best_avg = -1.0;
    if (seed >= adj.size() || max_hops == 0)
        return;

    const auto edge_ok = [&](uint32_t u, uint32_t v) -> bool {
        return graph_has_edge(adj, u, v) && used_page_edges.count(undirected_edge_key(u, v)) == 0;
    };

    if (max_hops <= 2)
    {
        for (uint32_t n1 : adj[seed])
        {
            if (used_page_edges.count(undirected_edge_key(seed, n1)) > 0)
                continue;
            if (target_eligible(n1))
            {
                std::vector<uint32_t> candidate = {seed, n1};
                double candidate_avg = -1.0;
                std::vector<uint32_t> trial;
                try_candidate_path_to_node(candidate, n1, used_page_edges, weights, candidate_avg, trial);
                if (!trial.empty() && candidate_avg > best_avg)
                {
                    best_avg = candidate_avg;
                    best_path = std::move(trial);
                    best_target = n1;
                }
            }
            if (n1 >= adj.size())
                continue;
            for (uint32_t n2 : adj[n1])
            {
                if (n2 == seed || used_page_edges.count(undirected_edge_key(n1, n2)) > 0)
                    continue;
                if (!target_eligible(n2))
                    continue;
                if (!edge_ok(seed, n1))
                    continue;
                std::vector<uint32_t> candidate = {seed, n1, n2};
                double candidate_avg = -1.0;
                std::vector<uint32_t> trial;
                try_candidate_path_to_node(candidate, n2, used_page_edges, weights, candidate_avg, trial);
                if (!trial.empty() && candidate_avg > best_avg)
                {
                    best_avg = candidate_avg;
                    best_path = std::move(trial);
                    best_target = n2;
                }
            }
        }
        return;
    }

    std::function<void(uint32_t, std::vector<uint32_t> &, size_t)> dfs = [&](uint32_t cur, std::vector<uint32_t> &path,
                                                                             size_t depth) {
        if (depth > max_hops)
            return;
        if (path.size() >= 2)
        {
            const uint32_t end = path.back();
            if (target_eligible(end))
            {
                double candidate_avg = -1.0;
                std::vector<uint32_t> trial;
                try_candidate_path_to_node(path, end, used_page_edges, weights, candidate_avg, trial);
                if (!trial.empty() && candidate_avg > best_avg)
                {
                    best_avg = candidate_avg;
                    best_path = trial;
                    best_target = end;
                }
            }
        }
        if (depth == max_hops || cur >= adj.size())
            return;
        for (uint32_t nbr : adj[cur])
        {
            if (used_page_edges.count(undirected_edge_key(cur, nbr)) > 0)
                continue;
            bool on_path = false;
            for (uint32_t n : path)
            {
                if (n == nbr)
                {
                    on_path = true;
                    break;
                }
            }
            if (on_path)
                continue;
            path.push_back(nbr);
            dfs(nbr, path, depth + 1);
            path.pop_back();
        }
    };
    std::vector<uint32_t> path = {seed};
    dfs(seed, path, 0);
}

static void try_path_profile_edge_targets(const std::vector<uint32_t> &path,
                                          const std::unordered_set<uint64_t> &profile_edge_in_list,
                                          const std::unordered_set<uint64_t> &used_page_edges,
                                          const std::unordered_map<uint64_t, uint64_t> &weights, double &best_avg,
                                          std::vector<uint32_t> &best_path, uint64_t &best_target_key)
{
    if (path.size() < 2)
        return;
    for (size_t i = 1; i < path.size(); i++)
    {
        const uint32_t ta = path[i - 1];
        const uint32_t tb = path[i];
        const uint64_t key = undirected_edge_key(ta, tb);
        if (profile_edge_in_list.count(key) == 0 || used_page_edges.count(key) > 0)
            continue;
        std::vector<uint32_t> trial;
        double candidate_avg = -1.0;
        try_candidate_path(path, ta, tb, used_page_edges, weights, candidate_avg, trial);
        if (!trial.empty() && candidate_avg > best_avg)
        {
            best_avg = candidate_avg;
            best_path = std::move(trial);
            best_target_key = key;
        }
    }
}

static void dfs_edge_paths_from_seed(uint32_t cur, std::vector<uint32_t> &path, size_t depth, uint32_t max_hops,
                                     const std::vector<std::vector<uint32_t>> &adj,
                                     const std::unordered_set<uint64_t> &profile_edge_in_list,
                                     const std::unordered_set<uint64_t> &used_page_edges,
                                     const std::unordered_map<uint64_t, uint64_t> &weights, double &best_avg,
                                     std::vector<uint32_t> &best_path, uint64_t &best_target_key)
{
    if (depth > max_hops || cur >= adj.size())
        return;

    try_path_profile_edge_targets(path, profile_edge_in_list, used_page_edges, weights, best_avg, best_path,
                                  best_target_key);

    if (depth == max_hops)
        return;

    for (uint32_t nbr : adj[cur])
    {
        const uint64_t ek = undirected_edge_key(cur, nbr);
        if (used_page_edges.count(ek) > 0)
            continue;
        bool on_path = false;
        for (uint32_t n : path)
        {
            if (n == nbr)
            {
                on_path = true;
                break;
            }
        }
        if (on_path)
            continue;
        path.push_back(nbr);
        dfs_edge_paths_from_seed(nbr, path, depth + 1, max_hops, adj, profile_edge_in_list, used_page_edges, weights,
                                 best_avg, best_path, best_target_key);
        path.pop_back();
    }
}

static void find_best_path_from_seeds_edge_limited(
    uint32_t seed_u, uint32_t seed_v, const std::vector<std::vector<uint32_t>> &adj,
    const std::unordered_map<uint64_t, uint64_t> &weights, const std::unordered_set<uint64_t> &used_page_edges,
    uint32_t max_hops, const std::unordered_set<uint64_t> &profile_edge_in_list, double &best_avg,
    std::vector<uint32_t> &best_path, uint64_t &best_target_key)
{
    best_path.clear();
    best_target_key = 0;
    best_avg = -1.0;
    if (max_hops == 0)
        return;

    for (uint32_t start : {seed_u, seed_v})
    {
        if (start >= adj.size())
            continue;
        std::vector<uint32_t> path = {start};
        dfs_edge_paths_from_seed(start, path, 0, max_hops, adj, profile_edge_in_list, used_page_edges, weights,
                                 best_avg, best_path, best_target_key);
    }
}

static void find_best_path_to_node(uint32_t start, uint32_t target, const std::vector<std::vector<uint32_t>> &adj,
                                   const std::unordered_map<uint64_t, uint64_t> &weights,
                                   const std::unordered_set<uint64_t> &used_page_edges, uint32_t max_hops,
                                   double &best_avg, std::vector<uint32_t> &best_path)
{
    if (start >= adj.size() || target >= adj.size() || max_hops == 0)
        return;

    if (max_hops <= 2)
    {
        find_best_path_to_node_k2(start, target, adj, weights, used_page_edges, best_avg, best_path);
        return;
    }

    struct PathState
    {
        uint32_t node;
        std::vector<uint32_t> nodes;
        size_t num_edges;
    };
    std::deque<PathState> q;
    q.push_back({start, {start}, 0});

    while (!q.empty())
    {
        PathState cur = std::move(q.front());
        q.pop_front();
        if (cur.num_edges >= max_hops)
            continue;

        try_candidate_path_to_node(cur.nodes, target, used_page_edges, weights, best_avg, best_path);

        if (cur.node >= adj.size())
            continue;
        for (uint32_t nbr : adj[cur.node])
        {
            const uint64_t ek = undirected_edge_key(cur.node, nbr);
            if (used_page_edges.count(ek) > 0)
                continue;

            bool on_path = false;
            for (uint32_t n : cur.nodes)
            {
                if (n == nbr)
                {
                    on_path = true;
                    break;
                }
            }
            if (on_path)
                continue;

            PathState next;
            next.node = nbr;
            next.nodes = cur.nodes;
            next.nodes.push_back(nbr);
            next.num_edges = cur.num_edges + 1;
            try_candidate_path_to_node(next.nodes, target, used_page_edges, weights, best_avg, best_path);
            q.push_back(std::move(next));
        }
    }
}

static std::unordered_set<uint32_t> collect_node_khop_neighborhood(uint32_t seed, uint32_t k_hops,
                                                                   const std::vector<int32_t> &dist,
                                                                   const std::function<bool(uint32_t)> &eligible_fn)
{
    std::unordered_set<uint32_t> neighborhood;
    for (uint32_t i = 0; i < dist.size(); i++)
    {
        if (dist[i] < 0 || static_cast<uint32_t>(dist[i]) > k_hops)
            continue;
        if (i == seed)
            continue;
        if (!eligible_fn(i))
            continue;
        neighborhood.insert(i);
    }
    return neighborhood;
}

static bool is_node_pending_pack(const std::vector<bool> &node_in_list, uint32_t node)
{
    return node < node_in_list.size() && node_in_list[node];
}

static void remove_path_from_lists(const std::vector<uint32_t> &path, std::unordered_set<uint64_t> &edge_in_list,
                                   std::vector<bool> &node_in_list)
{
    for (size_t i = 1; i < path.size(); i++)
    {
        const uint64_t key = undirected_edge_key(path[i - 1], path[i]);
        if (edge_in_list.count(key) > 0)
            edge_in_list.erase(key);
    }
    for (uint32_t node : path)
    {
        if (node < node_in_list.size() && node_in_list[node])
            node_in_list[node] = false;
    }
}

struct PagePackConfig
{
    uint64_t page_cap = 0;
    uint64_t max_output_nodes = 0; // 0 = full relayout; >0 = disk sidecar cap (e.g. 10%)
    const std::unordered_set<uint32_t> *skip_output = nullptr;
    const std::vector<uint64_t> *node_expand = nullptr;
    std::vector<uint32_t> *output_nodes = nullptr; // disk sidecar: collect nodes here directly
};

static bool disk_cache_node_eligible(uint32_t node, const PagePackConfig &cfg)
{
    if (cfg.skip_output != nullptr && cfg.skip_output->count(node) > 0)
        return false;
    if (cfg.node_expand != nullptr && node_heat(*cfg.node_expand, node) == 0)
        return false;
    return true;
}

static bool try_append_unique_disk_cache_node(uint32_t node, const PagePackConfig &cfg, uint64_t &output_count,
                                    std::unordered_set<uint32_t> &output_seen)
{
    if (cfg.output_nodes == nullptr)
        return false;
    if (!disk_cache_node_eligible(node, cfg))
        return false;
    if (output_count >= cfg.max_output_nodes)
        return false;
    if (output_seen.count(node) > 0)
        return false;
    cfg.output_nodes->push_back(node);
    output_seen.insert(node);
    output_count++;
    return true;
}

static void append_page_nodes_to_disk_cache_list(const std::vector<uint32_t> &page_nodes, const PagePackConfig &cfg,
                                      uint64_t &output_count, std::unordered_set<uint32_t> &output_seen)
{
    if (cfg.output_nodes == nullptr)
        return;
    for (uint32_t node : page_nodes)
    {
        if (output_count >= cfg.max_output_nodes)
            break;
        try_append_unique_disk_cache_node(node, cfg, output_count, output_seen);
    }
}

static void log_disk_cache_node_list_stats(const std::vector<uint32_t> &node_list, const char *layout_name)
{
    const std::unordered_set<uint32_t> uniq(node_list.begin(), node_list.end());
    diskann::cout << "MERIT disk-cache node list stats (" << layout_name << "): entries=" << node_list.size()
                  << " unique=" << uniq.size();
    if (uniq.size() != node_list.size())
        diskann::cout << " duplicates=" << (node_list.size() - uniq.size());
    diskann::cout << std::endl;
}

static void init_pending_pack_nodes(uint32_t num_points, const PagePackConfig &cfg, std::vector<bool> &node_in_list)
{
    node_in_list.assign(num_points, false);
    for (uint32_t i = 0; i < num_points; i++)
    {
        if (cfg.node_expand != nullptr && node_heat(*cfg.node_expand, i) == 0)
            continue;
        if (cfg.skip_output != nullptr && cfg.skip_output->count(i) > 0)
            continue;
        node_in_list[i] = true;
    }
}

static void init_pending_profile_edges(const std::unordered_map<uint64_t, uint64_t> &weights,
                                            std::unordered_set<uint64_t> &profile_edge_in_list)
{
    profile_edge_in_list.clear();
    profile_edge_in_list.reserve(weights.size());
    for (const auto &kv : weights)
    {
        if (kv.second == 0)
            continue;
        profile_edge_in_list.insert(kv.first);
    }
}

static void place_node_on_page(std::vector<uint32_t> &page_nodes, std::unordered_set<uint32_t> &on_page,
                                uint64_t page_cap, uint32_t node, const PagePackConfig *cfg = nullptr,
                                const std::unordered_set<uint32_t> *output_seen = nullptr)
{
    if (cfg != nullptr && !disk_cache_node_eligible(node, *cfg))
        return;
    if (output_seen != nullptr && output_seen->count(node) > 0)
        return;
    if (page_nodes.size() >= page_cap || on_page.count(node) > 0)
        return;
    page_nodes.push_back(node);
    on_page.insert(node);
}

static void place_path_on_page(std::vector<uint32_t> &page_nodes, std::unordered_set<uint32_t> &on_page,
                                     uint64_t page_cap, const std::vector<uint32_t> &path, const PagePackConfig *cfg,
                                     const std::unordered_set<uint32_t> *output_seen = nullptr)
{
    for (uint32_t node : path)
    {
        if (page_nodes.size() >= page_cap)
            break;
        place_node_on_page(page_nodes, on_page, page_cap, node, cfg, output_seen);
    }
}

static bool has_pending_pack_nodes(const std::vector<bool> &node_in_list)
{
    for (bool in : node_in_list)
    {
        if (in)
            return true;
    }
    return false;
}

static bool pick_highest_weight_seed_edge(const std::vector<std::pair<uint32_t, uint32_t>> &sorted_edges,
                                 const std::unordered_set<uint64_t> &edge_in_list, size_t &edge_cursor,
                                 uint32_t &seed_u, uint32_t &seed_v, uint64_t &seed_key,
                                 const std::vector<bool> *node_in_list = nullptr)
{
    while (edge_cursor < sorted_edges.size())
    {
        const auto &edge = sorted_edges[edge_cursor++];
        const uint64_t key = undirected_edge_key(edge.first, edge.second);
        if (edge_in_list.count(key) == 0)
            continue;
        if (node_in_list != nullptr)
        {
            const bool u_ok = is_node_pending_pack(*node_in_list, edge.first);
            const bool v_ok = is_node_pending_pack(*node_in_list, edge.second);
            if (!u_ok && !v_ok)
                continue;
        }
        seed_u = edge.first;
        seed_v = edge.second;
        seed_key = key;
        return true;
    }
    return false;
}

static std::vector<std::vector<std::pair<uint32_t, uint64_t>>> build_profile_edges_by_node(
    uint64_t num_points, const std::unordered_map<uint64_t, uint64_t> &weights)
{
    std::vector<std::vector<std::pair<uint32_t, uint64_t>>> by_node(num_points);
    for (const auto &kv : weights)
    {
        const uint32_t u = static_cast<uint32_t>(kv.first >> 32);
        const uint32_t v = static_cast<uint32_t>(kv.first & 0xFFFFFFFFu);
        if (u >= num_points || v >= num_points)
            continue;
        by_node[u].emplace_back(v, kv.first);
        by_node[v].emplace_back(u, kv.first);
    }
    return by_node;
}

static void remove_path_edges_from_set(const std::vector<uint32_t> &path,
                                                std::unordered_set<uint64_t> &neighborhood)
{
    for (size_t i = 1; i < path.size(); i++)
        neighborhood.erase(undirected_edge_key(path[i - 1], path[i]));
}

static int pack_pages_by_edge_importance(const VamanaGraph &graph, const std::unordered_map<uint64_t, uint64_t> &weights,
                            const std::vector<std::pair<uint32_t, uint32_t>> &sorted_edges, uint32_t k_hops,
                            const PagePackConfig &cfg, std::vector<std::vector<uint32_t>> &pages,
                            std::vector<bool> &assigned)
{
    if (graph.num_points == 0 || cfg.page_cap == 0)
        return -1;

    const auto adj = build_undirected_adjacency(graph);
    size_t edge_cursor = 0;
    std::unordered_set<uint64_t> profile_edge_in_list;
    std::vector<bool> node_in_list;
    const bool disk_sidecar = (cfg.max_output_nodes > 0);
    init_pending_profile_edges(weights, profile_edge_in_list);
    init_pending_pack_nodes(static_cast<uint32_t>(graph.num_points), cfg, node_in_list);
    assigned.assign(graph.num_points, false);

    std::vector<uint32_t> nodes_by_expand(static_cast<size_t>(graph.num_points));
    std::iota(nodes_by_expand.begin(), nodes_by_expand.end(), 0);
    if (cfg.node_expand != nullptr)
    {
        std::sort(nodes_by_expand.begin(), nodes_by_expand.end(), [&](uint32_t a, uint32_t b) {
            const uint64_t ca = node_heat(*cfg.node_expand, a);
            const uint64_t cb = node_heat(*cfg.node_expand, b);
            if (ca != cb)
                return ca > cb;
            return a < b;
        });
    }

    auto append_page = [&](const std::vector<uint32_t> &page_nodes) {
        if (page_nodes.empty())
            return;
        pages.push_back(page_nodes);
        for (uint32_t node : page_nodes)
            assigned[node] = true;
    };

    uint64_t output_count = 0;
    std::unordered_set<uint32_t> output_seen;
    if (disk_sidecar)
        output_seen.reserve(static_cast<size_t>(cfg.max_output_nodes));
    const std::unordered_set<uint32_t> *seen_ptr = disk_sidecar ? &output_seen : nullptr;

    auto fill_page_from_node_list = [&](std::vector<uint32_t> &page_nodes, std::unordered_set<uint32_t> &on_page) {
        for (uint32_t node : nodes_by_expand)
        {
            if (page_nodes.size() >= cfg.page_cap)
                break;
            if (!is_node_pending_pack(node_in_list, node))
                continue;
            place_node_on_page(page_nodes, on_page, cfg.page_cap, node, &cfg, seen_ptr);
            node_in_list[node] = false;
        }
    };

    while (true)
    {
        if (disk_sidecar && output_count >= cfg.max_output_nodes)
            break;
        if (!disk_sidecar && !has_pending_pack_nodes(node_in_list))
            break;

        uint32_t seed_u = 0, seed_v = 0;
        uint64_t seed_key = 0;
        const std::vector<bool> *pick_node_list = disk_sidecar ? &node_in_list : nullptr;
        if (!pick_highest_weight_seed_edge(sorted_edges, profile_edge_in_list, edge_cursor, seed_u, seed_v, seed_key,
                                  pick_node_list))
        {
            if (disk_sidecar)
                break;
            std::vector<uint32_t> page_nodes;
            std::unordered_set<uint32_t> on_page;
            fill_page_from_node_list(page_nodes, on_page);
            append_page(page_nodes);
            continue;
        }

        std::vector<uint32_t> page_nodes;
        std::unordered_set<uint32_t> on_page;
        page_nodes.reserve(static_cast<size_t>(cfg.page_cap));

        // (b) seed edge: place counted endpoints only, remove edge + endpoints from lists.
        place_node_on_page(page_nodes, on_page, cfg.page_cap, seed_u, &cfg, seen_ptr);
        place_node_on_page(page_nodes, on_page, cfg.page_cap, seed_v, &cfg, seen_ptr);
        profile_edge_in_list.erase(seed_key);
        if (seed_u < node_in_list.size())
            node_in_list[seed_u] = false;
        if (seed_v < node_in_list.size())
            node_in_list[seed_v] = false;

        std::unordered_set<uint64_t> used_page_edges;
        used_page_edges.insert(seed_key);

        while (page_nodes.size() < cfg.page_cap)
        {
            if (disk_sidecar && output_count >= cfg.max_output_nodes)
                break;

            std::vector<uint32_t> best_path;
            uint64_t best_target_key = 0;
            double best_avg = -1.0;

            find_best_path_from_seeds_edge_limited(seed_u, seed_v, adj, weights, used_page_edges, k_hops,
                                                   profile_edge_in_list, best_avg, best_path, best_target_key);

            if (best_path.empty())
                break;

            for (uint32_t node : best_path)
            {
                if (page_nodes.size() >= cfg.page_cap)
                    break;
                if (is_node_pending_pack(node_in_list, node))
                    place_node_on_page(page_nodes, on_page, cfg.page_cap, node, &cfg, seen_ptr);
            }

            remove_path_from_lists(best_path, profile_edge_in_list, node_in_list);
            for (size_t i = 1; i < best_path.size(); i++)
                used_page_edges.insert(undirected_edge_key(best_path[i - 1], best_path[i]));
        }

        append_page(page_nodes);
        append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
    }

    if (disk_sidecar && output_count < cfg.max_output_nodes)
    {
        for (uint32_t node : nodes_by_expand)
        {
            if (output_count >= cfg.max_output_nodes)
                break;
            if (!is_node_pending_pack(node_in_list, node))
                continue;
            try_append_unique_disk_cache_node(node, cfg, output_count, output_seen);
            node_in_list[node] = false;
        }
    }

    return 0;
}

static int hot_node_pack_pages(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                               const std::unordered_map<uint64_t, uint64_t> &weights, uint32_t k_hops,
                               const PagePackConfig &cfg_in, std::vector<std::vector<uint32_t>> &pages,
                               std::vector<bool> &assigned)
{
    if (graph.num_points == 0 || cfg_in.page_cap == 0)
        return -1;

    PagePackConfig cfg = cfg_in;
    cfg.node_expand = &node_expand;

    const auto adj = build_undirected_adjacency(graph);

    std::vector<bool> node_in_list;
    const bool disk_sidecar = (cfg.max_output_nodes > 0);
    init_pending_pack_nodes(static_cast<uint32_t>(graph.num_points), cfg, node_in_list);
    assigned.assign(graph.num_points, false);

    std::vector<uint32_t> nodes_by_expand(static_cast<size_t>(graph.num_points));
    std::iota(nodes_by_expand.begin(), nodes_by_expand.end(), 0);
    std::sort(nodes_by_expand.begin(), nodes_by_expand.end(), [&](uint32_t a, uint32_t b) {
        const uint64_t ca = node_heat(node_expand, a);
        const uint64_t cb = node_heat(node_expand, b);
        if (ca != cb)
            return ca > cb;
        return a < b;
    });

    auto append_page = [&](const std::vector<uint32_t> &page_nodes) {
        if (page_nodes.empty())
            return;
        pages.push_back(page_nodes);
        for (uint32_t node : page_nodes)
            assigned[node] = true;
    };

    const uint32_t invalid = static_cast<uint32_t>(graph.num_points);
    uint64_t output_count = 0;
    std::unordered_set<uint32_t> output_seen;
    if (disk_sidecar)
        output_seen.reserve(static_cast<size_t>(cfg.max_output_nodes));
    const std::unordered_set<uint32_t> *seen_ptr = disk_sidecar ? &output_seen : nullptr;

    auto fill_page_from_node_list = [&](std::vector<uint32_t> &page_nodes, std::unordered_set<uint32_t> &on_page) {
        for (uint32_t node : nodes_by_expand)
        {
            if (page_nodes.size() >= cfg.page_cap)
                break;
            if (!is_node_pending_pack(node_in_list, node))
                continue;
            place_node_on_page(page_nodes, on_page, cfg.page_cap, node, &cfg, seen_ptr);
            node_in_list[node] = false;
        }
    };

    while (true)
    {
        if (disk_sidecar && output_count >= cfg.max_output_nodes)
            break;
        if (!disk_sidecar && !has_pending_pack_nodes(node_in_list))
            break;

        uint32_t seed = invalid;
        for (uint32_t node : nodes_by_expand)
        {
            if (is_node_pending_pack(node_in_list, node))
            {
                seed = node;
                break;
            }
        }

        if (seed == invalid)
        {
            if (disk_sidecar)
                break;
            std::vector<uint32_t> page_nodes;
            std::unordered_set<uint32_t> on_page;
            fill_page_from_node_list(page_nodes, on_page);
            append_page(page_nodes);
            continue;
        }

        std::vector<uint32_t> page_nodes;
        std::unordered_set<uint32_t> on_page;
        page_nodes.reserve(static_cast<size_t>(cfg.page_cap));

        place_node_on_page(page_nodes, on_page, cfg.page_cap, seed, &cfg, seen_ptr);
        node_in_list[seed] = false;

        std::unordered_set<uint64_t> used_page_edges;
        const auto target_eligible = [&](uint32_t id) -> bool { return is_node_pending_pack(node_in_list, id); };

        while (page_nodes.size() < cfg.page_cap && has_pending_pack_nodes(node_in_list))
        {
            if (disk_sidecar && output_count >= cfg.max_output_nodes)
                break;

            std::vector<uint32_t> best_path;
            uint32_t best_target = invalid;
            double best_avg = -1.0;

            find_best_path_from_seed_limited(seed, adj, weights, used_page_edges, k_hops, target_eligible, best_avg,
                                             best_path, best_target);

            if (best_path.empty() || best_target == invalid)
                break;

            for (uint32_t node : best_path)
            {
                if (page_nodes.size() >= cfg.page_cap)
                    break;
                if (is_node_pending_pack(node_in_list, node))
                    place_node_on_page(page_nodes, on_page, cfg.page_cap, node, &cfg, seen_ptr);
            }

            for (uint32_t node : best_path)
            {
                if (node < node_in_list.size() && node_in_list[node])
                    node_in_list[node] = false;
            }
            for (size_t i = 1; i < best_path.size(); i++)
                used_page_edges.insert(undirected_edge_key(best_path[i - 1], best_path[i]));
        }

        append_page(page_nodes);
        append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
    }

    if (disk_sidecar && output_count < cfg.max_output_nodes)
    {
        for (uint32_t node : nodes_by_expand)
        {
            if (output_count >= cfg.max_output_nodes)
                break;
            if (!is_node_pending_pack(node_in_list, node))
                continue;
            try_append_unique_disk_cache_node(node, cfg, output_count, output_seen);
            node_in_list[node] = false;
            output_count++;
        }
    }

    return 0;
}

static void append_unassigned_nodes_to_pages(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                          uint64_t page_cap, std::vector<std::vector<uint32_t>> &pages,
                                          std::vector<bool> &assigned)
{
    std::vector<uint32_t> nodes_by_expand(static_cast<size_t>(graph.num_points));
    std::iota(nodes_by_expand.begin(), nodes_by_expand.end(), 0);
    std::sort(nodes_by_expand.begin(), nodes_by_expand.end(), [&](uint32_t a, uint32_t b) {
        const uint64_t ca = node_heat(node_expand, a);
        const uint64_t cb = node_heat(node_expand, b);
        if (ca != cb)
            return ca > cb;
        return a < b;
    });

    std::vector<uint32_t> tail;
    tail.reserve(static_cast<size_t>(page_cap));
    for (uint32_t node : nodes_by_expand)
    {
        if (assigned[node])
            continue;
        tail.push_back(node);
        assigned[node] = true;
        if (tail.size() >= page_cap)
        {
            pages.push_back(tail);
            tail.clear();
        }
    }
    if (!tail.empty())
        pages.push_back(tail);
}

int compute_edge_relayout_order(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                 const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                 uint64_t nnodes_per_sector, uint32_t k_hops, std::vector<uint32_t> &order)
{
    if (graph.num_points == 0 || nnodes_per_sector == 0)
        return -1;

    const auto weights = build_undirected_weights(directed_edges);
    const auto sorted_edges = sorted_undirected_edges(weights);

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.node_expand = &node_expand;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (pack_pages_by_edge_importance(graph, weights, sorted_edges, k_hops, cfg, pages, assigned) != 0)
        return -1;

    append_unassigned_nodes_to_pages(graph, node_expand, nnodes_per_sector, pages, assigned);

    order.clear();
    order.reserve(graph.num_points);
    for (const auto &page : pages)
        order.insert(order.end(), page.begin(), page.end());

    if (order.size() != graph.num_points)
        return -1;
    return 0;
}

int compute_edge_disk_cache_list(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                  const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                  uint64_t nnodes_per_sector, uint32_t k_hops, uint64_t max_nodes,
                                  const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    const auto weights = build_undirected_weights(directed_edges);
    const auto sorted_edges = sorted_undirected_edges(weights);

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (pack_pages_by_edge_importance(graph, weights, sorted_edges, k_hops, cfg, pages, assigned) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: edge-importance list empty (max_nodes=" << max_nodes << ", k_hops=" << k_hops
                      << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "edge");
    return 0;
}

int compute_hot_node_relayout_order(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                    uint64_t nnodes_per_sector, uint32_t k_hops, std::vector<uint32_t> &order)
{
    if (graph.num_points == 0 || nnodes_per_sector == 0)
        return -1;

    const auto weights = build_undirected_weights(directed_edges);

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = 0;
    cfg.node_expand = &node_expand;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (hot_node_pack_pages(graph, node_expand, weights, k_hops, cfg, pages, assigned) != 0)
        return -1;

    append_unassigned_nodes_to_pages(graph, node_expand, nnodes_per_sector, pages, assigned);

    order.clear();
    order.reserve(graph.num_points);
    for (const auto &page : pages)
        order.insert(order.end(), page.begin(), page.end());

    if (order.size() != graph.num_points)
        return -1;
    return 0;
}

int compute_hot_node_disk_cache_list(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                     const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                     uint64_t nnodes_per_sector, uint32_t k_hops, uint64_t max_nodes,
                                     const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    const auto weights = build_undirected_weights(directed_edges);

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (hot_node_pack_pages(graph, node_expand, weights, k_hops, cfg, pages, assigned) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: hot-node path list empty (max_nodes=" << max_nodes << ", k_hops=" << k_hops
                      << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "node");
    return 0;
}

void append_uncounted_nodes_to_disk_list(const std::vector<uint64_t> &node_expand, uint64_t max_nodes,
                                         const std::unordered_set<uint32_t> &exclude_ids,
                                         std::vector<uint32_t> &node_list)
{
    if (node_list.size() >= max_nodes || node_expand.empty())
        return;

    std::unordered_set<uint32_t> in_list(node_list.begin(), node_list.end());
    const size_t before = node_list.size();
    const uint32_t num_points = static_cast<uint32_t>(node_expand.size());

    for (uint32_t id = 0; id < num_points && node_list.size() < max_nodes; ++id)
    {
        if (exclude_ids.find(id) != exclude_ids.end())
            continue;
        if (in_list.find(id) != in_list.end())
            continue;
        node_list.push_back(id);
        in_list.insert(id);
    }

    if (node_list.size() > before)
    {
        diskann::cout << "MERIT disk-cache: appended " << (node_list.size() - before)
                      << " remaining nodes (ascending id order; total=" << node_list.size() << ")."
                      << std::endl;
    }
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
