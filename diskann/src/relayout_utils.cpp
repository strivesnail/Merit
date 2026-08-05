// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "relayout_utils.h"
#include "hotness_profiler.h"
#include "utils.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <map>
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
    if (layout == "d" || layout == "layoutd" || layout == "layout_d")
        layout = "frontier";
    if (layout == "e" || layout == "layoute" || layout == "layout_e" || layout == "layout-e")
        layout = "directed_beam";
    if (layout == "parent" || layout == "outedge" || layout == "layout_p" || layout == "p")
        layout = "parent";
    if (layout == "edge_dir" || layout == "edgedir" || layout == "directed_edge")
        layout = "edge_dir";
    if (layout == "edge_star" || layout == "edgestar")
        layout = "edge_star";
    if (layout == "edge_u" || layout == "edge_parent")
        layout = "edge_u";
    if (layout == "edge_pair" || layout == "edgepair")
        layout = "edge_pair";
    if (layout == "edge_clique" || layout == "edgeclique")
        layout = "edge_clique";
    if (layout == "edge_replica" || layout == "edgereplica" || layout == "replica")
        layout = "edge_replica";
    if (layout == "edge_star_dup" || layout == "stardup" || layout == "star_dup")
        layout = "edge_star_dup";
    if (layout == "frontier_page" || layout == "frontier_iso" || layout == "fpage")
        layout = "frontier_page";
    if (layout == "frontier_dup" || layout == "frontierdup")
        layout = "frontier_dup";
    if (layout == "directed_beam_pct30" || layout == "e_pct30" || layout == "dbeam_pct30")
        layout = "directed_beam_pct30";
    if (layout == "directed_beam_pct50" || layout == "e_pct50" || layout == "dbeam_pct50")
        layout = "directed_beam_pct50";
    if (layout == "directed_beam_pct80" || layout == "e_pct80" || layout == "dbeam_pct80")
        layout = "directed_beam_pct80";
    if (layout == "directed_beam_pct90" || layout == "e_pct90" || layout == "dbeam_pct90")
        layout = "directed_beam_pct90";
    if (layout == "directed_beam_pct100" || layout == "e_pct100" || layout == "dbeam_pct100")
        layout = "directed_beam_pct100";
    if (layout == "directed_seed_replica" || layout == "seed_replica" || layout == "seedreplica" ||
        layout == "seed_only")
        layout = "directed_seed_replica";
    if (layout == "directed_seed_replica_pct30" || layout == "seed_replica_pct30" || layout == "seed_pct30")
        layout = "directed_seed_replica_pct30";
    if (layout == "directed_seed_replica_pct50" || layout == "seed_replica_pct50" || layout == "seed_pct50")
        layout = "directed_seed_replica_pct50";
    if (layout == "directed_seed_replica_pct80" || layout == "seed_replica_pct80" || layout == "seed_pct80")
        layout = "directed_seed_replica_pct80";
    if (layout == "directed_seed_replica_pct90" || layout == "seed_replica_pct90" || layout == "seed_pct90")
        layout = "directed_seed_replica_pct90";
    if (layout == "directed_seed_replica_pct100" || layout == "seed_replica_pct100" || layout == "seed_pct100")
        layout = "directed_seed_replica_pct100";
    if (layout == "directed_beam" || layout == "dir_beam" || layout == "dbeam")
        layout = "directed_beam";
    if (layout == "directed_star" || layout == "dstar")
        layout = "directed_star";
    if (layout == "dir_edge_star" || layout == "des")
        layout = "dir_edge_star";
    if (layout == "directed_beam_hybrid" || layout == "dbeam_hybrid" || layout == "dbeam_h")
        layout = "directed_beam_hybrid";
    if (layout == "parent_star" || layout == "pstar")
        layout = "parent_star";
    if (layout == "dbeam_estar_split" || layout == "dbeam_estar" || layout == "desplit")
        layout = "dbeam_estar_split";
    if (layout == "directed_beam_dual" || layout == "dbeam_dual" || layout == "dbeam-dual")
        layout = "directed_beam_dual";
    if (layout == "directed_beam_inseed" || layout == "dbeam_inseed" || layout == "dbeam-inseed")
        layout = "directed_beam_inseed";
    if (layout == "directed_beam_tight" || layout == "dbeam_tight" || layout == "beam_tight")
        layout = "directed_beam_tight";
    if (layout == "directed_beam_starfill" || layout == "dbeam_starfill" || layout == "e_starfill")
        layout = "directed_beam_starfill";
    if (layout == "directed_beam_top4first" || layout == "dbeam_top4first" || layout == "e_top4first")
        layout = "directed_beam_top4first";
    if (layout == "node_top4first" || layout == "b_top4first" || layout == "hotnode_top4first")
        layout = "node_top4first";
    if (layout == "cooccur_star" || layout == "cooccur" || layout == "hop_cooccur")
        layout = "cooccur_star";
    if (layout == "dbeam_cooccur" || layout == "dbeam_coc" || layout == "beam_cooccur")
        layout = "dbeam_cooccur";
    if (layout == "frontier_topk" || layout == "ftopk" || layout == "frontier_top")
        layout = "frontier_topk";
    if (layout == "node-mp" || layout == "node_multi" || layout == "node-multi" || layout == "b2")
        layout = "node";
    return layout;
}

bool disk_cache_layout_allows_replicas(const std::string &layout)
{
    const std::string n = normalize_disk_cache_layout(layout);
    return n == "edge_replica" || n == "edge_star_dup" || n == "frontier_dup" ||
           n == "directed_seed_replica" || n == "directed_seed_replica_pct80" ||
           n == "directed_seed_replica_pct90" || n == "directed_seed_replica_pct100";
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

    if (max_hops == 1)
    {
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
        }
        return;
    }

    if (max_hops == 2)
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

    if (max_hops == 1)
    {
        for (uint32_t n1 : adj[seed])
        {
            if (used_page_edges.count(undirected_edge_key(seed, n1)) > 0)
                continue;
            if (!target_eligible(n1))
                continue;
            const std::vector<uint32_t> candidate = {seed, n1};
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
        return;
    }

    if (max_hops == 2)
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

    const auto try_end = [&](const std::vector<uint32_t> &candidate, uint32_t end) {
        if (!target_eligible(end))
            return;
        double candidate_avg = -1.0;
        std::vector<uint32_t> trial;
        try_candidate_path_to_node(candidate, end, used_page_edges, weights, candidate_avg, trial);
        if (!trial.empty() && candidate_avg > best_avg)
        {
            best_avg = candidate_avg;
            best_path = std::move(trial);
            best_target = end;
        }
    };

    if (max_hops == 3)
    {
        for (uint32_t n1 : adj[seed])
        {
            if (used_page_edges.count(undirected_edge_key(seed, n1)) > 0)
                continue;
            try_end({seed, n1}, n1);
            if (n1 >= adj.size())
                continue;
            for (uint32_t n2 : adj[n1])
            {
                if (n2 == seed || used_page_edges.count(undirected_edge_key(n1, n2)) > 0)
                    continue;
                if (!edge_ok(seed, n1))
                    continue;
                try_end({seed, n1, n2}, n2);
                if (n2 >= adj.size())
                    continue;
                for (uint32_t n3 : adj[n2])
                {
                    if (n3 == seed || n3 == n1 || used_page_edges.count(undirected_edge_key(n2, n3)) > 0)
                        continue;
                    try_end({seed, n1, n2, n3}, n3);
                }
            }
        }
        return;
    }

    if (max_hops == 4)
    {
        for (uint32_t n1 : adj[seed])
        {
            if (used_page_edges.count(undirected_edge_key(seed, n1)) > 0)
                continue;
            try_end({seed, n1}, n1);
            if (n1 >= adj.size())
                continue;
            for (uint32_t n2 : adj[n1])
            {
                if (n2 == seed || used_page_edges.count(undirected_edge_key(n1, n2)) > 0)
                    continue;
                if (!edge_ok(seed, n1))
                    continue;
                try_end({seed, n1, n2}, n2);
                if (n2 >= adj.size())
                    continue;
                for (uint32_t n3 : adj[n2])
                {
                    if (n3 == seed || n3 == n1 || used_page_edges.count(undirected_edge_key(n2, n3)) > 0)
                        continue;
                    try_end({seed, n1, n2, n3}, n3);
                    if (n3 >= adj.size())
                        continue;
                    for (uint32_t n4 : adj[n3])
                    {
                        if (n4 == seed || n4 == n1 || n4 == n2 ||
                            used_page_edges.count(undirected_edge_key(n3, n4)) > 0)
                            continue;
                        try_end({seed, n1, n2, n3, n4}, n4);
                    }
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

    const auto edge_ok = [&](uint32_t u, uint32_t v) -> bool {
        return graph_has_edge(adj, u, v) && used_page_edges.count(undirected_edge_key(u, v)) == 0;
    };

    if (max_hops == 1)
    {
        if (start == target)
            try_candidate_path_to_node({target}, target, used_page_edges, weights, best_avg, best_path);
        if (edge_ok(start, target))
            try_candidate_path_to_node({start, target}, target, used_page_edges, weights, best_avg, best_path);
        return;
    }

    if (max_hops == 2)
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
    uint64_t max_output_nodes = 0; // 0 = full relayout; >0 = disk cache cap (e.g. 10%)
    bool allow_duplicate_output = false;
    const std::unordered_set<uint32_t> *skip_output = nullptr;
    const std::vector<uint64_t> *node_expand = nullptr;
    std::vector<uint32_t> *output_nodes = nullptr; // disk cache: collect nodes here directly
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
    if (!cfg.allow_duplicate_output && output_seen.count(node) > 0)
        return false;
    cfg.output_nodes->push_back(node);
    if (!cfg.allow_duplicate_output)
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
    {
        if (cfg == nullptr || !cfg->allow_duplicate_output)
            return;
    }
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

static void star_fill_page_from_nodes(const VamanaGraph &graph,
                                      const std::unordered_map<uint64_t, uint64_t> &weights,
                                      const std::vector<std::vector<uint32_t>> &adj,
                                      std::unordered_set<uint64_t> &profile_edge_in_list, uint64_t page_cap,
                                      std::vector<uint32_t> &page_nodes, std::unordered_set<uint32_t> &on_page,
                                      const PagePackConfig &cfg, const std::unordered_set<uint32_t> *output_seen,
                                      std::vector<bool> &node_in_list);

static void fill_page_after_seed_directed_topk_and_starfill(
    uint32_t seed, const VamanaGraph &graph, const std::vector<std::vector<uint32_t>> &adj,
    const std::unordered_map<uint64_t, uint64_t> &weights,
    const std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> &out_edges,
    std::unordered_set<uint64_t> &profile_edge_in_list, uint32_t beam_width, const PagePackConfig &cfg,
    const std::unordered_set<uint32_t> *seen_ptr, std::vector<bool> &node_in_list, std::vector<uint32_t> &page_nodes,
    std::unordered_set<uint32_t> &on_page, uint64_t &output_count);

static std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> build_sorted_out_edges(
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t num_points)
{
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> out_edges;
    out_edges.reserve(directed_edges.size());
    for (const auto &edge : directed_edges)
    {
        const uint32_t parent = std::get<0>(edge);
        const uint32_t child = std::get<1>(edge);
        const uint64_t count = std::get<2>(edge);
        if (count == 0 || parent >= num_points || child >= num_points)
            continue;
        out_edges[parent].emplace_back(child, count);
    }
    for (auto &kv : out_edges)
    {
        auto &children = kv.second;
        std::sort(children.begin(), children.end(), [](const auto &a, const auto &b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
    }
    return out_edges;
}

static bool profile_out_topk_is_full_coverage(
    const std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> &out_edges, uint32_t node,
    uint32_t k)
{
    const auto it = out_edges.find(node);
    if (it == out_edges.end() || it->second.empty() || k == 0)
        return false;
    uint64_t total = 0;
    for (const auto &child_w : it->second)
        total += child_w.second;
    if (total == 0)
        return false;
    const size_t take = std::min(static_cast<size_t>(k), it->second.size());
    uint64_t top_sum = 0;
    for (size_t i = 0; i < take; i++)
        top_sum += it->second[i].second;
    return top_sum >= total;
}

static void partition_seeds_top4_full_first(
    const std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> &out_edges,
    std::vector<uint32_t> &seeds, uint32_t k = 4)
{
    std::vector<uint32_t> first;
    std::vector<uint32_t> rest;
    first.reserve(seeds.size());
    rest.reserve(seeds.size());
    for (uint32_t node : seeds)
    {
        if (profile_out_topk_is_full_coverage(out_edges, node, k))
            first.push_back(node);
        else
            rest.push_back(node);
    }
    seeds.clear();
    seeds.insert(seeds.end(), first.begin(), first.end());
    seeds.insert(seeds.end(), rest.begin(), rest.end());
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

static std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> sorted_directed_profile_edges(
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges)
{
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> sorted = directed_edges;
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
        const uint64_t ca = std::get<2>(a), cb = std::get<2>(b);
        if (ca != cb)
            return ca > cb;
        if (std::get<0>(a) != std::get<0>(b))
            return std::get<0>(a) < std::get<0>(b);
        return std::get<1>(a) < std::get<1>(b);
    });
    return sorted;
}

static void init_pending_directed_edges(const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                      std::unordered_set<uint64_t> &directed_in_list)
{
    directed_in_list.clear();
    directed_in_list.reserve(directed_edges.size());
    for (const auto &edge : directed_edges)
    {
        if (std::get<2>(edge) == 0)
            continue;
        directed_in_list.insert(directed_edge_key(std::get<0>(edge), std::get<1>(edge)));
    }
}

static bool pick_highest_directed_seed_edge(
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &sorted_directed,
    const std::unordered_set<uint64_t> &directed_in_list, size_t &edge_cursor, uint32_t &seed_u, uint32_t &seed_v,
    uint64_t &seed_key, const std::vector<bool> *node_in_list = nullptr)
{
    while (edge_cursor < sorted_directed.size())
    {
        const auto &edge = sorted_directed[edge_cursor++];
        const uint32_t u = std::get<0>(edge);
        const uint32_t v = std::get<1>(edge);
        const uint64_t key = directed_edge_key(u, v);
        if (directed_in_list.count(key) == 0)
            continue;
        if (node_in_list != nullptr)
        {
            const bool u_ok = is_node_pending_pack(*node_in_list, u);
            const bool v_ok = is_node_pending_pack(*node_in_list, v);
            if (!u_ok && !v_ok)
                continue;
        }
        seed_u = u;
        seed_v = v;
        seed_key = key;
        return true;
    }
    return false;
}

static std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> build_directed_out_edges(
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t num_points)
{
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> out_edges;
    out_edges.reserve(directed_edges.size());
    for (const auto &edge : directed_edges)
    {
        const uint32_t u = std::get<0>(edge);
        const uint32_t v = std::get<1>(edge);
        const uint64_t c = std::get<2>(edge);
        if (c == 0 || u >= num_points || v >= num_points)
            continue;
        out_edges[u].emplace_back(v, c);
    }
    for (auto &kv : out_edges)
    {
        auto &children = kv.second;
        std::sort(children.begin(), children.end(), [](const auto &a, const auto &b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
    }
    return out_edges;
}

enum class EdgePackVariant
{
    Khop,
    DirectedChild,
    Star,
    ParentOnly,
    PairOnly,
    Clique,
    DirectedSeedStar
};

static int pack_pages_edge_variant(const VamanaGraph &graph, const std::unordered_map<uint64_t, uint64_t> &weights,
                                   const std::vector<std::pair<uint32_t, uint32_t>> &sorted_undirected,
                                   const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                   uint32_t k_hops, EdgePackVariant variant, const PagePackConfig &cfg,
                                   std::vector<std::vector<uint32_t>> &pages, std::vector<bool> &assigned)
{
    if (graph.num_points == 0 || cfg.page_cap == 0)
        return -1;

    const auto adj = build_undirected_adjacency(graph);
    const auto sorted_directed = sorted_directed_profile_edges(directed_edges);
    const auto directed_out = build_directed_out_edges(directed_edges, graph.num_points);

    size_t undirected_cursor = 0;
    size_t directed_cursor = 0;
    std::unordered_set<uint64_t> profile_edge_in_list;
    std::unordered_set<uint64_t> directed_in_list;
    std::vector<bool> node_in_list;
    const bool disk_cache_capped = (cfg.max_output_nodes > 0);
    init_pending_profile_edges(weights, profile_edge_in_list);
    init_pending_directed_edges(directed_edges, directed_in_list);
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
    if (disk_cache_capped)
        output_seen.reserve(static_cast<size_t>(cfg.max_output_nodes));
    const std::unordered_set<uint32_t> *seen_ptr = disk_cache_capped ? &output_seen : nullptr;

    const std::vector<bool> *pick_node_list = &node_in_list;
    const bool use_directed_seed = (variant == EdgePackVariant::DirectedChild ||
                                   variant == EdgePackVariant::ParentOnly ||
                                   variant == EdgePackVariant::DirectedSeedStar);

    while (true)
    {
        if (disk_cache_capped && output_count >= cfg.max_output_nodes)
            break;
        if (!disk_cache_capped && !has_pending_pack_nodes(node_in_list))
            break;

        uint32_t seed_u = 0, seed_v = 0;
        uint64_t seed_key = 0;
        bool got_seed = false;
        if (use_directed_seed)
            got_seed = pick_highest_directed_seed_edge(sorted_directed, directed_in_list, directed_cursor, seed_u,
                                                       seed_v, seed_key, pick_node_list);
        else
            got_seed = pick_highest_weight_seed_edge(sorted_undirected, profile_edge_in_list, undirected_cursor, seed_u,
                                                     seed_v, seed_key, pick_node_list);

        if (!got_seed)
            break;

        std::vector<uint32_t> page_nodes;
        std::unordered_set<uint32_t> on_page;
        page_nodes.reserve(static_cast<size_t>(cfg.page_cap));

        place_node_on_page(page_nodes, on_page, cfg.page_cap, seed_u, &cfg, seen_ptr);
        place_node_on_page(page_nodes, on_page, cfg.page_cap, seed_v, &cfg, seen_ptr);
        profile_edge_in_list.erase(undirected_edge_key(seed_u, seed_v));
        directed_in_list.erase(directed_edge_key(seed_u, seed_v));
        directed_in_list.erase(directed_edge_key(seed_v, seed_u));
        if (seed_u < node_in_list.size())
            node_in_list[seed_u] = false;
        if (seed_v < node_in_list.size())
            node_in_list[seed_v] = false;

        std::unordered_set<uint64_t> used_page_edges;
        used_page_edges.insert(undirected_edge_key(seed_u, seed_v));

        if (variant != EdgePackVariant::PairOnly)
        {
            while (page_nodes.size() < cfg.page_cap)
            {
                if (disk_cache_capped && output_count >= cfg.max_output_nodes)
                    break;

                if (variant == EdgePackVariant::Khop)
                {
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
                    continue;
                }

                if (variant == EdgePackVariant::DirectedChild || variant == EdgePackVariant::ParentOnly)
                {
                    uint32_t best_child = static_cast<uint32_t>(graph.num_points);
                    uint64_t best_w = 0;
                    const auto consider_parent = [&](uint32_t parent) {
                        if (variant == EdgePackVariant::ParentOnly && parent != seed_u)
                            return;
                        const auto it = directed_out.find(parent);
                        if (it == directed_out.end())
                            return;
                        for (const auto &child_w : it->second)
                        {
                            const uint32_t child = child_w.first;
                            const uint64_t w = child_w.second;
                            const uint64_t dk = directed_edge_key(parent, child);
                            if (!directed_in_list.count(dk))
                                continue;
                            if (!is_node_pending_pack(node_in_list, child))
                                continue;
                            if (!graph_has_edge(adj, parent, child))
                                continue;
                            if (w > best_w || (w == best_w && (best_child == graph.num_points || child < best_child)))
                            {
                                best_w = w;
                                best_child = child;
                            }
                        }
                    };
                    consider_parent(seed_u);
                    if (variant == EdgePackVariant::DirectedChild)
                        consider_parent(seed_v);
                    if (best_child >= graph.num_points)
                        break;
                    place_node_on_page(page_nodes, on_page, cfg.page_cap, best_child, &cfg, seen_ptr);
                    node_in_list[best_child] = false;
                    profile_edge_in_list.erase(undirected_edge_key(seed_u, best_child));
                    profile_edge_in_list.erase(undirected_edge_key(seed_v, best_child));
                    directed_in_list.erase(directed_edge_key(seed_u, best_child));
                    directed_in_list.erase(directed_edge_key(best_child, seed_u));
                    used_page_edges.insert(undirected_edge_key(seed_u, best_child));
                    continue;
                }

                if (variant == EdgePackVariant::Star || variant == EdgePackVariant::DirectedSeedStar)
                {
                    uint32_t best_node = static_cast<uint32_t>(graph.num_points);
                    uint64_t best_w = 0;
                    for (uint32_t on : page_nodes)
                    {
                        if (on >= adj.size())
                            continue;
                        for (uint32_t nbr : adj[on])
                        {
                            const uint64_t ek = undirected_edge_key(on, nbr);
                            if (used_page_edges.count(ek) > 0)
                                continue;
                            if (profile_edge_in_list.count(ek) == 0)
                                continue;
                            if (!is_node_pending_pack(node_in_list, nbr))
                                continue;
                            const uint64_t w = edge_weight(weights, on, nbr);
                            if (w == 0)
                                continue;
                            if (w > best_w || (w == best_w && (best_node == graph.num_points || nbr < best_node)))
                            {
                                best_w = w;
                                best_node = nbr;
                            }
                        }
                    }
                    if (best_node >= graph.num_points)
                        break;
                    place_node_on_page(page_nodes, on_page, cfg.page_cap, best_node, &cfg, seen_ptr);
                    node_in_list[best_node] = false;
                    for (uint32_t on : page_nodes)
                    {
                        if (on == best_node)
                            continue;
                        profile_edge_in_list.erase(undirected_edge_key(on, best_node));
                        used_page_edges.insert(undirected_edge_key(on, best_node));
                    }
                    continue;
                }

                if (variant == EdgePackVariant::Clique)
                {
                    std::vector<std::pair<uint64_t, uint32_t>> candidates;
                    candidates.reserve(64);
                    for (uint32_t on : page_nodes)
                    {
                        if (on >= adj.size())
                            continue;
                        for (uint32_t nbr : adj[on])
                        {
                            const uint64_t ek = undirected_edge_key(on, nbr);
                            if (used_page_edges.count(ek) > 0 || on_page.count(nbr) > 0)
                                continue;
                            if (profile_edge_in_list.count(ek) == 0)
                                continue;
                            if (!is_node_pending_pack(node_in_list, nbr))
                                continue;
                            const uint64_t w = edge_weight(weights, on, nbr);
                            if (w == 0)
                                continue;
                            candidates.emplace_back(w, nbr);
                        }
                    }
                    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
                        return a.first > b.first || (a.first == b.first && a.second < b.second);
                    });
                    size_t added = 0;
                    for (const auto &cw : candidates)
                    {
                        if (page_nodes.size() >= cfg.page_cap)
                            break;
                        const uint32_t nbr = cw.second;
                        if (on_page.count(nbr) > 0)
                            continue;
                        place_node_on_page(page_nodes, on_page, cfg.page_cap, nbr, &cfg, seen_ptr);
                        if (!on_page.count(nbr))
                            continue;
                        node_in_list[nbr] = false;
                        for (uint32_t on : page_nodes)
                        {
                            if (on == nbr)
                                continue;
                            profile_edge_in_list.erase(undirected_edge_key(on, nbr));
                            used_page_edges.insert(undirected_edge_key(on, nbr));
                        }
                        added++;
                    }
                    if (added == 0)
                        break;
                }
            }
        }

        append_page(page_nodes);
        append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
    }

    if (disk_cache_capped && output_count < cfg.max_output_nodes)
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

static int pack_pages_edge_replica(const VamanaGraph &graph, const std::unordered_map<uint64_t, uint64_t> &weights,
                                   const std::vector<std::pair<uint32_t, uint32_t>> &sorted_undirected,
                                   const PagePackConfig &cfg, std::vector<std::vector<uint32_t>> &pages)
{
    if (graph.num_points == 0 || cfg.page_cap == 0 || cfg.output_nodes == nullptr)
        return -1;

    const auto adj = build_undirected_adjacency(graph);
    std::unordered_set<uint64_t> profile_edge_in_list;
    init_pending_profile_edges(weights, profile_edge_in_list);

    size_t undirected_cursor = 0;
    uint64_t output_count = 0;

    auto append_page_slots = [&](const std::vector<uint32_t> &page_nodes) {
        for (uint32_t node : page_nodes)
        {
            if (output_count >= cfg.max_output_nodes)
                break;
            if (!disk_cache_node_eligible(node, cfg))
                continue;
            cfg.output_nodes->push_back(node);
            output_count++;
        }
    };

    while (output_count < cfg.max_output_nodes)
    {
        uint32_t seed_u = 0, seed_v = 0;
        uint64_t seed_key = 0;
        if (!pick_highest_weight_seed_edge(sorted_undirected, profile_edge_in_list, undirected_cursor, seed_u, seed_v,
                                         seed_key, nullptr))
            break;

        std::vector<uint32_t> page_nodes;
        std::unordered_set<uint32_t> on_page;
        page_nodes.reserve(static_cast<size_t>(cfg.page_cap));
        place_node_on_page(page_nodes, on_page, cfg.page_cap, seed_u, &cfg, nullptr);
        place_node_on_page(page_nodes, on_page, cfg.page_cap, seed_v, &cfg, nullptr);
        profile_edge_in_list.erase(undirected_edge_key(seed_u, seed_v));

        std::unordered_set<uint64_t> used_page_edges;
        used_page_edges.insert(undirected_edge_key(seed_u, seed_v));

        while (page_nodes.size() < cfg.page_cap && output_count < cfg.max_output_nodes)
        {
            uint32_t best_node = static_cast<uint32_t>(graph.num_points);
            uint64_t best_w = 0;
            for (uint32_t on : page_nodes)
            {
                if (on >= adj.size())
                    continue;
                for (uint32_t nbr : adj[on])
                {
                    const uint64_t ek = undirected_edge_key(on, nbr);
                    if (used_page_edges.count(ek) > 0 || on_page.count(nbr) > 0)
                        continue;
                    if (profile_edge_in_list.count(ek) == 0)
                        continue;
                    if (!disk_cache_node_eligible(nbr, cfg))
                        continue;
                    const uint64_t w = edge_weight(weights, on, nbr);
                    if (w == 0)
                        continue;
                    if (w > best_w || (w == best_w && (best_node == graph.num_points || nbr < best_node)))
                    {
                        best_w = w;
                        best_node = nbr;
                    }
                }
            }
            if (best_node >= graph.num_points)
                break;
            place_node_on_page(page_nodes, on_page, cfg.page_cap, best_node, &cfg, nullptr);
            for (uint32_t on : page_nodes)
            {
                if (on == best_node)
                    continue;
                used_page_edges.insert(undirected_edge_key(on, best_node));
            }
        }

        if (!page_nodes.empty())
            pages.push_back(page_nodes);
        append_page_slots(page_nodes);
    }

    return 0;
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
                            const std::vector<std::pair<uint32_t, uint32_t>> &sorted_edges,
                            const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint32_t k_hops,
                            const PagePackConfig &cfg, std::vector<std::vector<uint32_t>> &pages,
                            std::vector<bool> &assigned)
{
    return pack_pages_edge_variant(graph, weights, sorted_edges, directed_edges, k_hops, EdgePackVariant::Khop, cfg,
                                   pages, assigned);
}

static int hot_node_pack_pages(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                               const std::unordered_map<uint64_t, uint64_t> &weights, uint32_t k_hops,
                               const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                               const PagePackConfig &cfg_in, std::vector<std::vector<uint32_t>> &pages,
                               std::vector<bool> &assigned, bool seed_top4_full_first = false)
{
    if (graph.num_points == 0 || cfg_in.page_cap == 0)
        return -1;

    PagePackConfig cfg = cfg_in;
    cfg.node_expand = &node_expand;

    const auto adj = build_undirected_adjacency(graph);
    const auto out_edges = build_sorted_out_edges(directed_edges, graph.num_points);
    std::unordered_set<uint64_t> profile_edge_in_list;
    init_pending_profile_edges(weights, profile_edge_in_list);

    const uint32_t beam_width = (k_hops > 0) ? k_hops : 1;

    std::vector<bool> node_in_list;
    const bool disk_cache_capped = (cfg.max_output_nodes > 0);
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

    std::vector<uint32_t> seeds_by_priority = nodes_by_expand;
    if (seed_top4_full_first)
        partition_seeds_top4_full_first(out_edges, seeds_by_priority, 4);

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
    if (disk_cache_capped)
        output_seen.reserve(static_cast<size_t>(cfg.max_output_nodes));
    const std::unordered_set<uint32_t> *seen_ptr = disk_cache_capped ? &output_seen : nullptr;

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
        if (disk_cache_capped && output_count >= cfg.max_output_nodes)
            break;
        if (!disk_cache_capped && !has_pending_pack_nodes(node_in_list))
            break;

        uint32_t seed = invalid;
        for (uint32_t node : seeds_by_priority)
        {
            if (is_node_pending_pack(node_in_list, node))
            {
                seed = node;
                break;
            }
        }

        if (seed == invalid)
        {
            if (disk_cache_capped)
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

        fill_page_after_seed_directed_topk_and_starfill(seed, graph, adj, weights, out_edges, profile_edge_in_list,
                                                        beam_width, cfg, seen_ptr, node_in_list, page_nodes, on_page,
                                                        output_count);

        append_page(page_nodes);
        append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
    }

    if (disk_cache_capped && output_count < cfg.max_output_nodes)
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

// Layout P: seed by total outgoing profile edge weight; co-locate parent + top profile children (directed).
static void star_fill_page_from_nodes(const VamanaGraph &graph,
                                      const std::unordered_map<uint64_t, uint64_t> &weights,
                                      const std::vector<std::vector<uint32_t>> &adj,
                                      std::unordered_set<uint64_t> &profile_edge_in_list, uint64_t page_cap,
                                      std::vector<uint32_t> &page_nodes, std::unordered_set<uint32_t> &on_page,
                                      const PagePackConfig &cfg, const std::unordered_set<uint32_t> *output_seen,
                                      std::vector<bool> &node_in_list);

static int profile_parent_pack_pages(const VamanaGraph &graph,
                                     const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                     const PagePackConfig &cfg_in, std::vector<std::vector<uint32_t>> &pages,
                                     std::vector<bool> &assigned,
                                     const std::unordered_map<uint64_t, uint64_t> *weights_for_star = nullptr)
{
    if (graph.num_points == 0 || cfg_in.page_cap == 0)
        return -1;

    PagePackConfig cfg = cfg_in;
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> out_edges;
    out_edges.reserve(directed_edges.size());
    for (const auto &edge : directed_edges)
    {
        const uint32_t parent = std::get<0>(edge);
        const uint32_t child = std::get<1>(edge);
        const uint64_t count = std::get<2>(edge);
        if (count == 0 || parent >= graph.num_points || child >= graph.num_points)
            continue;
        out_edges[parent].emplace_back(child, count);
    }

    std::vector<uint32_t> parents;
    parents.reserve(out_edges.size());
    for (auto &kv : out_edges)
    {
        auto &children = kv.second;
        std::sort(children.begin(), children.end(), [](const auto &a, const auto &b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
        parents.push_back(kv.first);
    }
    std::sort(parents.begin(), parents.end(), [&](uint32_t a, uint32_t b) {
        uint64_t wa = 0, wb = 0;
        for (const auto &p : out_edges[a])
            wa += p.second;
        for (const auto &p : out_edges[b])
            wb += p.second;
        if (wa != wb)
            return wa > wb;
        return a < b;
    });

    const auto adj = build_undirected_adjacency(graph);
    std::unordered_set<uint64_t> profile_edge_in_list;
    if (weights_for_star != nullptr)
        init_pending_profile_edges(*weights_for_star, profile_edge_in_list);
    std::vector<bool> node_in_list;
    const bool disk_cache_capped = (cfg.max_output_nodes > 0);
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
    if (disk_cache_capped)
        output_seen.reserve(static_cast<size_t>(cfg.max_output_nodes));
    const std::unordered_set<uint32_t> *seen_ptr = disk_cache_capped ? &output_seen : nullptr;

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
        if (disk_cache_capped && output_count >= cfg.max_output_nodes)
            break;
        if (!disk_cache_capped && !has_pending_pack_nodes(node_in_list))
            break;

        uint32_t parent = static_cast<uint32_t>(graph.num_points);
        for (uint32_t cand : parents)
        {
            if (is_node_pending_pack(node_in_list, cand))
            {
                parent = cand;
                break;
            }
        }

        if (parent >= graph.num_points)
        {
            if (disk_cache_capped)
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
        place_node_on_page(page_nodes, on_page, cfg.page_cap, parent, &cfg, seen_ptr);
        node_in_list[parent] = false;

        const auto it = out_edges.find(parent);
        if (it != out_edges.end())
        {
            for (const auto &child_w : it->second)
            {
                if (page_nodes.size() >= cfg.page_cap)
                    break;
                if (disk_cache_capped && output_count >= cfg.max_output_nodes)
                    break;
                const uint32_t child = child_w.first;
                if (!is_node_pending_pack(node_in_list, child))
                    continue;
                if (!graph_has_edge(adj, parent, child))
                    continue;
                place_node_on_page(page_nodes, on_page, cfg.page_cap, child, &cfg, seen_ptr);
                node_in_list[child] = false;
            }
        }

        if (weights_for_star != nullptr)
            star_fill_page_from_nodes(graph, *weights_for_star, adj, profile_edge_in_list, cfg.page_cap, page_nodes,
                                      on_page, cfg, seen_ptr, node_in_list);

        append_page(page_nodes);
        append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
    }

    if (disk_cache_capped && output_count < cfg.max_output_nodes)
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

int compute_profile_parent_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (profile_parent_pack_pages(graph, directed_edges, cfg, pages, assigned) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: profile-parent list empty (max_nodes=" << max_nodes << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "parent");
    return 0;
}

int compute_parent_star_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list)
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
    if (profile_parent_pack_pages(graph, directed_edges, cfg, pages, assigned, &weights) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: parent_star list empty (max_nodes=" << max_nodes << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "parent_star");
    return 0;
}

int compute_dbeam_estar_split_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;
    if (beam_width == 0)
        beam_width = 1;

    const uint64_t phase1_cap = max_nodes / 2;
    if (phase1_cap == 0)
        return compute_edge_variant_disk_cache_list(graph, node_expand, directed_edges, nnodes_per_sector, 1,
                                                    max_nodes, exclude_ids, "edge_star", node_list);

    std::vector<uint32_t> phase1;
    if (compute_directed_beam_disk_cache_list(graph, node_expand, directed_edges, nnodes_per_sector, beam_width,
                                              phase1_cap, exclude_ids, phase1) != 0)
        return -1;

    std::unordered_set<uint32_t> exclude2 = exclude_ids;
    exclude2.insert(phase1.begin(), phase1.end());

    std::vector<uint32_t> phase2;
    const uint64_t phase2_cap = (max_nodes > phase1.size()) ? (max_nodes - phase1.size()) : 0;
    if (phase2_cap > 0)
    {
        if (compute_edge_variant_disk_cache_list(graph, node_expand, directed_edges, nnodes_per_sector, 1, phase2_cap,
                                                 exclude2, "edge_star", phase2) != 0)
            return -1;
    }

    const size_t phase1_count = phase1.size();
    node_list = std::move(phase1);
    node_list.insert(node_list.end(), phase2.begin(), phase2.end());

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: dbeam_estar_split list empty (max_nodes=" << max_nodes << ")." << std::endl;
        return -1;
    }
    diskann::cout << "MERIT disk-cache dbeam_estar_split: phase1=" << phase1_count << " phase2=" << phase2.size()
                  << " total=" << node_list.size() << std::endl;
    log_disk_cache_node_list_stats(node_list, "dbeam_estar_split");
    return 0;
}

static void star_fill_page_from_nodes(const VamanaGraph &graph,
                                      const std::unordered_map<uint64_t, uint64_t> &weights,
                                      const std::vector<std::vector<uint32_t>> &adj,
                                      std::unordered_set<uint64_t> &profile_edge_in_list, uint64_t page_cap,
                                      std::vector<uint32_t> &page_nodes, std::unordered_set<uint32_t> &on_page,
                                      const PagePackConfig &cfg, const std::unordered_set<uint32_t> *output_seen,
                                      std::vector<bool> &node_in_list);

enum class DirectedBeamStarMode
{
    Undirected, // Layout E default: directed top-k + undirected star_fill
    StarFill,   // Same as Undirected (legacy alias)
    Directed,
    Hybrid,
    None
};

enum class DirectedBeamSeedMode
{
    ParentOut,
    ParentOutTop4First, // ParentOut order, but top-4=100% out-edge nodes first
    ChildIn
};

static std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> build_directed_in_edges(
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t num_points)
{
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> in_edges;
    in_edges.reserve(directed_edges.size());
    for (const auto &edge : directed_edges)
    {
        const uint32_t parent = std::get<0>(edge);
        const uint32_t child = std::get<1>(edge);
        const uint64_t count = std::get<2>(edge);
        if (count == 0 || parent >= num_points || child >= num_points)
            continue;
        in_edges[child].emplace_back(parent, count);
    }
    for (auto &kv : in_edges)
    {
        auto &parents = kv.second;
        std::sort(parents.begin(), parents.end(), [](const auto &a, const auto &b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
    }
    return in_edges;
}

static void directed_star_fill_page_from_nodes(
    const VamanaGraph &graph,
    const std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> &out_edges,
    const std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> &in_edges,
    std::unordered_set<uint64_t> &directed_in_list, uint64_t page_cap, std::vector<uint32_t> &page_nodes,
    std::unordered_set<uint32_t> &on_page, const PagePackConfig &cfg, const std::unordered_set<uint32_t> *output_seen,
    std::vector<bool> &node_in_list)
{
    while (page_nodes.size() < page_cap)
    {
        uint32_t best_node = static_cast<uint32_t>(graph.num_points);
        uint64_t best_w = 0;
        for (uint32_t on : page_nodes)
        {
            const auto out_it = out_edges.find(on);
            if (out_it != out_edges.end())
            {
                for (const auto &child_w : out_it->second)
                {
                    const uint32_t nbr = child_w.first;
                    const uint64_t w = child_w.second;
                    const uint64_t ek = directed_edge_key(on, nbr);
                    if (on_page.count(nbr) > 0 || directed_in_list.count(ek) == 0)
                        continue;
                    if (!is_node_pending_pack(node_in_list, nbr))
                        continue;
                    if (w > best_w || (w == best_w && (best_node == graph.num_points || nbr < best_node)))
                    {
                        best_w = w;
                        best_node = nbr;
                    }
                }
            }
            const auto in_it = in_edges.find(on);
            if (in_it != in_edges.end())
            {
                for (const auto &parent_w : in_it->second)
                {
                    const uint32_t nbr = parent_w.first;
                    const uint64_t w = parent_w.second;
                    const uint64_t ek = directed_edge_key(nbr, on);
                    if (on_page.count(nbr) > 0 || directed_in_list.count(ek) == 0)
                        continue;
                    if (!is_node_pending_pack(node_in_list, nbr))
                        continue;
                    if (w > best_w || (w == best_w && (best_node == graph.num_points || nbr < best_node)))
                    {
                        best_w = w;
                        best_node = nbr;
                    }
                }
            }
        }
        if (best_node >= graph.num_points)
            break;
        place_node_on_page(page_nodes, on_page, page_cap, best_node, &cfg, output_seen);
        node_in_list[best_node] = false;
        for (uint32_t on : page_nodes)
        {
            if (on == best_node)
                continue;
            directed_in_list.erase(directed_edge_key(on, best_node));
            directed_in_list.erase(directed_edge_key(best_node, on));
        }
    }
}

static void hybrid_star_fill_page_from_nodes(
    const VamanaGraph &graph, const std::unordered_map<uint64_t, uint64_t> &weights,
    const std::vector<std::vector<uint32_t>> &adj,
    const std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> &out_edges,
    const std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> &in_edges,
    std::unordered_set<uint64_t> &profile_edge_in_list, std::unordered_set<uint64_t> &directed_in_list,
    uint64_t page_cap, std::vector<uint32_t> &page_nodes, std::unordered_set<uint32_t> &on_page,
    const PagePackConfig &cfg, const std::unordered_set<uint32_t> *output_seen, std::vector<bool> &node_in_list)
{
    std::unordered_set<uint64_t> used_page_edges;
    for (size_t i = 0; i < page_nodes.size(); ++i)
    {
        for (size_t j = i + 1; j < page_nodes.size(); ++j)
            used_page_edges.insert(undirected_edge_key(page_nodes[i], page_nodes[j]));
    }

    while (page_nodes.size() < page_cap)
    {
        uint32_t best_dir_node = static_cast<uint32_t>(graph.num_points);
        uint64_t best_dir_w = 0;
        for (uint32_t on : page_nodes)
        {
            const auto out_it = out_edges.find(on);
            if (out_it != out_edges.end())
            {
                for (const auto &child_w : out_it->second)
                {
                    const uint32_t nbr = child_w.first;
                    const uint64_t w = child_w.second;
                    const uint64_t ek = directed_edge_key(on, nbr);
                    if (on_page.count(nbr) > 0 || directed_in_list.count(ek) == 0)
                        continue;
                    if (!is_node_pending_pack(node_in_list, nbr))
                        continue;
                    if (w > best_dir_w ||
                        (w == best_dir_w && (best_dir_node == graph.num_points || nbr < best_dir_node)))
                    {
                        best_dir_w = w;
                        best_dir_node = nbr;
                    }
                }
            }
            const auto in_it = in_edges.find(on);
            if (in_it != in_edges.end())
            {
                for (const auto &parent_w : in_it->second)
                {
                    const uint32_t nbr = parent_w.first;
                    const uint64_t w = parent_w.second;
                    const uint64_t ek = directed_edge_key(nbr, on);
                    if (on_page.count(nbr) > 0 || directed_in_list.count(ek) == 0)
                        continue;
                    if (!is_node_pending_pack(node_in_list, nbr))
                        continue;
                    if (w > best_dir_w ||
                        (w == best_dir_w && (best_dir_node == graph.num_points || nbr < best_dir_node)))
                    {
                        best_dir_w = w;
                        best_dir_node = nbr;
                    }
                }
            }
        }

        uint32_t best_node = static_cast<uint32_t>(graph.num_points);
        if (best_dir_node < graph.num_points)
        {
            best_node = best_dir_node;
        }
        else
        {
            uint64_t best_w = 0;
            for (uint32_t on : page_nodes)
            {
                if (on >= adj.size())
                    continue;
                for (uint32_t nbr : adj[on])
                {
                    const uint64_t ek = undirected_edge_key(on, nbr);
                    if (used_page_edges.count(ek) > 0 || on_page.count(nbr) > 0)
                        continue;
                    if (profile_edge_in_list.count(ek) == 0)
                        continue;
                    if (!is_node_pending_pack(node_in_list, nbr))
                        continue;
                    const uint64_t w = edge_weight(weights, on, nbr);
                    if (w == 0)
                        continue;
                    if (w > best_w || (w == best_w && (best_node == graph.num_points || nbr < best_node)))
                    {
                        best_w = w;
                        best_node = nbr;
                    }
                }
            }
        }

        if (best_node >= graph.num_points)
            break;
        place_node_on_page(page_nodes, on_page, page_cap, best_node, &cfg, output_seen);
        node_in_list[best_node] = false;
        for (uint32_t on : page_nodes)
        {
            if (on == best_node)
                continue;
            profile_edge_in_list.erase(undirected_edge_key(on, best_node));
            directed_in_list.erase(directed_edge_key(on, best_node));
            directed_in_list.erase(directed_edge_key(best_node, on));
            used_page_edges.insert(undirected_edge_key(on, best_node));
        }
    }
}

static int directed_beam_pack_pages(const VamanaGraph &graph,
                                      const std::vector<uint64_t> &node_expand,
                                      const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                      const std::unordered_map<uint64_t, uint64_t> &weights, uint32_t beam_width,
                                      DirectedBeamStarMode star_mode, DirectedBeamSeedMode seed_mode,
                                      const PagePackConfig &cfg_in, std::vector<std::vector<uint32_t>> &pages,
                                      std::vector<bool> &assigned, bool replica_no_dedup = false,
                                      std::vector<SeedPageGroup> *seed_groups_out = nullptr)
{
    if (graph.num_points == 0 || cfg_in.page_cap == 0)
        return -1;
    if (beam_width == 0)
        beam_width = 1;

    PagePackConfig cfg = cfg_in;
    if (replica_no_dedup)
        cfg.allow_duplicate_output = true;

    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> out_edges;
    out_edges.reserve(directed_edges.size());
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> in_edges;
    in_edges.reserve(directed_edges.size());
    for (const auto &edge : directed_edges)
    {
        const uint32_t parent = std::get<0>(edge);
        const uint32_t child = std::get<1>(edge);
        const uint64_t count = std::get<2>(edge);
        if (count == 0 || parent >= graph.num_points || child >= graph.num_points)
            continue;
        out_edges[parent].emplace_back(child, count);
        in_edges[child].emplace_back(parent, count);
    }

    std::vector<uint32_t> parents;
    if (seed_mode == DirectedBeamSeedMode::ChildIn)
    {
        parents.reserve(in_edges.size());
        for (auto &kv : in_edges)
        {
            auto &parents_of = kv.second;
            std::sort(parents_of.begin(), parents_of.end(), [](const auto &a, const auto &b) {
                return a.second > b.second || (a.second == b.second && a.first < b.first);
            });
            parents.push_back(kv.first);
        }
        std::sort(parents.begin(), parents.end(), [&](uint32_t a, uint32_t b) {
            uint64_t wa = 0, wb = 0;
            for (const auto &p : in_edges[a])
                wa += p.second;
            for (const auto &p : in_edges[b])
                wb += p.second;
            if (wa != wb)
                return wa > wb;
            return a < b;
        });
    }
    else
    {
        parents.reserve(out_edges.size());
        for (auto &kv : out_edges)
        {
            auto &children = kv.second;
            std::sort(children.begin(), children.end(), [](const auto &a, const auto &b) {
                return a.second > b.second || (a.second == b.second && a.first < b.first);
            });
            parents.push_back(kv.first);
        }
        std::sort(parents.begin(), parents.end(), [&](uint32_t a, uint32_t b) {
            uint64_t wa = 0, wb = 0;
            for (const auto &p : out_edges[a])
                wa += p.second;
            for (const auto &p : out_edges[b])
                wb += p.second;
            if (wa != wb)
                return wa > wb;
            return a < b;
        });
        if (seed_mode == DirectedBeamSeedMode::ParentOutTop4First)
            partition_seeds_top4_full_first(out_edges, parents, 4);
    }

    const auto adj = build_undirected_adjacency(graph);
    std::unordered_set<uint64_t> profile_edge_in_list;
    init_pending_profile_edges(weights, profile_edge_in_list);
    std::unordered_set<uint64_t> directed_in_list;
    init_pending_directed_edges(directed_edges, directed_in_list);
    const auto directed_in = build_directed_in_edges(directed_edges, graph.num_points);

    std::vector<bool> node_in_list;
    const bool disk_cache_capped = (cfg.max_output_nodes > 0);
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

    uint64_t output_count = 0;
    std::unordered_set<uint32_t> output_seen;
    if (disk_cache_capped)
        output_seen.reserve(static_cast<size_t>(cfg.max_output_nodes));
    const std::unordered_set<uint32_t> *seen_ptr =
        replica_no_dedup ? nullptr : (disk_cache_capped ? &output_seen : nullptr);

    auto can_pack_node = [&](uint32_t node) {
        if (replica_no_dedup)
            return disk_cache_node_eligible(node, cfg);
        return is_node_pending_pack(node_in_list, node);
    };

    std::vector<bool> replica_pending;
    if (replica_no_dedup)
        replica_pending.assign(static_cast<size_t>(graph.num_points), true);

    const size_t max_children = (beam_width > 0) ? static_cast<size_t>(beam_width - 1) : 0;

    for (uint32_t parent : parents)
    {
        if (disk_cache_capped && output_count >= cfg.max_output_nodes)
            break;
        if (!can_pack_node(parent))
            continue;

        if (replica_no_dedup)
            std::fill(replica_pending.begin(), replica_pending.end(), true);
        std::vector<bool> &pack_state = replica_no_dedup ? replica_pending : node_in_list;

        std::vector<uint32_t> page_nodes;
        std::unordered_set<uint32_t> on_page;
        page_nodes.reserve(static_cast<size_t>(cfg.page_cap));
        place_node_on_page(page_nodes, on_page, cfg.page_cap, parent, &cfg, seen_ptr);
        if (!replica_no_dedup)
            node_in_list[parent] = false;

        if (star_mode == DirectedBeamStarMode::Directed)
        {
            const auto it = out_edges.find(parent);
            if (it != out_edges.end())
            {
                size_t added = 0;
                for (const auto &child_w : it->second)
                {
                    if (added >= max_children || page_nodes.size() >= cfg.page_cap)
                        break;
                    if (disk_cache_capped && output_count >= cfg.max_output_nodes)
                        break;
                    const uint32_t child = child_w.first;
                    if (!can_pack_node(child))
                        continue;
                    if (!graph_has_edge(adj, parent, child))
                        continue;
                    const size_t before = page_nodes.size();
                    place_node_on_page(page_nodes, on_page, cfg.page_cap, child, &cfg, seen_ptr);
                    if (page_nodes.size() > before)
                    {
                        if (!replica_no_dedup)
                            node_in_list[child] = false;
                        added++;
                    }
                }
            }
            directed_star_fill_page_from_nodes(graph, out_edges, directed_in, directed_in_list, cfg.page_cap,
                                               page_nodes, on_page, cfg, seen_ptr, pack_state);
        }
        else if (star_mode == DirectedBeamStarMode::Hybrid)
        {
            const auto it = out_edges.find(parent);
            if (it != out_edges.end())
            {
                size_t added = 0;
                for (const auto &child_w : it->second)
                {
                    if (added >= max_children || page_nodes.size() >= cfg.page_cap)
                        break;
                    if (disk_cache_capped && output_count >= cfg.max_output_nodes)
                        break;
                    const uint32_t child = child_w.first;
                    if (!can_pack_node(child))
                        continue;
                    if (!graph_has_edge(adj, parent, child))
                        continue;
                    const size_t before = page_nodes.size();
                    place_node_on_page(page_nodes, on_page, cfg.page_cap, child, &cfg, seen_ptr);
                    if (page_nodes.size() > before)
                    {
                        if (!replica_no_dedup)
                            node_in_list[child] = false;
                        added++;
                    }
                }
            }
            hybrid_star_fill_page_from_nodes(graph, weights, adj, out_edges, directed_in, profile_edge_in_list,
                                             directed_in_list, cfg.page_cap, page_nodes, on_page, cfg, seen_ptr,
                                             pack_state);
        }
        else if (star_mode == DirectedBeamStarMode::None)
        {
            const auto it = out_edges.find(parent);
            if (it != out_edges.end())
            {
                size_t added = 0;
                for (const auto &child_w : it->second)
                {
                    if (added >= max_children || page_nodes.size() >= cfg.page_cap)
                        break;
                    if (disk_cache_capped && output_count >= cfg.max_output_nodes)
                        break;
                    const uint32_t child = child_w.first;
                    if (!can_pack_node(child))
                        continue;
                    if (!graph_has_edge(adj, parent, child))
                        continue;
                    const size_t before = page_nodes.size();
                    place_node_on_page(page_nodes, on_page, cfg.page_cap, child, &cfg, seen_ptr);
                    if (page_nodes.size() > before)
                    {
                        if (!replica_no_dedup)
                            node_in_list[child] = false;
                        added++;
                    }
                }
            }
        }
        else
        {
            fill_page_after_seed_directed_topk_and_starfill(parent, graph, adj, weights, out_edges,
                                                            profile_edge_in_list, beam_width, cfg, seen_ptr,
                                                            pack_state, page_nodes, on_page, output_count);
        }

        if (page_nodes.empty())
            continue;
        pages.push_back(page_nodes);
        for (uint32_t node : page_nodes)
            assigned[node] = true;
        append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
        if (seed_groups_out != nullptr)
        {
            SeedPageGroup grp;
            grp.seed = parent;
            grp.total_pages = 1;
            grp.pages.push_back(page_nodes);
            seed_groups_out->push_back(grp);
        }
    }

    if (!replica_no_dedup && disk_cache_capped && output_count < cfg.max_output_nodes)
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

static int compute_directed_beam_disk_cache_list_with_seed(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, DirectedBeamStarMode star_mode, DirectedBeamSeedMode seed_mode, uint64_t max_nodes,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list, const char *stats_label)
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
    if (directed_beam_pack_pages(graph, node_expand, directed_edges, weights, beam_width, star_mode, seed_mode, cfg,
                                 pages, assigned) != 0)
        return -1;

    if (node_list.empty())
        return -1;
    log_disk_cache_node_list_stats(node_list, stats_label);
    return 0;
}

int compute_directed_seed_replica_beam_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list, std::vector<SeedPageGroup> *seed_page_groups)
{
    node_list.clear();
    if (seed_page_groups != nullptr)
        seed_page_groups->clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    const auto weights = build_undirected_weights(directed_edges);
    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.allow_duplicate_output = true;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    std::vector<SeedPageGroup> groups;
    if (directed_beam_pack_pages(graph, node_expand, directed_edges, weights, beam_width,
                                 DirectedBeamStarMode::Undirected, DirectedBeamSeedMode::ParentOut, cfg, pages,
                                 assigned, true, seed_page_groups != nullptr ? &groups : nullptr) != 0)
        return -1;

    if (node_list.empty())
        return -1;
    if (seed_page_groups != nullptr)
        *seed_page_groups = std::move(groups);
    log_disk_cache_node_list_stats(node_list, "directed_seed_replica (Layout E replica)");
    return 0;
}

// max_pages=0 in split_directed_neighbors_into_pages => no page-count cap.
static constexpr uint8_t MERIT_DC_PCT_UNLIMITED_PAGES = 0;

static size_t directed_out_edges_for_pct_threshold(
    const std::vector<std::pair<uint32_t, uint64_t>> &sorted_out, double pct_threshold, uint64_t page_cap)
{
    uint64_t total = 0;
    for (const auto &edge : sorted_out)
        total += edge.second;
    if (total == 0)
        return 0;
    uint64_t cum = 0;
    size_t k = 0;
    for (const auto &edge : sorted_out)
    {
        cum += edge.second;
        k++;
        if (static_cast<double>(cum) / static_cast<double>(total) >= pct_threshold)
            break;
    }
    // Page-aligned threshold:
    // - Page0 carries seed + neighbors => first-page neighbor capacity = page_cap - 1.
    // - Later pages are pure neighbors => page_cap neighbors per page.
    // We keep adding neighbors until the page containing the threshold-crossing neighbor is full.
    if (page_cap == 0 || k == 0)
        return k;

    const size_t total_nbrs = sorted_out.size();
    const size_t first_cap = (page_cap > 0) ? static_cast<size_t>(page_cap - 1) : 0;
    size_t rounded = k;
    if (k <= first_cap)
    {
        rounded = first_cap;
    }
    else
    {
        const size_t rem = k - first_cap;
        const size_t pp = static_cast<size_t>(page_cap);
        const size_t pages_after = (rem + pp - 1) / pp;
        rounded = first_cap + pages_after * pp;
    }
    return std::min(rounded, total_nbrs);
}

static void split_directed_neighbors_into_pages(uint32_t seed, const std::vector<uint32_t> &directed_nbrs,
                                                uint64_t page_cap, uint8_t max_pages,
                                                std::vector<std::vector<uint32_t>> &pages)
{
    pages.clear();
    if (page_cap == 0)
        return;

    size_t idx = 0;
    std::vector<uint32_t> page0;
    page0.push_back(seed);
    while (page0.size() < page_cap && idx < directed_nbrs.size())
        page0.push_back(directed_nbrs[idx++]);
    pages.push_back(page0);

    while (idx < directed_nbrs.size() && (max_pages == 0 || pages.size() < max_pages))
    {
        std::vector<uint32_t> page;
        while (page.size() < page_cap && idx < directed_nbrs.size())
            page.push_back(directed_nbrs[idx++]);
        if (!page.empty())
            pages.push_back(page);
    }
}

static void split_nodes_into_pages(const std::vector<uint32_t> &nodes, uint64_t page_cap,
                                   std::vector<std::vector<uint32_t>> &pages)
{
    pages.clear();
    if (page_cap == 0 || nodes.empty())
        return;
    for (uint32_t node : nodes)
    {
        if (pages.empty() || pages.back().size() >= page_cap)
            pages.emplace_back();
        pages.back().push_back(node);
    }
}

static int directed_beam_pct_pack_pages(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                       const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                       double pct_threshold, const char *layout_name, const PagePackConfig &cfg_in,
                                       std::vector<std::vector<uint32_t>> &pages, std::vector<bool> &assigned,
                                       std::vector<SeedPageGroup> &seed_groups)
{
    if (graph.num_points == 0 || cfg_in.page_cap == 0)
        return -1;

    PagePackConfig cfg = cfg_in;
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> out_edges;
    out_edges.reserve(directed_edges.size());
    for (const auto &edge : directed_edges)
    {
        const uint32_t parent = std::get<0>(edge);
        const uint32_t child = std::get<1>(edge);
        const uint64_t count = std::get<2>(edge);
        if (count == 0 || parent >= graph.num_points || child >= graph.num_points)
            continue;
        out_edges[parent].emplace_back(child, count);
    }

    std::vector<uint32_t> parents;
    parents.reserve(out_edges.size());
    for (auto &kv : out_edges)
    {
        auto &children = kv.second;
        std::sort(children.begin(), children.end(), [](const auto &a, const auto &b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
        parents.push_back(kv.first);
    }
    std::sort(parents.begin(), parents.end(), [&](uint32_t a, uint32_t b) {
        uint64_t wa = 0, wb = 0;
        for (const auto &p : out_edges[a])
            wa += p.second;
        for (const auto &p : out_edges[b])
            wb += p.second;
        if (wa != wb)
            return wa > wb;
        return a < b;
    });

    const auto adj = build_undirected_adjacency(graph);
    std::vector<bool> node_in_list;
    const bool disk_cache_capped = (cfg.max_output_nodes > 0);
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

    uint64_t output_count = 0;
    std::unordered_set<uint32_t> output_seen;
    if (disk_cache_capped)
        output_seen.reserve(static_cast<size_t>(cfg.max_output_nodes));
    const std::unordered_set<uint32_t> *seen_ptr = disk_cache_capped ? &output_seen : nullptr;

    uint64_t multi_page_seeds = 0;
    uint64_t cold_tail_pages = 0;
    uint64_t cold_tail_nodes = 0;
    seed_groups.clear();

    for (uint32_t parent : parents)
    {
        if (disk_cache_capped && output_count >= cfg.max_output_nodes)
            break;
        if (!is_node_pending_pack(node_in_list, parent))
            continue;

        const auto it = out_edges.find(parent);
        if (it == out_edges.end())
            continue;

        const size_t k = directed_out_edges_for_pct_threshold(it->second, pct_threshold, cfg.page_cap);
        std::vector<uint32_t> directed_nbrs;
        directed_nbrs.reserve(k);
        for (size_t i = 0; i < k && i < it->second.size(); ++i)
        {
            const uint32_t child = it->second[i].first;
            if (!is_node_pending_pack(node_in_list, child))
                continue;
            if (!graph_has_edge(adj, parent, child))
                continue;
            directed_nbrs.push_back(child);
        }
        std::vector<std::vector<uint32_t>> template_pages;
        split_directed_neighbors_into_pages(parent, directed_nbrs, cfg.page_cap, MERIT_DC_PCT_UNLIMITED_PAGES,
                                            template_pages);

        std::vector<std::vector<uint32_t>> built_pages;
        built_pages.reserve(template_pages.size());
        for (size_t pi = 0; pi < template_pages.size(); ++pi)
        {
            if (disk_cache_capped && output_count >= cfg.max_output_nodes)
                break;

            std::vector<uint32_t> page_nodes;
            std::unordered_set<uint32_t> on_page;
            page_nodes.reserve(static_cast<size_t>(cfg.page_cap));

            for (uint32_t node : template_pages[pi])
            {
                if (disk_cache_capped && output_count >= cfg.max_output_nodes)
                    break;
                if (node == parent)
                {
                    if (!is_node_pending_pack(node_in_list, parent))
                        continue;
                    place_node_on_page(page_nodes, on_page, cfg.page_cap, parent, &cfg, seen_ptr);
                    node_in_list[parent] = false;
                    continue;
                }
                if (!is_node_pending_pack(node_in_list, node))
                    continue;
                if (!graph_has_edge(adj, parent, node))
                    continue;
                const size_t before = page_nodes.size();
                place_node_on_page(page_nodes, on_page, cfg.page_cap, node, &cfg, seen_ptr);
                if (page_nodes.size() > before)
                    node_in_list[node] = false;
            }

            if (page_nodes.empty())
                continue;
            const bool only_seed =
                (page_nodes.size() == 1 && page_nodes[0] == parent);
            if (only_seed)
                continue;
            built_pages.push_back(page_nodes);
        }

        if (built_pages.empty())
            continue;

        SeedPageGroup grp;
        grp.seed = parent;
        grp.total_pages = static_cast<uint8_t>(built_pages.size());
        grp.pages = built_pages;

        if (pct_threshold < 1.0 - 1e-9)
        {
            std::vector<uint32_t> cold_nbrs;
            cold_nbrs.reserve(it->second.size() - k);
            for (size_t i = k; i < it->second.size(); ++i)
            {
                const uint32_t child = it->second[i].first;
                if (!is_node_pending_pack(node_in_list, child))
                    continue;
                if (!graph_has_edge(adj, parent, child))
                    continue;
                cold_nbrs.push_back(child);
            }

            std::vector<std::vector<uint32_t>> cold_template;
            split_nodes_into_pages(cold_nbrs, cfg.page_cap, cold_template);
            for (const auto &template_page : cold_template)
            {
                if (disk_cache_capped && output_count >= cfg.max_output_nodes)
                    break;

                std::vector<uint32_t> page_nodes;
                std::unordered_set<uint32_t> on_page;
                page_nodes.reserve(static_cast<size_t>(cfg.page_cap));
                for (uint32_t node : template_page)
                {
                    if (disk_cache_capped && output_count >= cfg.max_output_nodes)
                        break;
                    if (!is_node_pending_pack(node_in_list, node))
                        continue;
                    if (!graph_has_edge(adj, parent, node))
                        continue;
                    const size_t before = page_nodes.size();
                    place_node_on_page(page_nodes, on_page, cfg.page_cap, node, &cfg, seen_ptr);
                    if (page_nodes.size() > before)
                        node_in_list[node] = false;
                }
                if (page_nodes.empty())
                    continue;
                grp.cold_pages.push_back(page_nodes);
            }
        }

        seed_groups.push_back(grp);
        if (grp.total_pages > 1)
            multi_page_seeds++;

        for (const auto &page_nodes : built_pages)
        {
            pages.push_back(page_nodes);
            for (uint32_t node : page_nodes)
                assigned[node] = true;
            append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
        }
        for (const auto &page_nodes : grp.cold_pages)
        {
            cold_tail_pages++;
            cold_tail_nodes += page_nodes.size();
            pages.push_back(page_nodes);
            for (uint32_t node : page_nodes)
                assigned[node] = true;
            append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
        }
    }

    if (disk_cache_capped && output_count < cfg.max_output_nodes)
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

    if (!seed_groups.empty())
    {
        diskann::cout << "MERIT disk-cache " << layout_name << ": seed_groups=" << seed_groups.size()
                      << " multi_page_seeds=" << multi_page_seeds;
        if (cold_tail_pages > 0)
            diskann::cout << " cold_tail_pages=" << cold_tail_pages << " cold_tail_nodes=" << cold_tail_nodes;
        diskann::cout << std::endl;
    }

    return 0;
}

struct SeedReplicaPackPlan
{
    uint32_t parent = 0;
    std::vector<std::vector<uint32_t>> template_pages;
    std::vector<std::vector<uint32_t>> cold_template_pages;
    size_t next_page = 0;
    size_t next_cold_page = 0;
    int group_idx = -1;
};

static bool seed_replica_materialize_page(const std::vector<std::vector<uint32_t>> &adj, uint32_t parent,
                                          const std::vector<uint32_t> &template_page, const PagePackConfig &cfg,
                                          std::vector<uint32_t> &page_nodes)
{
    page_nodes.clear();
    std::unordered_set<uint32_t> on_page;
    page_nodes.reserve(static_cast<size_t>(cfg.page_cap));

    for (uint32_t node : template_page)
    {
        if (node == parent)
        {
            if (!disk_cache_node_eligible(parent, cfg))
                continue;
            place_node_on_page(page_nodes, on_page, cfg.page_cap, parent, &cfg, nullptr);
            continue;
        }
        if (!disk_cache_node_eligible(node, cfg))
            continue;
        if (!graph_has_edge(adj, parent, node))
            continue;
        place_node_on_page(page_nodes, on_page, cfg.page_cap, node, &cfg, nullptr);
    }

    if (page_nodes.empty())
        return false;
    const bool only_seed = (page_nodes.size() == 1 && page_nodes[0] == parent);
    return !only_seed;
}

static void log_seed_replica_uncovered_expand_nodes(const std::vector<uint64_t> &node_expand,
                                                    const PagePackConfig &cfg,
                                                    const std::unordered_set<uint32_t> &on_seed_pages)
{
    if (cfg.node_expand == nullptr)
        return;

    uint64_t expand_total = 0;
    uint64_t expand_missing = 0;
    const uint32_t num_points = static_cast<uint32_t>(node_expand.size());
    for (uint32_t id = 0; id < num_points; ++id)
    {
        if (cfg.skip_output != nullptr && cfg.skip_output->count(id) > 0)
            continue;
        if (node_heat(node_expand, id) == 0)
            continue;
        expand_total++;
        if (on_seed_pages.count(id) == 0)
            expand_missing++;
    }

    if (expand_missing > 0)
    {
        diskann::cout << "MERIT disk-cache seed_replica: " << expand_missing << " / " << expand_total
                      << " expand nodes are not on any seed page (no flat append). Increase ratio or lower pct."
                      << std::endl;
    }
}

static int directed_beam_pct_seed_replica_pack_pages(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, double pct_threshold,
    const char *layout_name, const PagePackConfig &cfg_in, std::vector<std::vector<uint32_t>> &pages,
    std::vector<bool> &assigned, std::vector<SeedPageGroup> &seed_groups)
{
    if (graph.num_points == 0 || cfg_in.page_cap == 0)
        return -1;

    PagePackConfig cfg = cfg_in;
    cfg.allow_duplicate_output = true;

    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> out_edges;
    out_edges.reserve(directed_edges.size());
    for (const auto &edge : directed_edges)
    {
        const uint32_t parent = std::get<0>(edge);
        const uint32_t child = std::get<1>(edge);
        const uint64_t count = std::get<2>(edge);
        if (count == 0 || parent >= graph.num_points || child >= graph.num_points)
            continue;
        out_edges[parent].emplace_back(child, count);
    }

    std::vector<uint32_t> parents;
    parents.reserve(out_edges.size());
    for (auto &kv : out_edges)
    {
        auto &children = kv.second;
        std::sort(children.begin(), children.end(), [](const auto &a, const auto &b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        });
        parents.push_back(kv.first);
    }
    std::sort(parents.begin(), parents.end(), [&](uint32_t a, uint32_t b) {
        uint64_t wa = 0, wb = 0;
        for (const auto &p : out_edges[a])
            wa += p.second;
        for (const auto &p : out_edges[b])
            wb += p.second;
        if (wa != wb)
            return wa > wb;
        return a < b;
    });

    const auto adj = build_undirected_adjacency(graph);
    assigned.assign(graph.num_points, false);

    uint64_t output_count = 0;
    std::unordered_set<uint32_t> output_seen;
    if (cfg.max_output_nodes > 0)
        output_seen.reserve(static_cast<size_t>(cfg.max_output_nodes));
    std::unordered_set<uint32_t> unique_on_seed_pages;

    uint64_t multi_page_seeds = 0;
    uint64_t cold_tail_pages = 0;
    uint64_t cold_tail_nodes = 0;
    uint64_t skipped_parents = 0;
    uint64_t seeds_blocked_by_budget = 0;
    seed_groups.clear();

    std::vector<SeedReplicaPackPlan> plans;
    plans.reserve(parents.size());
    for (uint32_t parent : parents)
    {
        if (!disk_cache_node_eligible(parent, cfg))
        {
            skipped_parents++;
            continue;
        }

        const auto it = out_edges.find(parent);
        if (it == out_edges.end())
            continue;

        const size_t k = directed_out_edges_for_pct_threshold(it->second, pct_threshold, cfg.page_cap);
        std::vector<uint32_t> directed_nbrs;
        directed_nbrs.reserve(k);
        for (size_t i = 0; i < k && i < it->second.size(); ++i)
        {
            const uint32_t child = it->second[i].first;
            if (!disk_cache_node_eligible(child, cfg))
                continue;
            if (!graph_has_edge(adj, parent, child))
                continue;
            directed_nbrs.push_back(child);
        }

        SeedReplicaPackPlan plan;
        plan.parent = parent;
        split_directed_neighbors_into_pages(parent, directed_nbrs, cfg.page_cap, MERIT_DC_PCT_UNLIMITED_PAGES,
                                            plan.template_pages);
        if (plan.template_pages.empty())
            continue;

        if (pct_threshold < 1.0 - 1e-9)
        {
            std::vector<uint32_t> cold_nbrs;
            cold_nbrs.reserve(it->second.size() - k);
            for (size_t i = k; i < it->second.size(); ++i)
            {
                const uint32_t child = it->second[i].first;
                if (!disk_cache_node_eligible(child, cfg))
                    continue;
                if (!graph_has_edge(adj, parent, child))
                    continue;
                cold_nbrs.push_back(child);
            }
            split_nodes_into_pages(cold_nbrs, cfg.page_cap, plan.cold_template_pages);
        }
        plans.push_back(std::move(plan));
    }

    const auto budget_left = [&]() {
        return cfg.max_output_nodes == 0 || output_count < cfg.max_output_nodes;
    };

    const auto emit_plan_page = [&](SeedReplicaPackPlan &plan) -> bool {
        if (plan.next_page >= plan.template_pages.size())
            return false;
        if (!budget_left())
            return false;

        std::vector<uint32_t> page_nodes;
        if (!seed_replica_materialize_page(adj, plan.parent, plan.template_pages[plan.next_page], cfg, page_nodes))
        {
            plan.next_page++;
            return false;
        }

        if (plan.group_idx < 0)
        {
            SeedPageGroup grp;
            grp.seed = plan.parent;
            grp.total_pages = 1;
            grp.pages.push_back(page_nodes);
            seed_groups.push_back(grp);
            plan.group_idx = static_cast<int>(seed_groups.size()) - 1;
        }
        else
        {
            SeedPageGroup &grp = seed_groups[static_cast<size_t>(plan.group_idx)];
            const size_t pages_before = grp.pages.size();
            grp.pages.push_back(page_nodes);
            grp.total_pages = static_cast<uint8_t>(grp.pages.size());
            if (pages_before == 1 && grp.pages.size() == 2)
                multi_page_seeds++;
        }

        pages.push_back(page_nodes);
        for (uint32_t node : page_nodes)
        {
            assigned[node] = true;
            unique_on_seed_pages.insert(node);
        }
        append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
        plan.next_page++;
        return true;
    };

    const auto emit_plan_cold_page = [&](SeedReplicaPackPlan &plan) -> bool {
        if (plan.group_idx < 0 || plan.next_cold_page >= plan.cold_template_pages.size())
            return false;
        if (!budget_left())
            return false;

        std::vector<uint32_t> page_nodes;
        if (!seed_replica_materialize_page(adj, plan.parent, plan.cold_template_pages[plan.next_cold_page], cfg,
                                           page_nodes))
        {
            plan.next_cold_page++;
            return false;
        }

        SeedPageGroup &grp = seed_groups[static_cast<size_t>(plan.group_idx)];
        grp.cold_pages.push_back(page_nodes);
        pages.push_back(page_nodes);
        cold_tail_pages++;
        cold_tail_nodes += page_nodes.size();
        for (uint32_t node : page_nodes)
        {
            assigned[node] = true;
            unique_on_seed_pages.insert(node);
        }
        append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
        plan.next_cold_page++;
        return true;
    };

    // Phase 1: every eligible profile parent gets its first seed page before any seed grows multi-page.
    for (SeedReplicaPackPlan &plan : plans)
    {
        if (!budget_left())
            break;
        if (plan.next_page == 0)
            emit_plan_page(plan);
    }

    // Phase 2: round-robin extra neighbor pages until budget is exhausted.
    bool progress = true;
    while (progress && budget_left())
    {
        progress = false;
        for (SeedReplicaPackPlan &plan : plans)
        {
            if (!budget_left())
                break;
            if (plan.group_idx < 0 || plan.next_page >= plan.template_pages.size())
                continue;
            if (emit_plan_page(plan))
                progress = true;
        }
    }

    // Phase 3: place cold-tail pages (pct<100) next to each seed group on disk writeout.
    progress = true;
    while (progress && budget_left())
    {
        progress = false;
        for (SeedReplicaPackPlan &plan : plans)
        {
            if (!budget_left())
                break;
            if (emit_plan_cold_page(plan))
                progress = true;
        }
    }

    for (const SeedReplicaPackPlan &plan : plans)
    {
        if (plan.group_idx < 0)
            seeds_blocked_by_budget++;
    }

    if (!seed_groups.empty() || !plans.empty())
    {
        diskann::cout << "MERIT disk-cache " << layout_name << ": seed_groups=" << seed_groups.size()
                      << " profile_parents=" << parents.size() << " pack_plans=" << plans.size()
                      << " multi_page_seeds=" << multi_page_seeds << " skipped_parents=" << skipped_parents
                      << " seeds_blocked_by_budget=" << seeds_blocked_by_budget;
        if (cold_tail_pages > 0)
            diskann::cout << " cold_tail_pages=" << cold_tail_pages << " cold_tail_nodes=" << cold_tail_nodes;
        diskann::cout << std::endl;
    }
    log_seed_replica_uncovered_expand_nodes(node_expand, cfg, unique_on_seed_pages);

    return 0;
}

bool is_directed_beam_pct_layout(const std::string &layout_norm)
{
    return layout_norm == "directed_beam_pct30" || layout_norm == "directed_beam_pct50" ||
           layout_norm == "directed_beam_pct80" ||
           layout_norm == "directed_beam_pct90" || layout_norm == "directed_beam_pct100";
}

double directed_beam_pct_layout_threshold(const std::string &layout_norm)
{
    if (layout_norm == "directed_beam_pct30")
        return 0.30;
    if (layout_norm == "directed_beam_pct50")
        return 0.50;
    if (layout_norm == "directed_beam_pct90")
        return 0.90;
    if (layout_norm == "directed_beam_pct100")
        return 1.0;
    return 0.80;
}

int compute_directed_beam_pct_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list,
    double pct_threshold, const char *layout_label, std::vector<SeedPageGroup> *seed_page_groups)
{
    node_list.clear();
    if (seed_page_groups != nullptr)
        seed_page_groups->clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    std::vector<SeedPageGroup> groups;
    if (directed_beam_pct_pack_pages(graph, node_expand, directed_edges, pct_threshold, layout_label, cfg, pages,
                                     assigned, groups) != 0)
        return -1;

    if (node_list.empty())
        return -1;
    if (seed_page_groups != nullptr)
        *seed_page_groups = std::move(groups);
    log_disk_cache_node_list_stats(node_list, layout_label);
    return 0;
}

bool is_directed_seed_replica_layout(const std::string &layout_norm)
{
    return layout_norm == "directed_seed_replica" || is_directed_seed_replica_pct_layout(layout_norm);
}

bool is_directed_seed_replica_pct_layout(const std::string &layout_norm)
{
    return layout_norm == "directed_seed_replica_pct30" || layout_norm == "directed_seed_replica_pct50" ||
           layout_norm == "directed_seed_replica_pct80" || layout_norm == "directed_seed_replica_pct90" ||
           layout_norm == "directed_seed_replica_pct100";
}

bool is_seed_only_disk_cache_layout(const std::string &layout_norm)
{
    (void)layout_norm;
    // Seed-replica changes packing only; lookup/expand stay Layout-E unless MERIT_SEED_ONLY_* env opts in.
    return false;
}

double directed_seed_replica_layout_threshold(const std::string &layout_norm)
{
    if (layout_norm == "directed_seed_replica_pct30")
        return 0.30;
    if (layout_norm == "directed_seed_replica_pct50")
        return 0.50;
    if (layout_norm == "directed_seed_replica_pct90")
        return 0.90;
    if (layout_norm == "directed_seed_replica_pct80")
        return 0.80;
    return 1.0;
}

int compute_directed_seed_replica_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list,
    double pct_threshold, const char *layout_label, std::vector<SeedPageGroup> *seed_page_groups)
{
    node_list.clear();
    if (seed_page_groups != nullptr)
        seed_page_groups->clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.allow_duplicate_output = true;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    std::vector<SeedPageGroup> groups;
    if (directed_beam_pct_seed_replica_pack_pages(graph, node_expand, directed_edges, pct_threshold, layout_label, cfg,
                                                  pages, assigned, groups) != 0)
        return -1;

    if (node_list.empty())
        return -1;
    if (seed_page_groups != nullptr)
        *seed_page_groups = std::move(groups);
    log_disk_cache_node_list_stats(node_list, layout_label);
    return 0;
}

int compute_directed_beam_pct80_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list,
    std::vector<SeedPageGroup> *seed_page_groups)
{
    return compute_directed_beam_pct_disk_cache_list(graph, node_expand, directed_edges, nnodes_per_sector, max_nodes,
                                                     exclude_ids, node_list, 0.80, "Layout E (directed_beam_pct80)",
                                                     seed_page_groups);
}

int compute_directed_beam_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list)
{
    if (compute_directed_beam_disk_cache_list_with_seed(graph, node_expand, directed_edges, nnodes_per_sector,
                                                        beam_width, DirectedBeamStarMode::Undirected,
                                                        DirectedBeamSeedMode::ParentOut, max_nodes, exclude_ids,
                                                        node_list, "Layout E (directed_beam)") != 0)
    {
        diskann::cerr << "MERIT disk-cache: directed_beam list empty (max_nodes=" << max_nodes
                      << ", beam_width=" << beam_width << ")." << std::endl;
        return -1;
    }
    return 0;
}

int compute_directed_beam_top4first_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list)
{
    if (compute_directed_beam_disk_cache_list_with_seed(graph, node_expand, directed_edges, nnodes_per_sector,
                                                        beam_width, DirectedBeamStarMode::Undirected,
                                                        DirectedBeamSeedMode::ParentOutTop4First, max_nodes,
                                                        exclude_ids, node_list, "directed_beam_top4first") != 0)
    {
        diskann::cerr << "MERIT disk-cache: directed_beam_top4first list empty (max_nodes=" << max_nodes
                      << ", beam_width=" << beam_width << ")." << std::endl;
        return -1;
    }
    return 0;
}

int compute_directed_beam_starfill_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list)
{
    if (compute_directed_beam_disk_cache_list_with_seed(graph, node_expand, directed_edges, nnodes_per_sector,
                                                        beam_width, DirectedBeamStarMode::StarFill,
                                                        DirectedBeamSeedMode::ParentOut, max_nodes, exclude_ids,
                                                        node_list, "directed_beam_starfill (legacy E)") != 0)
    {
        diskann::cerr << "MERIT disk-cache: directed_beam_starfill list empty (max_nodes=" << max_nodes
                      << ", beam_width=" << beam_width << ")." << std::endl;
        return -1;
    }
    return 0;
}

int compute_directed_beam_inseed_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list)
{
    if (compute_directed_beam_disk_cache_list_with_seed(graph, node_expand, directed_edges, nnodes_per_sector,
                                                        beam_width, DirectedBeamStarMode::Undirected,
                                                        DirectedBeamSeedMode::ChildIn, max_nodes, exclude_ids,
                                                        node_list, "directed_beam_inseed") != 0)
    {
        diskann::cerr << "MERIT disk-cache: directed_beam_inseed list empty (max_nodes=" << max_nodes
                      << ", beam_width=" << beam_width << ")." << std::endl;
        return -1;
    }
    return 0;
}

int compute_directed_beam_dual_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    const uint64_t phase1_cap = max_nodes / 2;
    std::vector<uint32_t> phase1;
    if (phase1_cap > 0)
    {
        if (compute_directed_beam_disk_cache_list_with_seed(
                graph, node_expand, directed_edges, nnodes_per_sector, beam_width, DirectedBeamStarMode::Undirected,
                DirectedBeamSeedMode::ParentOut, phase1_cap, exclude_ids, phase1, "directed_beam_dual/parent") != 0)
            return -1;
    }

    std::unordered_set<uint32_t> exclude2 = exclude_ids;
    exclude2.insert(phase1.begin(), phase1.end());

    std::vector<uint32_t> phase2;
    const uint64_t phase2_cap = (max_nodes > phase1.size()) ? (max_nodes - phase1.size()) : 0;
    if (phase2_cap > 0)
    {
        if (compute_directed_beam_disk_cache_list_with_seed(
                graph, node_expand, directed_edges, nnodes_per_sector, beam_width, DirectedBeamStarMode::Undirected,
                DirectedBeamSeedMode::ChildIn, phase2_cap, exclude2, phase2, "directed_beam_dual/child") != 0)
            return -1;
    }

    node_list = std::move(phase1);
    node_list.insert(node_list.end(), phase2.begin(), phase2.end());
    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: directed_beam_dual list empty (max_nodes=" << max_nodes << ")."
                      << std::endl;
        return -1;
    }
    diskann::cout << "MERIT disk-cache directed_beam_dual: parent_phase=" << (node_list.size() - phase2.size())
                  << " child_phase=" << phase2.size() << std::endl;
    log_disk_cache_node_list_stats(node_list, "directed_beam_dual");
    return 0;
}

int compute_directed_star_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list)
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
    if (directed_beam_pack_pages(graph, node_expand, directed_edges, weights, beam_width,
                                 DirectedBeamStarMode::Directed, DirectedBeamSeedMode::ParentOut, cfg, pages,
                                 assigned) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: directed_star list empty (max_nodes=" << max_nodes
                      << ", beam_width=" << beam_width << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "directed_star");
    return 0;
}

int compute_directed_beam_hybrid_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list)
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
    if (directed_beam_pack_pages(graph, node_expand, directed_edges, weights, beam_width,
                                 DirectedBeamStarMode::Hybrid, DirectedBeamSeedMode::ParentOut, cfg, pages,
                                 assigned) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: directed_beam_hybrid list empty (max_nodes=" << max_nodes
                      << ", beam_width=" << beam_width << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "directed_beam_hybrid");
    return 0;
}

int compute_directed_beam_tight_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list)
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
    if (directed_beam_pack_pages(graph, node_expand, directed_edges, weights, beam_width,
                                 DirectedBeamStarMode::None, DirectedBeamSeedMode::ParentOut, cfg, pages, assigned) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: directed_beam_tight list empty (max_nodes=" << max_nodes
                      << ", beam_width=" << beam_width << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "directed_beam_tight");
    return 0;
}

static void flatten_pages_to_permutation_order(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                               const std::vector<std::vector<uint32_t>> &pages,
                                               std::vector<uint32_t> &order)
{
    order.clear();
    order.reserve(graph.num_points);
    std::vector<bool> placed(static_cast<size_t>(graph.num_points), false);
    for (const auto &page : pages)
    {
        for (uint32_t node : page)
        {
            if (node >= graph.num_points || placed[node])
                continue;
            placed[node] = true;
            order.push_back(node);
        }
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
    for (uint32_t node : nodes_by_expand)
    {
        if (placed[node])
            continue;
        placed[node] = true;
        order.push_back(node);
    }
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
    if (pack_pages_by_edge_importance(graph, weights, sorted_edges, directed_edges, k_hops, cfg, pages, assigned) != 0)
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

static EdgePackVariant edge_pack_variant_from_layout(const std::string &variant_layout)
{
    if (variant_layout == "edge_dir")
        return EdgePackVariant::DirectedChild;
    if (variant_layout == "edge_star")
        return EdgePackVariant::Star;
    if (variant_layout == "edge_u")
        return EdgePackVariant::ParentOnly;
    if (variant_layout == "edge_pair")
        return EdgePackVariant::PairOnly;
    if (variant_layout == "edge_clique")
        return EdgePackVariant::Clique;
    if (variant_layout == "dir_edge_star")
        return EdgePackVariant::DirectedSeedStar;
    return EdgePackVariant::Khop;
}

int compute_edge_variant_disk_cache_list(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                         const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                         uint64_t nnodes_per_sector, uint32_t k_hops, uint64_t max_nodes,
                                         const std::unordered_set<uint32_t> &exclude_ids,
                                         const std::string &variant_layout, std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    const auto weights = build_undirected_weights(directed_edges);
    const auto sorted_edges = sorted_undirected_edges(weights);
    const EdgePackVariant variant = edge_pack_variant_from_layout(variant_layout);

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (pack_pages_edge_variant(graph, weights, sorted_edges, directed_edges, k_hops, variant, cfg, pages, assigned) !=
        0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: edge-variant list empty (layout=" << variant_layout
                      << ", max_nodes=" << max_nodes << ", k_hops=" << k_hops << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, variant_layout.c_str());
    return 0;
}

int compute_edge_disk_cache_list(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                  const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                  uint64_t nnodes_per_sector, uint32_t k_hops, uint64_t max_nodes,
                                  const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list)
{
    return compute_edge_variant_disk_cache_list(graph, node_expand, directed_edges, nnodes_per_sector, k_hops,
                                                max_nodes, exclude_ids, "edge", node_list);
}

int compute_edge_replica_disk_cache_list(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                         const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                         uint64_t nnodes_per_sector, uint64_t max_slots,
                                         const std::unordered_set<uint32_t> &exclude_ids,
                                         std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_slots == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    const auto weights = build_undirected_weights(directed_edges);
    const auto sorted_edges = sorted_undirected_edges(weights);

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_slots;
    cfg.allow_duplicate_output = true;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    std::vector<std::vector<uint32_t>> pages;
    if (pack_pages_edge_replica(graph, weights, sorted_edges, cfg, pages) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: edge-replica list empty (max_slots=" << max_slots << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "edge_replica");
    return 0;
}

int compute_edge_star_dup_disk_cache_list(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                          const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                          uint64_t nnodes_per_sector, uint64_t max_slots,
                                          const std::unordered_set<uint32_t> &exclude_ids,
                                          std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_slots == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    const auto weights = build_undirected_weights(directed_edges);
    const auto sorted_edges = sorted_undirected_edges(weights);

    // Phase 1: unique edge_star packing (disk-cache mode => no cross-page node reuse).
    std::vector<uint32_t> unique_list;
    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_slots;
    cfg.allow_duplicate_output = false;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &unique_list;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (pack_pages_edge_variant(graph, weights, sorted_edges, directed_edges, 1, EdgePackVariant::Star, cfg, pages,
                                assigned) != 0)
        return -1;

    if (unique_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: edge_star_dup unique phase empty (max_slots=" << max_slots << ")."
                      << std::endl;
        return -1;
    }

    node_list = unique_list;
    const size_t unique_slots = node_list.size();

    // Phase 2: with leftover budget, duplicate hottest pages (earliest seed = hottest).
    size_t page_cursor = 0;
    uint64_t dup_slots = 0;
    while (node_list.size() < max_slots && !pages.empty())
    {
        const auto &page = pages[page_cursor % pages.size()];
        const size_t before = node_list.size();
        for (uint32_t node : page)
        {
            if (node_list.size() >= max_slots)
                break;
            node_list.push_back(node);
            dup_slots++;
        }
        page_cursor++;
        if (page_cursor > pages.size() && node_list.size() == before)
            break;
    }

    diskann::cout << "MERIT disk-cache edge_star_dup: unique_pages=" << pages.size()
                  << " unique_slots=" << unique_slots << " dup_slots=" << dup_slots
                  << " total_slots=" << node_list.size() << std::endl;
    log_disk_cache_node_list_stats(node_list, "edge_star_dup");
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
    if (hot_node_pack_pages(graph, node_expand, weights, k_hops, directed_edges, cfg, pages, assigned) != 0)
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

int compute_directed_beam_relayout_order(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                         const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                         uint64_t nnodes_per_sector, uint32_t beam_width, std::vector<uint32_t> &order)
{
    if (graph.num_points == 0 || nnodes_per_sector == 0)
        return -1;
    if (beam_width == 0)
        beam_width = 1;

    const auto weights = build_undirected_weights(directed_edges);

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = 0;
    cfg.node_expand = &node_expand;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (directed_beam_pack_pages(graph, node_expand, directed_edges, weights, beam_width,
                                 DirectedBeamStarMode::Undirected, DirectedBeamSeedMode::ParentOut, cfg, pages,
                                 assigned) != 0)
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

int compute_edge_star_relayout_order(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                     const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                     uint64_t nnodes_per_sector, std::vector<uint32_t> &order)
{
    if (graph.num_points == 0 || nnodes_per_sector == 0)
        return -1;

    const auto weights = build_undirected_weights(directed_edges);
    const auto sorted_edges = sorted_undirected_edges(weights);

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = 0;
    cfg.node_expand = &node_expand;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (pack_pages_edge_variant(graph, weights, sorted_edges, directed_edges, 1, EdgePackVariant::Star, cfg, pages,
                                assigned) != 0)
        return -1;

    flatten_pages_to_permutation_order(graph, node_expand, pages, order);

    if (order.size() != graph.num_points)
    {
        diskann::cerr << "edge_star relayout: order size " << order.size() << " != num_points " << graph.num_points
                      << std::endl;
        return -1;
    }
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
    if (hot_node_pack_pages(graph, node_expand, weights, k_hops, directed_edges, cfg, pages, assigned) != 0)
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

int compute_hot_node_top4first_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t k_hops, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list)
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
    if (hot_node_pack_pages(graph, node_expand, weights, k_hops, directed_edges, cfg, pages, assigned, true) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: hot-node top4first list empty (max_nodes=" << max_nodes
                      << ", k_hops=" << k_hops << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "node_top4first");
    return 0;
}

static std::vector<std::pair<uint64_t, std::vector<uint32_t>>> build_frontier_templates_from_profile_edges(
    const VamanaGraph &graph, const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
    uint32_t beam_width)
{
    if (beam_width == 0)
        beam_width = 1;

    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> by_parent;
    by_parent.reserve(directed_edges.size());
    for (const auto &edge : directed_edges)
    {
        const uint32_t parent = std::get<0>(edge);
        const uint32_t child = std::get<1>(edge);
        const uint64_t count = std::get<2>(edge);
        if (parent >= graph.num_points || child >= graph.num_points || count == 0)
            continue;
        by_parent[parent].emplace_back(child, count);
    }

    std::map<std::vector<uint32_t>, uint64_t> aggregated;
    for (auto &kv : by_parent)
    {
        auto &children = kv.second;
        std::sort(children.begin(), children.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second || (a.second == b.second && a.first < b.first); });
        const size_t take = std::min(static_cast<size_t>(beam_width), children.size());
        if (take == 0)
            continue;

        std::vector<uint32_t> frontier;
        frontier.reserve(take);
        uint64_t weight = 0;
        for (size_t i = 0; i < take; ++i)
        {
            frontier.push_back(children[i].first);
            weight += children[i].second;
        }
        std::sort(frontier.begin(), frontier.end());
        aggregated[frontier] += weight;
    }

    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> templates;
    templates.reserve(aggregated.size());
    for (auto &kv : aggregated)
        templates.emplace_back(kv.second, std::move(kv.first));
    return templates;
}

static void filter_frontier_templates_for_disk_cache(
    const PagePackConfig &cfg, std::vector<std::pair<uint64_t, std::vector<uint32_t>>> &templates)
{
    std::map<std::vector<uint32_t>, uint64_t> merged;
    for (const auto &tpl : templates)
    {
        std::vector<uint32_t> nodes;
        nodes.reserve(tpl.second.size());
        for (uint32_t node : tpl.second)
        {
            if (disk_cache_node_eligible(node, cfg))
                nodes.push_back(node);
        }
        if (nodes.size() < 2)
            continue;
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        if (nodes.size() < 2)
            continue;
        merged[nodes] += tpl.first;
    }
    templates.clear();
    templates.reserve(merged.size());
    for (auto &kv : merged)
        templates.emplace_back(kv.second, std::move(kv.first));
}

static int frontier_pack_pages(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                               const std::vector<std::pair<uint64_t, std::vector<uint32_t>>> &templates,
                               const PagePackConfig &cfg, std::vector<std::vector<uint32_t>> &pages,
                               std::vector<bool> &assigned)
{
    if (graph.num_points == 0 || cfg.page_cap == 0)
        return -1;

    assigned.assign(graph.num_points, false);
    pages.clear();

    const bool disk_cache_capped = (cfg.max_output_nodes > 0);
    std::vector<int32_t> node_page(graph.num_points, -1);

    auto page_room = [&](size_t page_idx) -> uint64_t {
        return (page_idx < pages.size()) ? (cfg.page_cap - pages[page_idx].size()) : 0;
    };

    auto assign_node_to_page = [&](size_t page_idx, uint32_t node) -> bool {
        if (page_idx >= pages.size() || node >= graph.num_points)
            return false;
        if (node_page[node] >= 0 || !disk_cache_node_eligible(node, cfg))
            return false;
        if (pages[page_idx].size() >= cfg.page_cap)
            return false;
        pages[page_idx].push_back(node);
        node_page[node] = static_cast<int32_t>(page_idx);
        assigned[node] = true;
        return true;
    };

    auto new_page = [&]() -> size_t {
        pages.emplace_back();
        return pages.size() - 1;
    };

    auto placed_count = [&]() -> uint64_t {
        uint64_t n = 0;
        for (int32_t p : node_page)
        {
            if (p >= 0)
                n++;
        }
        return n;
    };

    auto pick_page_for_group = [&](const std::vector<uint32_t> &group) -> size_t {
        auto group_fits = [&](size_t pi) -> bool { return page_room(pi) >= group.size(); };

        size_t best_page = static_cast<size_t>(-1);
        size_t best_overlap = 0;
        for (size_t pi = 0; pi < pages.size(); ++pi)
        {
            if (!group_fits(pi))
                continue;
            size_t overlap = 0;
            for (uint32_t node : pages[pi])
            {
                if (std::find(group.begin(), group.end(), node) != group.end())
                    overlap++;
            }
            if (overlap > best_overlap)
            {
                best_overlap = overlap;
                best_page = pi;
            }
        }
        if (best_overlap > 0 && best_page != static_cast<size_t>(-1))
            return best_page;

        for (size_t pi = 0; pi < pages.size(); ++pi)
        {
            if (pages[pi].empty() && group_fits(pi))
                return pi;
        }

        for (size_t pi = 0; pi < pages.size(); ++pi)
        {
            if (group_fits(pi))
                return pi;
        }
        return new_page();
    };

    auto place_group_on_one_page = [&](const std::vector<uint32_t> &group) {
        if (group.empty())
            return;
        if (disk_cache_capped && placed_count() >= cfg.max_output_nodes)
            return;

        std::vector<uint32_t> batch;
        batch.reserve(group.size());
        for (uint32_t node : group)
        {
            if (node_page[node] < 0 && disk_cache_node_eligible(node, cfg))
                batch.push_back(node);
        }
        if (batch.empty())
            return;

        while (!batch.empty())
        {
            if (disk_cache_capped && placed_count() >= cfg.max_output_nodes)
                break;

            const size_t chunk = std::min(batch.size(), static_cast<size_t>(cfg.page_cap));
            std::vector<uint32_t> piece(batch.begin(), batch.begin() + static_cast<std::ptrdiff_t>(chunk));
            const size_t page_idx = pick_page_for_group(piece);
            for (uint32_t node : piece)
            {
                if (disk_cache_capped && placed_count() >= cfg.max_output_nodes)
                    break;
                assign_node_to_page(page_idx, node);
            }
            batch.erase(batch.begin(), batch.begin() + static_cast<std::ptrdiff_t>(chunk));
        }
    };

    auto sorted_templates = templates;
    std::sort(sorted_templates.begin(), sorted_templates.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    for (const auto &tpl : sorted_templates)
    {
        if (disk_cache_capped && placed_count() >= cfg.max_output_nodes)
            break;
        place_group_on_one_page(tpl.second);
    }

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

    for (uint32_t node : nodes_by_expand)
    {
        if (disk_cache_capped && placed_count() >= cfg.max_output_nodes)
            break;
        if (node_page[node] >= 0 || !disk_cache_node_eligible(node, cfg))
            continue;
        assign_node_to_page(new_page(), node);
    }

    if (disk_cache_capped && cfg.output_nodes != nullptr)
    {
        cfg.output_nodes->clear();
        std::unordered_set<uint32_t> output_seen;
        output_seen.reserve(static_cast<size_t>(cfg.max_output_nodes));
        uint64_t output_count = 0;
        for (const auto &page_nodes : pages)
        {
            append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
            if (output_count >= cfg.max_output_nodes)
                break;
            }
        }

    return 0;
}

static int load_or_build_frontier_templates(const VamanaGraph &graph,
                                            const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                            const std::string &profile_prefix, uint32_t beam_width,
                                            std::vector<std::pair<uint64_t, std::vector<uint32_t>>> &templates)
{
    if (HotnessProfiler::load_frontier_templates(profile_prefix, templates) != 0)
    {
        templates = build_frontier_templates_from_profile_edges(graph, directed_edges, beam_width);
        diskann::cout << "MERIT disk-cache: no hop-frontier profile; synthesized " << templates.size()
                      << " frontier templates from directed edges (beam_width=" << beam_width << ")." << std::endl;
    }
    else
    {
        diskann::cout << "MERIT disk-cache: loaded " << templates.size() << " hop-frontier templates from profile."
                      << std::endl;
    }
    return templates.empty() ? -1 : 0;
}

static std::unordered_map<uint64_t, uint64_t> build_cooccur_weights_from_frontier_templates(
    const std::vector<std::pair<uint64_t, std::vector<uint32_t>>> &templates)
{
    std::unordered_map<uint64_t, uint64_t> weights;
    for (const auto &tpl : templates)
    {
        const uint64_t tpl_w = tpl.first;
        const auto &nodes = tpl.second;
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            for (size_t j = i + 1; j < nodes.size(); ++j)
            {
                const uint64_t ek = undirected_edge_key(nodes[i], nodes[j]);
                weights[ek] += tpl_w;
            }
        }
    }
    return weights;
}

static std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> cooccur_weights_to_directed_edges(
    const std::unordered_map<uint64_t, uint64_t> &weights)
{
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> edges;
    edges.reserve(weights.size() * 2);
    for (const auto &kv : weights)
    {
        const uint32_t u = static_cast<uint32_t>(kv.first >> 32);
        const uint32_t v = static_cast<uint32_t>(kv.first & 0xFFFFFFFFu);
        const uint64_t w = kv.second;
        if (w == 0)
            continue;
        edges.emplace_back(u, v, w);
        edges.emplace_back(v, u, w);
    }
    return edges;
}

static void truncate_frontier_templates_topk(std::vector<std::pair<uint64_t, std::vector<uint32_t>>> &templates,
                                             uint32_t top_k)
{
    if (top_k == 0 || templates.size() <= static_cast<size_t>(top_k))
        return;
    std::sort(templates.begin(), templates.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    templates.resize(static_cast<size_t>(top_k));
}

static void star_fill_page_from_nodes(const VamanaGraph &graph,
                                      const std::unordered_map<uint64_t, uint64_t> &weights,
                                      const std::vector<std::vector<uint32_t>> &adj,
                                      std::unordered_set<uint64_t> &profile_edge_in_list, uint64_t page_cap,
                                      std::vector<uint32_t> &page_nodes, std::unordered_set<uint32_t> &on_page,
                                      const PagePackConfig &cfg, const std::unordered_set<uint32_t> *output_seen,
                                      std::vector<bool> &node_in_list)
{
    std::unordered_set<uint64_t> used_page_edges;
    for (size_t i = 0; i < page_nodes.size(); ++i)
    {
        for (size_t j = i + 1; j < page_nodes.size(); ++j)
            used_page_edges.insert(undirected_edge_key(page_nodes[i], page_nodes[j]));
    }

    while (page_nodes.size() < page_cap)
    {
        uint32_t best_node = static_cast<uint32_t>(graph.num_points);
        uint64_t best_w = 0;
        for (uint32_t on : page_nodes)
        {
            if (on >= adj.size())
                continue;
            for (uint32_t nbr : adj[on])
            {
                const uint64_t ek = undirected_edge_key(on, nbr);
                if (used_page_edges.count(ek) > 0 || on_page.count(nbr) > 0)
                    continue;
                if (profile_edge_in_list.count(ek) == 0)
                    continue;
                if (!is_node_pending_pack(node_in_list, nbr))
                    continue;
                const uint64_t w = edge_weight(weights, on, nbr);
                if (w == 0)
                    continue;
                if (w > best_w || (w == best_w && (best_node == graph.num_points || nbr < best_node)))
                {
                    best_w = w;
                    best_node = nbr;
                }
            }
        }
        if (best_node >= graph.num_points)
            break;
        place_node_on_page(page_nodes, on_page, page_cap, best_node, &cfg, output_seen);
        node_in_list[best_node] = false;
        for (uint32_t on : page_nodes)
        {
            if (on == best_node)
                continue;
            profile_edge_in_list.erase(undirected_edge_key(on, best_node));
            used_page_edges.insert(undirected_edge_key(on, best_node));
        }
    }
}

static void fill_page_after_seed_directed_topk_and_starfill(
    uint32_t seed, const VamanaGraph &graph, const std::vector<std::vector<uint32_t>> &adj,
    const std::unordered_map<uint64_t, uint64_t> &weights,
    const std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint64_t>>> &out_edges,
    std::unordered_set<uint64_t> &profile_edge_in_list, uint32_t beam_width, const PagePackConfig &cfg,
    const std::unordered_set<uint32_t> *seen_ptr, std::vector<bool> &node_in_list, std::vector<uint32_t> &page_nodes,
    std::unordered_set<uint32_t> &on_page, uint64_t &output_count)
{
    if (beam_width == 0)
        beam_width = 1;
    const size_t max_children = static_cast<size_t>(beam_width - 1);
    const bool disk_cache_capped = (cfg.max_output_nodes > 0);

    const auto it = out_edges.find(seed);
    if (it != out_edges.end())
    {
        size_t added = 0;
        for (const auto &child_w : it->second)
        {
            if (added >= max_children || page_nodes.size() >= cfg.page_cap)
                break;
            if (disk_cache_capped && output_count >= cfg.max_output_nodes)
                break;
            const uint32_t child = child_w.first;
            if (!is_node_pending_pack(node_in_list, child))
                continue;
            if (!graph_has_edge(adj, seed, child))
                continue;
            const size_t before = page_nodes.size();
            place_node_on_page(page_nodes, on_page, cfg.page_cap, child, &cfg, seen_ptr);
            if (page_nodes.size() > before)
            {
                node_in_list[child] = false;
                added++;
            }
        }
    }

    star_fill_page_from_nodes(graph, weights, adj, profile_edge_in_list, cfg.page_cap, page_nodes, on_page, cfg,
                              seen_ptr, node_in_list);
}

static int frontier_page_pack(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                              const std::unordered_map<uint64_t, uint64_t> &weights,
                              const std::vector<std::pair<uint64_t, std::vector<uint32_t>>> &templates,
                              const PagePackConfig &cfg, std::vector<std::vector<uint32_t>> &pages,
                              std::vector<bool> &assigned)
{
    if (graph.num_points == 0 || cfg.page_cap == 0)
        return -1;

    const auto adj = build_undirected_adjacency(graph);
    std::unordered_set<uint64_t> profile_edge_in_list;
    init_pending_profile_edges(weights, profile_edge_in_list);

    std::vector<bool> node_in_list;
    const bool disk_cache_capped = (cfg.max_output_nodes > 0);
    init_pending_pack_nodes(static_cast<uint32_t>(graph.num_points), cfg, node_in_list);
    assigned.assign(graph.num_points, false);

    uint64_t output_count = 0;
    std::unordered_set<uint32_t> output_seen;
    if (disk_cache_capped)
        output_seen.reserve(static_cast<size_t>(cfg.max_output_nodes));
    const std::unordered_set<uint32_t> *seen_ptr = disk_cache_capped ? &output_seen : nullptr;

    auto sorted_templates = templates;
    std::sort(sorted_templates.begin(), sorted_templates.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    for (const auto &tpl : sorted_templates)
    {
        if (disk_cache_capped && output_count >= cfg.max_output_nodes)
            break;

        std::vector<uint32_t> page_nodes;
        std::unordered_set<uint32_t> on_page;
        page_nodes.reserve(static_cast<size_t>(cfg.page_cap));
        for (uint32_t node : tpl.second)
        {
            if (page_nodes.size() >= cfg.page_cap)
                break;
            if (!is_node_pending_pack(node_in_list, node))
                continue;
            const size_t before = page_nodes.size();
            place_node_on_page(page_nodes, on_page, cfg.page_cap, node, &cfg, seen_ptr);
            if (page_nodes.size() > before)
                node_in_list[node] = false;
        }
        if (page_nodes.size() < 2)
            continue;

        star_fill_page_from_nodes(graph, weights, adj, profile_edge_in_list, cfg.page_cap, page_nodes, on_page, cfg,
                                  seen_ptr, node_in_list);

        if (page_nodes.empty())
            continue;
        pages.push_back(page_nodes);
        for (uint32_t node : page_nodes)
            assigned[node] = true;
        append_page_nodes_to_disk_cache_list(page_nodes, cfg, output_count, output_seen);
    }

    if (disk_cache_capped && output_count < cfg.max_output_nodes)
    {
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

int compute_frontier_disk_cache_list(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                     const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                     const std::string &profile_prefix, uint64_t nnodes_per_sector,
                                     uint32_t beam_width, uint64_t max_nodes,
                                     const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> templates;
    if (load_or_build_frontier_templates(graph, directed_edges, profile_prefix, beam_width, templates) != 0)
    {
        diskann::cerr << "MERIT disk-cache: frontier template list empty." << std::endl;
        return -1;
    }

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    filter_frontier_templates_for_disk_cache(cfg, templates);
    if (templates.empty())
    {
        diskann::cerr << "MERIT disk-cache: no multi-node frontier templates after eligibility filter." << std::endl;
        return -1;
    }
    diskann::cout << "MERIT disk-cache: " << templates.size() << " eligible frontier templates (size>=2)."
                  << std::endl;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (frontier_pack_pages(graph, node_expand, templates, cfg, pages, assigned) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: frontier packing list empty (max_nodes=" << max_nodes
                      << ", beam_width=" << beam_width << ")." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "frontier");
    return 0;
}

int compute_frontier_page_disk_cache_list(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                          const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                          const std::string &profile_prefix, uint64_t nnodes_per_sector,
                                          uint32_t beam_width, uint64_t max_nodes,
                                          const std::unordered_set<uint32_t> &exclude_ids,
                                          std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> templates;
    if (load_or_build_frontier_templates(graph, directed_edges, profile_prefix, beam_width, templates) != 0)
    {
        diskann::cerr << "MERIT disk-cache: frontier_page template list empty." << std::endl;
        return -1;
    }

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    filter_frontier_templates_for_disk_cache(cfg, templates);
    if (templates.empty())
    {
        diskann::cerr << "MERIT disk-cache: frontier_page no eligible templates." << std::endl;
        return -1;
    }

    const auto weights = build_undirected_weights(directed_edges);
    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (frontier_page_pack(graph, node_expand, weights, templates, cfg, pages, assigned) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: frontier_page list empty." << std::endl;
        return -1;
    }
    log_disk_cache_node_list_stats(node_list, "frontier_page");
    return 0;
}

int compute_frontier_dup_disk_cache_list(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                         const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                         const std::string &profile_prefix, uint64_t nnodes_per_sector,
                                         uint32_t beam_width, uint64_t max_slots,
                                         const std::unordered_set<uint32_t> &exclude_ids,
                                         std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_slots == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> templates;
    if (load_or_build_frontier_templates(graph, directed_edges, profile_prefix, beam_width, templates) != 0)
        return -1;

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_slots;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    filter_frontier_templates_for_disk_cache(cfg, templates);
    if (templates.empty())
        return -1;

    const auto weights = build_undirected_weights(directed_edges);
    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (frontier_page_pack(graph, node_expand, weights, templates, cfg, pages, assigned) != 0)
        return -1;

    if (node_list.empty())
        return -1;

    const size_t unique_slots = node_list.size();
    size_t page_cursor = 0;
    uint64_t dup_slots = 0;
    while (node_list.size() < max_slots && !pages.empty())
    {
        const auto &page = pages[page_cursor % pages.size()];
        const size_t before = node_list.size();
        for (uint32_t node : page)
        {
            if (node_list.size() >= max_slots)
                break;
            node_list.push_back(node);
            dup_slots++;
        }
        page_cursor++;
        if (page_cursor > pages.size() && node_list.size() == before)
            break;
    }

    diskann::cout << "MERIT disk-cache frontier_dup: unique_pages=" << pages.size()
                  << " unique_slots=" << unique_slots << " dup_slots=" << dup_slots
                  << " total_slots=" << node_list.size() << std::endl;
    log_disk_cache_node_list_stats(node_list, "frontier_dup");
    return 0;
}

int compute_cooccur_star_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, const std::string &profile_prefix,
    uint64_t nnodes_per_sector, uint32_t beam_width, uint64_t max_nodes,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> templates;
    if (load_or_build_frontier_templates(graph, directed_edges, profile_prefix, beam_width, templates) != 0)
        return -1;

    const auto co_weights = build_cooccur_weights_from_frontier_templates(templates);
    if (co_weights.empty())
    {
        diskann::cerr << "MERIT disk-cache: co-occur weight map empty." << std::endl;
        return -1;
    }

    const auto co_edges = cooccur_weights_to_directed_edges(co_weights);
    const auto weights = build_undirected_weights(co_edges);

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (pack_pages_edge_variant(graph, weights, sorted_undirected_edges(weights), co_edges, 1, EdgePackVariant::Star,
                                cfg, pages, assigned) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: cooccur_star list empty (max_nodes=" << max_nodes << ")." << std::endl;
        return -1;
    }
    diskann::cout << "MERIT disk-cache cooccur_star: co_pairs=" << co_weights.size()
                  << " templates=" << templates.size() << std::endl;
    log_disk_cache_node_list_stats(node_list, "cooccur_star");
    return 0;
}

int compute_dbeam_cooccur_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, const std::string &profile_prefix,
    uint64_t nnodes_per_sector, uint32_t beam_width, uint64_t max_nodes,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    const uint64_t phase1_cap = max_nodes / 2;
    std::vector<uint32_t> phase1;
    if (compute_directed_beam_disk_cache_list(graph, node_expand, directed_edges, nnodes_per_sector, beam_width,
                                              phase1_cap, exclude_ids, phase1) != 0)
        return -1;

    std::unordered_set<uint32_t> exclude2 = exclude_ids;
    exclude2.insert(phase1.begin(), phase1.end());

    std::vector<uint32_t> phase2;
    const uint64_t phase2_cap = (max_nodes > phase1.size()) ? (max_nodes - phase1.size()) : 0;
    if (phase2_cap > 0)
    {
        if (compute_cooccur_star_disk_cache_list(graph, node_expand, directed_edges, profile_prefix, nnodes_per_sector,
                                                 beam_width, phase2_cap, exclude2, phase2) != 0)
            return -1;
    }

    node_list = std::move(phase1);
    node_list.insert(node_list.end(), phase2.begin(), phase2.end());
    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: dbeam_cooccur list empty (max_nodes=" << max_nodes << ")." << std::endl;
        return -1;
    }
    diskann::cout << "MERIT disk-cache dbeam_cooccur: phase1=" << (node_list.size() - phase2.size())
                  << " phase2=" << phase2.size() << " total=" << node_list.size() << std::endl;
    log_disk_cache_node_list_stats(node_list, "dbeam_cooccur");
    return 0;
}

int compute_frontier_topk_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, const std::string &profile_prefix,
    uint64_t nnodes_per_sector, uint32_t beam_width, uint64_t max_nodes,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list)
{
    node_list.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> templates;
    if (load_or_build_frontier_templates(graph, directed_edges, profile_prefix, beam_width, templates) != 0)
        return -1;

    const uint32_t top_k = (beam_width > 0) ? beam_width * 1000U : 5000U;
    truncate_frontier_templates_topk(templates, top_k);

    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;
    cfg.output_nodes = &node_list;

    filter_frontier_templates_for_disk_cache(cfg, templates);
    if (templates.empty())
    {
        diskann::cerr << "MERIT disk-cache: frontier_topk template list empty after filter." << std::endl;
        return -1;
    }

    const auto weights = build_undirected_weights(directed_edges);
    std::vector<std::vector<uint32_t>> pages;
    std::vector<bool> assigned;
    if (frontier_page_pack(graph, node_expand, weights, templates, cfg, pages, assigned) != 0)
        return -1;

    if (node_list.empty())
    {
        diskann::cerr << "MERIT disk-cache: frontier_topk list empty (max_nodes=" << max_nodes << ")." << std::endl;
        return -1;
    }
    diskann::cout << "MERIT disk-cache frontier_topk: top_k=" << top_k << " templates_used=" << templates.size()
                  << std::endl;
    log_disk_cache_node_list_stats(node_list, "frontier_topk");
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

    std::vector<uint32_t> remaining;
    remaining.reserve(num_points);
    for (uint32_t id = 0; id < num_points; ++id)
    {
        if (exclude_ids.find(id) != exclude_ids.end())
                continue;
        if (in_list.find(id) != in_list.end())
                continue;
        if (id >= node_expand.size() || node_expand[id] == 0)
            continue;
        remaining.push_back(id);
    }

    std::sort(remaining.begin(), remaining.end(), [&](uint32_t a, uint32_t b) {
        if (node_expand[a] != node_expand[b])
            return node_expand[a] > node_expand[b];
        return a < b;
    });

    for (uint32_t id : remaining)
    {
            if (node_list.size() >= max_nodes)
                break;
        node_list.push_back(id);
        in_list.insert(id);
    }

    if (node_list.size() > before)
    {
        diskann::cout << "MERIT disk-cache: appended " << (node_list.size() - before)
                      << " remaining hot nodes (by node_expand desc; total=" << node_list.size() << ")."
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

int merit_dump_disk_cache_pages(const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
                                const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges,
                                const std::string &layout_in, uint64_t nnodes_per_sector, uint32_t k_hops,
                                uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
                                std::vector<std::vector<uint32_t>> &pages)
{
    pages.clear();
    if (max_nodes == 0 || graph.num_points == 0 || nnodes_per_sector == 0)
        return 0;

    const std::string layout = normalize_disk_cache_layout(layout_in);
    const auto weights = build_undirected_weights(directed_edges);
    PagePackConfig cfg;
    cfg.page_cap = nnodes_per_sector;
    cfg.max_output_nodes = max_nodes;
    cfg.skip_output = &exclude_ids;
    cfg.node_expand = &node_expand;

    std::vector<uint32_t> node_list;
    std::vector<bool> assigned;

    if (layout == "node")
    {
        cfg.output_nodes = &node_list;
        if (hot_node_pack_pages(graph, node_expand, weights, k_hops, directed_edges, cfg, pages, assigned) != 0)
            return -1;
        return 0;
    }

    if (layout == "edge" || layout == "jiang")
    {
        const auto sorted_edges = sorted_undirected_edges(weights);
        cfg.output_nodes = &node_list;
        if (pack_pages_edge_variant(graph, weights, sorted_edges, directed_edges, k_hops, EdgePackVariant::Khop, cfg,
                                    pages, assigned) != 0)
            return -1;
        return 0;
    }

    if (layout == "directed_beam")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 1;
        cfg.output_nodes = &node_list;
        if (directed_beam_pack_pages(graph, node_expand, directed_edges, weights, beam_width,
                                     DirectedBeamStarMode::Undirected, DirectedBeamSeedMode::ParentOut, cfg, pages,
                                     assigned) != 0)
            return -1;
        return 0;
    }

    if (is_directed_beam_pct_layout(layout))
    {
        std::vector<SeedPageGroup> seed_groups;
        const double pct = directed_beam_pct_layout_threshold(layout);
        std::string label = std::string("Layout E (") + layout + ")";
        if (compute_directed_beam_pct_disk_cache_list(graph, node_expand, directed_edges, nnodes_per_sector, max_nodes,
                                                      exclude_ids, node_list, pct, label.c_str(), &seed_groups) != 0)
            return -1;
        for (const SeedPageGroup &group : seed_groups)
            for (const auto &page : group.pages)
                pages.push_back(page);
        return 0;
    }

    if (is_directed_seed_replica_layout(layout))
    {
        std::vector<SeedPageGroup> seed_groups;
        const double pct = directed_seed_replica_layout_threshold(layout);
        std::string label = std::string("seed_replica (") + layout + ")";
        if (compute_directed_seed_replica_disk_cache_list(graph, node_expand, directed_edges, nnodes_per_sector,
                                                          max_nodes, exclude_ids, node_list, pct, label.c_str(),
                                                          &seed_groups) != 0)
            return -1;
        for (const SeedPageGroup &group : seed_groups)
            for (const auto &page : group.pages)
                pages.push_back(page);
        return 0;
    }

    if (layout == "directed_beam_starfill")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 1;
        cfg.output_nodes = &node_list;
        if (directed_beam_pack_pages(graph, node_expand, directed_edges, weights, beam_width,
                                     DirectedBeamStarMode::StarFill, DirectedBeamSeedMode::ParentOut, cfg, pages,
                                     assigned) != 0)
            return -1;
        return 0;
    }

    if (layout == "directed_beam_tight")
    {
        const uint32_t beam_width = (k_hops > 0) ? k_hops : 1;
        cfg.output_nodes = &node_list;
        if (directed_beam_pack_pages(graph, node_expand, directed_edges, weights, beam_width,
                                     DirectedBeamStarMode::None, DirectedBeamSeedMode::ParentOut, cfg, pages,
                                     assigned) != 0)
            return -1;
        return 0;
    }

    if (layout == "parent")
    {
        cfg.output_nodes = &node_list;
        if (profile_parent_pack_pages(graph, directed_edges, cfg, pages, assigned) != 0)
            return -1;
        return 0;
    }

    diskann::cerr << "merit_dump_disk_cache_pages: unsupported layout " << layout_in << std::endl;
    return -1;
}

} // namespace diskann
