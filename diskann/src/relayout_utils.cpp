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

std::string normalize_relayout(std::string layout)
{
    for (char &c : layout)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (layout == "khop" || layout == "hotnode" || layout == "hot-node") return "node";
    if (layout == "c") return "edge";
    if (layout == "e" || layout == "layoute" || layout == "layout_e" || layout == "layout-e") return "directed_beam";
    if (layout == "edgestar") return "edge_star";
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

static void star_fill_page_from_nodes(const VamanaGraph &graph,
                                      const std::unordered_map<uint64_t, uint64_t> &weights,
                                      const std::vector<std::vector<uint32_t>> &adj,
                                      std::unordered_set<uint64_t> &profile_edge_in_list, uint64_t page_cap,
                                      std::vector<uint32_t> &page_nodes, std::unordered_set<uint32_t> &on_page,
                                      const PagePackConfig &cfg, const std::unordered_set<uint32_t> *output_seen,
                                      std::vector<bool> &node_in_list);

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
                                      std::vector<bool> &assigned, bool replica_no_dedup = false)
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
