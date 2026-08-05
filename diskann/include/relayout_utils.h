// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common_includes.h"
#include "windows_customizations.h"

#include <unordered_set>

namespace diskann
{

// Multi-page seed group for directed_beam_pct* (contiguous disk-cache sectors per seed).
struct SeedPageGroup
{
    uint32_t seed = 0;
    uint8_t total_pages = 1; // hot pct pages only (multiread span); cold tail excluded
    std::vector<std::vector<uint32_t>> pages;
    // Remaining profile children (pct < 100): packed in sectors immediately after hot pages.
    std::vector<std::vector<uint32_t>> cold_pages;
};

struct VamanaGraph
{
    uint32_t width = 0;
    uint64_t medoid = 0;
    uint64_t num_points = 0;
    std::vector<std::vector<uint32_t>> adjacency;
};

DISKANN_DLLEXPORT int load_vamana_graph(const std::string &mem_index_file, VamanaGraph &graph);

DISKANN_DLLEXPORT std::string normalize_disk_cache_layout(std::string layout);

// True when layout stores duplicate node ids across disk cache pages (space-for-time).
DISKANN_DLLEXPORT bool disk_cache_layout_allows_replicas(const std::string &layout);

// order[new_id] = old_id. Edge-importance seed + k-hop paths per page.
DISKANN_DLLEXPORT int compute_edge_relayout_order(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t k_hops, std::vector<uint32_t> &order);

// order[new_id] = old_id. Hot-node seed + k-hop neighborhood; page fill via highest avg edge-weight path.
DISKANN_DLLEXPORT int compute_hot_node_relayout_order(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t k_hops, std::vector<uint32_t> &order);

// order[new_id] = old_id. Layout E (directed_beam) page packing over full disk index.
DISKANN_DLLEXPORT int compute_directed_beam_relayout_order(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, std::vector<uint32_t> &order);

// order[new_id] = old_id. edge_star page packing over full disk index.
DISKANN_DLLEXPORT int compute_edge_star_relayout_order(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    std::vector<uint32_t> &order);

// MERIT disk cache: edge-importance page packing; capped at max_nodes after exclude_ids.
DISKANN_DLLEXPORT int compute_edge_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t k_hops, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: edge packing variants (edge_dir | edge_star | edge_u | edge_pair | edge_clique).
DISKANN_DLLEXPORT int compute_edge_variant_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t k_hops, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    const std::string &variant_layout, std::vector<uint32_t> &node_list);

// MERIT disk cache: one star-packed page per hot profile edge; nodes may repeat (space-for-time).
DISKANN_DLLEXPORT int compute_edge_replica_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_slots, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list);

// MERIT disk cache: unique edge_star pages first, then duplicate hottest pages into remaining slots.
DISKANN_DLLEXPORT int compute_edge_star_dup_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_slots, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list);

// MERIT disk cache: hot-node seed + k-hop paths ranked by avg edge weight; capped at max_nodes after exclude_ids.
DISKANN_DLLEXPORT int compute_hot_node_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t k_hops, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: like node, but seeds with top-4=100% profile out-edge coverage first.
DISKANN_DLLEXPORT int compute_hot_node_top4first_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t k_hops, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: profile parent + top directed profile children per page (Layout P).
DISKANN_DLLEXPORT int compute_profile_parent_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list);

// MERIT disk cache: parent + all directed children + undirected star fill.
DISKANN_DLLEXPORT int compute_parent_star_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list);

// MERIT disk cache: 50% directed_beam then 50% edge_star on disjoint nodes.
DISKANN_DLLEXPORT int compute_dbeam_estar_split_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache (Layout E): parent-outgoing-heat seed + top (beam_width-1) directed children + undirected star fill.
DISKANN_DLLEXPORT int compute_directed_beam_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: Layout E + pct outgoing-mass threshold, multi-page seed groups (directed only).
DISKANN_DLLEXPORT bool is_directed_beam_pct_layout(const std::string &layout_norm);
DISKANN_DLLEXPORT double directed_beam_pct_layout_threshold(const std::string &layout_norm);

DISKANN_DLLEXPORT int compute_directed_beam_pct_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list,
    double pct_threshold, const char *layout_label, std::vector<SeedPageGroup> *seed_page_groups = nullptr);

// MERIT disk cache: seed-centric replica packing (no global dedup); map indexes seed_id only.
DISKANN_DLLEXPORT bool is_directed_seed_replica_layout(const std::string &layout_norm);
DISKANN_DLLEXPORT bool is_directed_seed_replica_pct_layout(const std::string &layout_norm);
DISKANN_DLLEXPORT bool is_seed_only_disk_cache_layout(const std::string &layout_norm);
DISKANN_DLLEXPORT double directed_seed_replica_layout_threshold(const std::string &layout_norm);

DISKANN_DLLEXPORT int compute_directed_seed_replica_beam_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list, std::vector<SeedPageGroup> *seed_page_groups = nullptr);

DISKANN_DLLEXPORT int compute_directed_seed_replica_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list,
    double pct_threshold, const char *layout_label, std::vector<SeedPageGroup> *seed_page_groups = nullptr);

DISKANN_DLLEXPORT int compute_directed_beam_pct80_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list,
    std::vector<SeedPageGroup> *seed_page_groups = nullptr);

DISKANN_DLLEXPORT int compute_directed_beam_top4first_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: directed_beam with directed-profile star fill only.
DISKANN_DLLEXPORT int compute_directed_star_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: directed_beam with directed-first hybrid star fill.
DISKANN_DLLEXPORT int compute_directed_beam_hybrid_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: child in-edge heat as seed (directed_beam variant).
DISKANN_DLLEXPORT int compute_directed_beam_inseed_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: 50% parent-out seed pages + 50% child-in seed pages.
DISKANN_DLLEXPORT int compute_directed_beam_dual_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: parent + top (beam_width-1) children only; no star fill.
DISKANN_DLLEXPORT int compute_directed_beam_tight_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: legacy Layout E (directed top-k + undirected star_fill).
DISKANN_DLLEXPORT int compute_directed_beam_starfill_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, uint64_t nnodes_per_sector,
    uint32_t beam_width, uint64_t max_nodes, const std::unordered_set<uint32_t> &exclude_ids,
    std::vector<uint32_t> &node_list);

// MERIT disk cache: hop-frontier pairwise co-occurrence + undirected star fill.
DISKANN_DLLEXPORT int compute_cooccur_star_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, const std::string &profile_prefix,
    uint64_t nnodes_per_sector, uint32_t beam_width, uint64_t max_nodes,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list);

// MERIT disk cache: 50% directed_beam + 50% cooccur_star on disjoint nodes.
DISKANN_DLLEXPORT int compute_dbeam_cooccur_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, const std::string &profile_prefix,
    uint64_t nnodes_per_sector, uint32_t beam_width, uint64_t max_nodes,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list);

// MERIT disk cache: top (beam_width*1000) hop-frontier templates per page + star fill.
DISKANN_DLLEXPORT int compute_frontier_topk_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, const std::string &profile_prefix,
    uint64_t nnodes_per_sector, uint32_t beam_width, uint64_t max_nodes,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list);

// MERIT disk cache: beam frontier co-location (Layout D); beam_width caps template size.
DISKANN_DLLEXPORT int compute_frontier_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, const std::string &profile_prefix,
    uint64_t nnodes_per_sector, uint32_t beam_width, uint64_t max_nodes,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list);

// MERIT disk cache: one hop-frontier template per page + star fill (no cross-template merge).
DISKANN_DLLEXPORT int compute_frontier_page_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, const std::string &profile_prefix,
    uint64_t nnodes_per_sector, uint32_t beam_width, uint64_t max_nodes,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list);

// MERIT disk cache: frontier_page unique pass then duplicate hottest template pages.
DISKANN_DLLEXPORT int compute_frontier_dup_disk_cache_list(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, const std::string &profile_prefix,
    uint64_t nnodes_per_sector, uint32_t beam_width, uint64_t max_slots,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<uint32_t> &node_list);

// After hot/edge/flat/frontier packing: append remaining profile-hot nodes (expand>0) until max_nodes.
DISKANN_DLLEXPORT void append_uncounted_nodes_to_disk_list(const std::vector<uint64_t> &node_expand, uint64_t max_nodes,
                                                          const std::unordered_set<uint32_t> &exclude_ids,
                                                          std::vector<uint32_t> &node_list);

DISKANN_DLLEXPORT int save_relayout_order(const std::string &path, const std::vector<uint32_t> &order,
                                          uint64_t nnodes_per_sector);

DISKANN_DLLEXPORT int load_relayout_order(const std::string &path, std::vector<uint32_t> &order,
                                          uint64_t &nnodes_per_sector);

DISKANN_DLLEXPORT int read_disk_index_nnodes_per_sector(const std::string &disk_index_file, uint64_t &nnodes_per_sector);

// Disk-disk cache page packing (same as search_disk_index MERIT disk cache). Returns packed pages only.
DISKANN_DLLEXPORT int merit_dump_disk_cache_pages(
    const VamanaGraph &graph, const std::vector<uint64_t> &node_expand,
    const std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> &directed_edges, const std::string &layout_in,
    uint64_t nnodes_per_sector, uint32_t k_hops, uint64_t max_nodes,
    const std::unordered_set<uint32_t> &exclude_ids, std::vector<std::vector<uint32_t>> &pages);

} // namespace diskann
