// Generate the two GPS (Gorgeous) disk files from a DiskANN disk index without the GPS graph
// partitioner, whose memory use exceeds our machine at 100M points:
//   graph-only file  (<graph_prefix>_disk_graph.index + _partition.bin), nodes in ID order;
//   graph-replicated (<rep_prefix>_graph_rep.index + _partition.bin), one block per node, with the
//   same neighbor selection as GPS light_graph_replicated_partition (partitioner.h) and the same
//   block format as GPS index_relayout_free_mem (mode 1 and 3).
// Memory: flat graph N * (R + 1) * 4 bytes plus replicated layouts N * C * 4 bytes.
//
// Build: g++ -O3 -fopenmp -std=c++17 gps_layout.cpp -o gps_layout
// Usage: gps_layout <disk.index> <elem_size> <graph_prefix> <rep_prefix> [threads]

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;
static constexpr u64 kSector = 4096;

struct Meta {
  int nr = 0;
  std::vector<u64> v;
  char raw[kSector];
  u64 npts() const { return v[0]; }
  u64 dim() const { return v[1]; }
  u64 max_node_len() const { return v[3]; }
  u64 nnodes_per_sector() const { return v[4]; }
};

static Meta read_meta(const char *path) {
  Meta m;
  std::ifstream in(path, std::ios::binary);
  in.read(m.raw, kSector);
  if (!in) {
    std::cerr << "cannot read " << path << "\n";
    exit(1);
  }
  std::memcpy(&m.nr, m.raw, sizeof(int));
  if (m.nr != 9 && m.nr != 12) {
    std::cerr << "unsupported metadata (nr=" << m.nr << ")\n";
    exit(1);
  }
  m.v.resize(m.nr);
  std::memcpy(m.v.data(), m.raw + 8, sizeof(u64) * m.nr);
  return m;
}

static void write_header(std::ofstream &out, const Meta &m, u64 C, u64 n_blocks) {
  char buf[kSector];
  std::memcpy(buf, m.raw, kSector);
  u64 *meta = reinterpret_cast<u64 *>(buf + 8);
  meta[4] = C;
  meta[m.nr - 1] = n_blocks * kSector + kSector;
  out.write(buf, kSector);
}

// layouts == nullptr: partition p holds IDs [p * ids_per_part, (p + 1) * ids_per_part).
// Otherwise partition p holds layouts[p * C, p * C + sizes[p]).
static void write_partition(const std::string &path, u64 C, u64 n_parts, u64 nd, const std::vector<u32> *layouts,
                            const std::vector<uint8_t> *sizes, u64 ids_per_part) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(&C), sizeof(u64));
  out.write(reinterpret_cast<const char *>(&n_parts), sizeof(u64));
  out.write(reinterpret_cast<const char *>(&nd), sizeof(u64));
  std::vector<u32> ids;
  for (u64 p = 0; p < n_parts; p++) {
    if (layouts) {
      u32 s = (*sizes)[p];
      out.write(reinterpret_cast<const char *>(&s), sizeof(u32));
      out.write(reinterpret_cast<const char *>(layouts->data() + p * C), sizeof(u32) * s);
    } else {
      u64 st = p * ids_per_part, ed = std::min(nd, st + ids_per_part);
      u32 s = ed - st;
      ids.resize(s);
      for (u64 i = st; i < ed; i++) ids[i - st] = i;
      out.write(reinterpret_cast<const char *>(&s), sizeof(u32));
      out.write(reinterpret_cast<const char *>(ids.data()), sizeof(u32) * s);
    }
  }
  std::vector<u32> id2pid(nd);
  for (u64 i = 0; i < nd; i++) id2pid[i] = layouts ? i : i / ids_per_part;
  out.write(reinterpret_cast<const char *>(id2pid.data()), sizeof(u32) * nd);
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "usage: gps_layout <disk.index> <elem_size> <graph_prefix> <rep_prefix> [threads]\n";
    return 1;
  }
  const char *index_path = argv[1];
  const u64 elem = std::stoull(argv[2]);
  const std::string graph_prefix = argv[3], rep_prefix = argv[4];
  if (argc > 5) omp_set_num_threads(std::atoi(argv[5]));

  Meta m = read_meta(index_path);
  const u64 nd = m.npts(), dim = m.dim(), node_len = m.max_node_len(), nps = m.nnodes_per_sector();
  const u64 emb_len = dim * elem;
  const u64 graph_node_len = node_len - emb_len;  // u32 degree + R neighbors
  const u64 R = graph_node_len / sizeof(u32) - 1;
  const u64 n_index_sectors = (nd + nps - 1) / nps;
  std::cout << "nd=" << nd << " dim=" << dim << " max_node_len=" << node_len << " nps=" << nps << " R=" << R
            << std::endl;

  // Pass 1: load the flat graph.
  std::vector<u32> graph(nd * (R + 1));
  {
    std::ifstream in(index_path, std::ios::binary);
    in.seekg(kSector);
    const u64 batch = 65536;
    std::vector<char> buf(batch * kSector);
    for (u64 s = 0; s < n_index_sectors; s += batch) {
      u64 ns = std::min(batch, n_index_sectors - s);
      in.read(buf.data(), ns * kSector);
#pragma omp parallel for
      for (u64 k = 0; k < ns; k++) {
        for (u64 j = 0; j < nps; j++) {
          u64 id = (s + k) * nps + j;
          if (id >= nd) break;
          const char *node = buf.data() + k * kSector + j * node_len + emb_len;
          std::memcpy(&graph[id * (R + 1)], node, graph_node_len);
        }
      }
    }
    std::cout << "graph loaded" << std::endl;
  }
  auto nbrs = [&](u64 id) { return &graph[id * (R + 1) + 1]; };
  auto deg = [&](u64 id) { return graph[id * (R + 1)]; };

  // Graph-only file: C adjacency lists per page in ID order.
  {
    const u64 C = kSector / (graph_node_len + sizeof(u32));
    const u64 n_pages = (nd + C - 1) / C;
    std::ofstream out(graph_prefix + "_disk_graph.index", std::ios::binary);
    write_header(out, m, C, n_pages);
    const u64 batch = 65536;
    std::vector<char> buf(batch * kSector);
    for (u64 p0 = 0; p0 < n_pages; p0 += batch) {
      u64 np = std::min(batch, n_pages - p0);
      std::memset(buf.data(), 0, np * kSector);
#pragma omp parallel for
      for (u64 k = 0; k < np; k++) {
        char *page = buf.data() + k * kSector;
        u64 st = (p0 + k) * C, ed = std::min(nd, st + C);
        u32 s = ed - st;
        std::memcpy(page, &s, sizeof(u32));
        for (u64 i = st; i < ed; i++) {
          u32 id = i;
          std::memcpy(page + sizeof(u32) * (1 + i - st), &id, sizeof(u32));
          std::memcpy(page + sizeof(u32) * (1 + C) + (i - st) * graph_node_len, &graph[i * (R + 1)],
                      graph_node_len);
        }
      }
      out.write(buf.data(), np * kSector);
    }
    write_partition(graph_prefix + "_partition.bin", C, n_pages, nd, nullptr, nullptr, C);
    std::cout << "graph-only done, C=" << C << " pages=" << n_pages << std::endl;
  }

  // Graph-replicated file: one block per node = vector + layout + C adjacency lists.
  {
    const u64 C = (kSector - emb_len) / (graph_node_len + sizeof(u32));
    std::ofstream out(rep_prefix + "_graph_rep.index", std::ios::binary);
    write_header(out, m, C, nd);
    std::ifstream in(index_path, std::ios::binary);
    in.seekg(kSector);
    const u64 batch_sectors = 16384;
    std::vector<char> ibuf(batch_sectors * kSector), obuf(batch_sectors * nps * kSector);
    std::vector<u32> layouts(nd * C);
    std::vector<uint8_t> sizes(nd);
    for (u64 s = 0; s < n_index_sectors; s += batch_sectors) {
      u64 ns = std::min(batch_sectors, n_index_sectors - s);
      in.read(ibuf.data(), ns * kSector);
      u64 id0 = s * nps, id1 = std::min(nd, (s + ns) * nps);
      std::memset(obuf.data(), 0, (id1 - id0) * kSector);
#pragma omp parallel for schedule(dynamic, 256)
      for (u64 id = id0; id < id1; id++) {
        // Same selection as GPS light_graph_replicated_partition.
        std::vector<u32> part;
        part.reserve(C);
        part.push_back(id);
        std::vector<u32> c_nbrs;
        c_nbrs.reserve(64);
        for (size_t i = 0; i < part.size(); i++) {
          const u32 *nb = nbrs(part[i]);
          for (u32 t = 0; t < deg(part[i]); t++) c_nbrs.emplace_back(nb[t]);
          auto rng = std::default_random_engine{};
          std::shuffle(c_nbrs.begin(), c_nbrs.end(), rng);
          for (size_t j = 0; j < c_nbrs.size(); j++) {
            if (part.size() == C) break;
            part.push_back(c_nbrs[j]);
          }
          if (part.size() == C) break;
        }
        char *blk = obuf.data() + (id - id0) * kSector;
        u64 local = id - s * nps;
        std::memcpy(blk, ibuf.data() + (local / nps) * kSector + (local % nps) * node_len, emb_len);
        u64 off = emb_len;
        u32 sz = part.size();
        std::copy(part.begin(), part.end(), layouts.begin() + id * C);
        sizes[id] = sz;
        std::memcpy(blk + off, &sz, sizeof(u32));
        std::memcpy(blk + off + sizeof(u32), part.data(), sizeof(u32) * sz);
        off += sizeof(u32) * (1 + C);
        for (u32 j = 0; j < sz; j++) {
          std::memcpy(blk + off, &graph[(u64)part[j] * (R + 1)], graph_node_len);
          off += graph_node_len;
        }
      }
      out.write(obuf.data(), (id1 - id0) * kSector);
      if ((s / batch_sectors) % 64 == 0) std::cout << "rep " << id1 << "/" << nd << std::endl;
    }
    write_partition(rep_prefix + "_partition.bin", C, nd, nd, &layouts, &sizes, 0);
    std::cout << "graph-replicated done, C=" << C << std::endl;
  }
  return 0;
}
