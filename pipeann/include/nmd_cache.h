#pragma once

// Cold N + prefetch + D cache in front of PipeANN's SSD reads.
// Lookup order: N, then the in-memory prefetch page, then a page that
// prefetch previously evicted to the D file, then the base index.
// Only a node that search actually expands is admitted to N and heated.
// A hot node opens one prefetch page: slot 0 is itself, the other slots are
// its hottest neighbors that are already in N. The page is written to D only
// when prefetch evicts it.

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <queue>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace pipeann {

class NMDCache {
 public:
  NMDCache() = default;
  NMDCache(const NMDCache &) = delete;
  NMDCache &operator=(const NMDCache &) = delete;
  ~NMDCache() {
    if (dfd_ >= 0) {
      close(dfd_);
    }
  }

  bool enabled() const {
    return enabled_;
  }

  void init(uint64_t npoints, uint64_t node_len, uint64_t nnodes_per_sector, uint64_t vector_bytes,
            const std::string &dpath, const std::string &index_path, uint32_t entry_id) {
    const char *on = std::getenv("PIPEANN_NMD");
    if (on == nullptr || on[0] == '\0' || on[0] == '0') {
      return;
    }
    if (node_len == 0 || node_len > (1u << 20) || vector_bytes + 8 > node_len) {
      std::fprintf(stderr, "NMD disabled: bad node layout len=%lu vec=%lu\n", (unsigned long) node_len,
                   (unsigned long) vector_bytes);
      return;
    }

    node_len_ = (size_t) node_len;
    vector_bytes_ = vector_bytes;
    npoints_ = npoints;
    sector_nps_ = nnodes_per_sector;
    entry_id_ = entry_id;
    index_path_ = index_path;
    slots_ = nnodes_per_sector > 1 ? (int) std::min<uint64_t>(nnodes_per_sector, 16) : 8;
    page_bytes_ = (size_t) slots_ * node_len_;
    heat_th_ = env_u32("PIPEANN_NMD_HEAT", 2);
    if (heat_th_ < 1) {
      heat_th_ = 1;
    }

    uint32_t nmb = env_u32("PIPEANN_NMD_NMB", 32);
    uint32_t pmb = env_u32("PIPEANN_NMD_PMB", 4);
    ncap_ = std::max<size_t>(32, (size_t) nmb * 1024 * 1024 / node_len_);
    pcap_ = std::max<size_t>(4, (size_t) pmb * 1024 * 1024 / page_bytes_);
    // A full id->heat array is 4 bytes per point. Past 8M points, keep a map.
    if (npoints > 0 && npoints <= 8000000) {
      heat_vec_.assign((size_t) npoints, 0);
    }

    pages_.clear();
    pages_.resize(pcap_);
    free_.clear();
    free_.reserve(pcap_);
    for (uint32_t i = 0; i < (uint32_t) pcap_; ++i) {
      free_.push_back(i);
    }

    std::string path = dpath;
    if (const char *ep = std::getenv("PIPEANN_NMD_PATH")) {
      if (ep[0] != '\0') {
        path = ep;
      }
    }
    dfd_ = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (dfd_ < 0) {
      std::fprintf(stderr, "NMD D file open failed (%s); prefetch will drop pages on eviction\n", path.c_str());
    }
    path_ = path;
    enabled_ = true;
    const char *pre = std::getenv("PIPEANN_NMD_PREFILL");
    if (pre != nullptr && pre[0] != '\0' && pre[0] != '0') {
      prefill_bfs(env_u32("PIPEANN_NMD_DMB", 64));
    }
    std::fprintf(stderr,
                 "NMD on: N=%uMB (%zu nodes) prefetch=%uMB (%zu pages x %d slots) heat>=%u node=%zu points=%lu "
                 "prefill=%u pages (%lu ids)\n",
                 nmb, ncap_, pmb, pcap_, slots_, heat_th_, node_len_, (unsigned long) npoints, prefill_pages_,
                 (unsigned long) prefill_ids_);
  }

  // Copy one node's bytes into dst. kind: 1 N, 2 prefetch, 3 D.
  bool lookup(uint32_t id, char *dst, int &kind) {
    if (!enabled_) {
      return false;
    }
    uint32_t dpage = 0;
    uint16_t dslot = 0;
    uint32_t cov_mem[16];
    int cov_count = 0;
    bool want_cov = false;
    {
      std::lock_guard<std::mutex> g(mu_);
      auto nit = nid_.find(id);
      if (nit != nid_.end()) {
        std::memcpy(dst, nslots_[nit->second].bytes.data(), node_len_);
        kind = 1;
        n_n_.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      auto pit = pref_.find(id);
      if (pit != pref_.end()) {
        Page &page = pages_[pit->second.page];
        std::memcpy(dst, page.bytes.data() + (size_t) pit->second.slot * node_len_, node_len_);
        kind = 2;
        n_p_.fetch_add(1, std::memory_order_relaxed);
        if (pit->second.slot == 0) {
          note_coverage(dst, page.mem, page.count);
        }
        return true;
      }
      auto dit = dloc_.find(id);
      if (dit == dloc_.end() || dfd_ < 0) {
        n_miss_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      dpage = dit->second.page;
      dslot = dit->second.slot;
      if (dslot == 0 && dpage < dmeta_.size()) {
        want_cov = true;
        cov_count = dmeta_[dpage].count;
        if (cov_count > 16) {
          cov_count = 16;
        }
        std::memcpy(cov_mem, dmeta_[dpage].mem, sizeof(uint32_t) * (size_t) cov_count);
      }
    }

    std::vector<char> page(page_bytes_);
    ssize_t n = pread(dfd_, page.data(), page_bytes_, (off_t) dpage * (off_t) page_bytes_);
    if (n != (ssize_t) page_bytes_) {
      n_miss_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    std::memcpy(dst, page.data() + (size_t) dslot * node_len_, node_len_);
    kind = 3;
    n_d_.fetch_add(1, std::memory_order_relaxed);
    if (want_cov) {
      note_coverage(dst, cov_mem, cov_count);
    }
    return true;
  }

  void on_expand(uint32_t id, const char *bytes, const uint32_t *nbrs, uint32_t nn) {
    if (!enabled_ || bytes == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> g(mu_);
    uint32_t h = bump(id);
    admit_locked(id, bytes, h);
    n_expand_.fetch_add(1, std::memory_order_relaxed);

    auto rit = rev_.find(id);
    if (rit != rev_.end()) {
      std::vector<uint32_t> seeds = rit->second;
      for (uint32_t seed : seeds) {
        if (seed != id) {
          upgrade_locked(seed, id, h);
        }
      }
    }
    if (h >= heat_th_) {
      rebuild_locked(id, bytes, nbrs, nn);
    }
  }

  void print() const {
    if (!enabled_) {
      std::fprintf(stderr, "NMD disabled\n");
      return;
    }
    uint64_t cov_pages = cov_pages_.load(std::memory_order_relaxed);
    uint64_t cov_nbrs = cov_nbrs_.load(std::memory_order_relaxed);
    uint64_t cov_hit = cov_hit_.load(std::memory_order_relaxed);
    uint64_t fill_pages = fill_pages_.load(std::memory_order_relaxed);
    uint64_t fill_nbrs = fill_nbrs_.load(std::memory_order_relaxed);
    uint64_t fill_hit = fill_hit_.load(std::memory_order_relaxed);
    uint64_t fill_slots = fill_slots_.load(std::memory_order_relaxed);
    double ratio = cov_nbrs == 0 ? 0.0 : (double) cov_hit / (double) cov_nbrs;
    double mean_cov = cov_pages == 0 ? 0.0 : (double) cov_hit / (double) cov_pages;
    double fill_ratio = fill_nbrs == 0 ? 0.0 : (double) fill_hit / (double) fill_nbrs;
    double fill_mean = fill_pages == 0 ? 0.0 : (double) fill_hit / (double) fill_pages;
    double slot_prec = fill_slots == 0 ? 0.0 : (double) fill_hit / (double) fill_slots;
    std::lock_guard<std::mutex> g(mu_);
    uint64_t live_pages = 0, live_nbrs = 0, live_hit = 0, live_slots = 0;
    for (const Page &page : pages_) {
      if (page.seed == UINT32_MAX || page.count < 2 || page.bytes.size() < node_len_) {
        continue;
      }
      if (vector_bytes_ + 4 > node_len_) {
        break;
      }
      const char *node = page.bytes.data();
      uint16_t nn = *reinterpret_cast<const uint16_t *>(node + vector_bytes_);
      if (nn == 0 || nn > 512 || vector_bytes_ + 4 + (uint64_t) nn * 4 > node_len_) {
        continue;
      }
      const uint32_t *nbrs = reinterpret_cast<const uint32_t *>(node + vector_bytes_ + 4);
      uint32_t hit = 0;
      for (uint16_t i = 0; i < nn; ++i) {
        for (int s = 1; s < page.count; ++s) {
          if (page.mem[s] == nbrs[i]) {
            hit++;
            break;
          }
        }
      }
      live_pages++;
      live_nbrs += nn;
      live_hit += hit;
      live_slots += (uint64_t) (page.count - 1);
    }
    double live_ratio = live_nbrs == 0 ? 0.0 : (double) live_hit / (double) live_nbrs;
    double live_mean = live_pages == 0 ? 0.0 : (double) live_hit / (double) live_pages;
    std::fprintf(stderr,
                 "NMD n_hit=%lu pref_hit=%lu d_hit=%lu miss=%lu expand=%lu d_pages=%lu | "
                 "N=%zu prefetch_pages=%zu/%zu\n"
                 "NMD seed_read_pages=%lu read_coverage=%.4f mean_covered=%.2f\n"
                 "NMD page_fills=%lu fill_coverage=%.4f mean_covered=%.2f slot_precision=%.4f\n"
                 "NMD prefetch_now=%lu live_coverage=%.4f live_mean_covered=%.2f of %.1f\n",
                 (unsigned long) n_n_.load(std::memory_order_relaxed),
                 (unsigned long) n_p_.load(std::memory_order_relaxed),
                 (unsigned long) n_d_.load(std::memory_order_relaxed),
                 (unsigned long) n_miss_.load(std::memory_order_relaxed),
                 (unsigned long) n_expand_.load(std::memory_order_relaxed),
                 (unsigned long) d_pages_.load(std::memory_order_relaxed), nslots_.size(), seed_page_.size(), pcap_,
                 (unsigned long) cov_pages, ratio, mean_cov, (unsigned long) fill_pages, fill_ratio, fill_mean,
                 slot_prec, (unsigned long) live_pages, live_ratio, live_mean,
                 live_pages == 0 ? 0.0 : (double) live_nbrs / (double) live_pages);
  }

 private:
  struct NSlot {
    uint32_t id = 0;
    uint32_t heat = 0;
    std::vector<char> bytes;
  };
  struct Loc {
    uint32_t page = 0;
    uint16_t slot = 0;
  };
  struct Page {
    uint32_t seed = UINT32_MAX;
    uint32_t seed_heat = 0;
    uint32_t mem[16] = {};
    uint32_t heat[16] = {};
    int count = 0;
    std::vector<char> bytes;
  };
  struct DMeta {
    uint32_t mem[16] = {};
    int count = 0;
  };

  static uint32_t env_u32(const char *name, uint32_t fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
      return fallback;
    }
    return (uint32_t) std::strtoul(v, nullptr, 10);
  }

  uint32_t heat_of(uint32_t id) const {
    if (!heat_vec_.empty()) {
      return id < heat_vec_.size() ? heat_vec_[id] : 0;
    }
    auto it = heat_map_.find(id);
    return it == heat_map_.end() ? 0 : it->second;
  }

  uint32_t bump(uint32_t id) {
    if (!heat_vec_.empty()) {
      if (id >= heat_vec_.size()) {
        return 1;
      }
      return ++heat_vec_[id];
    }
    return ++heat_map_[id];
  }

  void admit_locked(uint32_t id, const char *bytes, uint32_t h) {
    auto it = nid_.find(id);
    if (it != nid_.end()) {
      NSlot &s = nslots_[it->second];
      std::memcpy(s.bytes.data(), bytes, node_len_);
      s.heat = h;
      return;
    }
    if (nslots_.size() < ncap_) {
      NSlot s;
      s.id = id;
      s.heat = h;
      s.bytes.assign(bytes, bytes + node_len_);
      nid_[id] = (uint32_t) nslots_.size();
      nslots_.push_back(std::move(s));
      return;
    }
    size_t best = nhand_ % nslots_.size();
    for (int i = 1; i < 32; ++i) {
      size_t j = (nhand_ + (size_t) i) % nslots_.size();
      if (nslots_[j].heat < nslots_[best].heat) {
        best = j;
      }
    }
    nhand_ = (best + 1) % nslots_.size();
    nid_.erase(nslots_[best].id);
    nslots_[best].id = id;
    nslots_[best].heat = h;
    std::memcpy(nslots_[best].bytes.data(), bytes, node_len_);
    nid_[id] = (uint32_t) best;
  }

  bool copy_n_to(uint32_t id, char *dst) const {
    auto it = nid_.find(id);
    if (it == nid_.end()) {
      return false;
    }
    std::memcpy(dst, nslots_[it->second].bytes.data(), node_len_);
    return true;
  }

  void unindex_locked(uint32_t pi) {
    Page &page = pages_[pi];
    for (int s = 0; s < page.count; ++s) {
      auto it = pref_.find(page.mem[s]);
      if (it != pref_.end() && it->second.page == pi) {
        pref_.erase(it);
      }
    }
    if (page.seed != UINT32_MAX) {
      auto sit = seed_page_.find(page.seed);
      if (sit != seed_page_.end() && sit->second == pi) {
        seed_page_.erase(sit);
      }
    }
  }

  void evict_locked(uint32_t pi) {
    Page &page = pages_[pi];
    if (page.seed == UINT32_MAX) {
      return;
    }
    if (dfd_ >= 0 && page.count >= 1 && page.bytes.size() == page_bytes_) {
      uint32_t dp = dnext_++;
      ssize_t wr = pwrite(dfd_, page.bytes.data(), page_bytes_, (off_t) dp * (off_t) page_bytes_);
      if (wr == (ssize_t) page_bytes_) {
        if (dmeta_.size() <= dp) {
          dmeta_.resize((size_t) dp + 1);
        }
        int n = page.count > 16 ? 16 : page.count;
        dmeta_[dp].count = n;
        std::memcpy(dmeta_[dp].mem, page.mem, sizeof(uint32_t) * (size_t) n);
        for (int s = 0; s < n; ++s) {
          dloc_[page.mem[s]] = Loc{dp, (uint16_t) s};
        }
        d_pages_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    unindex_locked(pi);
    page.seed = UINT32_MAX;
    page.count = 0;
    page.seed_heat = 0;
  }

  uint32_t pick_victim_locked() const {
    uint32_t best = 0;
    uint32_t best_h = UINT32_MAX;
    bool found = false;
    for (uint32_t i = 0; i < (uint32_t) pages_.size(); ++i) {
      if (pages_[i].seed == UINT32_MAX) {
        continue;
      }
      if (!found || pages_[i].seed_heat < best_h ||
          (pages_[i].seed_heat == best_h && pages_[i].seed < pages_[best].seed)) {
        found = true;
        best = i;
        best_h = pages_[i].seed_heat;
      }
    }
    return best;
  }

  uint32_t alloc_page_locked() {
    uint32_t pi;
    if (!free_.empty()) {
      pi = free_.back();
      free_.pop_back();
    } else {
      pi = pick_victim_locked();
      evict_locked(pi);
    }
    pages_[pi].bytes.assign(page_bytes_, 0);
    pages_[pi].count = 0;
    pages_[pi].seed = UINT32_MAX;
    pages_[pi].seed_heat = 0;
    return pi;
  }

  void link_rev(uint32_t seed, const uint32_t *nbrs, uint32_t nn) {
    uint32_t lim = nn > 512 ? 512 : nn;
    for (uint32_t i = 0; i < lim; ++i) {
      auto &v = rev_[nbrs[i]];
      if (v.size() >= 8) {
        continue;
      }
      bool found = false;
      for (uint32_t s : v) {
        if (s == seed) {
          found = true;
          break;
        }
      }
      if (!found) {
        v.push_back(seed);
      }
    }
  }

  void upgrade_locked(uint32_t seed, uint32_t id, uint32_t h) {
    auto sit = seed_page_.find(seed);
    if (sit == seed_page_.end()) {
      return;
    }
    if (nid_.find(id) == nid_.end()) {
      return;
    }
    uint32_t pi = sit->second;
    Page &page = pages_[pi];
    for (int s = 0; s < page.count; ++s) {
      if (page.mem[s] == id) {
        page.heat[s] = h;
        copy_n_to(id, page.bytes.data() + (size_t) s * node_len_);
        if (s == 0) {
          page.seed_heat = h;
        }
        return;
      }
    }
    int dest = -1;
    if (page.count < slots_) {
      dest = page.count;
      page.count++;
    } else {
      int cold = -1;
      uint32_t cold_h = UINT32_MAX;
      for (int s = 1; s < page.count; ++s) {
        if (page.heat[s] < cold_h) {
          cold_h = page.heat[s];
          cold = s;
        }
      }
      if (cold < 0 || h <= cold_h) {
        return;
      }
      auto old = pref_.find(page.mem[cold]);
      if (old != pref_.end() && old->second.page == pi && old->second.slot == (uint16_t) cold) {
        pref_.erase(old);
      }
      dest = cold;
    }
    if (!copy_n_to(id, page.bytes.data() + (size_t) dest * node_len_)) {
      return;
    }
    page.mem[dest] = id;
    page.heat[dest] = h;
    pref_[id] = Loc{pi, (uint16_t) dest};
  }

  void rebuild_locked(uint32_t seed, const char *seed_bytes, const uint32_t *nbrs, uint32_t nn) {
    if (pages_.empty() || seed_bytes == nullptr) {
      return;
    }
    auto it = seed_page_.find(seed);
    bool created = it == seed_page_.end();
    uint32_t pi = created ? alloc_page_locked() : it->second;
    if (!created) {
      unindex_locked(pi);
    }
    Page &page = pages_[pi];
    page.bytes.assign(page_bytes_, 0);
    std::memcpy(page.bytes.data(), seed_bytes, node_len_);
    page.seed = seed;
    page.mem[0] = seed;
    page.heat[0] = heat_of(seed);
    page.seed_heat = page.heat[0];
    page.count = 1;
    pref_[seed] = Loc{pi, 0};
    seed_page_[seed] = pi;

    struct Item {
      uint32_t id;
      uint32_t h;
    };
    std::vector<Item> cand;
    uint32_t lim = nn > 512 ? 512 : nn;
    cand.reserve(lim);
    for (uint32_t i = 0; i < lim; ++i) {
      uint32_t nid = nbrs[i];
      if (nid == seed || nid_.find(nid) == nid_.end()) {
        continue;
      }
      cand.push_back(Item{nid, heat_of(nid)});
    }
    std::sort(cand.begin(), cand.end(), [](const Item &a, const Item &b) {
      return a.h > b.h || (a.h == b.h && a.id < b.id);
    });
    for (const Item &c : cand) {
      if (page.count >= slots_) {
        break;
      }
      int s = page.count;
      if (!copy_n_to(c.id, page.bytes.data() + (size_t) s * node_len_)) {
        continue;
      }
      page.mem[s] = c.id;
      page.heat[s] = c.h;
      pref_[c.id] = Loc{pi, (uint16_t) s};
      page.count++;
    }
    if (created && nbrs != nullptr && nn > 0) {
      link_rev(seed, nbrs, nn);
    }
    note_page_fill(seed_bytes, page.mem, page.count);
  }

  bool read_disk_node(int fd, uint32_t id, char *dst) const {
    if (fd < 0 || id >= npoints_ || node_len_ == 0) {
      return false;
    }
    const uint64_t nps = sector_nps_;
    const uint64_t sector = 1 + (nps > 0 ? (uint64_t) id / nps : (uint64_t) id);
    const uint64_t inner = nps > 0 ? ((uint64_t) id % nps) * node_len_ : 0;
    const off_t off = (off_t) (sector * 4096ull + inner);
    return pread(fd, dst, node_len_, off) == (ssize_t) node_len_;
  }

  // Before any query: BFS from the entry, one page per visited node, written
  // straight into D. N and prefetch stay empty. There is no runtime heat yet,
  // so the other slots are the seed's neighbor-list prefix, up to a full page.
  void prefill_bfs(uint32_t dmb) {
    if (dfd_ < 0 || npoints_ == 0 || entry_id_ >= npoints_ || page_bytes_ == 0 || dmb == 0) {
      std::fprintf(stderr, "NMD prefill skipped\n");
      return;
    }
    const uint32_t cap = (uint32_t) ((uint64_t) dmb * 1024ull * 1024ull / page_bytes_);
    if (cap == 0) {
      return;
    }
    const int fd = open(index_path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      std::fprintf(stderr, "NMD prefill could not open %s\n", index_path_.c_str());
      return;
    }
    std::vector<char> node(node_len_);
    std::vector<char> page(page_bytes_, 0);
    std::vector<char> seen((size_t) npoints_, 0);
    std::queue<uint32_t> bfs;
    bfs.push(entry_id_);
    seen[entry_id_] = 1;
    uint64_t covered = 0, nbrs_total = 0;
    while (!bfs.empty() && prefill_pages_ < cap) {
      const uint32_t id = bfs.front();
      bfs.pop();
      if (!read_disk_node(fd, id, node.data())) {
        continue;
      }
      uint16_t nn = 0;
      const uint32_t *nbrs = nullptr;
      if (vector_bytes_ + 4 <= node_len_) {
        nn = *reinterpret_cast<const uint16_t *>(node.data() + vector_bytes_);
        if (nn > 512 || vector_bytes_ + 4 + (uint64_t) nn * 4 > node_len_) {
          nn = 0;
        } else {
          nbrs = reinterpret_cast<const uint32_t *>(node.data() + vector_bytes_ + 4);
        }
      }
      for (uint16_t i = 0; i < nn; ++i) {
        const uint32_t nbr = nbrs[i];
        if (nbr < npoints_ && !seen[nbr]) {
          seen[nbr] = 1;
          bfs.push(nbr);
        }
      }
      std::fill(page.begin(), page.end(), 0);
      std::memcpy(page.data(), node.data(), node_len_);
      uint32_t mem[16];
      mem[0] = id;
      int count = 1;
      for (uint16_t i = 0; i < nn && count < slots_ && count < 16; ++i) {
        const uint32_t nbr = nbrs[i];
        if (nbr == id || nbr >= npoints_) {
          continue;
        }
        bool dup = false;
        for (int s = 0; s < count; ++s) {
          if (mem[s] == nbr) {
            dup = true;
            break;
          }
        }
        if (dup || !read_disk_node(fd, nbr, page.data() + (size_t) count * node_len_)) {
          continue;
        }
        mem[count++] = nbr;
      }
      const uint32_t dp = dnext_++;
      if (pwrite(dfd_, page.data(), page_bytes_, (off_t) dp * (off_t) page_bytes_) != (ssize_t) page_bytes_) {
        break;
      }
      if (dmeta_.size() <= dp) {
        dmeta_.resize((size_t) dp + 1);
      }
      dmeta_[dp].count = count;
      std::memset(dmeta_[dp].mem, 0, sizeof(dmeta_[dp].mem));
      std::memcpy(dmeta_[dp].mem, mem, sizeof(uint32_t) * (size_t) count);
      for (int s = 0; s < count; ++s) {
        dloc_[mem[s]] = Loc{dp, (uint16_t) s};
      }
      prefill_pages_++;
      covered += (uint64_t) (count > 0 ? count - 1 : 0);
      nbrs_total += nn;
    }
    close(fd);
    prefill_ids_ = dloc_.size();
    const double ratio = nbrs_total == 0 ? 0.0 : (double) covered / (double) nbrs_total;
    std::fprintf(stderr, "NMD prefill BFS: %u pages, %lu ids, fill_coverage=%.4f mean_covered=%.2f DMB=%u\n",
                 prefill_pages_, (unsigned long) prefill_ids_, ratio,
                 prefill_pages_ == 0 ? 0.0 : (double) covered / (double) prefill_pages_, dmb);
  }

  void note_coverage(const char *node, const uint32_t *mem, int count) {
    if (count < 2 || vector_bytes_ + 4 > node_len_) {
      return;
    }
    uint16_t nn = *reinterpret_cast<const uint16_t *>(node + vector_bytes_);
    if (nn == 0 || nn > 512) {
      return;
    }
    if (vector_bytes_ + 4 + (uint64_t) nn * 4 > node_len_) {
      return;
    }
    const uint32_t *nbrs = reinterpret_cast<const uint32_t *>(node + vector_bytes_ + 4);
    uint32_t hit = 0;
    for (uint16_t i = 0; i < nn; ++i) {
      for (int s = 1; s < count; ++s) {
        if (mem[s] == nbrs[i]) {
          hit++;
          break;
        }
      }
    }
    cov_pages_.fetch_add(1, std::memory_order_relaxed);
    cov_nbrs_.fetch_add(nn, std::memory_order_relaxed);
    cov_hit_.fetch_add(hit, std::memory_order_relaxed);
  }

  void note_page_fill(const char *node, const uint32_t *mem, int count) {
    if (vector_bytes_ + 4 > node_len_) {
      return;
    }
    uint16_t nn = *reinterpret_cast<const uint16_t *>(node + vector_bytes_);
    if (nn == 0 || nn > 512) {
      return;
    }
    if (vector_bytes_ + 4 + (uint64_t) nn * 4 > node_len_) {
      return;
    }
    const uint32_t *nbrs = reinterpret_cast<const uint32_t *>(node + vector_bytes_ + 4);
    uint32_t hit = 0;
    int members = count > 0 ? count - 1 : 0;
    for (uint16_t i = 0; i < nn; ++i) {
      for (int s = 1; s < count; ++s) {
        if (mem[s] == nbrs[i]) {
          hit++;
          break;
        }
      }
    }
    fill_pages_.fetch_add(1, std::memory_order_relaxed);
    fill_nbrs_.fetch_add(nn, std::memory_order_relaxed);
    fill_hit_.fetch_add(hit, std::memory_order_relaxed);
    fill_slots_.fetch_add((uint64_t) members, std::memory_order_relaxed);
  }

  bool enabled_ = false;
  size_t node_len_ = 0;
  uint64_t vector_bytes_ = 0;
  uint64_t npoints_ = 0;
  uint64_t sector_nps_ = 0;
  uint32_t entry_id_ = 0;
  uint32_t prefill_pages_ = 0;
  uint64_t prefill_ids_ = 0;
  std::string index_path_;
  int slots_ = 1;
  size_t page_bytes_ = 0;
  size_t ncap_ = 0;
  size_t pcap_ = 0;
  uint32_t heat_th_ = 2;
  int dfd_ = -1;
  uint32_t dnext_ = 0;
  size_t nhand_ = 0;
  std::string path_;

  std::vector<uint32_t> heat_vec_;
  std::unordered_map<uint32_t, uint32_t> heat_map_;
  std::vector<NSlot> nslots_;
  std::unordered_map<uint32_t, uint32_t> nid_;
  std::vector<Page> pages_;
  std::vector<uint32_t> free_;
  std::unordered_map<uint32_t, Loc> pref_;
  std::unordered_map<uint32_t, uint32_t> seed_page_;
  std::unordered_map<uint32_t, std::vector<uint32_t>> rev_;
  std::unordered_map<uint32_t, Loc> dloc_;
  std::vector<DMeta> dmeta_;

  mutable std::mutex mu_;
  std::atomic<uint64_t> n_n_{0}, n_p_{0}, n_d_{0}, n_miss_{0}, n_expand_{0}, d_pages_{0};
  std::atomic<uint64_t> cov_pages_{0}, cov_nbrs_{0}, cov_hit_{0};
  std::atomic<uint64_t> fill_pages_{0}, fill_nbrs_{0}, fill_hit_{0}, fill_slots_{0};
};

}  // namespace pipeann
