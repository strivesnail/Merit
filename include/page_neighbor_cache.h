#pragma once

// Online N-cache and seed pages for Starling/MARGO page search.
// A hit is served before id2page_ is read.
// NEIGHBORCACHE_DCACHE_PREFILL=1 writes seed pages into the D-cache before
// search and reads them back from that file. Otherwise a seed page stays in
// the prefetch buffer and is written only when that buffer evicts it.

#include "tsl/robin_map.h"

#include <libaio.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <unistd.h>
#include <utility>
#include <vector>

namespace diskann {

template <typename T> class PageNeighborCache {
 public:
  PageNeighborCache() = default;
  ~PageNeighborCache()
  {
    report();
    if (fd_ >= 0)
      ::close(fd_);
  }

  bool init(uint64_t max_node_len, uint64_t disk_bytes, uint64_t max_degree, const std::string &dyn_path)
  {
    const uint64_t cap_nodes = env_u64("NEIGHBORCACHE_NCACHE_CAP_NODES", 0);
    if (cap_nodes == 0 || max_node_len == 0 || disk_bytes == 0)
      return false;
    node_len_ = max_node_len;
    disk_bytes_ = disk_bytes;
    max_degree_ = max_degree;
    slots_per_page_ = static_cast<uint32_t>(4096 / max_node_len);
    if (slots_per_page_ > 10)
      slots_per_page_ = 10;
    if (slots_per_page_ == 0)
      slots_per_page_ = 1;
    const uint64_t page_budget = env_u64("NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES", 25600);
    prefill_ = env_u64("NEIGHBORCACHE_DCACHE_PREFILL", 0) != 0;
    dcache_off_ = env_u64("NEIGHBORCACHE_DCACHE_OFF", 0) != 0;
    if (dcache_off_)
      prefill_ = false;
    if (prefill_)
    {
      prefill_pages_ = page_budget == 0 ? 25600 : page_budget;
      prefetch_pages_ = env_u64("NEIGHBORCACHE_DCACHE_READ_PAGES", 6400);
    }
    else
    {
      prefetch_pages_ = page_budget;
      prefill_pages_ = 0;
    }
    if (prefetch_pages_ < kShards)
      prefetch_pages_ = kShards;
    physical_cap_ = env_u64("NEIGHBORCACHE_DCACHE_CAP", 1500000);
    node_cap_ = (cap_nodes / kShards) * kShards;
    if (node_cap_ == 0)
      node_cap_ = kShards;
    init_ncache(node_cap_);
    init_prefetch(prefetch_pages_);
    dirty_.reset(new std::atomic<uint8_t>[physical_cap_]);
    for (uint64_t i = 0; i < physical_cap_; ++i)
      dirty_[i].store(0, std::memory_order_relaxed);
#ifndef _WINDOWS
    if (!dcache_off_)
    {
      fd_ = ::open(dyn_path.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_DIRECT | O_LARGEFILE, 0644);
      if (fd_ < 0)
        fd_ = ::open(dyn_path.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_LARGEFILE, 0644);
      if (fd_ >= 0 && ::ftruncate(fd_, static_cast<off_t>(physical_cap_ * 4096)) != 0)
      {
        ::close(fd_);
        fd_ = -1;
      }
    }
#endif
    locs_.reset(new LocShard[kShards]);
    std::cout << "PageNeighborCache nodes=" << cap_nodes << " bytes/node=" << node_len_
              << " prefetch_pages=" << prefetch_pages_ << " prefill_pages=" << prefill_pages_
              << " slots/page=" << slots_per_page_
              << " prefill=" << (prefill_ ? 1 : 0)
              << " dcache_off=" << (dcache_off_ ? 1 : 0)
              << " dyn=" << dyn_path << " fd=" << fd_ << std::endl;
    enabled_ = true;
    return true;
  }

  bool prefill_enabled() const
  {
    return prefill_;
  }

  uint64_t prefill_page_budget() const
  {
    return prefill_pages_;
  }

  uint32_t slots_per_page() const
  {
    return slots_per_page_;
  }

  bool contains(uint32_t id)
  {
    return ncache_contains(id);
  }

  bool enabled() const
  {
    return enabled_;
  }

  // Copies one node record into dst (node_len_ bytes). N-cache is checked first.
  // from_disk is set when the bytes were read from the D-cache file.
  bool lookup(uint32_t id, char *dst, bool *from_disk = nullptr)
  {
    if (from_disk != nullptr)
      *from_disk = false;
    if (!enabled_ || dst == nullptr)
      return false;
    if (ncache_copy(id, dst))
    {
      ncache_hits_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    if (prefill_ && lookup_prefill(id, dst, from_disk))
      return true;
    std::vector<uint32_t> seeds;
    {
      OwnerShard &own = owners_[id % kShards];
      std::lock_guard<std::mutex> lock(own.mu);
      auto it = own.by_node.find(id);
      if (it == own.by_node.end())
      {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      seeds = it.value();
    }
    for (uint32_t seed_id : seeds)
    {
      Seed *seed = find_seed(seed_id);
      if (seed == nullptr)
        continue;
      std::lock_guard<std::mutex> lock(seed->mu);
      int slot = -1;
      for (size_t i = 0; i < seed->members.size(); ++i)
      {
        if (seed->members[i] == id)
        {
          slot = static_cast<int>(i);
          break;
        }
      }
      if (slot < 0)
        continue;
      std::vector<char> page(4096);
      if (!prefetch_get(seed->page, page.data()))
        continue;
      std::memcpy(dst, page.data() + static_cast<uint64_t>(slot) * node_len_, node_len_);
      seed_hits_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    misses_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  struct DiskNode
  {
    uint32_t id = kInvalid;
    uint32_t page = kInvalid;
    uint32_t slot = 0;
  };

  enum class Probe
  {
    Miss,
    Hit,
    Disk
  };

  // N-cache and prefetch hits copy into dst. A D-cache page that is not
  // resident is returned as Disk so the caller can submit several pages at once.
  Probe probe(uint32_t id, char *dst, DiskNode *disk)
  {
    if (disk != nullptr)
      *disk = DiskNode();
    if (!enabled_ || dst == nullptr)
      return Probe::Miss;
    if (ncache_copy(id, dst))
    {
      ncache_hits_.fetch_add(1, std::memory_order_relaxed);
      return Probe::Hit;
    }
    if (prefill_ && locs_ != nullptr)
    {
      uint32_t packed = kInvalid;
      {
        LocShard &sh = locs_[id % kShards];
        std::lock_guard<std::mutex> lock(sh.mu);
        auto it = sh.map.find(id);
        if (it != sh.map.end())
          packed = it.value();
      }
      if (packed != kInvalid)
      {
        const uint32_t page = packed >> 8;
        const uint32_t slot = packed & 0xffu;
        if (slot < slots_per_page_)
        {
          static thread_local void *scratch = nullptr;
          if (scratch == nullptr && posix_memalign(&scratch, 4096, 4096) != 0)
            return Probe::Miss;
          if (prefetch_get(page, static_cast<char *>(scratch)))
          {
            std::memcpy(dst, static_cast<char *>(scratch) + static_cast<size_t>(slot) * node_len_, node_len_);
            ncache_admit(id, dst);
            seed_hits_.fetch_add(1, std::memory_order_relaxed);
            return Probe::Hit;
          }
          if (fd_ >= 0 && disk != nullptr)
          {
            disk->id = id;
            disk->page = page;
            disk->slot = slot;
            return Probe::Disk;
          }
        }
      }
    }
    bool ignored = false;
    if (lookup(id, dst, &ignored))
      return Probe::Hit;
    return Probe::Miss;
  }

  // Starts one io_submit for the distinct D-cache pages. Returns how many 4KB reads were issued.
  // Call finish_disk_nodes before using the nodes.
  uint32_t submit_disk_nodes(const std::vector<DiskNode> &nodes)
  {
    DiskBatch &batch = disk_batch();
    batch.reset();
    if (nodes.empty() || fd_ < 0)
      return 0;
    batch.nodes = nodes;
    for (const DiskNode &node : nodes)
    {
      bool seen = false;
      for (uint32_t page : batch.pages)
      {
        if (page == node.page)
        {
          seen = true;
          break;
        }
      }
      if (!seen)
        batch.pages.push_back(node.page);
    }
    const int npages = static_cast<int>(batch.pages.size());
    if (npages <= 0 || posix_memalign(&batch.mem, 4096, static_cast<size_t>(npages) * 4096) != 0)
      return 0;
    std::memset(batch.mem, 0, static_cast<size_t>(npages) * 4096);
    batch.npages = npages;
    if (!batch.ready || npages > 128)
    {
      for (int i = 0; i < npages; ++i)
        ::pread(fd_, static_cast<char *>(batch.mem) + static_cast<size_t>(i) * 4096, 4096,
                static_cast<off_t>(batch.pages[i]) * 4096);
      batch.sync = true;
      batch.pending = true;
      return static_cast<uint32_t>(npages);
    }
    batch.cbs.resize(static_cast<size_t>(npages));
    batch.cbps.resize(static_cast<size_t>(npages));
    for (int i = 0; i < npages; ++i)
    {
      io_prep_pread(&batch.cbs[i], fd_, static_cast<char *>(batch.mem) + static_cast<size_t>(i) * 4096, 4096,
                    static_cast<off_t>(batch.pages[i]) * 4096);
      batch.cbps[i] = &batch.cbs[i];
    }
    const int submitted = io_submit(batch.ctx, npages, batch.cbps.data());
    if (submitted != npages)
    {
      if (submitted > 0)
      {
        std::vector<io_event> events(static_cast<size_t>(submitted));
        io_getevents(batch.ctx, submitted, submitted, events.data(), nullptr);
      }
      for (int i = 0; i < npages; ++i)
        ::pread(fd_, static_cast<char *>(batch.mem) + static_cast<size_t>(i) * 4096, 4096,
                static_cast<off_t>(batch.pages[i]) * 4096);
      batch.sync = true;
    }
    batch.pending = true;
    return static_cast<uint32_t>(npages);
  }

  // Waits for the submitted pages, then copies each node into hits. Returns the number of 4KB reads.
  uint32_t finish_disk_nodes(std::vector<std::pair<uint32_t, std::vector<char>>> &hits)
  {
    DiskBatch &batch = disk_batch();
    if (!batch.pending)
      return 0;
    if (!batch.sync)
    {
      std::vector<io_event> events(static_cast<size_t>(batch.npages));
      const int got = io_getevents(batch.ctx, batch.npages, batch.npages, events.data(), nullptr);
      if (got != batch.npages)
      {
        for (int i = 0; i < batch.npages; ++i)
          ::pread(fd_, static_cast<char *>(batch.mem) + static_cast<size_t>(i) * 4096, 4096,
                  static_cast<off_t>(batch.pages[i]) * 4096);
      }
    }
    for (int i = 0; i < batch.npages; ++i)
    {
      char *page_buf = static_cast<char *>(batch.mem) + static_cast<size_t>(i) * 4096;
      prefetch_put(batch.pages[static_cast<size_t>(i)], page_buf);
      dcache_reads_.fetch_add(1, std::memory_order_relaxed);
    }
    for (const DiskNode &node : batch.nodes)
    {
      int index = -1;
      for (int i = 0; i < batch.npages; ++i)
      {
        if (batch.pages[static_cast<size_t>(i)] == node.page)
        {
          index = i;
          break;
        }
      }
      if (index < 0 || node.slot >= slots_per_page_)
        continue;
      const char *page_buf = static_cast<char *>(batch.mem) + static_cast<size_t>(index) * 4096;
      std::vector<char> record(node_len_);
      std::memcpy(record.data(), page_buf + static_cast<size_t>(node.slot) * node_len_, node_len_);
      ncache_admit(node.id, record.data());
      seed_hits_.fetch_add(1, std::memory_order_relaxed);
      hits.emplace_back(node.id, std::move(record));
    }
    const uint32_t nread = static_cast<uint32_t>(batch.npages);
    batch.reset();
    return nread;
  }

  uint64_t node_capacity() const
  {
    return node_cap_;
  }

  // BFS preload fills the N-cache. Prefill mode does not call this; only the D-cache is installed beforehand.
  void preload(uint32_t id, const char *node_buf)
  {
    if (!enabled_ || node_buf == nullptr || id == kInvalid)
      return;
    ncache_admit(id, node_buf);
  }

  void observe_node(uint32_t id, const char *node_buf)
  {
    if (!enabled_ || node_buf == nullptr || id == kInvalid)
      return;
    const bool seen = ncache_contains(id);
    ncache_admit(id, node_buf);
    if (!dcache_off_ && !prefill_ && seen && find_seed(id) == nullptr && try_reserve_open())
      open_seed(id, node_buf);
    note(id);
  }

  void observe_edge(uint32_t parent, uint32_t child)
  {
    if (!enabled_ || parent == kInvalid || child == kInvalid || parent == child)
      return;
    HeatShard &sh = heats_[parent % kShards];
    std::lock_guard<std::mutex> lock(sh.mu);
    auto &edges = sh.edges[parent];
    for (auto &edge : edges)
    {
      if (edge.first == child)
      {
        if (edge.second < 65535)
          ++edge.second;
        return;
      }
    }
    if (edges.size() < 64)
      edges.emplace_back(child, 1);
  }

  void report() const
  {
    if (!enabled_)
      return;
    std::cout << "PageNeighborCache ncache_hits=" << ncache_hits_.load() << " seed_hits=" << seed_hits_.load()
              << " dcache_reads=" << dcache_reads_.load()
              << " misses=" << misses_.load() << " admits=" << admits_.load() << " opens=" << opens_.load()
              << " appends=" << appends_.load() << " replaces=" << replaces_.load()
              << " evicts=" << evicts_.load() << " written=" << written_.load() << std::endl;
  }

  // ids[0] is the seed. Each buffer is node_len_ bytes. The page is written to the D-cache file now.
  bool commit_prefill_page(const std::vector<uint32_t> &ids, const std::vector<const char *> &bufs)
  {
    if (!enabled_ || !prefill_ || fd_ < 0 || ids.empty() || ids.size() != bufs.size())
      return false;
    const uint32_t n = static_cast<uint32_t>(std::min<size_t>(ids.size(), slots_per_page_));
    const uint32_t page = alloc_page();
    if (page == kInvalid || page > 0x00ffffffu)
      return false;
    void *aligned = nullptr;
    if (posix_memalign(&aligned, 4096, 4096) != 0)
      return false;
    std::memset(aligned, 0, 4096);
    for (uint32_t s = 0; s < n; ++s)
    {
      if (bufs[s] == nullptr)
        continue;
      std::memcpy(static_cast<char *>(aligned) + static_cast<size_t>(s) * node_len_, bufs[s], node_len_);
    }
    const ssize_t wr = ::pwrite(fd_, aligned, 4096, static_cast<off_t>(page) * 4096);
    std::free(aligned);
    if (wr != 4096)
      return false;
    for (uint32_t s = 0; s < n; ++s)
    {
      if (bufs[s] == nullptr)
        continue;
      LocShard &sh = locs_[ids[s] % kShards];
      std::lock_guard<std::mutex> lock(sh.mu);
      sh.map[ids[s]] = (page << 8) | s;
    }
    opens_.fetch_add(1, std::memory_order_relaxed);
    written_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

 private:
  static constexpr uint32_t kShards = 64;
  static constexpr uint32_t kInvalid = 0xffffffffu;

  struct Slot
  {
    uint32_t id = kInvalid;
    uint32_t prev = kInvalid;
    uint32_t next = kInvalid;
  };
  struct NShard
  {
    std::mutex mu;
    std::vector<char> bytes;
    std::vector<Slot> slots;
    std::vector<uint32_t> free_list;
    tsl::robin_map<uint32_t, uint32_t> id_to_slot;
    uint32_t head = kInvalid;
    uint32_t tail = kInvalid;
    uint32_t cap = 0;
  };
  struct PShard
  {
    std::mutex mu;
    char *data = nullptr;
    std::vector<Slot> slots;
    std::vector<uint32_t> free_list;
    tsl::robin_map<uint32_t, uint32_t> key_to_slot;
    uint32_t head = kInvalid;
    uint32_t tail = kInvalid;
    uint32_t cap = 0;
    ~PShard()
    {
      std::free(data);
    }
  };
  struct Seed
  {
    std::mutex mu;
    uint32_t page = kInvalid;
    std::vector<uint32_t> members;
  };
  struct OwnerShard
  {
    std::mutex mu;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> by_node;
  };
  struct HeatShard
  {
    std::mutex mu;
    tsl::robin_map<uint32_t, std::vector<std::pair<uint32_t, uint16_t>>> edges;
  };

  static uint64_t env_u64(const char *name, uint64_t fallback)
  {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0')
      return fallback;
    return std::strtoull(v, nullptr, 10);
  }

  void init_ncache(uint64_t cap_nodes)
  {
    const uint32_t each = static_cast<uint32_t>(std::max<uint64_t>(1, cap_nodes / kShards));
    for (uint32_t s = 0; s < kShards; ++s)
    {
      NShard &sh = ncache_[s];
      sh.cap = each;
      sh.bytes.assign(static_cast<size_t>(each) * node_len_, 0);
      sh.slots.resize(each);
      sh.free_list.reserve(each);
      for (uint32_t i = each; i > 0; --i)
        sh.free_list.push_back(i - 1);
    }
  }

  void init_prefetch(uint64_t pages)
  {
    const uint32_t each = static_cast<uint32_t>(std::max<uint64_t>(1, pages / kShards));
    for (uint32_t s = 0; s < kShards; ++s)
    {
      PShard &sh = prefetch_[s];
      sh.cap = each;
      sh.data = static_cast<char *>(aligned_alloc(4096, static_cast<size_t>(each) * 4096));
      std::memset(sh.data, 0, static_cast<size_t>(each) * 4096);
      sh.slots.resize(each);
      for (uint32_t i = each; i > 0; --i)
        sh.free_list.push_back(i - 1);
    }
  }

  void touch(std::vector<Slot> &slots, uint32_t &head, uint32_t &tail, uint32_t slot)
  {
    Slot &s = slots[slot];
    if (head == slot)
      return;
    if (s.prev != kInvalid)
      slots[s.prev].next = s.next;
    if (s.next != kInvalid)
      slots[s.next].prev = s.prev;
    if (tail == slot)
      tail = s.prev;
    s.prev = kInvalid;
    s.next = head;
    if (head != kInvalid)
      slots[head].prev = slot;
    head = slot;
    if (tail == kInvalid)
      tail = slot;
  }

  bool ncache_copy(uint32_t id, char *dst)
  {
    NShard &sh = ncache_[id % kShards];
    std::lock_guard<std::mutex> lock(sh.mu);
    auto it = sh.id_to_slot.find(id);
    if (it == sh.id_to_slot.end())
      return false;
    const uint32_t slot = it.value();
    std::memcpy(dst, sh.bytes.data() + static_cast<size_t>(slot) * node_len_, node_len_);
    touch(sh.slots, sh.head, sh.tail, slot);
    return true;
  }

  bool ncache_contains(uint32_t id)
  {
    NShard &sh = ncache_[id % kShards];
    std::lock_guard<std::mutex> lock(sh.mu);
    return sh.id_to_slot.find(id) != sh.id_to_slot.end();
  }

  void ncache_admit(uint32_t id, const char *buf)
  {
    NShard &sh = ncache_[id % kShards];
    std::lock_guard<std::mutex> lock(sh.mu);
    auto it = sh.id_to_slot.find(id);
    if (it != sh.id_to_slot.end())
    {
      const uint32_t slot = it.value();
      std::memcpy(sh.bytes.data() + static_cast<size_t>(slot) * node_len_, buf, node_len_);
      touch(sh.slots, sh.head, sh.tail, slot);
      return;
    }
    uint32_t slot = kInvalid;
    if (!sh.free_list.empty())
    {
      slot = sh.free_list.back();
      sh.free_list.pop_back();
    }
    else if (sh.tail != kInvalid)
    {
      slot = sh.tail;
      const uint32_t old = sh.slots[slot].id;
      sh.id_to_slot.erase(old);
      if (sh.slots[slot].prev != kInvalid)
        sh.slots[sh.slots[slot].prev].next = kInvalid;
      else
        sh.head = kInvalid;
      sh.tail = sh.slots[slot].prev;
      sh.slots[slot] = Slot();
    }
    else
      return;
    std::memcpy(sh.bytes.data() + static_cast<size_t>(slot) * node_len_, buf, node_len_);
    sh.slots[slot].id = id;
    sh.id_to_slot[id] = slot;
    touch(sh.slots, sh.head, sh.tail, slot);
    admits_.fetch_add(1, std::memory_order_relaxed);
  }

  bool prefetch_get(uint32_t page, char *dst)
  {
    if (page == kInvalid)
      return false;
    PShard &sh = prefetch_[page % kShards];
    std::lock_guard<std::mutex> lock(sh.mu);
    auto it = sh.key_to_slot.find(page);
    if (it == sh.key_to_slot.end())
      return false;
    const uint32_t slot = it.value();
    std::memcpy(dst, sh.data + static_cast<size_t>(slot) * 4096, 4096);
    touch(sh.slots, sh.head, sh.tail, slot);
    return true;
  }

  bool prefetch_put(uint32_t page, const char *src)
  {
    if (page == kInvalid || src == nullptr)
      return false;
    PShard &sh = prefetch_[page % kShards];
    std::lock_guard<std::mutex> lock(sh.mu);
    auto it = sh.key_to_slot.find(page);
    if (it != sh.key_to_slot.end())
    {
      const uint32_t slot = it.value();
      std::memcpy(sh.data + static_cast<size_t>(slot) * 4096, src, 4096);
      touch(sh.slots, sh.head, sh.tail, slot);
      return true;
    }
    uint32_t slot = kInvalid;
    if (!sh.free_list.empty())
    {
      slot = sh.free_list.back();
      sh.free_list.pop_back();
    }
    else if (sh.tail != kInvalid)
    {
      slot = sh.tail;
      const uint32_t old = sh.slots[slot].id;
      write_if_dirty(old, sh.data + static_cast<size_t>(slot) * 4096);
      sh.key_to_slot.erase(old);
      if (sh.slots[slot].prev != kInvalid)
        sh.slots[sh.slots[slot].prev].next = kInvalid;
      else
        sh.head = kInvalid;
      sh.tail = sh.slots[slot].prev;
      sh.slots[slot] = Slot();
    }
    else
      return false;
    std::memcpy(sh.data + static_cast<size_t>(slot) * 4096, src, 4096);
    sh.slots[slot].id = page;
    sh.key_to_slot[page] = slot;
    touch(sh.slots, sh.head, sh.tail, slot);
    return true;
  }

  void write_if_dirty(uint32_t page, const char *bytes)
  {
    if (page >= physical_cap_ || fd_ < 0 || bytes == nullptr)
      return;
    uint8_t expected = 1;
    if (!dirty_[page].compare_exchange_strong(expected, 0, std::memory_order_acq_rel))
      return;
    evicts_.fetch_add(1, std::memory_order_relaxed);
    const off_t off = static_cast<off_t>(page) * 4096;
    ssize_t n = ::pwrite(fd_, bytes, 4096, off);
    if (n != 4096)
      n = ::pwrite(fd_, bytes, 4096, off);
    if (n == 4096)
      written_.fetch_add(1, std::memory_order_relaxed);
  }

  void mark_dirty(uint32_t page)
  {
    if (page < physical_cap_)
      dirty_[page].store(1, std::memory_order_release);
  }

  bool try_reserve_open()
  {
    std::lock_guard<std::mutex> lock(page_mu_);
    if (reserved_opens_ >= prefetch_pages_)
      return false;
    ++reserved_opens_;
    return true;
  }

  void unreserve_open()
  {
    std::lock_guard<std::mutex> lock(page_mu_);
    if (reserved_opens_ > 0)
      --reserved_opens_;
  }

  uint32_t alloc_page()
  {
    std::lock_guard<std::mutex> lock(page_mu_);
    if (!free_pages_.empty())
    {
      const uint32_t page = free_pages_.back();
      free_pages_.pop_back();
      return page;
    }
    if (next_page_ >= physical_cap_)
      return kInvalid;
    return next_page_++;
  }

  Seed *find_seed(uint32_t id)
  {
    std::lock_guard<std::mutex> lock(seeds_mu_);
    auto it = seeds_.find(id);
    if (it == seeds_.end())
      return nullptr;
    return it.value().get();
  }

  void remember_owner(uint32_t node, uint32_t seed_id)
  {
    OwnerShard &own = owners_[node % kShards];
    std::lock_guard<std::mutex> lock(own.mu);
    auto &list = own.by_node[node];
    for (uint32_t existing : list)
    {
      if (existing == seed_id)
        return;
    }
    if (list.size() < 32)
      list.push_back(seed_id);
  }

  std::vector<uint32_t> heats_of(uint32_t id)
  {
    HeatShard &sh = heats_[id % kShards];
    std::lock_guard<std::mutex> lock(sh.mu);
    auto it = sh.edges.find(id);
    if (it == sh.edges.end())
      return {};
    std::vector<uint32_t> order;
    auto edges = it.value();
    std::sort(edges.begin(), edges.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
    order.reserve(edges.size());
    for (const auto &edge : edges)
      order.push_back(edge.first);
    return order;
  }

  void open_seed(uint32_t id, const char *buf)
  {
    const uint32_t page = alloc_page();
    if (page == kInvalid)
    {
      unreserve_open();
      return;
    }
    Seed *seed = nullptr;
    {
      std::lock_guard<std::mutex> lock(seeds_mu_);
      if (seeds_.find(id) != seeds_.end())
      {
        std::lock_guard<std::mutex> page_lock(page_mu_);
        free_pages_.push_back(page);
        if (reserved_opens_ > 0)
          --reserved_opens_;
        return;
      }
      auto created = std::make_unique<Seed>();
      seed = created.get();
      seed->page = page;
      seeds_.emplace(id, std::move(created));
    }
    std::lock_guard<std::mutex> lock(seed->mu);
    std::vector<char> page_bytes(4096, 0);
    std::memcpy(page_bytes.data(), buf, node_len_);
    seed->members.push_back(id);
    remember_owner(id, id);
    unsigned *nhood = reinterpret_cast<unsigned *>(const_cast<char *>(buf) + disk_bytes_);
    const unsigned nnbrs = nhood[0] > max_degree_ ? static_cast<unsigned>(max_degree_) : nhood[0];
    std::vector<uint32_t> nbrs(nhood + 1, nhood + 1 + nnbrs);
    const std::vector<uint32_t> hot = heats_of(id);
    std::vector<uint32_t> ordered = hot;
    for (uint32_t nbr : nbrs)
    {
      if (std::find(ordered.begin(), ordered.end(), nbr) == ordered.end())
        ordered.push_back(nbr);
    }
    for (uint32_t nbr : ordered)
    {
      if (seed->members.size() >= slots_per_page_)
        break;
      if (!ncache_contains(nbr))
        continue;
      std::vector<char> payload(node_len_);
      if (!ncache_copy(nbr, payload.data()))
        continue;
      std::memcpy(page_bytes.data() + seed->members.size() * node_len_, payload.data(), node_len_);
      seed->members.push_back(nbr);
    }
    for (uint32_t nbr : nbrs)
      remember_owner(nbr, id);
    if (!prefetch_put(page, page_bytes.data()))
    {
      unreserve_open();
      return;
    }
    mark_dirty(page);
    opens_.fetch_add(1, std::memory_order_relaxed);
  }

  void note(uint32_t id)
  {
    if (!ncache_contains(id))
      return;
    std::vector<char> payload(node_len_);
    if (!ncache_copy(id, payload.data()))
      return;
    std::vector<uint32_t> seeds;
    {
      OwnerShard &own = owners_[id % kShards];
      std::lock_guard<std::mutex> lock(own.mu);
      auto it = own.by_node.find(id);
      if (it == own.by_node.end())
        return;
      seeds = it.value();
    }
    for (uint32_t seed_id : seeds)
    {
      if (seed_id == id)
        continue;
      Seed *seed = find_seed(seed_id);
      if (seed == nullptr)
        continue;
      std::lock_guard<std::mutex> lock(seed->mu);
      bool present = false;
      for (uint32_t member : seed->members)
      {
        if (member == id)
        {
          present = true;
          break;
        }
      }
      if (present || seed->members.empty())
        continue;
      std::vector<char> page(4096);
      if (!prefetch_get(seed->page, page.data()))
        continue;
      const bool replace = seed->members.size() >= slots_per_page_;
      if (replace && slots_per_page_ < 2)
        continue;
      const size_t slot = replace ? seed->members.size() - 1 : seed->members.size();
      std::memcpy(page.data() + slot * node_len_, payload.data(), node_len_);
      if (!prefetch_put(seed->page, page.data()))
        continue;
      mark_dirty(seed->page);
      if (replace)
      {
        seed->members[slot] = id;
        replaces_.fetch_add(1, std::memory_order_relaxed);
      }
      else
      {
        seed->members.push_back(id);
        appends_.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  bool enabled_ = false;
  uint64_t node_cap_ = 0;
  uint64_t node_len_ = 0;
  uint64_t disk_bytes_ = 0;
  uint64_t max_degree_ = 0;
  uint32_t slots_per_page_ = 1;
  uint64_t prefetch_pages_ = 0;
  uint64_t physical_cap_ = 0;
  int fd_ = -1;
  uint32_t next_page_ = 0;
  uint64_t reserved_opens_ = 0;
  std::mutex page_mu_;
  std::vector<uint32_t> free_pages_;
  std::mutex seeds_mu_;
  tsl::robin_map<uint32_t, std::unique_ptr<Seed>> seeds_;
  std::unique_ptr<std::atomic<uint8_t>[]> dirty_;
  NShard ncache_[kShards];
  PShard prefetch_[kShards];
  OwnerShard owners_[kShards];
  HeatShard heats_[kShards];
  std::atomic<uint64_t> ncache_hits_{0};
  std::atomic<uint64_t> seed_hits_{0};
  std::atomic<uint64_t> misses_{0};
  std::atomic<uint64_t> admits_{0};
  std::atomic<uint64_t> opens_{0};
  std::atomic<uint64_t> appends_{0};
  std::atomic<uint64_t> replaces_{0};
  std::atomic<uint64_t> evicts_{0};
  std::atomic<uint64_t> written_{0};
  std::atomic<uint64_t> dcache_reads_{0};
  bool prefill_ = false;
  bool dcache_off_ = false;
  uint64_t prefill_pages_ = 0;
  struct LocShard
  {
    std::mutex mu;
    tsl::robin_map<uint32_t, uint32_t> map;
  };
  std::unique_ptr<LocShard[]> locs_;

  bool lookup_prefill(uint32_t id, char *dst, bool *from_disk)
  {
    if (locs_ == nullptr)
      return false;
    uint32_t packed = kInvalid;
    {
      LocShard &sh = locs_[id % kShards];
      std::lock_guard<std::mutex> lock(sh.mu);
      auto it = sh.map.find(id);
      if (it == sh.map.end())
        return false;
      packed = it.value();
    }
    const uint32_t page = packed >> 8;
    const uint32_t slot = packed & 0xffu;
    if (slot >= slots_per_page_)
      return false;
    static thread_local void *scratch = nullptr;
    if (scratch == nullptr && posix_memalign(&scratch, 4096, 4096) != 0)
      return false;
    bool hit = false;
    if (prefetch_get(page, static_cast<char *>(scratch)))
      hit = true;
    else
    {
      const ssize_t n = ::pread(fd_, scratch, 4096, static_cast<off_t>(page) * 4096);
      if (n == 4096)
      {
        hit = true;
        prefetch_put(page, static_cast<char *>(scratch));
        dcache_reads_.fetch_add(1, std::memory_order_relaxed);
        if (from_disk != nullptr)
          *from_disk = true;
      }
    }
    if (hit)
    {
      std::memcpy(dst, static_cast<char *>(scratch) + static_cast<size_t>(slot) * node_len_, node_len_);
      ncache_admit(id, dst);
    }
    if (!hit)
      return false;
    seed_hits_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  struct DiskBatch
  {
    io_context_t ctx = 0;
    bool ready = false;
    bool pending = false;
    bool sync = false;
    int npages = 0;
    std::vector<DiskNode> nodes;
    std::vector<uint32_t> pages;
    std::vector<struct iocb> cbs;
    std::vector<struct iocb *> cbps;
    void *mem = nullptr;

    DiskBatch()
    {
      if (io_setup(128, &ctx) == 0)
        ready = true;
    }

    ~DiskBatch()
    {
      reset();
      if (ready)
        io_destroy(ctx);
    }

    void reset()
    {
      if (mem != nullptr)
      {
        std::free(mem);
        mem = nullptr;
      }
      nodes.clear();
      pages.clear();
      cbs.clear();
      cbps.clear();
      pending = false;
      sync = false;
      npages = 0;
    }
  };

  DiskBatch &disk_batch()
  {
    static thread_local DiskBatch batch;
    return batch;
  }
};

}  // namespace diskann
