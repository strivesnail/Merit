#pragma once
#include "nc_backend.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <functional>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace neighborcache {
inline void verify_file_record(const std::string &path, uint64_t offset, size_t bytes, uint32_t id, const char *raw) {
    std::vector<char> expected(bytes);
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    ssize_t n = fd < 0 ? -1 : pread(fd, expected.data(), bytes, offset);
    if (fd >= 0) close(fd);
    if (n != static_cast<ssize_t>(bytes) || std::memcmp(raw, expected.data(), bytes))
        throw std::runtime_error("NeighborCache full-record verification failed, node=" + std::to_string(id));
}

inline uint64_t env(const char *key, uint64_t fallback) {
    const char *s = std::getenv(key);
    return s && *s ? std::stoull(s) : fallback;
}
class Client {
    const NCBackend *api_ = nullptr;
    void *cache_ = nullptr;
    size_t bytes_ = 0;
    bool parent_metadata_ = true;
    using PageMembers = int (*)(void *,void *,uint32_t,uint32_t *,char *,uint32_t);
    PageMembers page_members_ = nullptr;
    uint32_t page_slots_ = 0;
    std::function<void(uint32_t, const char *)> verifier_;
    static std::unordered_map<const Client *, void *> &queries() {
        static thread_local std::unordered_map<const Client *, void *> q; return q;
    }
    void *query() const {
        auto it = queries().find(this);
        if (it == queries().end()) throw std::logic_error("NeighborCache access outside a query");
        return it->second;
    }
    int check(int n) const {
        if (n < 0) throw std::runtime_error(api_->error());
        return n;
    }
  public:
    Client() = default;
    Client(const Client &) = delete;
    Client &operator=(const Client &) = delete;
    ~Client() { if (cache_) api_->destroy(cache_); }
    void set_verifier(std::function<void(uint32_t, const char *)> f) { verifier_ = std::move(f); }
    bool enabled() const { return cache_ != nullptr; }
    size_t node_len() const { return bytes_; }
    bool needs_parent_metadata() const { return parent_metadata_; }
    void init(const NCConfig &c) {
        const char *path = std::getenv("NEIGHBORCACHE_BACKEND_LIBRARY");
        if (!path) path = "libneighborcache_backend.so";
        // Kept loaded for the process lifetime: backend TLS can outlive a cache.
        void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw std::runtime_error(std::string("NeighborCache backend: ") + dlerror());
        auto get = reinterpret_cast<const NCBackend *(*)()>(dlsym(lib, "nc_backend_api"));
        if (!get || !(api_ = get()) || api_->abi != NC_BACKEND_ABI)
            throw std::runtime_error("NeighborCache backend ABI mismatch");
        if (env("NEIGHBORCACHE_EXPAND_PREFETCH_PAGE",0)) {
            page_members_=reinterpret_cast<PageMembers>(dlsym(lib,"nc_prefetch_page_members"));
            if (!page_members_) throw std::runtime_error("Backend lacks prefetch-page expansion extension");
        }
        // One seed may span two immutable pages; duplicate seed/padding slots
        // are removed by the backend before returning the snapshot.
        page_slots_=c.nodes_per_page * (env("NEIGHBORCACHE_DCACHE_SECOND_PAGE",0) ? 2u : 1u);
        parent_metadata_ = env("NEIGHBORCACHE_FREEZE_MAINTENANCE", 2) != 1 ||
            env("NEIGHBORCACHE_RESOLVE_PARENT", 0) != 0;
        bytes_ = c.record_bytes;
        cache_ = api_->create(&c);
        if (!cache_) throw std::runtime_error(api_->error());
    }
    void begin() {
        if (!enabled()) return;
        if (queries().count(this)) throw std::logic_error("Nested NeighborCache query");
        void *q = api_->begin(cache_);
        if (!q) throw std::runtime_error(api_->error());
        queries()[this] = q;
    }
    void end() {
        if (!enabled()) return;
        check(api_->end(cache_, query()));
        queries().erase(this);
    }
    class QueryGuard {
        Client *client_;
      public:
        explicit QueryGuard(Client *c) : client_(c) { if (client_) client_->begin(); }
        explicit QueryGuard(Client &c) : QueryGuard(&c) {}
        QueryGuard(const QueryGuard &) = delete;
        ~QueryGuard() {
            if (client_) {
                try { client_->end(); }
                catch (const std::exception &e) { std::fprintf(stderr, "NeighborCache query failure: %s\n", e.what()); std::terminate(); }
            }
        }
    };
    void prepare_parent(uint32_t parent, uint32_t child) {
        if (enabled() && parent_metadata_) check(api_->visit(cache_, query(), parent, child));
    }
    int resolve(uint32_t id, char *dst, NCRead &r, uint32_t parent = UINT32_MAX) {
        if (!enabled()) return 0;
        prepare_parent(parent, id);
        int k = check(api_->resolve(cache_, query(), id, dst, &r));
        if ((k == 1 || k == 2) && verifier_) verifier_(id, dst);
        return k;
    }
    bool expands_prefetch_page() const { return page_members_ != nullptr; }
    void append_prefetch_members(uint32_t target,std::vector<std::pair<uint32_t,std::vector<char>>> &hits) {
        if (!page_members_) return;
        std::vector<uint32_t> ids(page_slots_);
        std::vector<char> records(size_t(page_slots_)*bytes_);
        const int n=check(page_members_(cache_,query(),target,ids.data(),records.data(),page_slots_));
        for (int i=0;i<n;++i) {
            const char *raw=records.data()+size_t(i)*bytes_;
            if (verifier_) verifier_(ids[i],raw);
            hits.emplace_back(ids[i],std::vector<char>(raw,raw+bytes_));
        }
    }
    bool complete(NCRead &r, int64_t got, char *dst) {
        bool ok = check(api_->complete(cache_, query(), &r, got, dst)) != 0;
        if (ok && verifier_) verifier_(r.node_id, dst);
        return ok;
    }
    void expand(uint32_t id, const char *raw, uint32_t hop = 0, uint32_t parent = UINT32_MAX) {
        if (enabled()) {
            prepare_parent(parent, id);
            check(api_->expand(cache_, query(), id, raw, hop));
        }
    }
    void base_reads(uint32_t n, double us = 0) {
        if (enabled()) check(api_->base_reads(cache_, query(), n, us));
    }
    bool contains(uint32_t id) { return enabled() && check(api_->contains(cache_, id)); }
    NCStats stats() const { NCStats s{}; if (enabled()) check(api_->stats(cache_, &s)); return s; }
    void print() const {
        if (!enabled()) return;
        auto s = stats();
        std::fprintf(stderr, "NeighborCache shared backend: N=%lu prefetch=%lu D_attempts=%lu D_hits=%lu "
            "D_reads=%lu base_miss=%lu seeds=%lu/%lu M=%lu opens=%lu appends=%lu replaces=%lu writes=%lu\n",
            s.n_hits, s.prefetch_hits, s.d_attempts, s.d_hits, s.d_reads, s.base_misses,
            s.seeds, s.capacity, s.m_entries, s.opens, s.appends, s.replaces, s.writes);
    }
};

class NMDCache : public Client {
    struct ProfileShard {
        mutable std::mutex mu;
        std::unordered_map<uint32_t, uint32_t> counts;
    };
    std::array<ProfileShard, 64> profile_;
    std::string profile_path_;
    uint64_t points_ = 0;
  public:
    ~NMDCache() { write_profile(); }
    struct PendingDRead : NCRead {
        // Compatibility fields; the backend owns page decoding and coverage.
        uint16_t slot = 0, cov_count = 0;
        uint32_t cov[16]{};
        uint32_t len = 0;
        PendingDRead() : NCRead{-1, 0, 0, nullptr, nullptr, UINT32_MAX} {}
    };
    void init(const char *prefix, uint32_t count_bytes, uint64_t points, uint64_t record,
              uint64_t nps, uint64_t vector, uint64_t degree, const std::string &path, uint32_t medoid) {
        std::string p(prefix);
        if (!env(p.c_str(), 0)) return;
        points_ = points;
        const char *profile = std::getenv((p + "_PROFILE").c_str());
        if (profile) profile_path_ = profile;
        if (env((p + "_PREFILL").c_str(), 0))
            throw std::runtime_error("Legacy NMD PREFILL is not the online NeighborCache experiment; set PREFILL=0");
        uint64_t ncap = env("NEIGHBORCACHE_NCACHE_CAP_NODES", env((p + "_NMB").c_str(), 32) * 1024 * 1024 / record);
        uint64_t pcap = env("NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES", env((p + "_PMB").c_str(), 4) * 256);
        const char *override_path = std::getenv((p + "_PATH").c_str());
        std::string use_path = override_path ? override_path : path;
        NCConfig c{NC_BACKEND_ABI, count_bytes, points, record, vector, degree, nps,
            ncap, pcap, env("NEIGHBORCACHE_THREADS", 64), use_path.c_str(), &medoid, 1};
        Client::init(c);
    }
    int resolve(uint32_t id, char *dst, PendingDRead &r, uint32_t parent = UINT32_MAX) {
        int k = Client::resolve(id, dst, r, parent);
        r.len = r.bytes;
        return k;
    }
    bool lookup(uint32_t id, char *dst, int &kind) {
        PendingDRead r;
        kind = resolve(id, dst, r);
        if (kind != 3) return kind != 0;
        ssize_t got = pread(r.fd, r.buffer, r.bytes, r.offset);
        return complete(r, got, dst);
    }
    void on_expand(uint32_t id, const char *raw, const uint32_t *, uint32_t, uint32_t parent = UINT32_MAX) {
        if (!profile_path_.empty()) {
            auto &shard = profile_[id % profile_.size()];
            std::lock_guard<std::mutex> lock(shard.mu);
            auto &n = shard.counts[id];
            if (n != UINT32_MAX) ++n;
        }
        expand(id, raw, 0, parent);
    }
    void count_d_hit(const char *, const uint32_t *, int) {}
    void count_d_read() {}
    void count_d_read_failure() {}
    // Same on-disk visit profile as the original adapter: u64 points, u32 counts.
    // Called after the host has joined all query threads.
    void write_profile() const {
        if (profile_path_.empty()) return;
        FILE *f = std::fopen(profile_path_.c_str(), "wb");
        if (!f) { std::perror("NeighborCache profile open"); return; }
        bool ok = std::fwrite(&points_, sizeof(points_), 1, f) == 1;
        std::vector<uint32_t> counts(65536);
        std::array<std::unique_lock<std::mutex>, 64> locks;
        for (size_t s = 0; s < profile_.size(); ++s)
            locks[s] = std::unique_lock<std::mutex>(profile_[s].mu);
        for (uint64_t start = 0; ok && start < points_; start += counts.size()) {
            size_t n = std::min<uint64_t>(counts.size(), points_ - start);
            std::fill(counts.begin(), counts.end(), 0);
            for (size_t j = 0; j < n; ++j) {
                uint32_t id = start + j;
                const auto &map = profile_[id % profile_.size()].counts;
                auto it = map.find(id);
                if (it != map.end()) counts[j] = it->second;
            }
            ok = std::fwrite(counts.data(), sizeof(uint32_t), n, f) == n;
        }
        if (std::fclose(f) != 0) ok = false;
        if (!ok) std::fprintf(stderr, "NeighborCache profile write failed: %s\n", profile_path_.c_str());
    }
};
} // namespace neighborcache
