#pragma once
#include "nc_client.h"
#include <libaio.h>
#include <memory>

namespace diskann {
template<class T> class PageNeighborCache : public neighborcache::Client {
    struct Batch {
        io_context_t ctx = 0;
        std::vector<NCRead> reads;
        std::vector<uint32_t> ids;
        std::vector<iocb> cbs;
        std::vector<iocb *> ptrs;
        std::vector<int64_t> results;
        std::vector<size_t> primary;
        size_t submitted = 0;
        ~Batch() { if (ctx) io_destroy(ctx); }
    };
    static Batch &batch() { static thread_local Batch b; return b; }
  public:
    ~PageNeighborCache() { print(); }
    bool init(uint64_t record, uint64_t vector, uint64_t degree, const std::string &path,
              uint64_t points, uint64_t nps, const uint32_t *medoids, uint64_t count) {
        using neighborcache::env;
        const auto capacity = env("NEIGHBORCACHE_NCACHE_CAP_NODES", 0);
        if (!capacity) return false;
        NCConfig c{NC_BACKEND_ABI, 4, points, record, vector, degree, nps, capacity,
            env("NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES", 25600), env("NEIGHBORCACHE_THREADS", 64), path.c_str(), medoids, count};
        Client::init(c);
        return true;
    }
    struct DiskNode { uint32_t id = UINT32_MAX; NCRead read{}; };
    enum class Probe { Miss, Hit, Disk };
    Probe probe(uint32_t id, char *dst, DiskNode *d, uint32_t parent = UINT32_MAX) {
        NCRead r{};
        int k = resolve(id, dst, r, parent);
        if (k == 3) { *d = DiskNode{id, r}; return Probe::Disk; }
        return k ? Probe::Hit : Probe::Miss;
    }
    uint32_t submit_disk_nodes(const std::vector<DiskNode> &nodes) {
        Batch &b = batch();
        if (!b.reads.empty()) throw std::logic_error("Overlapping NeighborCache D batches");
        b.results.assign(nodes.size(), -1);
        b.cbs.resize(nodes.size());
        b.ptrs.clear();
        b.primary.clear();
        for (size_t i = 0; i < nodes.size(); ++i) {
            b.reads.push_back(nodes[i].read);
            b.ids.push_back(nodes[i].id);
            auto &r = b.reads.back();
            size_t primary = i;
            for (size_t j = 0; j < i; ++j) {
                if (b.primary[j] == j && b.reads[j].fd == r.fd && b.reads[j].offset == r.offset &&
                    share_path_read(b.reads[j], r)) {
                    primary = j;
                    break;
                }
            }
            b.primary.push_back(primary);
            if (primary == i) {
                io_prep_pread(&b.cbs[i], r.fd, r.buffer, r.bytes, r.offset);
                b.cbs[i].data = reinterpret_cast<void *>(i);
                b.ptrs.push_back(&b.cbs[i]);
            }
        }
        if (!b.ctx && io_setup(128, &b.ctx)) b.ctx = 0;
        b.submitted = 0;
        while (b.ctx && b.submitted < b.ptrs.size()) {
            int n = io_submit(b.ctx, b.ptrs.size() - b.submitted, b.ptrs.data() + b.submitted);
            if (n == -EINTR) continue;
            if (n <= 0) break;
            b.submitted += n;
        }
        for (size_t j = b.submitted; j < b.ptrs.size(); ++j) {
            const size_t i = reinterpret_cast<size_t>(b.ptrs[j]->data);
            auto &r = b.reads[i];
            do {
                b.results[i] = pread(r.fd, r.buffer, r.bytes, r.offset);
            } while (b.results[i] < 0 && errno == EINTR);
        }
        return b.ptrs.size();
    }
    uint32_t finish_disk_nodes(std::vector<std::pair<uint32_t, std::vector<char>>> &hits) {
        Batch &b = batch();
        size_t done = 0;
        std::vector<io_event> events(b.submitted);
        while (done < b.submitted) {
            int n = io_getevents(b.ctx, 1, b.submitted - done, events.data(), nullptr);
            if (n == -EINTR) continue;
            if (n <= 0) throw std::runtime_error("NeighborCache io_getevents failed");
            for (int j = 0; j < n; ++j) b.results[reinterpret_cast<size_t>(events[j].data)] = events[j].res;
            done += n;
        }
        bool failed = false;
        for (size_t i = 0; i < b.reads.size(); ++i) {
            std::vector<char> raw(node_len());
            if (complete(b.reads[i], b.results[b.primary[i]], raw.data())) hits.emplace_back(b.ids[i], std::move(raw));
            else failed = true;
        }
        uint32_t count = b.ptrs.size();
        b.reads.clear(); b.ids.clear(); b.submitted = 0;
        if (failed) throw std::runtime_error("NeighborCache D read failed; no unverified record was returned");
        return count;
    }
    void observe_node(uint32_t id, const char *raw, uint32_t parent = UINT32_MAX) { expand(id, raw, 0, parent); }
    void report() const { print(); }
};
}
