#include "neighborcache_nodes.h"
#include "neighborcache_path_seed.h"
#include "nc_backend.h"
#include <omp.h>

namespace diskann
{
class NeighborCacheAdapter
{
    size_t width;
    NeighborCacheNodes nodes;
    PathSeedCache pages;
    std::atomic<uint64_t> nh{0}, ph{0}, da{0}, dh{0}, dr{0}, bm{0}, acquired{0}, parents{0};

  public:
    struct Query
    {
        PathSeedCache::Query path;
        uint32_t child = UINT32_MAX, parent = UINT32_MAX;
        int worker = omp_get_thread_num();
        std::unordered_map<uint32_t, int> sources;
        std::unordered_map<void *, std::unique_ptr<PathSeedCache::Location>> tickets;
        uint32_t parent_of(uint32_t id) const
        {
            return child == id ? parent : UINT32_MAX;
        }
    };
    static size_t record_size(const NCConfig &c)
    {
        if (c.abi != NC_BACKEND_ABI || !c.path || !c.points || !c.record_bytes || c.record_bytes > 2048 ||
            !c.nodes_per_page || c.nodes_per_page > 4096 / c.record_bytes || !c.n_capacity ||
            (c.count_bytes != 2 && c.count_bytes != 4))
            throw std::invalid_argument("Invalid path-cache configuration");
        return c.record_bytes;
    }
    explicit NeighborCacheAdapter(const NCConfig &c)
        : width(record_size(c)), nodes(c.n_capacity, width),
          pages(std::string(c.path) + "_pathseed.data", width, c.prefetch_pages, neighborcache_pin_background_thread)
    {
    }
    Query *begin()
    {
        return new Query();
    }
    void end(Query *q)
    {
        std::unique_ptr<Query> owner(q);
        if (!q->tickets.empty())
            throw std::logic_error("Unfinished D reads");
        pages.commit(q->path, q->worker);
    }
    void visit(Query *q, uint32_t parent, uint32_t child)
    {
        q->parent = parent;
        q->child = child;
    }
    bool contains(uint32_t id)
    {
        return nodes.contains(id);
    }
    int resolve(Query *q, uint32_t id, char *dst, NCRead *out)
    {
        *out = NCRead{-1, 0, 0, nullptr, nullptr, id};
        if (nodes.lookup(id, dst))
        {
            ++nh;
            return q->sources[id] = 1;
        }
        ++da;
        auto loc = pages.lookup(id, q->parent_of(id), q->worker);
        if (!loc.p)
        {
            ++bm;
            return q->sources[id] = 0;
        }
        if (loc.image.bytes)
        {
            memcpy(dst, pages.record(loc), width);
            ++ph;
            return q->sources[id] = 2;
        }
        auto ticket = std::make_unique<PathSeedCache::Location>(std::move(loc));
        pages.prepare_external(*ticket, q->worker, out->fd, out->offset);
        out->bytes = 4096;
        out->buffer = ticket->image.bytes->data;
        out->ticket = ticket.get();
        q->tickets.emplace(out->ticket, std::move(ticket));
        ++dr;
        ++dh;
        return q->sources[id] = 3;
    }
    int share_path_read(Query *q, const NCRead &first, NCRead &other)
    {
        if (first.ticket == other.ticket)
            return 0;
        auto &loc = *q->tickets.at(other.ticket);
        if (!pages.share_external(*q->tickets.at(first.ticket), loc))
            return 0;
        other.buffer = loc.image.bytes->data;
        --dr;
        return 1;
    }
    int complete(Query *q, NCRead *r, int64_t got, char *dst)
    {
        auto ticket = q->tickets.extract(r->ticket);
        if (ticket.empty())
            throw std::logic_error("Invalid D ticket");
        r->ticket = nullptr;
        if (got != int64_t(r->bytes))
        {
            ++bm;
            q->sources[r->node_id] = 0;
            return 0;
        }
        auto &loc = *ticket.mapped();
        pages.publish_external(loc, q->worker);
        memcpy(dst, pages.record(loc), width);
        return 1;
    }
    void expand(Query *q, uint32_t id, const char *raw, uint32_t)
    {
        if (q->path.index.count(id))
            return;
        auto parent = q->parent_of(id);
        // A host can expand a static-cache record without passing its parent
        // through this backend. Start a new path at the first observed record.
        if (!q->path.index.count(parent))
            parent = UINT32_MAX;
        q->path.add_raw(id, parent, raw, width);
        ++acquired;
        parents += parent != UINT32_MAX;
        int source = q->sources[id];
        if (source != 1 && source != 2)
            nodes.admit(id, raw);
    }
    void stats(NCStats *s)
    {
        *s = {nh.load(), ph.load(), da.load(), dh.load(), dr.load(),           bm.load(),       pages.page_count(), 0,
              0,         0,         0,         0,         pages.write_count(), acquired.load(), parents.load()};
    }
    void base_reads(Query *, uint32_t, double)
    {
    }
};
} // namespace diskann
namespace
{
thread_local std::string last_error;
using Adapter = diskann::NeighborCacheAdapter;
using Query = Adapter::Query;
template <class F> int checked(F f)
{
    try
    {
        return f();
    }
    catch (const std::exception &e)
    {
        last_error = e.what();
        return -1;
    }
}
const NCBackend api = {
    NC_BACKEND_ABI,
    [] { return last_error.c_str(); },
    [](const NCConfig *c) -> void * {
        try
        {
            return new Adapter(*c);
        }
        catch (const std::exception &e)
        {
            last_error = e.what();
            return nullptr;
        }
    },
    [](void *p) { delete static_cast<Adapter *>(p); },
    [](void *p) -> void * {
        try
        {
            return static_cast<Adapter *>(p)->begin();
        }
        catch (const std::exception &e)
        {
            last_error = e.what();
            return nullptr;
        }
    },
    [](void *p, void *q) {
        return checked([&] {
            static_cast<Adapter *>(p)->end(static_cast<Query *>(q));
            return 0;
        });
    },
    [](void *p, void *q, uint32_t a, uint32_t b) {
        return checked([&] {
            static_cast<Adapter *>(p)->visit(static_cast<Query *>(q), a, b);
            return 0;
        });
    },
    [](void *p, void *q, uint32_t id, char *dst, NCRead *r) {
        return checked([&] { return static_cast<Adapter *>(p)->resolve(static_cast<Query *>(q), id, dst, r); });
    },
    [](void *p, void *q, NCRead *r, int64_t n, char *dst) {
        return checked([&] { return static_cast<Adapter *>(p)->complete(static_cast<Query *>(q), r, n, dst); });
    },
    [](void *p, void *q, uint32_t id, const char *raw, uint32_t hop) {
        return checked([&] {
            static_cast<Adapter *>(p)->expand(static_cast<Query *>(q), id, raw, hop);
            return 0;
        });
    },
    [](void *p, NCStats *s) {
        return checked([&] {
            static_cast<Adapter *>(p)->stats(s);
            return 0;
        });
    },
    [](void *p, uint32_t id) { return checked([&] { return int(static_cast<Adapter *>(p)->contains(id)); }); },
    [](void *p, void *q, uint32_t n, double us) {
        return checked([&] {
            static_cast<Adapter *>(p)->base_reads(static_cast<Query *>(q), n, us);
            return 0;
        });
    }};
} // namespace
extern "C" __attribute__((visibility("default"))) const NCBackend *nc_backend_api()
{
    return &api;
}
extern "C" __attribute__((visibility("default"))) int nc_share_path_read(void *cache, void *query,
                                                                         const NCRead *primary, NCRead *other)
{
    return checked(
        [&] { return static_cast<Adapter *>(cache)->share_path_read(static_cast<Query *>(query), *primary, *other); });
}
