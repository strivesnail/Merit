// Compile the actual NeighborCache implementation, not a policy reimplementation.
// The private namespace and export map prevent interposition with host DiskANN forks.
#define NEIGHBORCACHE_BACKEND_ONLY
#include "../../diskann/src/pq_flash_index.cpp"
#include "nc_backend.h"
#include <stdexcept>

namespace diskann {
class NeighborCacheAdapter {
    using Index = PQFlashIndex<uint8_t>;
    NCConfig config;
    std::shared_ptr<AlignedFileReader> reader;
    Index index;
    size_t coords_bytes, tail_offset;
    std::vector<uint32_t> prefill_seeds;
    std::vector<std::pair<uint32_t, uint8_t>> prefill_spans;
    std::atomic<uint64_t> acquired_records{0}, recorded_parents{0};
    std::atomic<uint64_t> d_read_bytes{0};
    bool expand_prefetch_page = false;
    std::atomic<uint64_t> expanded_page_hits{0}, delivered_page_members{0};
    std::atomic<uint64_t> nh{0}, ph{0}, da{0}, dh{0}, dr{0}, bm{0};
  public:
    struct Ticket {
        std::vector<Index::NeighborCacheReadPending> pending;
        std::unordered_map<uint32_t, std::vector<size_t>> groups;
        char *buffer = nullptr;
        uint32_t id;
        uint64_t used = 0;
        explicit Ticket(uint32_t n) : id(n) {
            alloc_aligned(reinterpret_cast<void **>(&buffer), 128 * 4096, 4096);
        }
        ~Ticket() { aligned_free(buffer); }
    };
    struct Query {
        SSDQueryScratch<uint8_t> scratch;
        std::unique_ptr<NeighborCacheSharedPageCache::ReadSection> section;
        QueryStats stats{};
        uint64_t acquired = 0, parents = 0;
        uint64_t d_bytes = 0;
        uint32_t snapshot_target = UINT32_MAX;
        std::vector<uint32_t> snapshot_ids;
        std::vector<char> snapshot_records;
        uint64_t page_hits = 0, page_members = 0;
        std::unordered_map<uint32_t, int> source;
        std::unordered_set<Ticket *> tickets;
        explicit Query(Index &i) : scratch(i._aligned_dim, 4096) {}
        void reset(Index &i) {
            section = std::make_unique<NeighborCacheSharedPageCache::ReadSection>(i._neighborcache_shared_cache);
            scratch.reset();
            i.neighborcache_dyn_prepare_query(&scratch);
        }
        ~Query() { for (auto *t : tickets) delete t; }
    };

    explicit NeighborCacheAdapter(const NCConfig &c)
        : config(c), reader(std::make_shared<LinuxAlignedFileReader>()), index(reader, Metric::L2) {
        if (c.abi != NC_BACKEND_ABI || !c.points || !c.degree || !c.path ||
            (c.count_bytes != 2 && c.count_bytes != 4) || c.record_bytes > 4096 ||
            c.record_bytes < c.vector_bytes + 4 + 4 * c.degree || !c.n_capacity || !c.prefetch_pages)
            throw std::runtime_error("Invalid NeighborCache online configuration or record layout");
        // Move AiSAQ's opaque inline-PQ tail into the coordinate payload. Both
        // directions preserve it verbatim; the graph still has its original degree.
        tail_offset = c.vector_bytes + 4 + 4 * c.degree;
        coords_bytes = c.record_bytes - 4 - 4 * c.degree;
        index._num_points = c.points;
        index._data_dim = coords_bytes;
        index._aligned_dim = (coords_bytes + 7) & ~uint64_t(7);
        index._disk_bytes_per_point = coords_bytes;
        index._max_node_len = c.record_bytes;
        index._max_degree = c.degree;
        index._nnodes_per_sector = c.nodes_per_page;
        if (!c.nodes_per_page || c.nodes_per_page * c.record_bytes > 4096)
            throw std::runtime_error("NeighborCache requires records packed within a 4KiB page");
        index._max_nthreads = std::max<uint64_t>(1, c.threads);
        if (c.medoid_count) {
            index._num_medoids = c.medoid_count;
            index._medoids = new uint32_t[c.medoid_count];
            std::copy(c.medoids, c.medoids + c.medoid_count, index._medoids);
        }
        index._neighborcache_mem_pool = std::make_unique<NeighborCacheMemoryPool<uint8_t>>();
        index._neighborcache_mem_pool->init(c.n_capacity, index._aligned_dim, c.degree, c.points);
        index._neighborcache_mem_pool->commit_initial_load({}, {});
        index._query_sector_cache_enabled = true;
        if (!index._neighborcache_shared_cache.init(c.prefetch_pages, c.threads))
            throw std::runtime_error("Cannot initialize NeighborCache prefetch");
        std::fprintf(stderr,"NeighborCache prefetch policy: mode=%s total_pages=%lu query_threads=%lu pages_per_worker_min=%lu pages_per_worker_max=%lu\n",
            index._neighborcache_shared_cache.thread_private()?"thread_private":"shared",
            index._neighborcache_shared_cache.capacity_pages(),c.threads,
            c.prefetch_pages/std::max<uint64_t>(1,c.threads),(c.prefetch_pages+std::max<uint64_t>(1,c.threads)-1)/std::max<uint64_t>(1,c.threads));
        index._neighborcache_shared_cache_dyn = true;
        load_hot_prefill();
        index.enable_neighborcache_dynamic_3cache(true, c.path);
        expand_prefetch_page = neighborcache_env_u64("NEIGHBORCACHE_EXPAND_PREFETCH_PAGE", 0) != 0;
        if (expand_prefetch_page && (!index._neighborcache_static_no_mcache ||
            (index._neighborcache_dcache_second_page && neighborcache_env_u64("NEIGHBORCACHE_DCACHE_MAX_PAGES", 5) != 2)))
            throw std::runtime_error("Prefetch expansion requires frozen one- or two-page D groups");
        // Publish immutable offline page groups before any query can start.
        // The existing span IO path fetches a group in one contiguous request.
        for (const auto &[base, span] : prefill_spans) {
            for (uint32_t i = 0; i < span; ++i) {
                index._neighborcache_dyn_page_span_base.at(base + i) = base;
                index._neighborcache_dyn_page_span_len.at(base + i) = span;
                index._neighborcache_dyn_page_secondary.at(base + i) = i != 0;
            }
        }
        if (!index._neighborcache_static_no_mcache && !prefill_seeds.empty()) {
            std::ifstream pages(index._neighborcache_dc_path, std::ios::binary);
            std::vector<char> page(4096);
            for (uint32_t seed : prefill_seeds) {
                pages.read(page.data(), page.size());
                uint32_t degree = 0;
                std::memcpy(&degree, page.data() + coords_bytes, sizeof(degree));
                if (!pages || degree > c.degree) throw std::runtime_error("Invalid prefill seed record");
                std::vector<uint32_t> graph(degree);
                std::memcpy(graph.data(), page.data() + coords_bytes + 4, degree * 4);
                index.neighborcache_seed_page_link(seed, graph);
            }
        }
        index.neighborcache_dynamic_begin_measurement();
        // Do not mix offline M/heap import cost into query-phase lock samples.
        for (auto &m : neighborcache_lock_metrics()) {
            m.samples.store(0); m.wait_ns.store(0); m.hold_ns.store(0);
        }
    }

    ~NeighborCacheAdapter() {
        if (neighborcache_env_u64("NEIGHBORCACHE_BACKEND_REPORT", 0)) {
            std::fprintf(stderr, "NeighborCache acquired-parent: records=%lu parents=%lu\n", acquired_records.load(), recorded_parents.load());
            std::fprintf(stderr, "NeighborCache D IO: requests=%lu bytes=%lu\n", dr.load(), d_read_bytes.load());
            std::fprintf(stderr, "NeighborCache prefetch expansion: enabled=%u page_hits=%lu delivered_members=%lu\n",
                unsigned(expand_prefetch_page), expanded_page_hits.load(), delivered_page_members.load());
            index.print_neighborcache_dynamic_3cache_stats();
            index.print_neighborcache_memory_cache_stats();
            index.print_neighborcache_shared_cache_stats();
        }
    }

    // Offline pages use the same canonical record codec as online admission.
    // Only the seed choice and initial D contents come from the profile. The
    // reference importer constructs M seed states, heaps and dynamic locations.
    void load_hot_prefill() {
        const char *prefix = std::getenv("NEIGHBORCACHE_HOT_PREFILL_PREFIX");
        if (!prefix || !*prefix) return;
        if (!neighborcache_env_u64("NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC", 0))
            throw std::runtime_error("HOT_PREFILL_PREFIX requires DYNAMIC_PREFILL_STATIC=1");
        std::ifstream in(std::string(prefix) + ".slots", std::ios::binary);
        char magic[8]; uint64_t h[6];
        in.read(magic, sizeof(magic)); in.read(reinterpret_cast<char *>(h), sizeof(h));
        const bool grouped = in && std::memcmp(magic, "NCHOT2P\0", 8) == 0;
        if (!in || (!grouped && std::memcmp(magic, "NCHOT20\0", 8)) || h[0] != config.points ||
            h[1] != config.record_bytes || h[2] != config.vector_bytes || h[3] != config.degree ||
            h[4] != config.nodes_per_page || !h[5] || h[5] > config.points)
            throw std::runtime_error("Invalid hot-prefill layout/header");
        if (grouped && (neighborcache_env_u64("NEIGHBORCACHE_FREEZE_MAINTENANCE", 2) != 1 ||
                        neighborcache_env_u64("NEIGHBORCACHE_DCACHE_SECOND_PAGE", 0) != 1 ||
                        neighborcache_env_u64("NEIGHBORCACHE_DCACHE_MAX_PAGES", 5) != 2))
            throw std::runtime_error("Grouped hot prefill requires frozen maintenance and a two-page limit");
        index._neighborcache_dc_path = std::string(prefix) + ".pages";
        if (get_file_size(index._neighborcache_dc_path) != h[5] * 4096)
            throw std::runtime_error("Truncated hot-prefill pages");
        std::vector<uint32_t> ids(h[4]);
        std::unordered_set<uint32_t> seeds;
        for (uint64_t page = 0; page < h[5]; ++page) {
            in.read(reinterpret_cast<char *>(ids.data()), ids.size() * sizeof(uint32_t));
            if (!in) throw std::runtime_error("Truncated hot-prefill slots");
            const bool new_seed = seeds.insert(ids[0]).second;
            if (!new_seed && (!grouped || prefill_seeds.back() != ids[0] || prefill_spans.back().second != 1))
                throw std::runtime_error("Duplicate, noncontiguous, or oversized hot-prefill seed group");
            if (new_seed) {
                prefill_seeds.push_back(ids[0]);
                if (grouped) prefill_spans.emplace_back(static_cast<uint32_t>(page), 1);
            } else {
                prefill_spans.back().second = 2;
            }
            for (uint16_t slot = 0; slot < ids.size(); ++slot) {
                if (ids[slot] >= config.points) throw std::runtime_error("Invalid prefill member ID");
                Index::NeighborCacheDiskLoc loc;
                loc.sector = page; loc.slot = slot; loc.nsectors = 1;
                index._neighborcache_dc_map[ids[slot]].push_back(loc);
            }
        }
        if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing prefill metadata");
        std::fprintf(stderr, "NeighborCache hot prefill: seeds=%lu pages=%lu N/P initially empty\n", seeds.size(), h[5]);
    }

    void encode(const char *raw, char *canonical) const {
        std::memset(canonical, 0, config.record_bytes);
        std::memcpy(canonical, raw, config.vector_bytes);
        std::memcpy(canonical + config.vector_bytes, raw + tail_offset, config.record_bytes - tail_offset);
        uint32_t degree = 0;
        std::memcpy(&degree, raw + config.vector_bytes, config.count_bytes);
        if (degree > config.degree) throw std::runtime_error("Invalid graph degree in cache admission");
        std::memcpy(canonical + coords_bytes, &degree, 4);
        std::memcpy(canonical + coords_bytes + 4, raw + config.vector_bytes + 4, degree * 4);
    }
    void decode(const char *canonical, char *raw) const {
        std::memset(raw, 0, config.record_bytes);
        std::memcpy(raw, canonical, config.vector_bytes);
        std::memcpy(raw + tail_offset, canonical + config.vector_bytes, config.record_bytes - tail_offset);
        uint32_t degree;
        std::memcpy(&degree, canonical + coords_bytes, 4);
        if (degree > config.degree) throw std::runtime_error("Invalid graph degree in cached record");
        std::memcpy(raw + config.vector_bytes, &degree, config.count_bytes);
        std::memcpy(raw + config.vector_bytes + 4, canonical + coords_bytes + 4, degree * 4);
    }
    void unpack(const std::vector<uint8_t> &coords, const std::vector<uint32_t> &nbrs, char *raw) const {
        std::vector<char> canonical(config.record_bytes, 0);
        std::memcpy(canonical.data(), coords.data(), coords_bytes);
        uint32_t degree = nbrs.size();
        std::memcpy(canonical.data() + coords_bytes, &degree, 4);
        std::memcpy(canonical.data() + coords_bytes + 4, nbrs.data(), degree * 4);
        decode(canonical.data(), raw);
    }
    std::mutex pool_mu;
    std::vector<std::unique_ptr<Query>> free_queries;
    std::unordered_set<std::thread::id> registered;
    Query *begin() {
        std::unique_ptr<Query> q;
        {
            std::lock_guard<std::mutex> lock(pool_mu);
            if (registered.insert(std::this_thread::get_id()).second)
                index._neighborcache_dyn_reader->register_thread();
            if (!free_queries.empty()) { q = std::move(free_queries.back()); free_queries.pop_back(); }
        }
        if (!q) q = std::make_unique<Query>(index);
        q->reset(index);
        return q.release();
    }
    void end(Query *q) {
        if (!q->tickets.empty()) throw std::runtime_error("NeighborCache query ended with unfinished D IO");
        for (uint32_t s : q->scratch.neighborcache_dyn_pinned_sectors)
            index.neighborcache_dyn_release_page_pin(s);
        q->scratch.neighborcache_dyn_pinned_sectors.clear();
        index.neighborcache_dyn_on_query_end(&q->stats, &q->scratch);
        index._neighborcache_mem_pool->on_query_end(q->stats.n_disk_reads, q->stats.io_us);
        q->section.reset();
        q->source.clear();
        q->stats = QueryStats{};
        acquired_records.fetch_add(q->acquired); recorded_parents.fetch_add(q->parents);
        d_read_bytes.fetch_add(q->d_bytes, std::memory_order_relaxed);
        q->d_bytes = 0;
        expanded_page_hits.fetch_add(q->page_hits, std::memory_order_relaxed);
        delivered_page_members.fetch_add(q->page_members, std::memory_order_relaxed);
        q->page_hits = q->page_members = 0;
        q->snapshot_ids.clear(); q->snapshot_records.clear(); q->snapshot_target = UINT32_MAX;
        q->acquired = q->parents = 0;
        std::lock_guard<std::mutex> lock(pool_mu);
        free_queries.emplace_back(q);
    }
    void visit(Query *q, uint32_t parent, uint32_t child) {
        q->scratch.neighborcache_candidate_id = child;
        q->scratch.neighborcache_candidate_parent = parent;
    }
    void acquired_parent(Query *q, uint32_t child) {
        ++q->acquired;
        if (!q->scratch.neighborcache_dcache_maintenance_enabled) return;
        const uint32_t parent = q->scratch.neighborcache_candidate_id == child
            ? q->scratch.neighborcache_candidate_parent : UINT32_MAX;
        if (parent == UINT32_MAX || !q->scratch.profile_parent.insert({child, parent}).second) return;
        ++q->parents;
        if (q->scratch.neighborcache_dcache_maintenance_enabled && !index._neighborcache_real_io_coaccess) {
            const auto tr = index._neighborcache_mcache.on_edge(parent, child, index._neighborcache_score_unit.load());
            if (tr.present) {
                index.neighborcache_dyn_note_touch(tr, &q->scratch);
                index.neighborcache_mark_refresh_needed();
            }
        }
    }
    void selected(Query *q, uint32_t id) {
        q->scratch.neighborcache_expand_count++;
        if (!q->scratch.neighborcache_dcache_maintenance_enabled || index._neighborcache_coread_pack) return;
        const auto decision = index._neighborcache_mcache.plan_expand_update(id,
            index._neighborcache_dcache_net_gate_enabled && q->scratch.neighborcache_dcache_probe,
            index._neighborcache_dcache_net_gate_enabled && !q->scratch.neighborcache_dcache_lookup_enabled);
        if (decision.update) {
            const auto tr = index._neighborcache_mcache.on_expand(id, index._neighborcache_score_unit);
            index.neighborcache_dyn_note_touch(tr, &q->scratch);
        }
    }
    int finish(Query *q, Ticket &t, char *raw) {
        index.complete_neighborcache_disk_cache_io(&q->scratch, t.pending, t.groups);
        std::vector<std::pair<uint32_t, char *>> ready;
        index.finalize_neighborcache_pending_nodes(t.pending, {t.id}, &q->scratch,
            t.buffer, t.used, ready, &q->stats);
        if (ready.empty()) return 0;
        decode(index.offset_to_node(ready.front().second, t.id), raw);
        return 1;
    }
    void capture_prefetch_page(Query *q, uint32_t target, const Index::NeighborCacheDiskLoc &loc, const char *page) {
        if (!page || !index.neighborcache_dyn_is_sector(loc.sector)) return;
        const uint32_t p = loc.sector - Index::NEIGHBORCACHE_DYN_SECTOR_BASE;
        if (!index.neighborcache_dyn_page_live(p)) return;
        const size_t begin = size_t(p) * index._neighborcache_dyn_members_per_page;
        const size_t count = index._neighborcache_dyn_page_member_count.at(p);
        if (loc.slot >= count || index._neighborcache_dyn_page_members.at(begin + loc.slot) != target)
            throw std::runtime_error("Prefetch snapshot does not contain its requested record");
        q->snapshot_target = target;
        const uint32_t base = index._neighborcache_dcache_second_page
            ? index._neighborcache_dyn_page_span_base.at(p) : p;
        const uint32_t span = index._neighborcache_dcache_second_page
            ? index._neighborcache_dyn_page_span_len.at(base) : 1;
        if (!span || span > 2 || p < base || p - base >= span ||
            uint64_t(base) + span > index._neighborcache_dyn_physical_cap)
            throw std::runtime_error("Invalid immutable prefetch page group");
        std::array<char,4096> companion;
        for (uint32_t member_page = base; member_page < base + span; ++member_page) {
            const char *bytes = page;
            if (member_page != p) {
                // Expand only resident companion pages. A missing/evicted page
                // never causes an extra disk read just to expand the group.
                if (!index._neighborcache_shared_cache.get_range(
                    Index::NEIGHBORCACHE_DYN_SECTOR_BASE + member_page,1,companion.data())) continue;
                bytes = companion.data();
            }
            const size_t member_begin = size_t(member_page) * index._neighborcache_dyn_members_per_page;
            const size_t member_count = index._neighborcache_dyn_page_member_count.at(member_page);
            for (size_t slot = 0; slot < member_count; ++slot) {
                const uint32_t id = index._neighborcache_dyn_page_members.at(member_begin + slot);
                if (id >= config.points || id == target ||
                    std::find(q->snapshot_ids.begin(),q->snapshot_ids.end(),id) != q->snapshot_ids.end()) continue;
                q->snapshot_ids.push_back(id);
                const size_t offset = q->snapshot_records.size();
                q->snapshot_records.resize(offset + config.record_bytes);
                decode(bytes + slot * config.record_bytes,q->snapshot_records.data() + offset);
            }
        }
    }
    bool expanded_prefetch_lookup(Query *q, uint32_t id, char *raw) {
        Index::NeighborCacheDiskLoc loc;
        if (!index.neighborcache_resolve_disk_cache_loc(id,nullptr,loc) || !index.neighborcache_dyn_is_sector(loc.sector)) return false;
        NeighborCacheTimedSharedMutexGuard lock(index._neighborcache_dyn_mu, NeighborCacheLockKind::DynamicShared, false);
        const uint32_t p = loc.sector - Index::NEIGHBORCACHE_DYN_SECTOR_BASE;
        if (p >= index._neighborcache_dyn_physical_cap || !index.neighborcache_dyn_page_live(p) ||
            loc.slot >= index._neighborcache_dyn_page_member_count[p] ||
            index._neighborcache_dyn_page_members[size_t(p)*index._neighborcache_dyn_members_per_page+loc.slot] != id) return false;
        std::array<char,4096> page;
        if (!index._neighborcache_shared_cache.get_range(loc.sector,1,page.data())) return false;
        decode(page.data() + loc.slot * config.record_bytes,raw);
        capture_prefetch_page(q,id,loc,page.data());
        index._neighborcache_seed_page_hits.fetch_add(1,std::memory_order_relaxed);
        return true;
    }
    int prefetch_members(Query *q,uint32_t target,uint32_t *ids,char *records,uint32_t capacity) {
        if (!expand_prefetch_page || q->snapshot_target != target) return 0;
        const size_t n=q->snapshot_ids.size();
        if (capacity<n || !ids || !records) throw std::runtime_error("Prefetch snapshot output buffer too small");
        std::copy(q->snapshot_ids.begin(),q->snapshot_ids.end(),ids);
        if (n) std::memcpy(records,q->snapshot_records.data(),n*config.record_bytes);
        for (uint32_t id:q->snapshot_ids) q->source[id]=2;
        ++q->page_hits; q->page_members+=n;
        q->snapshot_ids.clear();q->snapshot_records.clear();q->snapshot_target=UINT32_MAX;
        return static_cast<int>(n);
    }
    int resolve(Query *q, uint32_t id, char *raw, NCRead *out) {
        q->snapshot_ids.clear();q->snapshot_records.clear();q->snapshot_target=UINT32_MAX;
        *out = NCRead{-1, 0, 0, nullptr, nullptr, id};
        std::vector<uint8_t> coords;
        std::vector<uint32_t> nbrs;
        uint8_t *ptr = nullptr;
        std::pair<uint32_t, uint32_t *> hood;
        int kind = 0;
        bool frontier = false;
        // Copies occur while holding the reference N shard lock.
        if (index._neighborcache_mem_pool->lookup(id, ptr, hood, true, &coords, &nbrs)) {
            unpack(coords, nbrs, raw); ++nh; kind = 1;
        } else if (expand_prefetch_page && expanded_prefetch_lookup(q,id,raw)) {
            ++ph;kind=2;
        } else if (!expand_prefetch_page && index.neighborcache_seed_page_lookup(id, coords, nbrs)) {
            unpack(coords, nbrs, raw); ++ph; kind = 2;
        } else {
            const bool local = index.neighborcache_dyn_query_local_dcache_hit(id, &q->scratch);
            NeighborCacheDemandProbe probe;
            bool found = local;
            if (!local) {
                NeighborCacheDemandProbeScope scope(probe);
                found = index.neighborcache_disk_cache_lookup_hit(id, &q->scratch);
            }
            if (probe.attempted) { ++da; if (probe.hit) ++dh; }
            if (found) {
                frontier = true;
                auto loc = q->scratch.neighborcache_dcache_loc_cache.find(id);
                if (loc != q->scratch.neighborcache_dcache_loc_cache.end())
                    index.neighborcache_pack_dyn_page_into_query_prefetch(uint32_t(loc->second >> 32), &q->scratch);
                auto t = std::make_unique<Ticket>(id);
                std::vector<AlignedRead> reads;
                uint32_t ios = 0;
                index.prepare_neighborcache_disk_cache_io({id}, &q->scratch, t->buffer, t->used, 1,
                    t->pending, reads, t->groups, &q->stats, ios);
                if (reads.empty()) {
                    if (finish(q, *t, raw)) {
                        kind = 2; ++ph;
                        if (expand_prefetch_page && !t->pending.empty()) {
                            const auto &p=t->pending.front();
                            // Dynamic node_ptr can be a single-record spill, not a page.
                            // Reacquire the immutable full page; eviction means no extras.
                            std::array<char,4096> page;
                            if (index._neighborcache_shared_cache.get_range(p.loc.sector,1,page.data()))
                                capture_prefetch_page(q,id,p.loc,page.data());
                        }
                    }
                } else {
                    if (reads.size() != 1) throw std::runtime_error("Unexpected multiple reads for a single cache lookup");
                    out->fd = static_cast<LinuxAlignedFileReader *>(index._neighborcache_dyn_reader.get())->get_file_desc();
                    out->offset = reads[0].offset;
                    out->bytes = reads[0].len;
                    q->d_bytes += reads[0].len;
                    out->buffer = reads[0].buf;
                    out->ticket = t.get();
                    q->tickets.insert(t.release());
                    ++dr; kind = 3;
                }
            }
            if (!kind) ++bm;
        }
        q->source[id] = kind == 2 && frontier ? 4 : kind;
        return kind;
    }
    int complete(Query *q, NCRead *r, int64_t got, char *raw) {
        auto *t = static_cast<Ticket *>(r->ticket);
        if (!t || !q->tickets.erase(t)) throw std::runtime_error("Invalid or duplicate D IO completion");
        std::unique_ptr<Ticket> owned(t);
        r->ticket = nullptr;
        if (got != int64_t(r->bytes)) { q->source[t->id] = 0; ++bm; return 0; }
        return finish(q, *t, raw);
    }
    void expand(Query *q, uint32_t id, const char *raw, uint32_t hop) {
        acquired_parent(q, id);
        // Starling also expands other nodes from a page it already fetched.
        selected(q, id);
        auto src = q->source.find(id);
        if (src == q->source.end()) { q->source[id] = 0; src = q->source.find(id); }
        if (src->second == 1 || src->second == 2) {
            uint8_t *coords = nullptr; std::pair<uint32_t, uint32_t *> hood;
            index._neighborcache_mem_pool->lookup(id, coords, hood);
            return;
        }
        std::vector<char> canonical(config.record_bytes);
        encode(raw, canonical.data());
        index.neighborcache_dyn_admit_node(id, canonical.data(), &q->stats, &q->scratch, hop);
        if (!q->scratch.neighborcache_dcache_maintenance_enabled) return;
        uint32_t parent = UINT32_MAX;
        auto p = q->scratch.profile_parent.find(id);
        if (p != q->scratch.profile_parent.end()) parent = p->second;
        index.neighborcache_dyn_note_base_load(id, parent, canonical.data(), &q->scratch);
    }
    void stats(NCStats *s) {
        *s = {nh.load(), ph.load(), da.load(), dh.load(), dr.load(), bm.load(),
            index.neighborcache_dynamic_page_count(), index.neighborcache_dynamic_page_capacity(),
            index._neighborcache_mcache.size(), index._neighborcache_seed_page_opens.load(),
            index._neighborcache_seed_page_appends.load(), index._neighborcache_seed_page_replaces.load(),
            index._neighborcache_seed_page_written.load(),
            acquired_records.load(), recorded_parents.load()};
    }
    void base_reads(Query *q, uint32_t n, double us) {
        q->stats.n_disk_reads += n;
        q->stats.io_us += us;
    }
    bool contains(uint32_t id) { return index._neighborcache_mem_pool->contains(id); }
};
} // namespace diskann

namespace {
thread_local std::string last_error;
using Adapter = diskann::NeighborCacheAdapter;
using Query = Adapter::Query;
template<class F> int checked(F f) {
    try { return f(); } catch (const std::exception &e) { last_error = e.what(); return -1; }
}
const NCBackend api = {
    NC_BACKEND_ABI, [] { return last_error.c_str(); },
    [](const NCConfig *c) -> void * { try { return new Adapter(*c); } catch (const std::exception &e) { last_error = e.what(); return nullptr; } },
    [](void *p) { delete static_cast<Adapter *>(p); },
    [](void *p) -> void * { try { return static_cast<Adapter *>(p)->begin(); } catch (const std::exception &e) { last_error = e.what(); return nullptr; } },
    [](void *p, void *q) { return checked([&] { static_cast<Adapter *>(p)->end(static_cast<Query *>(q)); return 0; }); },
    [](void *p, void *q, uint32_t a, uint32_t b) { return checked([&] { static_cast<Adapter *>(p)->visit(static_cast<Query *>(q), a, b); return 0; }); },
    [](void *p, void *q, uint32_t id, char *dst, NCRead *r) { return checked([&] { return static_cast<Adapter *>(p)->resolve(static_cast<Query *>(q), id, dst, r); }); },
    [](void *p, void *q, NCRead *r, int64_t n, char *dst) { return checked([&] { return static_cast<Adapter *>(p)->complete(static_cast<Query *>(q), r, n, dst); }); },
    [](void *p, void *q, uint32_t id, const char *raw, uint32_t hop) { return checked([&] { static_cast<Adapter *>(p)->expand(static_cast<Query *>(q), id, raw, hop); return 0; }); },
    [](void *p, NCStats *s) { return checked([&] { static_cast<Adapter *>(p)->stats(s); return 0; }); },
    [](void *p, uint32_t id) { return checked([&] { return int(static_cast<Adapter *>(p)->contains(id)); }); },
    [](void *p, void *q, uint32_t n, double us) { return checked([&] { static_cast<Adapter *>(p)->base_reads(static_cast<Query *>(q), n, us); return 0; }); }
};
}
extern "C" __attribute__((visibility("default"))) const NCBackend *nc_backend_api() { return &api; }
// Optional extension: ABI 2 hosts that do not request it keep their old behavior.
extern "C" __attribute__((visibility("default"))) int nc_prefetch_page_members(
    void *cache,void *query,uint32_t target,uint32_t *ids,char *records,uint32_t capacity) {
    return checked([&] { return static_cast<Adapter *>(cache)->prefetch_members(static_cast<Query *>(query),target,ids,records,capacity); });
}
