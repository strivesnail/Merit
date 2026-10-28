// Each query worker owns its page slots, lookup table and LRU. Dynamic D
// generations are shared metadata, never a shared prefetch payload or LRU.
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include "defaults.h"

namespace diskann {
class NeighborCacheThreadPageCache {
    static constexpr uint32_t none = UINT32_MAX;
    using Bytes = std::array<char, defaults::SECTOR_LEN>;
    struct Slot {
        Bytes bytes;
        uint64_t version = 0;
        uint32_t key = none, prev = none, next = none;
        bool dirty = false;
    };
    struct Bucket { uint32_t key = none, slot = none; };
    struct alignas(64) Local {
        size_t depth = 0, pins_used = 0;
        uint32_t oldest = none, newest = none;
        std::vector<Slot> slots;
        std::vector<Bucket> table;
        std::vector<uint32_t> free;
        // Pointer API returns query-owned copies, reused at the next query.
        // This scratch is separate from the bounded resident page payload.
        std::deque<Bytes> pins;
        std::atomic<uint64_t> hits{0}, misses{0}, inserts{0}, evictions{0};
    };
    struct Binding { uint64_t generation; Local *local; };
    static inline std::atomic<uint64_t> next_generation{1};
    static inline thread_local std::unordered_map<const NeighborCacheThreadPageCache *, Binding> bindings;
    const uint64_t generation_ = next_generation.fetch_add(1, std::memory_order_relaxed);
    mutable std::atomic<size_t> next_worker_{0};
    std::unique_ptr<Local[]> locals_;
    size_t threads_ = 0;
    uint64_t capacity_ = 0;
    bool reset_ = false, immutable_ = true;
    uint32_t dynamic_base_ = 0;
    uint64_t dynamic_count_ = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> versions_;
    // Only mutation/invalidation/writeback handoff uses these short metadata
    // locks. Hits never acquire a shared cache lock.
    std::array<std::mutex, 256> mutation_;
    using Hook = bool (*)(void *, uint32_t, const char *, bool);
    Hook hook_ = nullptr;
    void *hook_context_ = nullptr;
    Local *local(bool create) const {
        auto it = bindings.find(this);
        if (it != bindings.end() && it->second.generation == generation_) return it->second.local;
        if (!create) return nullptr;
        const size_t worker = next_worker_.fetch_add(1, std::memory_order_relaxed);
        if (worker >= threads_) throw std::runtime_error("Private prefetch: more query workers than configured");
        Local *l = &locals_[worker];
        bindings[this] = {generation_, l};
        return l;
    }
    static size_t hash(uint32_t key) { return uint64_t(key) * 11400714819323198485ull; }
    static uint32_t find(const Local &l, uint32_t key) {
        size_t i = hash(key) & (l.table.size()-1);
        for (;;) {
            const auto &b = l.table[i];
            if (b.key == none) return none;
            if (b.key == key) return b.slot;
            i = (i+1) & (l.table.size()-1);
        }
    }
    static void index_add(Local &l, uint32_t key, uint32_t slot) {
        size_t i = hash(key) & (l.table.size()-1);
        while (l.table[i].key != none) i = (i+1) & (l.table.size()-1);
        l.table[i] = {key, slot};
    }
    static void index_remove(Local &l, uint32_t key) {
        const size_t mask = l.table.size()-1;
        size_t hole = hash(key) & mask;
        while (l.table[hole].key != key) hole = (hole+1) & mask;
        l.table[hole] = {};
        for (size_t i=(hole+1)&mask; l.table[i].key!=none; i=(i+1)&mask) {
            const auto b=l.table[i];
            const size_t home=hash(b.key)&mask;
            if (((hole-home)&mask) < ((i-home)&mask)) {
                l.table[hole]=b; l.table[i]={}; hole=i;
            }
        }
    }
    static void unlink(Local &l, uint32_t i) {
        auto &s=l.slots[i];
        if(s.prev!=none) l.slots[s.prev].next=s.next; else l.oldest=s.next;
        if(s.next!=none) l.slots[s.next].prev=s.prev; else l.newest=s.prev;
    }
    static void touch(Local &l, uint32_t i) {
        auto &s=l.slots[i];
        if(l.newest==i) return;
        unlink(l,i); s.prev=l.newest; s.next=none;
        if(l.newest!=none) l.slots[l.newest].next=i; else l.oldest=i;
        l.newest=i;
    }
    bool dynamic(uint32_t key) const { return versions_ && key>=dynamic_base_ && uint64_t(key-dynamic_base_)<dynamic_count_; }
    uint64_t version(uint32_t key) const {
        return dynamic(key) ? versions_[key-dynamic_base_].load(std::memory_order_acquire) : 0;
    }
    bool valid(const Slot &s) const { return s.version==version(s.key); }
    bool remove(Local &l, uint32_t i, bool discard=false) {
        auto &s=l.slots[i];
        if(s.dirty && dynamic(s.key)) {
            std::lock_guard<std::mutex> lock(mutation_[s.key%mutation_.size()]);
            // An obsolete replica must never write its bytes over the current D page.
            if(valid(s) && hook_ && !hook_(hook_context_,s.key,s.bytes.data(),discard)) return false;
        }
        index_remove(l,s.key); unlink(l,i); s.key=none; s.dirty=false; l.free.push_back(i);
        return true;
    }
    uint32_t reserve(Local &l) {
        if(l.free.empty()) {
            if(!remove(l,l.oldest)) return none;
            l.evictions.fetch_add(1,std::memory_order_relaxed);
        }
        uint32_t i=l.free.back(); l.free.pop_back(); return i;
    }
  public:
    bool init(uint64_t pages,size_t threads,bool reset=false,bool immutable=true) {
        if(locals_ || !threads || pages<threads || pages>=none)
            throw std::runtime_error("Private prefetch requires at least one page per worker and one initialization");
        threads_=threads; capacity_=pages; reset_=reset; immutable_=immutable;
        locals_=std::make_unique<Local[]>(threads);
        for(size_t t=0;t<threads;t++) {
            auto &l=locals_[t]; const size_t n=pages/threads+(t<pages%threads);
            l.slots.resize(n); l.free.reserve(n);
            size_t buckets=1; while(buckets<2*n) buckets*=2;
            l.table.resize(buckets);
            for(size_t i=n;i>0;i--) l.free.push_back(uint32_t(i-1));
        }
        return true;
    }
    // Configure once, before queries start. Static D requires no version array.
    void configure_dynamic(uint32_t base,uint64_t count) {
        if(immutable_) return;
        if(versions_ || uint64_t(base)+count>none) throw std::runtime_error("Invalid private D version range");
        dynamic_base_=base; dynamic_count_=count;
        versions_=std::make_unique<std::atomic<uint64_t>[]>(count);
        for(uint64_t i=0;i<count;i++) versions_[i].store(0,std::memory_order_relaxed);
    }
    void set_evict_hook(Hook h,void *ctx) { hook_=h; hook_context_=ctx; }
    bool enabled() const { return bool(locals_); }
    uint64_t capacity_pages() const { return capacity_; }
    void enter() {
        auto &l=*local(true);
        if(l.depth++==0 && reset_)
            while(l.oldest!=none) if(!remove(l,l.oldest)) throw std::runtime_error("Private reset writeback rejected");
    }
    void exit() {
        auto &l=*local(true);
        if(!l.depth) throw std::logic_error("Private prefetch unbalanced ReadSection");
        if(--l.depth==0) l.pins_used=0;
    }
    bool contains_range(uint32_t base,uint16_t n) const {
        auto *l=local(immutable_); if(!l) return false;
        for(uint16_t k=0;k<n;k++) { auto i=find(*l,base+k); if(i==none || !valid(l->slots[i])) return false; }
        return true;
    }
    bool get_range(uint32_t base,uint16_t n,char *dst) {
        auto *l=local(immutable_); if(!l) return false;
        for(uint16_t k=0;k<n;k++) {
            auto i=find(*l,base+k);
            if(i==none || !valid(l->slots[i])) { l->misses.fetch_add(1,std::memory_order_relaxed); return false; }
        }
        for(uint16_t k=0;k<n;k++) {
            auto i=find(*l,base+k); auto &s=l->slots[i];
            std::memcpy(dst+size_t(k)*defaults::SECTOR_LEN,s.bytes.data(),defaults::SECTOR_LEN);
            if(!valid(s)) { l->misses.fetch_add(1,std::memory_order_relaxed); return false; }
            touch(*l,i);
        }
        l->hits.fetch_add(1,std::memory_order_relaxed); return true;
    }
    const char *get_ptr(uint32_t key) {
        auto *l=local(false); if(!l || !l->depth) return nullptr;
        if(l->pins_used==l->pins.size()) l->pins.emplace_back();
        auto &copy=l->pins[l->pins_used];
        if(!get_range(key,1,copy.data())) return nullptr;
        ++l->pins_used; return copy.data();
    }
    void erase(uint32_t key) {
        if(dynamic(key)) {
            std::lock_guard<std::mutex> lock(mutation_[key%mutation_.size()]);
            versions_[key-dynamic_base_].fetch_add(1,std::memory_order_release);
        }
        auto *l=local(false); if(!l) return;
        auto i=find(*l,key); if(i!=none) remove(*l,i,true);
    }
    bool put(uint32_t key,const char *src,bool overwrite=false,std::atomic<uint64_t> *dirty=nullptr,
             const std::atomic<uint64_t> *disk_dirty=nullptr) {
        if(key==none) return false;
        if(dirty && (!dynamic(key) || !hook_)) throw std::runtime_error("Private dirty page needs configured D range and asynchronous eviction hook");
        if(dirty) overwrite=true; // A dirty admission always publishes a new version.
        auto *l=local(immutable_);
        // Background maintenance never consumes a query worker's cache budget.
        // Dirty maintenance snapshots are handed directly to the existing writer.
        if(!l) {
            if(!dynamic(key) || !overwrite) return false;
            std::lock_guard<std::mutex> lock(mutation_[key%mutation_.size()]);
            const uint64_t v=versions_[key-dynamic_base_].fetch_add(1,std::memory_order_acq_rel)+1;
            if(!dirty) return false;
            dirty->store(v,std::memory_order_release);
            return hook_(hook_context_,key,src,false);
        }
        uint32_t i=find(*l,key);
        if(i!=none && !valid(l->slots[i])) { remove(*l,i,true); i=none; }
        if(i!=none && !overwrite) { touch(*l,i); return true; }
        if(i!=none && immutable_ && overwrite && std::memcmp(l->slots[i].bytes.data(),src,defaults::SECTOR_LEN))
            throw std::runtime_error("Cannot rewrite immutable private pages");
        const bool inserting=i==none;
        if(inserting) { i=reserve(*l); if(i==none) return false; }
        std::unique_lock<std::mutex> lock;
        if(dynamic(key)) lock=std::unique_lock<std::mutex>(mutation_[key%mutation_.size()]);
        if(disk_dirty && disk_dirty->load(std::memory_order_acquire)) {
            if(inserting) l->free.push_back(i);
            return false;
        }
        auto &s=l->slots[i];
        uint64_t v=version(key);
        if(dynamic(key) && overwrite) v=versions_[key-dynamic_base_].fetch_add(1,std::memory_order_acq_rel)+1;
        std::memcpy(s.bytes.data(),src,defaults::SECTOR_LEN); s.version=v; s.dirty=dirty!=nullptr;
        if(dirty) dirty->store(v,std::memory_order_release);
        // Clean admissions race only metadata changes: stale data remains private
        // and is rejected on its next lookup. In-flight D reads retain core pins.
        if(inserting) {
            s.key=key; s.prev=l->newest; s.next=none;
            if(l->newest!=none) l->slots[l->newest].next=i; else l->oldest=i;
            l->newest=i; index_add(*l,key,i); l->inserts.fetch_add(1,std::memory_order_relaxed);
        } else touch(*l,i);
        return true;
    }
    uint64_t hits() const { uint64_t n=0;for(size_t i=0;i<threads_;i++) n+=locals_[i].hits.load(std::memory_order_relaxed);return n; }
    uint64_t misses() const { uint64_t n=0;for(size_t i=0;i<threads_;i++) n+=locals_[i].misses.load(std::memory_order_relaxed);return n; }
    uint64_t inserts() const { uint64_t n=0;for(size_t i=0;i<threads_;i++) n+=locals_[i].inserts.load(std::memory_order_relaxed);return n; }
    uint64_t evictions() const { uint64_t n=0;for(size_t i=0;i<threads_;i++) n+=locals_[i].evictions.load(std::memory_order_relaxed);return n; }
};
}
