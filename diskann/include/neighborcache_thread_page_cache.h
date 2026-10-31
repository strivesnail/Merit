// Each query worker owns its page slots, lookup table and LRU. Dynamic D
// generations are shared metadata, never a shared prefetch payload or LRU.
#pragma once
#include <algorithm>
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
#include <utility>
#include "defaults.h"
#include "tsl/robin_map.h"

namespace diskann {
class NeighborCacheThreadPageCache {
    static constexpr uint32_t none = UINT32_MAX;
    using Bytes = std::array<char, defaults::SECTOR_LEN>;
    class PageBytes {
        std::unique_ptr<char[]> data_;
        uint16_t stored_=0;
      public:
        PageBytes() = default;
        PageBytes(const PageBytes &) = delete;
        PageBytes &operator=(const PageBytes &) = delete;
        PageBytes(PageBytes &&other) noexcept : data_(std::move(other.data_)),stored_(std::exchange(other.stored_,0)) {}
        PageBytes &operator=(PageBytes &&other) noexcept {
            if(this!=&other) { data_=std::move(other.data_); stored_=std::exchange(other.stored_,0); }
            return *this;
        }
        bool full() const { return stored_==defaults::SECTOR_LEN; }
        const char *data() const { return data_.get(); }
        void clear() { data_.reset(); stored_=0; }
        void copy_to(char *dst,size_t offset=0,size_t count=defaults::SECTOR_LEN) const {
            const size_t copied=offset<stored_ ? std::min(count,size_t(stored_)-offset) : 0;
            if(copied) std::memcpy(dst,data_.get()+offset,copied);
            if(copied<count) std::memset(dst+copied,0,count-copied);
        }
        bool equals(const char *src) const {
            Bytes image;
            copy_to(image.data());
            return std::memcmp(image.data(),src,defaults::SECTOR_LEN)==0;
        }
        void assign(const char *src,bool trim) {
            size_t count=defaults::SECTOR_LEN;
            if(trim) {
                while(count>=sizeof(uint64_t)) {
                    uint64_t word;
                    std::memcpy(&word,src+count-sizeof(word),sizeof(word));
                    if(word) break;
                    count-=sizeof(word);
                }
                while(count && src[count-1]==0) --count;
                // Small size classes avoid reallocating for insignificant tail changes.
                count=((count+511)/512)*512;
            }
            if(count!=stored_) {
                std::unique_ptr<char[]> fresh;
                if(count) { fresh.reset(new char[count]); std::memcpy(fresh.get(),src,count); }
                data_=std::move(fresh);
                stored_=static_cast<uint16_t>(count);
            } else if(count) std::memcpy(data_.get(),src,count);
        }
    };
    struct Slot {
        PageBytes bytes;
        std::vector<uint32_t> members;
        uint64_t version = 0;
        uint32_t key = none, prev = none, next = none;
        bool dirty = false;
    };
    struct Bucket { uint32_t key = none, slot = none; };
    struct RecordLocation { uint32_t page_slot, record_slot; };
    // One compact word handles the common single-page case. Multiple
    // locations use a reusable vector pool and preserve registration order.
    class RecordDirectory {
        tsl::robin_map<uint32_t,uint64_t> heads_;
        std::vector<std::vector<RecordLocation>> groups_;
        std::vector<uint32_t> free_groups_;
        static uint64_t encode(RecordLocation x) {
            return (uint64_t(x.page_slot)<<32)|x.record_slot;
        }
        static RecordLocation decode(uint64_t x) {
            return {uint32_t(x>>32),uint32_t(x)};
        }
        void release(uint32_t group) {
            groups_[group].clear();
            free_groups_.push_back(group);
        }
      public:
        RecordDirectory() { heads_.max_load_factor(0.7f); }
        void add(uint32_t id,RecordLocation loc) {
            auto it=heads_.find(id);
            if(it==heads_.end()) { heads_.insert({id,encode(loc)}); return; }
            const uint64_t old=it.value();
            if(uint32_t(old)==none) { groups_[uint32_t(old>>32)].push_back(loc); return; }
            uint32_t group;
            if(free_groups_.empty()) {
                if(groups_.size()>=none) throw std::length_error("Prefetch location pool exhausted");
                group=uint32_t(groups_.size()); groups_.emplace_back();
            } else { group=free_groups_.back(); free_groups_.pop_back(); }
            auto &items=groups_[group];
            try { items.push_back(decode(old)); items.push_back(loc); }
            catch(...) { release(group); throw; }
            it.value()=(uint64_t(group)<<32)|none;
        }
        void remove_page(uint32_t id,uint32_t page) {
            auto it=heads_.find(id); if(it==heads_.end()) return;
            const uint64_t entry=it.value();
            if(uint32_t(entry)!=none) {
                if(decode(entry).page_slot==page) heads_.erase(it);
                return;
            }
            const uint32_t group=uint32_t(entry>>32);
            auto &items=groups_[group];
            items.erase(std::remove_if(items.begin(),items.end(),[page](RecordLocation loc) {
                return loc.page_slot==page;
            }),items.end());
            if(items.size()>1) return;
            if(items.empty()) heads_.erase(it);
            else it.value()=encode(items.front());
            release(group);
        }
        template<class Visitor> bool newest_first(uint32_t id,Visitor &&visit) const {
            auto it=heads_.find(id); if(it==heads_.end()) return false;
            const uint64_t entry=it.value();
            if(uint32_t(entry)!=none) return visit(decode(entry));
            const auto &items=groups_[uint32_t(entry>>32)];
            for(auto loc=items.rbegin();loc!=items.rend();++loc)
                if(visit(*loc)) return true;
            return false;
        }
    };
    struct alignas(64) Local {
        size_t depth = 0, pins_used = 0;
        uint32_t oldest = none, newest = none;
        std::vector<Slot> slots;
        std::vector<Bucket> table;
        std::vector<uint32_t> free;
        // Both the directory and its referenced pages belong to this worker.
        RecordDirectory records;
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
    bool reset_ = false, immutable_ = true, trim_zero_tail_ = true;
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
    static void forget_records(Local &l, uint32_t page_slot) {

        auto &page=l.slots[page_slot];
        for(uint32_t id:page.members) l.records.remove_page(id,page_slot);
        page.members.clear();
    }
    bool remove(Local &l, uint32_t i, bool discard=false) {

        auto &s=l.slots[i];
        if(s.dirty && dynamic(s.key)) {
            std::unique_lock<std::mutex> lock(mutation_[s.key%mutation_.size()],std::defer_lock);
            lock.lock();
            // An obsolete replica must never write its bytes over the current D page.
            if(valid(s) && hook_) {
                Bytes image;
                const char *data=s.bytes.data();
                if(!s.bytes.full()) { s.bytes.copy_to(image.data()); data=image.data(); }
                if(!hook_(hook_context_,s.key,data,discard)) return false;
            }
        }
        forget_records(l,i);
        index_remove(l,s.key); unlink(l,i); s.key=none; s.dirty=false; s.bytes.clear(); l.free.push_back(i);
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
        const char *trim=std::getenv("NEIGHBORCACHE_PREFETCH_TRIM_ZERO_TAIL");
        trim_zero_tail_=trim==nullptr || std::strcmp(trim,"0")!=0;
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
    // Called after the page payload and its membership have been published.
    // No query worker can add entries to another worker's directory.
    void register_records(uint32_t key, const uint32_t *ids, size_t count) {

        auto *l=local(false); if(!l) return;
        const uint32_t i=find(*l,key);
        if(i==none || !valid(l->slots[i])) return;
        forget_records(*l,i);
        if(!count) return;
        if(count>defaults::SECTOR_LEN) throw std::length_error("Too many records for a prefetch page");
        auto &s=l->slots[i];
        s.members.assign(ids,ids+count);
        for(uint32_t slot=0;slot<count;slot++)
            l->records.add(ids[slot],{i,slot});
    }
    bool record_location(uint32_t id, uint32_t &key, uint32_t &record_slot, bool seed_only=false,
                         bool current_version=true) const {
        auto *l=local(false); if(!l) return false;
        return l->records.newest_first(id,[&](RecordLocation loc) {
            const auto &s=l->slots[loc.page_slot];
            if(s.key!=none && (!current_version || valid(s)) && loc.record_slot<s.members.size() &&
               s.members[loc.record_slot]==id && (!seed_only || loc.record_slot==0)) {
                key=s.key; record_slot=loc.record_slot; return true;
            }
            return false;
        });
    }
    bool seed_snapshot(uint32_t seed, uint32_t &key, std::vector<uint32_t> &ids, char *dst,
                       uint32_t missing_record=none) {

        uint32_t slot=0;
        if(!record_location(seed,key,slot,true,false)) return false;
        auto *l=local(false);
        const uint32_t i=find(*l,key);
        const auto &s=l->slots[i];
        // Preserve the old recency update even when no page update is needed,
        // but avoid copying/allocating a snapshot for an existing member.
        touch(*l,i);
        if(missing_record!=none && std::find(s.members.begin(),s.members.end(),missing_record)!=s.members.end())
            return false;
        ids=s.members;
        s.bytes.copy_to(dst);
        return true;
    }
    // The base-index records are immutable. Repacking a shared D page changes
    // its placement/version, but does not invalidate records already copied
    // into this worker's bounded private snapshot. Version checks still apply
    // to sector reads, page updates and writeback.
    bool copy_record(uint32_t id, size_t record_bytes, char *dst, bool current_version=false) {
        auto *l=local(false); if(!l) return false;
        const bool found=l->records.newest_first(id,[&](RecordLocation loc) {
            const auto &s=l->slots[loc.page_slot];
            const size_t off=size_t(loc.record_slot)*record_bytes;
            if(s.key!=none && (!current_version || valid(s)) && loc.record_slot<s.members.size() &&
               s.members[loc.record_slot]==id && record_bytes<=defaults::SECTOR_LEN &&
               off<=defaults::SECTOR_LEN-record_bytes) {
                s.bytes.copy_to(dst,off,record_bytes);
                touch(*l,loc.page_slot);
                l->hits.fetch_add(1,std::memory_order_relaxed);
                return true;
            }
            return false;
        });
        if(found) return true;
        l->misses.fetch_add(1,std::memory_order_relaxed); return false;
    }
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
            s.bytes.copy_to(dst+size_t(k)*defaults::SECTOR_LEN);
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
            std::unique_lock<std::mutex> lock(mutation_[key%mutation_.size()],std::defer_lock);
            lock.lock();
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
            std::unique_lock<std::mutex> lock(mutation_[key%mutation_.size()],std::defer_lock);
            lock.lock();
            const uint64_t v=versions_[key-dynamic_base_].fetch_add(1,std::memory_order_acq_rel)+1;
            if(!dirty) return false;
            dirty->store(v,std::memory_order_release);
            return hook_(hook_context_,key,src,false);
        }
        uint32_t i=find(*l,key);
        if(i!=none && !valid(l->slots[i])) { remove(*l,i,true); i=none; }
        if(i!=none && !overwrite) { touch(*l,i); return true; }
        if(i!=none && immutable_ && overwrite && !l->slots[i].bytes.equals(src))
            throw std::runtime_error("Cannot rewrite immutable private pages");
        const bool inserting=i==none;
        if(inserting) { i=reserve(*l); if(i==none) return false; }
        std::unique_lock<std::mutex> lock;
        if(dynamic(key)) {  lock=std::unique_lock<std::mutex>(mutation_[key%mutation_.size()]); }
        if(disk_dirty && disk_dirty->load(std::memory_order_acquire)) {
            if(inserting) l->free.push_back(i);
            return false;
        }
        auto &s=l->slots[i];
        uint64_t v=version(key);
        if(dynamic(key) && overwrite) v=versions_[key-dynamic_base_].fetch_add(1,std::memory_order_acq_rel)+1;
        s.bytes.assign(src,trim_zero_tail_); s.version=v; s.dirty=dirty!=nullptr;
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
