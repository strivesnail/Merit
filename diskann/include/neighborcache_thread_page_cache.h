// Thread-owned LRU for immutable prefetch pages. Metadata and payload are private;
// the backing D file and its routing directory remain shared.
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include "defaults.h"

namespace diskann {
class NeighborCacheThreadPageCache {
    struct alignas(defaults::SECTOR_LEN) Page { std::array<char, defaults::SECTOR_LEN> bytes; };
    struct Entry { std::shared_ptr<Page> page; std::list<uint32_t>::iterator lru; };
    struct alignas(64) Local {
        size_t capacity=0, depth=0;
        std::list<uint32_t> lru;
        std::unordered_map<uint32_t,Entry> pages;
        // Only get_ptr pins bytes; copies do not prolong residency. Pins are
        // query scratch, outside the bounded resident-page payload budget.
        std::vector<std::shared_ptr<Page>> pins;
        std::atomic<uint64_t> hits{0},misses{0},inserts{0},evictions{0};
    };
    struct Binding { uint64_t generation; Local *local; };
    static inline std::atomic<uint64_t> next_generation{1};
    static inline thread_local std::unordered_map<const NeighborCacheThreadPageCache *,Binding> bindings;
    const uint64_t generation_=next_generation.fetch_add(1,std::memory_order_relaxed);
    mutable std::atomic<size_t> next_worker_{0};
    std::unique_ptr<Local[]> locals_;
    size_t threads_=0;
    uint64_t capacity_=0;
    bool reset_=false;
    Local &local() const {
        auto it=bindings.find(this);
        if(it!=bindings.end() && it->second.generation==generation_)return *it->second.local;
        const size_t worker=next_worker_.fetch_add(1,std::memory_order_relaxed);
        if(worker>=threads_)throw std::runtime_error("Private prefetch: more worker threads than configured");
        Local *l=&locals_[worker];bindings[this]={generation_,l};return *l;
    }
    static void touch(Local &l,std::unordered_map<uint32_t,Entry>::iterator p) {
        l.lru.splice(l.lru.end(),l.lru,p->second.lru);
    }
    static void clear(Local &l) { l.pages.clear();l.lru.clear(); }
  public:
    bool init(uint64_t pages,size_t threads,bool reset=false) {
        if(locals_ || !threads || pages<threads)throw std::runtime_error("Private prefetch requires at least one page per worker and one initialization");
        threads_=threads;capacity_=pages;reset_=reset;locals_=std::make_unique<Local[]>(threads);
        for(size_t i=0;i<threads;i++){
            auto &l=locals_[i];l.capacity=pages/threads+(i<pages%threads);
            l.pages.reserve(l.capacity);l.pins.reserve(128);
        }
        return true;
    }
    bool enabled() const{return bool(locals_);}
    uint64_t capacity_pages() const{return capacity_;}
    void enter(){auto &l=local();if(l.depth++==0 && reset_)clear(l);}
    void exit(){auto &l=local();if(!l.depth)throw std::logic_error("Private prefetch unbalanced ReadSection");if(--l.depth==0)l.pins.clear();}
    bool contains_range(uint32_t base,uint16_t n) const {
        auto &l=local();for(uint16_t i=0;i<n;i++)if(!l.pages.count(base+i))return false;return true;
    }
    bool get_range(uint32_t base,uint16_t n,char *dst) {
        auto &l=local();
        for(uint16_t i=0;i<n;i++)if(!l.pages.count(base+i)){l.misses.fetch_add(1,std::memory_order_relaxed);return false;}
        for(uint16_t i=0;i<n;i++){auto p=l.pages.find(base+i);std::memcpy(dst+size_t(i)*defaults::SECTOR_LEN,p->second.page->bytes.data(),defaults::SECTOR_LEN);touch(l,p);}
        l.hits.fetch_add(1,std::memory_order_relaxed);return true;
    }
    const char *get_ptr(uint32_t key) {
        auto &l=local();if(!l.depth)return nullptr;auto p=l.pages.find(key);
        if(p==l.pages.end()){l.misses.fetch_add(1,std::memory_order_relaxed);return nullptr;}
        touch(l,p);l.hits.fetch_add(1,std::memory_order_relaxed);l.pins.push_back(p->second.page);return p->second.page->bytes.data();
    }
    void erase(uint32_t key){auto &l=local();auto p=l.pages.find(key);if(p!=l.pages.end()){l.lru.erase(p->second.lru);l.pages.erase(p);}}
    bool put(uint32_t key,const char *src,bool overwrite=false) {
        auto &l=local();auto p=l.pages.find(key);
        if(p!=l.pages.end()){
            if(overwrite && std::memcmp(p->second.page->bytes.data(),src,defaults::SECTOR_LEN))
                throw std::runtime_error("Private prefetch cannot rewrite immutable backing pages");
            touch(l,p);return true;
        }
        auto bytes=std::make_shared<Page>();std::memcpy(bytes->bytes.data(),src,defaults::SECTOR_LEN);
        if(l.pages.size()==l.capacity){auto victim=l.lru.front();l.pages.erase(victim);l.lru.pop_front();l.evictions.fetch_add(1,std::memory_order_relaxed);}
        l.lru.push_back(key);l.pages.emplace(key,Entry{std::move(bytes),std::prev(l.lru.end())});
        l.inserts.fetch_add(1,std::memory_order_relaxed);return true;
    }
    uint64_t hits() const{uint64_t n=0;for(size_t i=0;i<threads_;i++)n+=locals_[i].hits.load(std::memory_order_relaxed);return n;}
    uint64_t misses() const{uint64_t n=0;for(size_t i=0;i<threads_;i++)n+=locals_[i].misses.load(std::memory_order_relaxed);return n;}
    uint64_t inserts() const{uint64_t n=0;for(size_t i=0;i<threads_;i++)n+=locals_[i].inserts.load(std::memory_order_relaxed);return n;}
    uint64_t evictions() const{uint64_t n=0;for(size_t i=0;i<threads_;i++)n+=locals_[i].evictions.load(std::memory_order_relaxed);return n;}
};
}
