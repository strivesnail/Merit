#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace residual {
using Clock = std::chrono::steady_clock;
inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
inline std::atomic<int64_t> query_end_ns{0};
inline std::atomic<unsigned> writing{0};
inline std::atomic<uint64_t> writer_batches{0};
inline std::atomic<uint64_t> writer_pages{0};
inline thread_local bool overlap=false;
enum Part { Beam, Nlookup, Dlookup, ReadPath, Expand, PathRecord, NinsertD, NinsertBase,
            CpuRemainder, Commit, SeedMetadata, OldRead, Enqueue, FullQueue, Submit, Reap, Batches, Pages, Reaps, SlowWait, SlowCount, InflightJoin, OverlapBatches, OverlapWait, OverlapPages, OverlapSubmit, Count };
inline constexpr const char* names[] = {"beam","n_lookup","d_lookup","read_path","expand","path_record",
    "n_insert_d","n_insert_base","cpu_remainder","commit","seed_metadata","old_read","enqueue","full_queue","submit_ns","reap_ns","submits","read_pages","reaps","slow_reap_ns","slow_reaps","join_ns","overlap_batches","overlap_wait_ns","overlap_pages","overlap_submit_ns"};
struct alignas(64) Row { uint64_t id=0; std::array<uint64_t,Count> ns{}; };
inline thread_local Row* active=nullptr;
inline void select(std::vector<Row>& rows, uint64_t i) {
    uint64_t bucket=i/32;
    uint64_t x=(bucket+1)*UINT64_C(0x9e3779b97f4a7c15);
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    active=(i%32==(x>>59)) ? &rows[bucket] : nullptr;
    if(active) active->id=i;
}
struct Scope {
    Row* row;
    Part part;
    Clock::time_point start;
    explicit Scope(Part p): row(active), part(p) { if(row) start=Clock::now(); }
    void stop() { if(row) {
        row->ns[part]+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
        row=nullptr;
    } }
    ~Scope() { stop(); }
};
// Match the original timer and record the part discarded by duration_cast<microseconds>.
struct CpuTimer {
    using Source=std::chrono::high_resolution_clock;
    Source::time_point start=Source::now();
    void reset(){start=Source::now();}
    long long elapsed() const {
        auto duration=Source::now()-start;
        auto us=std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        if(active) active->ns[CpuRemainder]+=std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()-1000*us;
        return us;
    }
};
inline void dump(const std::vector<Row>& rows,const std::string& path) {
    std::ofstream out(path);
    if(!out) throw std::runtime_error("cannot write residual sample");
    out<<"query_id"; for(auto name:names)out<<','<<name;out<<'\n';
    for(auto& r:rows) {out<<r.id; for(auto val:r.ns)out<<','<<val;out<<'\n';}
}
}
