"""Prepare a separate instrumented backend; never overwrite the timing library.

Each thread samples one invocation in 128 per node stage and 8 per query stage.
Durations include waiting;
they are not CPU time or disjoint percentages of benchmark wall time. Stages are
non-nested except seed_note and n_evict, which are children of admit_N_and_seed.
Native async I/O wait is outside these stages.
"""
from pathlib import Path
import hashlib
import json
import re

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/mnt/nvme2/nc_hot20_20261027')
OUT = ROOT / 'stage_profile_source'
OUT.mkdir(exist_ok=True)
header = r'''
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
namespace nc_stage {
enum Stage { Begin, Visit, Resolve, Complete, Selected, Admit, NoteBase, End, Contains, SeedNote, NEvict, GlobalWait, Count };
const char *names[] = {"begin", "visit", "resolve", "complete", "selected_M", "admit_N_and_seed", "note_base_stash", "query_end", "contains", "seed_note", "n_evict", "global_dyn_write_wait"};
unsigned stride(size_t s) { return s==Begin || s==End ? 8 : 128; }
struct Metric { std::atomic<unsigned long long> samples{0}, ns{0}; };
std::array<Metric, Count> metrics;
struct Scope {
    Stage stage; bool sampled;
    std::chrono::steady_clock::time_point start;
    explicit Scope(Stage s) : stage(s) {
        static thread_local std::array<unsigned long long, Count> sequence{};
        sampled = (++sequence[s] % stride(s)) == 0;
        if (sampled) start = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (!sampled) return;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        metrics[stage].ns.fetch_add(ns, std::memory_order_relaxed);
        metrics[stage].samples.fetch_add(1, std::memory_order_relaxed);
    }
};
void report() {
    for (size_t s=0; s<Count; ++s) {
        auto n=metrics[s].samples.load(), ns=metrics[s].ns.load();
        std::fprintf(stderr, "NC_STAGE stage=%s samples=%llu mean_us=%.6f estimated_thread_seconds=%.6f stride=%u\n", names[s], n, n ? double(ns)/n/1000 : 0, double(ns)*stride(s)/1e9, stride(s));
    }
}
}
'''
source = (REPO/'neighborcache_backend/src/backend.cpp').read_text()
source = source.replace('#include "../../diskann/src/pq_flash_index.cpp"',
                        '#include "stage_profile.h"\n#include "pq_profile.cpp"')
reference = (REPO/'diskann/src/pq_flash_index.cpp').read_text()
for signature, stage in [
    ('template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::neighborcache_seed_prefetch_note(uint32_t node_id)\n{', 'SeedNote'),
    ('void PQFlashIndex<T, LabelT>::neighborcache_dyn_on_ncache_evict(uint32_t node_id, QueryStats *stats)\n{', 'NEvict'),
]:
    assert reference.count(signature) == 1
    reference = reference.replace(signature, signature+'\n    nc_stage::Scope stage_profile(nc_stage::'+stage+');')
reference, count = re.subn(
    r'NeighborCacheTimedSharedMutexGuard (\w+)\(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true\);',
    lambda m: 'auto '+m[1]+' = [&] { nc_stage::Scope wait_profile(nc_stage::GlobalWait); return NeighborCacheTimedSharedMutexGuard(_neighborcache_dyn_mu, NeighborCacheLockKind::DynamicUnique, true); }();', reference)
assert count > 10, count
(OUT/'pq_profile.cpp').write_text(reference)
replacements = {
    'm.samples.store(0); m.wait_ns.store(0); m.hold_ns.store(0);': 'm.samples.store(0); m.wait_ns.store(0); m.hold_ns.store(0);\n            for (auto &s : nc_stage::metrics) { s.samples.store(0); s.ns.store(0); }',
    '~NeighborCacheAdapter() {': '~NeighborCacheAdapter() {\n        nc_stage::report();',
    'Query *begin() {': 'Query *begin() {\n        nc_stage::Scope profile(nc_stage::Begin);',
    'void end(Query *q) {': 'void end(Query *q) {\n        nc_stage::Scope profile(nc_stage::End);',
    'void visit(Query *q, uint32_t parent, uint32_t child) {': 'void visit(Query *q, uint32_t parent, uint32_t child) {\n        nc_stage::Scope profile(nc_stage::Visit);',
    'void selected(Query *q, uint32_t id) {': 'void selected(Query *q, uint32_t id) {\n        nc_stage::Scope profile(nc_stage::Selected);',
    'int resolve(Query *q, uint32_t id, char *raw, NCRead *out) {': 'int resolve(Query *q, uint32_t id, char *raw, NCRead *out) {\n        nc_stage::Scope profile(nc_stage::Resolve);',
    'int complete(Query *q, NCRead *r, int64_t got, char *raw) {': 'int complete(Query *q, NCRead *r, int64_t got, char *raw) {\n        nc_stage::Scope profile(nc_stage::Complete);',
    'index.neighborcache_dyn_admit_node(id, canonical.data(), &q->stats, &q->scratch, hop);': '{ nc_stage::Scope profile(nc_stage::Admit); index.neighborcache_dyn_admit_node(id, canonical.data(), &q->stats, &q->scratch, hop); }',
    'index.neighborcache_dyn_note_base_load(id, parent, canonical.data(), &q->scratch);': '{ nc_stage::Scope profile(nc_stage::NoteBase); index.neighborcache_dyn_note_base_load(id, parent, canonical.data(), &q->scratch); }',
    'bool contains(uint32_t id) {': 'bool contains(uint32_t id) { nc_stage::Scope profile(nc_stage::Contains);',
}
for old, new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
(OUT/'backend.cpp').write_text(source)
(OUT/'stage_profile.h').write_text(header)
cmake = (REPO/'neighborcache_backend/CMakeLists.txt').read_text().split('include(CTest)')[0]
cmake = cmake.replace('set(REF ${CMAKE_CURRENT_SOURCE_DIR}/../diskann)', f'set(REF {REPO}/diskann)')
cmake = cmake.replace('src/backend.cpp', 'backend.cpp').replace('${REF}/include include', f'${{REF}}/include {REPO}/neighborcache_backend/include')
cmake = cmake.replace('${CMAKE_CURRENT_SOURCE_DIR}/exports.map', str(REPO/'neighborcache_backend/exports.map'))
(OUT/'CMakeLists.txt').write_text(cmake)
inputs = [REPO/'neighborcache_backend/src/backend.cpp', REPO/'diskann/src/pq_flash_index.cpp', OUT/'backend.cpp', OUT/'stage_profile.h', OUT/'pq_profile.cpp']
(OUT/'sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, indent=2)+'\n')
print(OUT)
