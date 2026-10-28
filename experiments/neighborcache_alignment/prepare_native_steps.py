"""Instrument an isolated copy; never change the production executable or sources."""
from pathlib import Path
import hashlib,json,re,shutil
REPO=Path(__file__).resolve().parents[2];R=Path('/mnt/nvme2/sift100m_access_steps_20261027');S=R/'source';S.mkdir(parents=True,exist_ok=True)
shutil.copytree(REPO/'diskann/include',S/'include',dirs_exist_ok=True)
header=r'''#pragma once
#include <array>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <vector>
namespace nc_step {
enum Stage { Query, Setup, Loop, Select, NLookup, DRoute, Prefetch, Plan, IOFrame, Submit, Wait, Complete, Put, CopyPage, Evict, Decode, NAdmit, Exact, PQ, Discover, Finish, Count };
inline const char *names[]={"query", "query_setup", "loop_other", "select_candidate", "n_lookup", "d_directory", "prefetch_lookup", "d_request_plan", "io_prepare_dispatch", "io_submit", "io_wait", "d_completion", "prefetch_put", "page_copy", "prefetch_eviction", "decode_record", "n_admit", "exact_distance", "pq_distance", "discover_candidates", "query_finish"};
struct Metric { unsigned long long calls=0, inclusive=0, exclusive=0; };
struct Scope;
struct State { std::array<Metric,Count> m{}; Scope *top=nullptr; };
inline std::mutex registry_mu;
inline std::vector<State*> registry;
inline State &state() { static thread_local State *s=[] { auto *p=new State; std::lock_guard<std::mutex> lock(registry_mu); registry.push_back(p); return p; }(); return *s; }
inline unsigned long long now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Scope {
 State *s=nullptr; Scope *parent=nullptr; Stage stage; unsigned long long begin=0,children=0;
 explicit Scope(Stage id):stage(id) { auto &st=state(); if (id!=Query && !st.top) return; s=&st;parent=s->top;s->top=this;begin=now(); }
 void stop() { if (!s) return; auto elapsed=now()-begin; auto &m=s->m[stage];++m.calls;m.inclusive+=elapsed;m.exclusive+=elapsed-children;if(parent)parent->children+=elapsed;s->top=parent;s=nullptr; }
 ~Scope(){stop();}
};
inline void report(){
 std::array<Metric,Count> sum{}; std::lock_guard<std::mutex> lock(registry_mu);
 for(auto *s:registry)for(size_t i=0;i<Count;++i){sum[i].calls+=s->m[i].calls;sum[i].inclusive+=s->m[i].inclusive;sum[i].exclusive+=s->m[i].exclusive;}
 for(size_t i=0;i<Count;++i) std::fprintf(stderr,"NC_STEP stage=%s calls=%llu inclusive_ns=%llu exclusive_ns=%llu\n",names[i],sum[i].calls,sum[i].inclusive,sum[i].exclusive);
}
}
'''
(S/'include/nc_step_profile.h').write_text(header)
def function_scope(s,needle,stage):
 assert s.count(needle)==1,(needle,s.count(needle));pos=s.index('{',s.index(needle));return s[:pos+1]+'\n    nc_step::Scope step_scope(nc_step::'+stage+');'+s[pos+1:]
def replace(s,a,b,n=1):
 assert s.count(a)==n,(a,s.count(a),n);return s.replace(a,b)
s=(REPO/'diskann/src/pq_flash_index.cpp').read_text();s='#include "nc_step_profile.h"\n'+s
query_sig='''                                                 const bool use_filter, const LabelT &filter_label,
                                                 const uint32_t io_limit, const bool use_reorder_data,
                                                 QueryStats *stats)
{'''
s=replace(s,query_sig,query_sig+'\n    nc_step::Scope root_step(nc_step::Query);\n    nc_step::Scope setup_step(nc_step::Setup);')
s=replace(s,'        while (retset.has_unexpanded_node() && num_ios < io_limit)\n        {','        setup_step.stop();\n        while (retset.has_unexpanded_node() && num_ios < io_limit)\n        {\n            nc_step::Scope iteration_step(nc_step::Loop);')
s=replace(s,'Neighbor nbr = retset.closest_unexpanded();','Neighbor nbr = [&] { nc_step::Scope t(nc_step::Select); return retset.closest_unexpanded(); }();')
s=replace(s,'float *dists_out) {','float *dists_out) {\n        nc_step::Scope distance_step(nc_step::PQ);')
start=s.index(query_sig);end=s.index('// range search returns',start);part=s[start:end]
part,count=re.subn(r'(cur_expanded_dist\s*=\s*)(_dist_cmp->compare\(.*?\));',r'\1[&] { nc_step::Scope t(nc_step::Exact); return \2; }();',part,flags=re.S);assert count==4,count
part,count=re.subn(r'(for \(uint64_t m = 0; m < nnbrs; \+\+m\)\s*\{)',r'\1\n                    nc_step::Scope candidate_step(nc_step::Discover);',part);assert count==4,count
# Instrument batches of neighbor discovery instead of every neighbor to reduce timer overhead.
part=part.replace('for (uint64_t m = 0; m < nnbrs; ++m)\n                    {\n                    nc_step::Scope candidate_step(nc_step::Discover);','for (uint64_t m = 0; m < nnbrs; ++m)\n                    {\n                    nc_step::Scope candidate_step(nc_step::Discover);')
# Move the discovery timer outside each loop using balanced braces.
needle='nc_step::Scope candidate_step(nc_step::Discover);'
while needle in part:
 timer=part.index(needle);loop=part.rfind('for (uint64_t m = 0;',0,timer);brace=part.index('{',loop);depth=1;j=brace+1
 while depth:
  if part[j]=='{':depth+=1
  elif part[j]=='}':depth-=1
  j+=1
 block=part[loop:j].replace(needle,'')
 part=part[:loop]+'{ nc_step::Scope discover_step(nc_step::Discover);\n'+block+'\n}'+part[j:]
part=replace(part,'        hops++;\n    }','        hops++;\n    }\n    nc_step::Scope finish_step(nc_step::Finish);')
s=s[:start]+part+s[end:]
for needle,stage in [('inline float issue_neighborcache_and_base_disk_reads(', 'IOFrame'),('bool PQFlashIndex<T, LabelT>::neighborcache_resolve_disk_cache_loc(', 'DRoute'),('void PQFlashIndex<T, LabelT>::prepare_neighborcache_disk_cache_io(', 'Plan'),('void PQFlashIndex<T, LabelT>::complete_neighborcache_disk_cache_io(', 'Complete'),('void PQFlashIndex<T, LabelT>::finalize_neighborcache_pending_nodes(', 'Decode'),('const char *PQFlashIndex<T, LabelT>::neighborcache_shared_node_ptr(', 'Prefetch'),('uint32_t PQFlashIndex<T, LabelT>::neighborcache_dyn_admit_node(', 'NAdmit')]:s=function_scope(s,needle,stage)
(S/'pq_flash_index.cpp').write_text(s)
s='#include "nc_step_profile.h"\n'+(REPO/'diskann/src/neighborcache_memory_pool.cpp').read_text();s=function_scope(s,'bool NeighborCacheMemoryPool<T>::lookup(', 'NLookup');(S/'neighborcache_memory_pool.cpp').write_text(s)
s='#include "nc_step_profile.h"\n'+(REPO/'diskann/src/linux_aligned_file_reader.cpp').read_text();s=function_scope(s,'int64_t wait_for_events(', 'Wait')
s=replace(s,'int64_t ret = io_submit(ctx, (int64_t)n_ops, cbs.data());','int64_t ret = [&] { nc_step::Scope t(nc_step::Submit); return io_submit(ctx, (int64_t)n_ops, cbs.data()); }();',2);(S/'linux_aligned_file_reader.cpp').write_text(s)
p=S/'include/neighborcache_shared_page_cache.h';s='#include "nc_step_profile.h"\n'+p.read_text();s=function_scope(s,'bool put(uint32_t key, const char *src,', 'Put');s=function_scope(s,'bool retire_lru_locked(Shard &sh)', 'Evict')
s=replace(s,'std::memcpy(sh.data + static_cast<size_t>(existing) * defaults::SECTOR_LEN, src, defaults::SECTOR_LEN);','{ nc_step::Scope t(nc_step::CopyPage); std::memcpy(sh.data + static_cast<size_t>(existing) * defaults::SECTOR_LEN, src, defaults::SECTOR_LEN); }')
s=replace(s,'std::memcpy(sh.data + static_cast<size_t>(slot) * defaults::SECTOR_LEN, src, defaults::SECTOR_LEN);','{ nc_step::Scope t(nc_step::CopyPage); std::memcpy(sh.data + static_cast<size_t>(slot) * defaults::SECTOR_LEN, src, defaults::SECTOR_LEN); }');p.write_text(s)
s='#include "nc_step_profile.h"\n'+(REPO/'diskann/apps/search_disk_index.cpp').read_text();s=replace(s,'        auto e = std::chrono::high_resolution_clock::now();','        auto e = std::chrono::high_resolution_clock::now();\n        nc_step::report();');(S/'search_disk_index.cpp').write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(R/'source_hashes.json').write_text(json.dumps({'production_binary_sha256':sha(REPO/'diskann/build/apps/search_disk_index'),'instrumented_sources':{str(p):sha(p) for p in S.glob('*.cpp')}},indent=2)+'\n')
print(R)
