from pathlib import Path
import concurrent.futures
import difflib
import hashlib
import json
import os
import shlex
import shutil
import subprocess

REPO=Path(__file__).resolve().parents[3]
S=Path(__file__).resolve().parent
R=Path('/mnt/nvme2/nc_fullvisited_seed_single_20261031')
B=Path('/mnt/nvme2/nc_release_20261031/native')
R.mkdir(exist_ok=True)
shutil.copyfile(S/'full_visited_cache.h',R/'full_visited_cache.h')
s=(REPO/'diskann/src/pq_flash_index.cpp').read_text();original=s
s='#include "full_visited_cache.h"\n'+s
needle='    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector'
assert s.count(needle)==1
s=s.replace(needle,'    if (FullVisitedCache::instance().active()) return FullVisitedCache::instance().page_for(node_id);\n'+needle)
needle='    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);'
assert s.count(needle)==1
s=s.replace(needle,'    if (FullVisitedCache::instance().active()) return sector_buf;\n'+needle)
old='(_neighborcache_seed_page_lru || _neighborcache_seed_prefetch) && query_scratch != nullptr'
assert s.count(old)>=4
s=s.replace(old,'(_neighborcache_seed_page_lru || _neighborcache_seed_prefetch || FullVisitedCache::instance().active()) && query_scratch != nullptr')
needle='''                        NeighborCacheDemandProbeScope probe_scope(demand_probe);
                        sp_hit = neighborcache_seed_page_lookup(nbr.id, sp_coords, sp_nbrs);'''
assert s.count(needle)==1
s=s.replace(needle,'''                        if (FullVisitedCache::instance().active())
                        {
                            if (sizeof(T) != 1 || _max_node_len != 388 || _disk_bytes_per_point != 128)
                                throw std::runtime_error("Fullvisited requires uint8 SIFT 128D R64");
                            alignas(64) char record[388];
                            sp_hit = FullVisitedCache::instance().lookup(nbr.id,record);
                            if (sp_hit)
                            {
                                uint32_t degree;std::memcpy(&degree,record+128,4);
                                if(degree>64)throw std::runtime_error("Invalid fullvisited cached degree");
                                sp_coords.resize(128);std::memcpy(sp_coords.data(),record,128);
                                const auto *hood=reinterpret_cast<const uint32_t*>(record+132);
                                sp_nbrs.assign(hood,hood+degree);
                            }
                        }
                        else
                        {
                            NeighborCacheDemandProbeScope probe_scope(demand_probe);
                            sp_hit = neighborcache_seed_page_lookup(nbr.id, sp_coords, sp_nbrs);
                        }''')
needle='''                if (!mem_hit)
                {
                    g_dyn_probe.ncache_miss++;'''
assert s.count(needle)==1
s=s.replace(needle,'''                if (!mem_hit && FullVisitedCache::instance().active())
                {
                    // Every profiled target has its own immutable D page. N/P
                    // hits skip this lookup and do not enter its denominator.
                    FullVisitedCache::instance().page_for(nbr.id);
                    frontier.push_back(nbr.id);
                    if(stats){++stats->n_neighborcache_dcache_demand_probes;++stats->n_neighborcache_dcache_demand_hits;}
                    return;
                }
'''+needle)
for needle,node,page in [
 ('            char *node_disk_buf = offset_to_node(frontier_nhood.second, frontier_nhood.first);','frontier_nhood.first','frontier_nhood.second'),
 ('                char *node_disk_buf = offset_to_node(disk_it->second, expand_id);','expand_id','disk_it->second')]:
 assert s.count(needle)==1,needle
 block=f'''\n                if (FullVisitedCache::instance().active())
                {{
                    FullVisitedCache::instance().put({node},{page});
                    if(stats){{++stats->n_neighborcache_disk_reads;++stats->n_neighborcache_dyn_disk_reads;}}
                }}'''
 s=s.replace(needle,needle+block)
(R/'pq_flash_index.cpp').write_text(s)
(R/'pq.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='native/pq_flash_index.cpp',tofile='fullvisited/pq_flash_index.cpp')))
s=(REPO/'diskann/apps/search_disk_index.cpp').read_text();original=s
s='#include "full_visited_cache.h"\n'+s
needle='    log_rss("after_dyn_init");'
assert s.count(needle)==1
s=s.replace(needle,needle+'''
    if (const char *full=std::getenv("NC_FULLVISITED_DATA"))
    {
        if(sizeof(T)!=1 || num_threads!=1 || dyn3_cache || num_nodes_to_cache!=0)
            throw std::runtime_error("Fullvisited requires single-worker uint8 and no dynamic maintenance");
        const char *ids=std::getenv("NC_FULLVISITED_IDS");
        if(!ids)throw std::runtime_error("Missing fullvisited IDs");
        FullVisitedCache::instance().init(ids,2560,num_threads);
        if(get_file_size(full)!=FullVisitedCache::instance().page_count()*4096ULL)
            throw std::runtime_error("Fullvisited D size mismatch");
        reader->close();reader->open(full);
        log_rss("after_fullvisited_init");
    }
''')
needle="<< s.n_unique_sectors << ',' << s.n_unique_neighborcache_sectors << ',' << s.n_neighborcache_dc_hits << ','"
assert s.count(needle)==1
s=s.replace(needle,"<< (FullVisitedCache::instance().active() ? 0 : s.n_unique_sectors) << ',' << (FullVisitedCache::instance().active() ? s.n_unique_sectors : s.n_unique_neighborcache_sectors) << ',' << s.n_neighborcache_dc_hits << ','")
(R/'search_disk_index.cpp').write_text(s)
(R/'app.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='native/search_disk_index.cpp',tofile='fullvisited/search_disk_index.cpp')))
env=os.environ.copy();env.update(CPATH='/tmp/pipeann-inc:/home/jianz/miniconda3/include:/tmp/aisaq-nmd-deps/root/usr/include',LIBRARY_PATH='/home/jianz/miniconda3/lib:/tmp/aisaq-nmd-deps/root/usr/lib/x86_64-linux-gnu')
def compile_one(target,source,obj):
 flags={}
 for line in (B/target/'flags.make').read_text().splitlines():
  if ' = ' in line:k,v=line.split(' = ',1);flags[k]=shlex.split(v)
 cmd=['g++','-I'+str(R)]+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-c',str(R/source),'-o',str(R/obj)]
 (R/(source+'.compile.json')).write_text(json.dumps(cmd,indent=2))
 with (R/(source+'.build.log')).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('COMPILED',source,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for f in [pool.submit(compile_one,'src/CMakeFiles/diskann.dir','pq_flash_index.cpp','pq_flash_index.cpp.o'),pool.submit(compile_one,'apps/CMakeFiles/search_disk_index.dir','search_disk_index.cpp','app.o')]:f.result()
shutil.copyfile(B/'src/libdiskann.a',R/'libdiskann.a')
subprocess.run(['ar','r',str(R/'libdiskann.a'),str(R/'pq_flash_index.cpp.o')],check=True)
cmd=shlex.split((B/'apps/CMakeFiles/search_disk_index.dir/link.txt').read_text())
cmd=[str(R/'app.o') if x.endswith('search_disk_index.cpp.o') else str(R/'libdiskann.a') if x=='../src/libdiskann.a' else x for x in cmd]
cmd[cmd.index('-o')+1]=str(R/'search_disk_index')
(R/'link.json').write_text(json.dumps(cmd,indent=2));subprocess.run(cmd,env=env,cwd=B/'apps',check=True)
(R/'source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [REPO/'diskann/src/pq_flash_index.cpp',REPO/'diskann/apps/search_disk_index.cpp',REPO/'diskann/include/neighborcache_thread_page_cache.h',R/'full_visited_cache.h',R/'search_disk_index',R/'pq_flash_index.cpp']},indent=2))
print('BUILT',flush=True)
