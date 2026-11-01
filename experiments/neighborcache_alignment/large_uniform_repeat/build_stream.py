from pathlib import Path
import hashlib,json,os,shlex,subprocess
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];R=Path(os.environ.get('NC_LARGE_ROOT','/mnt/nvme3/nc_sift100m_uniform8mx10_20261101'));B=R/'build';B.mkdir(parents=True,exist_ok=True)
base=REPO/'diskann/apps/search_disk_index.cpp';s=base.read_text()
needle='            _pFlashIndex->cache_bfs_levels(max_nodes, node_list);'
assert s.count(needle)==1
s=s.replace(needle,'''            const char *cold = std::getenv("NEIGHBORCACHE_NCACHE_COLD");
            if ((ncache_only || dyn3_cache) && cold && std::strcmp(cold, "0") != 0)
            {
                // Cold loader ignores IDs and obtains exact capacity from the
                // explicit node-cap environment variable. No BFS I/O or prefill.
                node_list.assign(1, 0);
                diskann::cout << "COLD_CACHE_SKIP_BFS: empty exact-capacity N initialization" << std::endl;
            }
            else
                _pFlashIndex->cache_bfs_levels(max_nodes, node_list);''')
needle='    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);'
assert s.count(needle)==1
s=s.replace(needle,'''    const char *batch_env=std::getenv("NC_STREAM_BATCH_QUERIES");
    const uint64_t stream_batch=batch_env?std::strtoull(batch_env,nullptr,10):0;
    if(stream_batch) {
        std::ifstream f(query_file,std::ios::binary);uint32_t n=0,d=0;
        f.read(reinterpret_cast<char*>(&n),4);f.read(reinterpret_cast<char*>(&d),4);
        if(!f || !n || !d)throw std::runtime_error("Invalid streaming query header");
        query_num=n;query_dim=d;query_aligned_dim=(d+7)/8*8;
    } else
        diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);''')
needle='            _pFlashIndex->enable_neighborcache_dynamic_3cache(true, result_output_prefix);';assert s.count(needle)==1
s=s.replace(needle,'''            const char *d_prefix=std::getenv("NC_STREAM_D_PREFIX");
            _pFlashIndex->enable_neighborcache_dynamic_3cache(true,d_prefix?std::string(d_prefix):result_output_prefix);''')
# Define the helper after the existing CPU-pinning function and before search_disk_index templates.
needle='template <typename T, typename LabelT = uint32_t>'
pos=s.index(needle);s=s[:pos]+'#include "stream_query_search.h"\n'+s[pos:]
needle='    uint32_t optimized_beamwidth = 2;';assert s.count(needle)==1
s=s.replace(needle,needle+'''
    if(stream_batch) {
        if(filtered_search || Lvec.size()!=1 || beamwidth<=0 || WARMUP || calc_recall_flag)
            throw std::invalid_argument("Streaming driver requires one L, fixed beam, no filters/GT/warmup");
        return run_stream_query_search(*_pFlashIndex,query_file,result_output_prefix,
            query_num,query_dim,query_aligned_dim,num_threads,recall_at,Lvec.front(),
            static_cast<uint32_t>(beamwidth),use_reorder_data,stream_batch);
    }
''')
(B/'search_disk_index.cpp').write_text(s)
native=REPO/'diskann/build/apps';flags={}
for line in (native/'CMakeFiles/search_disk_index.dir/flags.make').read_text().splitlines():
 if ' = ' in line:
  key,value=line.split(' = ',1);flags[key]=shlex.split(value)
cmd=['g++','-I'+str(HERE)]+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-c',str(B/'search_disk_index.cpp'),'-o',str(B/'app.o')]
env=os.environ.copy();env.update(CPATH='/home/jianz/miniconda3/include',LIBRARY_PATH='/home/jianz/miniconda3/lib:/tmp/merit-link-deps/root/usr/lib/x86_64-linux-gnu')
with (B/'compile.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
cmd=shlex.split((native/'CMakeFiles/search_disk_index.dir/link.txt').read_text())
cmd=[str(B/'app.o') if x.endswith('search_disk_index.cpp.o') else x for x in cmd if not x.startswith('-Wl,--dependency-file=')];cmd[cmd.index('-o')+1]=str(B/'search_disk_index')
with (B/'link.log').open('w') as f:subprocess.run(cmd,cwd=native,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
(R/'build_provenance.json').write_text(json.dumps(dict(source=str(base),engine_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),binary_sha256=hashlib.sha256((B/'search_disk_index').read_bytes()).hexdigest(),batch_cache_reset=False,worker_mapping='global query index modulo threads; stable OS workers checked'),indent=2)+'\n')
print('Streaming driver built',flush=True)
