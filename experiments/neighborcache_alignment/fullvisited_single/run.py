from pathlib import Path
import csv
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
import time
import numpy as np

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'experiments/neighborcache_alignment/n_capacity_match'))
from cache_control import clear_files,prefixed_files
R=Path('/mnt/nvme2/nc_fullvisited_seed_single_20261031')
BASE=Path('/mnt/nvme2/nc_n_capacity_match_20261031/nonly_0_0/command.json')
REFERENCE=Path('/mnt/nvme2/nc_private_prefetch_fix_20261031/cost_final/bfs_1')

def write(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def validate_pages():
 ids=np.fromfile(R/'visited_seed_ids.u32',dtype='<u4')
 visited=np.zeros(100000000,dtype=bool);visited[ids]=True
 dpath=R/'visited_seeds.pages';assert dpath.stat().st_size==len(ids)*4096
 assert dpath.stat().st_blocks*512>=dpath.stat().st_size
 base=os.open('/mnt/nvme_single/sift100m/sift100m_index_disk.index',os.O_RDONLY)
 data=os.open(dpath,os.O_RDONLY);records=0
 positions=sorted(set([0,len(ids)-1]+random.Random(20261031).sample(range(len(ids)),10000)))
 def raw(n):return os.pread(base,388,(1+n//10)*4096+(n%10)*388)
 for pos in positions:
  seed=int(ids[pos]);page=os.pread(data,4096,pos*4096);n=struct.unpack_from('<I',page,4052)[0]
  assert 1<=n<=10
  members=struct.unpack_from(f'<{n}I',page,4056);assert members[0]==seed and len(set(members))==n
  seedrecord=raw(seed);degree=struct.unpack_from('<I',seedrecord,128)[0];assert degree<=64
  expected=[seed]
  for node in struct.unpack_from(f'<{degree}I',seedrecord,132):
   if visited[node] and node not in expected:expected.append(node)
   if len(expected)==10:break
  assert list(members)==expected
  for slot,node in enumerate(members):assert page[slot*388:(slot+1)*388]==raw(node);records+=1
 os.close(base);os.close(data)
 write(R/'prefill_validation.json',dict(passed=True,visited_seeds=len(ids),sampled_pages=len(positions),records_verified=records,d_bytes=dpath.stat().st_size,allocated_bytes=dpath.stat().st_blocks*512,order='seed then up to 9 distinct visited graph-order neighbors',unchanged_full_vectors_and_adjacencies=True))
 print('PREFILL VALIDATED',len(positions),records,flush=True)

def run(name,queries=20000,passes=2):
 out=R/name
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 out.mkdir(exist_ok=False)
 c=json.loads(BASE.read_text());cmd=c['command'];env=c['environment']
 cmd[3]=str(R/'search_disk_index')
 for key,value in {'--result_path':str(out/'result'),'--dump_query_stats':str(out/'query_stats.csv'),'--neighborcache_memory_gb':'2.0'}.items():cmd[cmd.index(key)+1]=value
 if passes==1:del cmd[cmd.index('--search_list')+1]
 if queries!=20000:
  at=cmd.index('--query_file')+1;original=Path(cmd[at]).read_bytes();query=out/'queries.u8bin'
  query.write_bytes(struct.pack('<II',queries,128)+original[8:8+queries*128]);cmd[at]=str(query)
 env.update(NEIGHBORCACHE_NCACHE_ONLY='1',NEIGHBORCACHE_NCACHE_CAP_NODES='2048',NEIGHBORCACHE_NCACHE_COLD='1',NC_FULLVISITED_DATA=str(R/'visited_seeds.pages'),NC_FULLVISITED_IDS=str(R/'visited_seed_ids.u32'),NEIGHBORCACHE_PREFETCH_TRIM_ZERO_TAIL='1')
 c.update(command=cmd,environment=env,binary_sha256=hashlib.sha256((R/'search_disk_index').read_bytes()).hexdigest(),input_sha256=hashlib.sha256(Path(cmd[cmd.index('--query_file')+1]).read_bytes()).hexdigest(),query_threads=1,n_nodes=2048,prefetch_pages=2560,prefetch_mib=10,m_entries=0,seed_scope='all full-vector visited nodes from the same 20K workload; offline oracle',D_updates='none',warmup_queries=queries if passes==2 else 0,measured_queries=queries)
 write(out/'command.json',c)
 files=prefixed_files(cmd[cmd.index('--index_path_prefix')+1])+[cmd[cmd.index('--query_file')+1],R/'visited_seeds.pages',R/'visited_seed_ids.u32']
 write(out/'cache_clear.json',clear_files(files))
 actual={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','NC_','OMP_'))};actual.update(env)
 print('START',name,flush=True);started=time.monotonic()
 with (out/'run.out').open('w') as log:subprocess.run(cmd,env=actual,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
 log=(out/'run.out').read_text();import re
 rows=re.findall(r'^\s*128\s+4\s+([\d.]+).*$',log,re.M);assert len(rows)==passes
 assert 'private_LRUs=1 M=0 writes=0' in log
 assert 'dynamic 3-cache enabled' not in log
 for suffix in ('idx_uint32.bin','dists_float.bin'):
  got=(out/('result_128_'+suffix)).read_bytes();expected=(REFERENCE/('result_128_'+suffix)).read_bytes()
  assert got[8:]==expected[8:8+queries*10*4],suffix
 with (out/'query_stats.csv').open() as f:q=list(csv.DictReader(f))
 assert len(q)==queries
 fields=('n_beam_lookups','n_mem_hits','n_neighborcache_prefetch_hits','n_disk_reads','n_base_pages','n_disk_cache_pages','n_neighborcache_dyn_disk_reads','n_neighborcache_dcache_demand_probes','n_neighborcache_dcache_demand_hits')
 counts={key:sum(int(row[key]) for row in q) for key in fields}
 a,n,p,d=(counts[k] for k in ('n_beam_lookups','n_mem_hits','n_neighborcache_prefetch_hits','n_disk_reads'))
 assert a==n+p+d,counts
 assert counts['n_base_pages']==0
 assert d==counts['n_disk_cache_pages']==counts['n_neighborcache_dyn_disk_reads']==counts['n_neighborcache_dcache_demand_probes']==counts['n_neighborcache_dcache_demand_hits']
 if queries==20000:assert a==2907400
 rss={stage:int(kib) for stage,kib in re.findall(r'MEM_STAGE (\w+) rss=\s*(\d+) kB',log)}
 result=dict(name=name,qps=float(rows[-1]),first_pass_qps=float(rows[0]),queries=queries,passes=passes,counts=counts,n_hit_pct=100*n/a,p_hit_pct=100*p/a,p_after_n_miss_pct=100*p/(a-n),d_lookup_hit_pct=100.0,d_coverage_including_prefetch_after_n_miss_pct=100.0,d_reads_per_query=d/queries,base_reads_per_query=0.0,extra_rss_mib=(rss['after_search']-rss['after_load'])/1024,rss_kib=rss,ids_and_distances_equal_reference=True,elapsed_seconds=time.monotonic()-started)
 write(out/'result.json',result);print('DONE',json.dumps(result),flush=True)
 return result

if __name__=='__main__':
 os.sched_setaffinity(0,{23});validate_pages()
 run('smoke',1000,1)
 results=[run('run1'),run('run2')]
 write(R/'results.json',results)
 print('COMPLETE',flush=True)
