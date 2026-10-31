from pathlib import Path
import sys,os,json,subprocess,csv,numpy as np,hashlib
sys.path.insert(0,'/home/jianz/workload/code/Merit/experiments/neighborcache_alignment/n_capacity_match')
from cache_control import clear_files,prefixed_files
R=Path('/mnt/nvme2/nc_fullvisited_seed_single_20261031');P=R/'profile';P.mkdir(exist_ok=False)
c=json.loads(Path('/mnt/nvme2/nc_n_capacity_match_20261031/nonly_0_0/command.json').read_text());cmd=c['command'];i=cmd.index('--search_list')+1;assert cmd[i:i+2]==['128','128'];del cmd[i]
cmd[cmd.index('--result_path')+1]=str(P/'result');cmd[cmd.index('--dump_query_stats')+1]=str(P/'query_stats.csv');cmd+=['--dump_base_nodes_path',str(P/'full_vector_frontier.csv')]
c.update(command=cmd,warmup_queries=0,measured_queries=20000,profiling_queries=20000,purpose='Untimed full-vector visited-ID collection: no N/M/P/D, actual base-frontier records; PQ candidates excluded')
(P/'command.json').write_text(json.dumps(c,indent=2));(P/'cache_clear.json').write_text(json.dumps(clear_files(prefixed_files(cmd[cmd.index('--index_path_prefix')+1])+[cmd[cmd.index('--query_file')+1]]),indent=2))
env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','NC_','OMP_'))};env.update(c['environment'])
with (P/'run.out').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
counts=np.zeros(100000000,dtype=np.uint32);events=0
with (P/'full_vector_frontier.csv').open() as f:
 for row in csv.DictReader(f):counts[int(row['node_id'])]+=1;events+=1
with (P/'query_stats.csv').open() as f:q=list(csv.DictReader(f))
assert events==sum(int(row['n_beam_lookups']) for row in q)==2907400
assert all(int(row['n_mem_hits'])==int(row['n_neighborcache_prefetch_hits'])==int(row['n_neighborcache_dyn_disk_reads'])==0 for row in q)
ref=Path('/mnt/nvme2/nc_private_prefetch_fix_20261031/cost_final/bfs_1')
for suffix in ('idx_uint32.bin','dists_float.bin'):assert (P/('result_128_'+suffix)).read_bytes()==(ref/('result_128_'+suffix)).read_bytes()
seeds=np.flatnonzero(counts).astype('<u4');seeds.tofile(R/'visited_seed_ids.u32')
meta=dict(queries=20000,full_vector_node_accesses=events,unique_visited_nodes=len(seeds),d_pages=len(seeds),d_bytes=len(seeds)*4096,d_mib=len(seeds)/256,input_sha256=hashlib.sha256(Path(cmd[cmd.index('--query_file')+1]).read_bytes()).hexdigest(),ids_and_distances_equal_reference=True,scope='Same-workload offline oracle; PQ-only candidates excluded; sorted unique actual full-record-expanded target IDs')
(R/'visited_summary.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
