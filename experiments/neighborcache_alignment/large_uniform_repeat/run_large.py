"""Cold 80M-query NC and measured hit-matched N-only controls, serially."""
from pathlib import Path
import csv,hashlib,json,math,os,re,shutil,subprocess,sys,time
R=Path(os.environ.get('NC_LARGE_ROOT','/mnt/nvme3/nc_sift100m_uniform8mx10_20261101'))
BASE=Path('/mnt/nvme_single/sift100m/sift100m_index');DROOT=Path('/mnt/nvme_single/sift100m/runs/nc_uniform8mx10_20261101')
BIN=R/'build/search_disk_index';SMALL=Path('/mnt/nvme2/nc_unbounded_same_disk_20261101')
sys.path.insert(0,'/home/jianz/workload/code/Merit/experiments/neighborcache_alignment/n_capacity_match')
from cache_control import clear_files,prefixed_files
N=42474;P=3200;RECORD=388

def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def hashfile(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 return h.hexdigest()

def active_searches():
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:first=(p/'cmdline').read_bytes().split(b'\0')[0]
  except OSError:continue
  if Path(os.fsdecode(first)).name=='search_disk_index':found.append(int(p.name))
 return found

def run(mode,nodes,limit=None):
 manifest=json.loads((R/'workload_manifest.json').read_text());q=Path(manifest['query_file']);count=manifest['total_queries']
 if limit is not None:q=q.parent/f'prefix{limit}.u8bin';count=limit
 tag=f'{mode}_n{nodes}'+(f'_prefix{limit}' if limit is not None else '')
 out=R/'runs'/tag;data=DROOT/tag
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 assert not active_searches(),'Another search benchmark is active'
 out.mkdir(parents=True,exist_ok=False);data.mkdir(parents=True,exist_ok=False)
 assert Path(str(BASE)+'_disk.index').stat().st_dev==data.stat().st_dev
 cfg=json.loads((SMALL/'synthetic/runs/full_nc_n42474_1/command.json').read_text())
 env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','NC_','OMP_'))};env.update(cfg['environment'])
 env.update(NEIGHBORCACHE_NCACHE_CAP_NODES=str(nodes),NC_STREAM_BATCH_QUERIES='10000',NC_STREAM_D_PREFIX=str(data/'D'))
 if mode!='nc':
  for k in ['NEIGHBORCACHE_DYNAMIC_3CACHE','NEIGHBORCACHE_SHARED_CACHE_DYN','NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES']:env.pop(k,None)
  if nodes:env['NEIGHBORCACHE_NCACHE_ONLY']='1'
 else:
  env['NEIGHBORCACHE_DCACHE_UNBOUNDED']='1'
  env.pop('NEIGHBORCACHE_DCACHE_CAP',None);env.pop('NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES',None)
 cmd=cfg['command'].copy();cmd[cmd.index('--data_type')-1]=str(BIN)
 for flag,value in [('--query_file',q),('--result_path',out/'result'),('--dump_query_stats',out/'unused.csv')]:cmd[cmd.index(flag)+1]=str(value)
 if not nodes:cmd[cmd.index('--neighborcache_memory_gb')+1]='0'
 write(out/'command.json',dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith(('NEIGHBORCACHE_','NC_','OMP_','LD_'))},binary_sha256=hashfile(BIN),query_count=count,cold=True,query_CPUs='0-9',background_CPUs='20-23',D_same_device_as_base=True,N_nodes=nodes,P_pages=P if mode=='nc' else 0,D_quota='none',memory_bar='N + P payload only; M/directories/PQ excluded as requested'))
 write(out/'cache_clear.json',clear_files(prefixed_files(BASE)+[q]))
 print('START',mode,nodes,count,flush=True);started=time.time()
 with (out/'run.out').open('w') as log,(out/'telemetry.jsonl').open('w') as telemetry:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  prefix_checked=False
  try:
   while proc.poll() is None:
    sample=dict(time=time.time(),pid=proc.pid,D_free_bytes=shutil.disk_usage(DROOT).free,result_free_bytes=shutil.disk_usage(R).free)
    try:
     status=(Path('/proc')/str(proc.pid)/'status').read_text();sample['VmRSS_kB']=int(re.search(r'VmRSS:\s+(\d+)',status)[1])
     sample['mem_available_kB']=int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])
     sample['process_io']=(Path('/proc')/str(proc.pid)/'io').read_text()
     sample['affinity']={p.name:next(v.split(':',1)[1].strip() for v in (p/'status').read_text().splitlines() if v.startswith('Cpus_allowed_list:')) for p in (Path('/proc')/str(proc.pid)/'task').iterdir()}
    except (OSError,TypeError,StopIteration):pass
    telemetry.write(json.dumps(sample)+'\n');telemetry.flush()
    checkpoint=out/'result_stream.json'
    if mode=='nc' and not prefix_checked and checkpoint.exists():
     try:progress=json.loads(checkpoint.read_text())
     except (OSError,json.JSONDecodeError):progress={}
     if progress.get('queries_completed',0)>=30000:
      for suffix in ['idx_uint32.bin','dists_float.bin']:
       with Path(str(out/'result')+'_128_'+suffix).open('rb') as f:f.seek(8);actual=f.read(20000*40)
       expected=Path(str(R/'runs/reference_n0_prefix20000/result')+'_128_'+suffix).read_bytes()[8:]
       assert len(actual)==len(expected)==20000*40 and actual==expected,'NC prefix differs from uncached reference'
      prefix_checked=True;write(out/'uncached_prefix_equality.json',dict(queries=20000,IDs_and_distances_equal=True))
    if min(sample['D_free_bytes'],sample['result_free_bytes'])<2**30:raise RuntimeError('Physical disk nearly full; no D quota was imposed. Run incomplete.')
    if sample.get('mem_available_kB',9999999)<1024*1024 and sample.get('VmRSS_kB',0)>54*1024*1024:raise RuntimeError('Process memory exceeded available machine RAM; run incomplete.')
    time.sleep(10)
   if proc.returncode:raise RuntimeError(f'Search exited {proc.returncode}')
  except BaseException as err:
   if proc.poll() is None:proc.terminate();proc.wait(timeout=60)
   write(out/'failure.json',dict(reason=str(err),incomplete=True,time=time.time()));raise
 result=json.loads((out/'result_stream.json').read_text());assert result['complete'] and result['queries_completed']==count
 assert not result['cache_reset_between_batches']
 log=(out/'run.out').read_text()
 if nodes:assert f'cold start, capacity {nodes} nodes, 0 preloaded' in log
 hashes={}
 for suffix in ['idx_uint32.bin','dists_float.bin']:
  p=Path(str(out/'result')+'_128_'+suffix);assert p.stat().st_size==8+count*40;hashes[suffix]=hashfile(p)
 if mode=='nc':
  assert 'd-cache budget=unbounded' in log and 'thread-private LRUs' in log
  line=next(v for v in log.splitlines() if v.startswith('NEIGHBORCACHE async_commit:'));assert re.search(r'\bwriter_failed=0\b',line)
 elif mode=='nonly':
  nc=json.loads((R/f'runs/nc_n{N}/result.json').read_text());assert hashes==nc['result_hashes'],'Full output mismatch'
  assert result['counts']['n_beam_lookups']==nc['counts']['n_beam_lookups']
 dfile=Path(str(data/'D')+'_neighborcache_dc.dyn.data');st=dfile.stat() if dfile.exists() else None
 result.update(mode=mode,N_nodes=nodes,N_payload_bytes=nodes*RECORD,P_payload_capacity_bytes=P*4096 if mode=='nc' else 0,memory_payload_bytes=nodes*RECORD+(P*4096 if mode=='nc' else 0),D_file=str(dfile) if st else None,D_file_allocated_bytes=st.st_blocks*512 if st else 0,D_file_logical_bytes=st.st_size if st else 0,result_hashes=hashes,elapsed_seconds=time.time()-started,result_prefix=str(out/'result'),all_caches_initially_empty=True)
 write(out/'result.json',result);print('DONE',mode,nodes,result['qps'],result['memory_hit_percent'],flush=True);return result

if __name__=='__main__':
 os.sched_setaffinity(0,{23})
 membership=json.loads((R/'index_membership.json').read_text());assert membership['matching_query_count']==0 and membership['query_vectors']==8000000
 run('reference',0,limit=20000)
 nc=run('nc',N);target=nc['memory_hit_percent']
 # Use previous complete-stream measurements solely as an initial guess.
 small=json.loads((SMALL/'synthetic/matched.json').read_text())
 selected=round(small['selected_N_nodes']*(100000000/21593794)/128)*128
 selected=max(128,min(100000000,selected));low=dict(N_nodes=0,memory_hit_percent=0);high=None;trials=[]
 for attempt in range(10):
  a=run('nonly',selected);trials.append(a);write(R/'N_only_trials.json',trials)
  error=a['memory_hit_percent']-target
  if 0<=error<=.25:
   write(R/'matched.json',dict(target_percent=target,matched_percent=a['memory_hit_percent'],NC=[nc],N_only=[a],selected_N_nodes=selected,memory_accounting='N + P data capacities only; excludes M, directories, PQ and other process memory; D is allocated SSD blocks',match_tolerance_percentage_point=.25))
   break
  if error<0:low=a
  else:high=a
  if high is None:selected=min(100000000,selected*2)
  else:
   selected=round((low['N_nodes']+(target+.08-low['memory_hit_percent'])*(high['N_nodes']-low['N_nodes'])/(high['memory_hit_percent']-low['memory_hit_percent']))/128)*128
   selected=max(low['N_nodes']+128,min(high['N_nodes']-128,selected))
 else:raise RuntimeError('Full measured N-only hit matching did not converge')
 spec=(SMALL/'plot.py').read_text().replace("R=Path(__file__).resolve().parent;name=sys.argv[1];o=R/name","R=Path(__file__).resolve().parent;name='large';o=R")
 (R/'plot_results.py').write_text(spec);subprocess.run(['/home/jianz/miniconda3/bin/python',str(R/'plot_results.py')],check=True)
 print('80M EXPERIMENT AND THREE-BAR FIGURE COMPLETE',flush=True)
