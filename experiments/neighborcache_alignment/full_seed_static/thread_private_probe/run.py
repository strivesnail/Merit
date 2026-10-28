from pathlib import Path
import os,json,hashlib,signal,subprocess,time,re,csv,ctypes
import numpy as np
R=Path(__file__).resolve().parent
BASE=Path('/mnt/nvme2/nc_fullseed_prefetch_20261028')
CAPACITIES=[2060]
OLD=Path('/mnt/nvme2/nc_ab_zipf_repeat_verify_20261028')
DATA=BASE/'visited_seeds.pages'
META=json.loads((BASE/'visited_summary.json').read_text());SEEDS=META['unique_visited_nodes'];INDEX=str(BASE/'index/sift100m_index')
QUERY=Path('/mnt/nvme2/sift100m_query_ab_20261027/A_then_B_zipf0p8_40k.u8bin')
EXE=R/'search_disk_index'
src=Path('/mnt/nvme2/sift100m_7modes_3runs_20261027/run.py').read_text();exec(src[src.index('lib=ctypes.CDLL'):src.index('\nconfigs=[]')])
assert f'COMPLETE seeds={SEEDS}' in (BASE/'visited_prefill_build.log').read_text()
assert DATA.stat().st_size==SEEDS*4096
expected=np.load(OLD/'expected_ids.npy')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
os.sched_setaffinity(0,{22,23})
active=None

def stop(sig,frame):
 if active is not None and active.poll() is None:active.terminate();active.wait(timeout=20)
 raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
rows=[]
for rep in range(1):
 for mode,pages,reset in [('fullseed',p,x) for p in CAPACITIES for x in [False,True]]:
  out=R/'runs'/f'{"private_reset" if reset else "shared"}_p{pages}_{rep}';out.mkdir(parents=True,exist_ok=True)
  assert not (out/'run.out').exists()
  config=json.loads((OLD/'runs'/('nc_0' if mode=='fullseed' else 'bfs_0')/'command.json').read_text())
  cmd=config['command'][:];cmd[3]=str(EXE);cmd[2]='0-19';cmd[cmd.index('--num_threads')+1]='20'
  for arg,val in [('--index_path_prefix',INDEX),('--result_path',str(out/'result')),('--dump_query_stats',str(out/'query_stats.csv'))]:cmd[cmd.index(arg)+1]=val
  if mode=='fullseed':cmd[cmd.index('--neighborcache_memory_gb')+1]='0'
  settings={k:v for k,v in config['environment'].items() if not k.startswith(('NEIGHBORCACHE_','NC_'))}
  settings.update(NEIGHBORCACHE_QUERY_CPUS='0-19',OMP_NUM_THREADS='20',OMP_DYNAMIC='FALSE',OMP_PLACES='{0}:20',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NEIGHBORCACHE_QUERY_TIMELINE=str(out/'timeline'),NC_EXPERIMENT_PHASE_QUERIES='20000')
  if mode=='fullseed':settings.update(NC_FULLSEED_DATA=str(DATA),NC_FULLSEED_IDS=str(BASE/'visited_seed_ids.u32'),NC_FULLSEED_PREFETCH_PAGES=str(pages),NC_FULLSEED_RESET_EACH_QUERY=str(int(reset)),NC_FULLSEED_THREAD_PRIVATE=str(int(reset)))
  env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','NC_','MERIT_','PIPEANN_NMD','AISAQ_NMD','LD_PRELOAD'))};env.update(settings)
  files=list(map(str,prefixed_files(INDEX)))+[str(QUERY)]+([str(DATA)] if mode=='fullseed' else [])
  print('CLEAR','private_reset' if reset else 'shared',pages,rep,flush=True);(out/'cache_clear.json').write_text(json.dumps(clear_files(files),indent=2))
  manifest=dict(command=cmd,environment=settings,binary_sha256=sha(EXE),mode='private_reset' if reset else 'shared',prefetch_pages=pages,node_record_slots=pages*10,rep=rep)
  (out/'command.json').write_text(json.dumps(manifest,indent=2))
  print('START','private_reset' if reset else 'shared',pages,rep,flush=True)
  with (out/'run.out').open('w') as f:
   active=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
   time.sleep(1)
   tasks=[]
   for task in Path(f'/proc/{active.pid}/task').glob('*'):
    try:
     status=(task/'status').read_text()
     tasks.append(dict(tid=int(task.name),cpu_allowed=re.search(r'Cpus_allowed_list:\s*(.*)',status)[1]))
    except FileNotFoundError:pass
   (out/'observed_tasks.json').write_text(json.dumps(tasks,indent=2))
   assert tasks
   for task in tasks:
    allowed=set()
    for part in task['cpu_allowed'].split(','):
     bounds=part.split('-');allowed.update(range(int(bounds[0]),int(bounds[-1])+1))
    assert allowed and allowed.issubset(set(range(20))),task
   code=active.wait(timeout=600);active=None
  assert code==0,(mode,rep,code)
  log=(out/'run.out').read_text();m=re.search(r'^\s*128\s+4\s+(.+)$',log,re.M);assert m
  qps=float(m[1].split()[0])
  reset_record=re.search(r'FULLSEED query_reset=(\d+) reset_calls=(\d+) reset_removed_pages=(\d+) reset_ns=(\d+)',log);assert reset_record
  reset_mode,reset_calls,removed_pages,reset_ns=map(int,reset_record.groups())
  assert reset_mode==int(reset) and reset_calls==(40000 if reset else 0)

  phases=[dict(zip(['begin','end','start_ns','end_ns'],map(int,p))) for p in re.findall(r'Experiment phase: begin=(\d+) end=(\d+) start_ns=(\d+) end_ns=(\d+)',log)]
  assert [(p['begin'],p['end']) for p in phases]==[(0,20000),(20000,40000)]
  q=np.genfromtxt(out/'query_stats.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
  t=np.genfromtxt(out/'timeline.L128.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
  ids=np.fromfile(out/'result_128_idx_uint32.bin',dtype='<u4')[2:].reshape(40000,10)
  assert np.array_equal(ids,expected),'Search results changed'
  previous=np.genfromtxt(Path('/mnt/nvme2/nc_query_reset_103pages_20261028/runs/reset_p103_0/query_stats.csv'),delimiter=',',names=True,dtype=None,encoding='utf8')
  assert np.array_equal(q['n_beam_lookups'],previous['n_beam_lookups'])
  if reset:
   for field in ['n_neighborcache_prefetch_hits','n_disk_reads']:
    assert np.array_equal(q[field],previous[field]),field

  assert np.array_equal(q['query_id'],t['query_id']) and len(q)==40000
  assert t['completed_ns'][:20000].max()<=phases[0]['end_ns']<=t['completed_ns'][20000:].min()
  visits=q['n_beam_lookups'];hits=q['n_mem_hits']+q['n_neighborcache_prefetch_hits']
  assert np.all(visits==hits+q['n_disk_reads'])
  assert np.all(q['n_neighborcache_mcache_hits']==0) and np.all(q['n_neighborcache_mcache_misses']==0)
  if mode=='fullseed':
   assert f'prefetch_pages={pages} N=0 M=0' in log
   if reset:
    assert 'private_LRU=20 query_threads=20 pages_per_thread_min=103 pages_per_thread_max=103' in log
   else:
    assert 'shard_LRU=128 query_threads=20' in log
   assert np.all(q['n_mem_hits']==0) and np.all(q['n_base_pages']==0)
   assert np.all(q['n_disk_reads']==q['n_neighborcache_dyn_disk_reads'])
   assert np.all(q['n_disk_reads']==q['n_neighborcache_dcache_demand_hits'])
   assert np.all(q['n_disk_reads']==q['n_neighborcache_dcache_demand_probes'])
   assert f'initial_d_pages={SEEDS} final_d_pages={SEEDS}' in log
  else:assert 'Caching 478680 nodes around medoid(s)' in log
  def metrics(ix):
   return dict(memory_hit_pct=100*float(hits[ix].sum())/int(visits[ix].sum()),n_hits_per_query=float(q['n_mem_hits'][ix].mean()),prefetch_hits_per_query=float(q['n_neighborcache_prefetch_hits'][ix].mean()),target_visits_per_query=float(visits[ix].mean()),disk_reads_per_query=float(q['n_disk_reads'][ix].mean()),io_us_per_query=float(q['io_us'][ix].mean()),latency_us_per_query=float(q['total_us'][ix].mean()))
  windows=[]
  for pi,p in enumerate(phases):
   ix=np.arange(p['begin'],p['end']);order=ix[np.argsort(t['completed_ns'][ix],kind='stable')];previous=p['start_ns']
   for end in range(200,len(order)+1,200):
    chosen=order[end-200:end];stamp=int(t['completed_ns'][order[end-1]]);dt=(stamp-previous)/1e9;previous=stamp
    windows.append(dict(phase='AB'[pi],completed_queries=p['begin']+end,elapsed_seconds=dt,qps=200/dt,**metrics(chosen)))
   p.update(name='AB'[pi],qps=20000e9/(p['end_ns']-p['start_ns']),**metrics(ix))
  row=dict(mode='private_reset' if reset else 'shared',prefetch_pages=pages,node_record_slots=pages*10,rep=rep,qps=qps,phases=phases,windows=windows,metrics=metrics(np.arange(40000)),result_ids_match_reference=True,thread_private=reset,query_threads=20,reset_each_query=reset,reset_calls=reset_calls,reset_removed_pages=removed_pages,reset_ns=reset_ns,reset_us_per_query=reset_ns/40000/1000)
  rows.append(row);(out/'measurement.json').write_text(json.dumps(row,indent=2));(R/'results_partial.json').write_text(json.dumps(rows,indent=2))
  print('DONE','private_reset' if reset else 'shared',pages,rep,qps,row['metrics'],flush=True)
(R/'results.json').write_text(json.dumps(rows,indent=2));print('COMPLETE',flush=True)
