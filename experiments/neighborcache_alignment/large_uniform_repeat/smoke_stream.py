from pathlib import Path
import os,json,subprocess,sys
R=Path(os.environ.get('NC_LARGE_ROOT','/mnt/nvme3/nc_sift100m_uniform8mx10_20261101'))
old=Path('/mnt/nvme2/nc_sift100m_uniform200kx10_hitmatch_20261101')
sys.path.insert(0,'/home/jianz/workload/code/Merit/experiments/neighborcache_alignment/n_capacity_match')
from cache_control import clear_files,prefixed_files
cfg=json.loads((old/'runs/smoke_nc_n42474_1/command.json').read_text());ref=json.loads((old/'runs/smoke_reference_n0_1/result.json').read_text())
base=Path('/mnt/nvme_single/sift100m/sift100m_index');droot=Path('/mnt/nvme_single/sift100m/runs/nc_uniform8mx10_20261101/smoke');droot.mkdir(parents=True,exist_ok=True)
os.sched_setaffinity(0,{23});results={}
for mode in ['nonly','nc']:
 out=R/'smoke'/mode;out.mkdir(parents=True,exist_ok=True);data=droot/mode;data.mkdir(exist_ok=True)
 env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','NC_','OMP_'))};env.update(cfg['environment'])
 env.pop('NEIGHBORCACHE_DCACHE_CAP',None);env.pop('NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES',None)
 env.update(NEIGHBORCACHE_DCACHE_UNBOUNDED='1',NC_STREAM_BATCH_QUERIES='10000',NC_STREAM_D_PREFIX=str(data/'D'))
 if mode=='nonly':
  for k in ['NEIGHBORCACHE_DYNAMIC_3CACHE','NEIGHBORCACHE_SHARED_CACHE_DYN','NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES']:env.pop(k,None)
  env['NEIGHBORCACHE_NCACHE_ONLY']='1'
 cmd=cfg['command'].copy();cmd[cmd.index('--data_type')-1]=str(R/'build/search_disk_index');cmd[cmd.index('--result_path')+1]=str(out/'result');cmd[cmd.index('--dump_query_stats')+1]=str(out/'unused.csv')
 (out/'command.json').write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith(('NEIGHBORCACHE_','NC_','OMP_','LD_'))}),indent=2)+'\n')
 (out/'cache_clear.json').write_text(json.dumps(clear_files(prefixed_files(base)+[Path(cmd[cmd.index('--query_file')+1])]),indent=2))
 with (out/'run.out').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
 result=json.loads((out/'result_stream.json').read_text());assert result['complete'] and result['queries_completed']==20000 and not result['cache_reset_between_batches']
 assert result['counts']['n_beam_lookups']==ref['counts']['n_beam_lookups']
 for suffix in ['idx_uint32.bin','dists_float.bin']:
  assert Path(str(out/'result')+'_128_'+suffix).read_bytes()==Path(ref['result_prefix']+'_128_'+suffix).read_bytes(),(mode,suffix)
 rows=(out/'result_windows.csv').read_text().splitlines();assert len(rows)==3
 if mode=='nc':
  assert result['counts']['n_neighborcache_prefetch_hits']>0
  assert Path(str(data/'D')+'_neighborcache_dc.dyn.data').stat().st_dev==Path(str(base)+'_disk.index').stat().st_dev
 else:assert result['counts']['n_neighborcache_prefetch_hits']==result['counts']['n_neighborcache_dyn_disk_reads']==0
 results[mode]=dict(result,all_ids_and_distances_equal_uncached=True,windows=2,stable_worker_OS_threads=True)
 print('STREAM SMOKE PASS',mode,result['qps'],flush=True)
(R/'stream_smoke_validation.json').write_text(json.dumps(results,indent=2)+'\n')
