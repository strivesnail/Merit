"""Visited hot20 D prefill: static measurement by default, dynamics for diagnosis."""
from pathlib import Path
import argparse,hashlib,json,os,re,statistics,subprocess,time
ROOT=Path('/mnt/nvme2/nc_read_parent_20261027')
QPS=Path('/mnt/nvme2/nc_shared_qps_20261027')
VALID=Path('/mnt/nvme2/nc_shared_backend_20261027/final')

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(engine,work,mode,rep,verify=False):
 variant=mode
 mode="off" if variant=="off" else "hot_fixed"
 source=(VALID/(f'aisaq_{work}_online' if engine=='aisaq' else f'{engine}_online') if verify else QPS/f'{engine}_{work}_online_0')
 cfg=json.loads((source/'command.json').read_text());cmd=cfg['command'].copy();settings=cfg['environment'].copy()
 out=ROOT/(f'verified_static_{engine}' if verify else f'{engine}_{work}_{variant}_{rep}');out.mkdir(exist_ok=False)
 for k in list(settings):
  if 'VERIFY' in k or 'PROFILE' in k or k=='PIPEANN_DUMP':del settings[k]
 env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','AISAQ_NMD','PIPEANN_NMD'))}
 meta=json.loads((ROOT/'prefill'/f'{engine}.json').read_text());pages=meta['pages']
 settings.update(NEIGHBORCACHE_BACKEND_LIBRARY=str(ROOT/'libneighborcache_backend.so'),
  NEIGHBORCACHE_DCACHE_CAP=str(pages),NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES=str(pages),
  NEIGHBORCACHE_MCACHE_CAP=str(pages*2),NEIGHBORCACHE_BACKEND_REPORT='1',
  NEIGHBORCACHE_FREEZE_MAINTENANCE='1' if mode=='hot_fixed' else '0')
 if mode.startswith('hot'):
  settings.update(NEIGHBORCACHE_HOT_PREFILL_PREFIX=str(ROOT/'prefill'/engine),NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC='1')
 if verify:settings.update(NEIGHBORCACHE_VERIFY='1',PIPEANN_NMD_VERIFY='1',AISAQ_NMD_VERIFY='1')
 if engine=='pipeann':
  settings.update(PIPEANN_NMD='0' if mode=='off' else '1',PIPEANN_NMD_PATH=str(out/'cache'))
  # Native benchmark does not write IDs unless requested; timing leaves it off.
 else:
  cmd[cmd.index('--result_path')+1]=str(out/'result')
  if engine=='aisaq':settings.update(AISAQ_NMD='0' if mode=='off' else '1',AISAQ_NMD_PATH=str(out/'cache'))
  elif mode=='off':settings['NEIGHBORCACHE_NCACHE_CAP_NODES']='0'
 if variant=='before':
  old_binary=str(ROOT/'before'/engine/'search_disk_index')
  cmd=[old_binary if x.endswith('/search_disk_index') else x for x in cmd]
  settings['NEIGHBORCACHE_BACKEND_LIBRARY']=str(ROOT/'before/libneighborcache_backend.so')
 env.update(settings);binary=next(x for x in cmd if x.endswith('/search_disk_index'))
 cfg=dict(command=cmd,environment=settings,binary_sha256=sha(binary),backend_sha256=sha(settings['NEIGHBORCACHE_BACKEND_LIBRARY']),
          engine=engine,workload=work,mode=variant,rep=rep,verification=verify,prefill=meta)
 dump(out/'command.json',cfg);print('START',out.name,flush=True);start=time.time()
 with (out/'run.out').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 cfg.update(exit_code=p.returncode,wall_seconds=time.time()-start);dump(out/'command.json',cfg)
 if p.returncode:raise RuntimeError(f'{out}: exit {p.returncode}')
 log=(out/'run.out').read_text();L=64 if engine=='aisaq' else 128
 fields=re.findall(r'^\s*'+str(L)+r'\s+4\s+(.+)$',log,re.M);assert len(fields)==1
 fields=fields[0].split();n=(1000 if verify else 10000)*(5 if work=='x5' else 1)
 if engine=='aisaq':n=1000*(5 if work=='x5' else 1)
 result=dict(engine=engine,workload=work,mode=variant,rep=rep,queries=n,verification=verify,qps=float(fields[0]),
             recall=float(fields[-1]),reported_ios_per_query=float(fields[4] if engine=='pipeann' else fields[3]))
 nc=re.search(r'NeighborCache shared backend: (.+)',log)
 result['cache_stats']={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',nc[1])} if nc else {}
 if mode!='off':assert nc
 if mode.startswith('hot'):
  imported=re.search(r'dynamic BFS prefill: (\d+)/(\d+) pages',log);assert imported and int(imported[1])==pages and int(imported[2])==pages
  assert ('freeze_maintenance=on' if mode=='hot_fixed' else 'freeze_maintenance=off') in log
  result['prefilled_pages']=pages
 if mode=='hot_fixed':
  assert all(result['cache_stats'][x]==0 for x in ['opens','appends','replaces','writes']),result
 result['reference_stats']=[x for x in log.splitlines() if x.startswith('NEIGHBORCACHE ') or 'lock sample=' in x]
 ap=re.search(r'NeighborCache acquired-parent: records=(\d+) parents=(\d+)',log)
 if ap: result.update(acquired_records=int(ap[1]),recorded_parents=int(ap[2]))
 if variant=='after':
  assert result['cache_stats']['M']==0 and result['recorded_parents']==0,result
 dump(out/'measurement.json',result);print('DONE',out.name,'QPS',result['qps'],'recall',result['recall'],flush=True)
 # Every process recreates its own D file; retain logs and seeds, remove temporary query-phase copies.
 if engine in ['pipeann','aisaq']:
  for f in out.glob('cache_neighborcache_dc.dyn.data'):f.unlink()
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['verify','timing','diagnose','static'],required=True);p.add_argument('--engines',nargs='+',default=['pipeann','starling','margo','aisaq']);p.add_argument('--modes',nargs='+',default=['before','after','off']);p.add_argument('--workloads',nargs='+',default=['public','x5']);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
 rows=[]
 if a.phase=='verify':
  for e in a.engines:
   for mode in ['after']:rows.append(run(e,'public',mode,0,True))
 else:
  for e in a.engines:
   for w in a.workloads:
    for rep in range(a.repeats):
     order=a.modes[rep%len(a.modes):]+a.modes[:rep%len(a.modes)]
     for mode in order:
      rows.append(run(e,w,mode,rep));dump(ROOT/f'{a.phase}_latest.json',rows)
 dump(ROOT/f'{a.phase}_latest.json',rows)
if __name__=='__main__':main()
