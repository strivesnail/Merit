"""Replay the nine screenshot configurations with the current native DiskANN."""
from pathlib import Path
import hashlib,json,os,re,subprocess,time
R=Path('/mnt/nvme2/sift100m_figure_repro_20261027')
configs=json.loads((R/'configurations.json').read_text())
exe=Path('/home/jianz/workload/code/Merit/diskann/build/apps/search_disk_index')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
binary_sha=sha(exe)
rows=[]
for rep in range(3):
 for w in ['pub','x5','zipf']:
  modes=['diskann','neighbor','bfs'];modes=modes[rep:]+modes[:rep]
  for mode in modes:
   cfg=next(x.copy() for x in configs if (x['workload'],x['mode'])==(w,mode))
   out=R/f'{w}_{mode}_{rep}';out.mkdir(exist_ok=False)
   cmd=cfg['command'].copy();cmd[cmd.index('--result_path')+1]=str(out/'result')
   env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','PIPEANN_NMD','AISAQ_NMD','MERIT_'))};env.update(cfg['environment'])
   assert sha(exe)==binary_sha,'binary changed during comparison'
   cfg.update(command=cmd,rep=rep,status='running',binary_sha256=binary_sha)
   (out/'command.json').write_text(json.dumps(cfg,indent=2)+'\n')
   print('START',out.name,flush=True);start=time.time()
   with (out/'run.out').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
   cfg.update(exit_code=p.returncode,wall_seconds=time.time()-start,status='completed' if p.returncode==0 else 'failed');(out/'command.json').write_text(json.dumps(cfg,indent=2)+'\n')
   assert p.returncode==0,out
   log=(out/'run.out').read_text();data=re.findall(r'^\s*128\s+4\s+(.+)$',log,re.M);assert len(data)==1
   v=data[0].split();breakdown=re.findall(r'breakdown: (.+)',log)
   counters={k:float(n) for k,n in re.findall(r'(\w+)=([\d.]+)',breakdown[-1])} if breakdown else {}
   row=dict(workload=w,mode=mode,rep=rep,queries=cfg['queries'],qps=float(v[0]),recall=float(v[-1]),ios_per_query=float(v[3]),historical_qps=cfg['historical_qps'],historical_recall=cfg['historical_recall'],breakdown=counters)
   assert row['recall']>95
   if mode=='neighbor':
    assert 'cold start, capacity 212371 nodes, 0 preloaded' in log
    assert 'page cache: 10112 pages' in log
    assert 'disk-cache loaded (reuse): 3185570 nodes' in log
    assert 'dynamic 3-cache enabled' not in log
   if mode=='bfs':assert 'Caching 1081006 nodes around medoid(s)' in log
   (out/'measurement.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
   (R/'results_partial.json').write_text(json.dumps(rows,indent=2)+'\n')
   print('DONE',out.name,'QPS',row['qps'],'recall',row['recall'],'IO',row['ios_per_query'],flush=True)
assert len(rows)==27
(R/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
print('ALLDONE 27 runs',flush=True)
