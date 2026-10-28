from pathlib import Path
import hashlib,json,os,re,subprocess,time
R=Path('/mnt/nvme2/sift100m_access_steps_20261027');ORIG=Path('/mnt/nvme2/sift100m_figure_repro_20261027');exe=R/'build/search_disk_index'
configs=json.loads((ORIG/'configurations.json').read_text());allrows=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
production=Path('/home/jianz/workload/code/Merit/diskann/build/apps/search_disk_index');saved=json.loads((R/'source_hashes.json').read_text())['production_binary_sha256'];assert sha(production)==saved
for work in ['pub','x5','zipf']:
 cfg=next(c.copy() for c in configs if c['workload']==work and c['mode']=='neighbor');out=R/work;out.mkdir(exist_ok=True)
 cmd=[str(exe) if a.endswith('/search_disk_index') else a for a in cfg['command']];cmd[cmd.index('--result_path')+1]=str(out/'result')
 env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','PIPEANN_NMD','AISAQ_NMD','MERIT_'))};env.update(cfg['environment'])
 cfg.update(command=cmd,diagnostic_only=True,binary_sha256=sha(exe),production_binary_sha256=saved,instrumentation='All queries; nested inclusive/exclusive steady-clock thread timers; no timing atomics')
 if (out/'command.json').exists() and json.loads((out/'command.json').read_text()).get('exit_code')==0:
  print('REUSE completed diagnostic',work,flush=True)
 else:
  (out/'command.json').write_text(json.dumps(cfg,indent=2)+'\n');start=time.time();print('START',work,flush=True)
  with (out/'run.out').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  cfg.update(exit_code=p.returncode,wall_seconds=time.time()-start);(out/'command.json').write_text(json.dumps(cfg,indent=2)+'\n');assert p.returncode==0
 log=(out/'run.out').read_text();stats={name:dict(calls=int(n),inclusive_ns=int(inc),exclusive_ns=int(exc)) for name,n,inc,exc in re.findall(r'NC_STEP stage=(\w+) calls=(\d+) inclusive_ns=(\d+) exclusive_ns=(\d+)',log)}
 assert len(stats)==21,len(stats)
 assert stats['query']['calls']==cfg['queries'],stats['query']
 assert sum(v['exclusive_ns'] for v in stats.values())==stats['query']['inclusive_ns']
 assert 'cold start, capacity 212371 nodes, 0 preloaded' in log and 'page cache: 10112 pages' in log
 assert 'disk-cache loaded (reuse): 3185570 nodes' in log and 'dynamic 3-cache enabled' not in log
 fields=re.findall(r'^\s*128\s+4\s+(.+)$',log,re.M)[-1].split();recall=float(fields[-1]);assert recall==cfg['historical_recall']
 row=dict(workload=work,queries=cfg['queries'],recall=recall,diagnostic_qps=float(fields[0]),stages=stats)
 (out/'measurement.json').write_text(json.dumps(row,indent=2)+'\n');allrows.append(row);(R/'measurements.json').write_text(json.dumps(allrows,indent=2)+'\n');print('DONE',work,'queries',cfg['queries'],'recall',recall,flush=True)
assert sha(production)==saved
print('COMPLETE: production executable unchanged',flush=True)
