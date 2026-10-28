"""Alternate baseline/NMD order; measure request counts, not physical SSD traffic."""
from pathlib import Path
import json,os,re,subprocess,statistics
r=Path('/mnt/nvme2/aisaq_nmd_trial_20261027');records=[]
for work in ['public','x5']:
 for rep in range(3):
  for mode in (['baseline','nmd'] if rep%2==0 else ['nmd','baseline']):
   cfg=json.loads((r/f'{work}_{mode}/command.json').read_text());out=r/f'repeat_{work}_{mode}_{rep}';out.mkdir(exist_ok=True)
   cmd=cfg['command'];cmd[cmd.index('--result_path')+1]=str(out/'result')
   settings=cfg['environment'];settings['AISAQ_NMD_PATH']=str(out/'cache.nmd')
   env={k:v for k,v in os.environ.items() if not k.startswith(('AISAQ_NMD','PIPEANN_NMD','NEIGHBORCACHE_'))};env.update(settings)
   (out/'command.json').write_text(json.dumps(cfg,indent=2))
   with (out/'run.out').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   text=(out/'run.out').read_text();fields=re.findall(r'^\s*64\s+4\s+(.+)$',text,re.M)[-1].split()
   dp=re.search(r'NMD demand D (?:pread|read) requests=(\d+)',text);nc=re.search(r'NMD n_hit=(\d+) pref_hit=(\d+) d_hit=(\d+) miss=(\d+) expand=(\d+) d_pages=(\d+)',text)
   n=1000 if work=='public' else 5000
   result=dict(workload=work,mode=mode,rep=rep,qps=float(fields[0]),recall=float(fields[-1]),mean_base_reads=float(fields[3]),d_reads=int(dp[1]) if dp else 0,d_writes=int(nc[6]) if nc else 0)
   if nc:result['mean_base_reads']=int(nc[4])/n
   result['mean_total_reads']=result['mean_base_reads']+result['d_reads']/n
   for suffix in ['idx_uint32.bin','dists_float.bin']:
    assert (out/f'result_64_{suffix}').read_bytes()==(r/f'{work}_baseline'/f'result_64_{suffix}').read_bytes()
   records.append(result);(out/'result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
   if (out/'cache.nmd').exists():(out/'cache.nmd').unlink()
(r/'repeats.json').write_text(json.dumps(records,indent=2))
for w in ['public','x5']:
 for m in ['baseline','nmd']:
  rows=[x for x in records if x['workload']==w and x['mode']==m]
  print(w,m,'median QPS',statistics.median(x['qps'] for x in rows),'mean read requests/query',statistics.mean(x['mean_total_reads'] for x in rows),'mean D writes',statistics.mean(x['d_writes'] for x in rows),flush=True)
