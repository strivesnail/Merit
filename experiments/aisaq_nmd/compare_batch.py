"""Paired sync/batched/off validation; request counts are not physical SSD I/O."""
import argparse, hashlib, json, os, re, statistics, subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--input-root',type=Path,default=Path('/mnt/nvme2/aisaq_nmd_trial_20261027'))
p.add_argument('--output-root',type=Path,default=Path('/mnt/nvme2/aisaq_nmd_batch_20261027'))
p.add_argument('--binary',type=Path,default=Path('/tmp/nmd-candidates-aisaq/build/apps/search_disk_index'))
a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
records=[]
def run(work,mode,rep,verify=False):
    cfg=json.loads((a.input_root/f'{work}_nmd/command.json').read_text())
    cmd=cfg['command'];cmd[0]=str(a.output_root/'search_disk_index_sync' if mode=='sync' else a.binary)
    out=a.output_root/f'{work}_{mode}_{rep}';out.mkdir(exist_ok=True)
    cmd[cmd.index('--result_path')+1]=str(out/'result')
    settings=cfg['environment'];settings['AISAQ_NMD']='0' if mode=='off' else '1'
    settings['AISAQ_NMD_PATH']=str(out/'cache.nmd')
    if verify:settings['AISAQ_NMD_VERIFY']='1'
    cfg['binary_sha256']=hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest()
    (out/'command.json').write_text(json.dumps(cfg,indent=2))
    env={k:v for k,v in os.environ.items() if not k.startswith(('AISAQ_NMD','PIPEANN_NMD','NEIGHBORCACHE_'))};env.update(settings)
    with (out/'run.out').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    for suffix in ['idx_uint32.bin','dists_float.bin']:
        assert (out/f'result_64_{suffix}').read_bytes()==(a.input_root/f'{work}_baseline'/f'result_64_{suffix}').read_bytes(),out
    text=(out/'run.out').read_text();fields=re.findall(r'^\s*64\s+4\s+(.+)$',text,re.M)[-1].split()
    d=re.search(r'NMD demand D (?:pread|read) requests=(\d+)',text)
    nc=re.search(r'NMD n_hit=(\d+) pref_hit=(\d+) d_hit=(\d+) miss=(\d+) expand=(\d+) d_pages=(\d+)',text)
    fail=re.search(r'NMD D read failures=(\d+)',text)
    n=1000 if work=='public' else 5000
    result=dict(workload=work,mode=mode,rep=rep,verify=verify,qps=float(fields[0]),recall=float(fields[-1]),base_reads_per_query=float(fields[3]),d_reads=int(d[1]) if d else 0,d_writes=int(nc[6]) if nc else 0,d_read_failures=int(fail[1]) if fail else 0,outputs_identical=True)
    if nc:result.update(n_hits=int(nc[1]),prefetch_hits=int(nc[2]),d_hits=int(nc[3]),base_reads_per_query=int(nc[4])/n)
    result['total_reads_per_query']=result['base_reads_per_query']+result['d_reads']/n
    result['writes_per_query']=result['d_writes']/n
    assert result['d_read_failures']==0,result
    if verify:assert all(result[k]>0 for k in ['n_hits','prefetch_hits','d_hits']),result
    (out/'result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    (out/'cache.nmd').unlink(missing_ok=True)
    return result
for work in ['public','x5']:run(work,'batch','verify',True)
for work in ['public','x5']:
    for rep in range(3):
        order=['off','sync','batch'];order=order[rep:]+order[:rep]
        for mode in order:records.append(run(work,mode,rep))
(a.output_root/'repeats.json').write_text(json.dumps(records,indent=2))
summary=[]
for work in ['public','x5']:
    for mode in ['off','sync','batch']:
        rows=[r for r in records if r['workload']==work and r['mode']==mode]
        summary.append(dict(workload=work,mode=mode,median_qps=statistics.median(r['qps'] for r in rows),qps_runs=[r['qps'] for r in rows],mean_reads_per_query=statistics.mean(r['total_reads_per_query'] for r in rows),mean_writes_per_query=statistics.mean(r['writes_per_query'] for r in rows)))
(a.output_root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
