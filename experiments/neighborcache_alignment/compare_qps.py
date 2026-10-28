"""Paired off/online timing of the shared backend. Run engines sequentially.
Uses the previous correctness configurations and immutable local indices, turns
verification/profiling off, and records commands, hashes, raw logs and repeats.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def repeated_inputs(query, gt, out, repeats):
    qb, gb = Path(query).read_bytes(), Path(gt).read_bytes()
    n, d = struct.unpack_from('<II', qb)
    gn, k = struct.unpack_from('<II', gb)
    assert gn == n and len(qb) == 8+n*d and len(gb) == 8+n*k*8
    qp, gp = out/'query_x5.u8bin', out/'gt_x5.bin'
    qp.write_bytes(struct.pack('<II', n*repeats, d) + qb[8:]*repeats)
    ids, distances = gb[8:8+n*k*4], gb[8+n*k*4:]
    gp.write_bytes(struct.pack('<II', n*repeats, k) + ids*repeats + distances*repeats)
    return str(qp), str(gp)


def summarize(records, root):
    rows = []
    for engine, workload in dict.fromkeys((r['engine'], r['workload']) for r in records):
        groups = {mode: [r for r in records if (r['engine'],r['workload'],r['mode']) == (engine,workload,mode)]
                  for mode in ['off','online']}
        if not all(groups.values()):
            continue
        off, online = (statistics.median(r['qps'] for r in groups[m]) for m in ['off','online'])
        row = dict(engine=engine, workload=workload, dataset='SIFT100K' if engine=='aisaq' else 'SIFT10M',
                   queries=groups['off'][0]['queries'], off_qps=off, online_qps=online,
                   change_pct=(online/off-1)*100)
        for mode, runs in groups.items():
            row[mode+'_qps_runs'] = [r['qps'] for r in runs]
            row[mode+'_recall_runs'] = [r['recall'] for r in runs]
            row[mode+'_recall_median'] = statistics.median(row[mode+'_recall_runs'])
            row[mode+'_reported_ios_per_query'] = statistics.median(r['reported_ios_per_query'] for r in runs)
            row[mode+'_qps_min'] = min(row[mode+'_qps_runs'])
            row[mode+'_qps_max'] = max(row[mode+'_qps_runs'])
        rows.append(row)
    dump(root/'summary.json', rows)
    if rows:
        fields = [k for k in rows[0] if not k.endswith('_runs')]
        with (root/'summary.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f,fields);writer.writeheader()
            writer.writerows({k:r[k] for k in fields} for r in rows)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--validation',type=Path,default=Path('/mnt/nvme2/nc_shared_backend_20261027/final'))
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--engines',nargs='+',default=['pipeann','starling','margo','aisaq'])
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=False)
    query='/home/jianz/workload/data/sift10m/query.public.10K.u8bin'
    gt='/home/jianz/workload/data/sift10m/bigann-10M'
    x5=repeated_inputs(query,gt,a.root,5)
    records=[]
    dump(a.root/'protocol.json',dict(repeats=a.repeats,engines=a.engines,
        query_order='public then exact concatenation repeated five times',
        cache_start='fresh process, empty online N/prefetch/D, no prefill or extra warmup',
        process_order='off/online for even repetition, online/off for odd repetition',
        timer='native search QPS; startup and shutdown outside native timer',
        verification=False,profiling=False,concurrent_benchmarks=False,os_page_cache_dropped=False,
        script_sha256=sha(__file__),validation_root=str(a.validation)))
    for engine in a.engines:
        for workload in ['public','x5']:
            template = f'aisaq_{workload}_online' if engine=='aisaq' else f'{engine}_online'
            base=json.loads((a.validation/template/'command.json').read_text())
            binary=next(s for s in base['command'] if s.endswith('/search_disk_index'))
            backend=base['environment']['NEIGHBORCACHE_BACKEND_LIBRARY']
            assert sha(binary)==base['binary_sha256'],f'binary changed since validation: {engine}'
            assert sha(backend)==base['backend_sha256'],'backend changed since validation'
            for rep in range(a.repeats):
                for mode in (['off','online'] if rep%2==0 else ['online','off']):
                    out=a.root/f'{engine}_{workload}_{mode}_{rep}';out.mkdir()
                    # Normalize the launcher: same CPU affinity and priority within each pair.
                    cmd=base['command'][base['command'].index(binary):].copy()
                    settings=base['environment'].copy()
                    for key in list(settings):
                        if 'VERIFY' in key or 'PROFILE' in key or key=='PIPEANN_DUMP':del settings[key]
                    # Some host checks test presence, so VERIFY must be absent, not '0'.
                    threads=8 if engine=='aisaq' else 20
                    settings.update(OMP_PROC_BIND='close',OMP_PLACES=f'{{0}}:{threads}',
                                    NEIGHBORCACHE_QUERY_CPUS=f'0-{threads-1}')
                    if engine=='pipeann':
                        q,g=(query,gt) if workload=='public' else x5
                        cmd[5],cmd[6]=q,g
                        settings.update(PIPEANN_NMD='0' if mode=='off' else '1',PIPEANN_NMD_PATH=str(out/'cache'))
                        l=128
                    else:
                        if engine!='aisaq':
                            q,g=(query,gt) if workload=='public' else x5
                            cmd[cmd.index('--query_file')+1]=q;cmd[cmd.index('--gt_file')+1]=g
                            settings['NEIGHBORCACHE_NCACHE_CAP_NODES']='0' if mode=='off' else base['environment']['NEIGHBORCACHE_NCACHE_CAP_NODES']
                        else:
                            settings.update(AISAQ_NMD='0' if mode=='off' else '1',AISAQ_NMD_PATH=str(out/'cache'))
                        q=cmd[cmd.index('--query_file')+1];g=cmd[cmd.index('--gt_file')+1]
                        cmd[cmd.index('--result_path')+1]=str(out/'result');l=int(cmd[cmd.index('-L')+1])
                    with open(q,'rb') as f: nq=struct.unpack('<II',f.read(8))[0]
                    fullcmd=['taskset','-c',f'0-{threads-1}']+cmd
                    env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','AISAQ_NMD','PIPEANN_NMD'))}
                    env.update(settings)
                    config=dict(command=fullcmd,environment=settings,binary_sha256=sha(binary),
                                backend_sha256=sha(backend),query_sha256=sha(q),gt_sha256=sha(g),
                                engine=engine,workload=workload,mode=mode,rep=rep,queries=nq)
                    dump(out/'command.json',config)
                    print('START',out.name,flush=True)
                    started=time.time()
                    with (out/'run.out').open('w') as f:
                        proc=subprocess.run(fullcmd,env=env,stdout=f,stderr=subprocess.STDOUT)
                    config.update(exit_code=proc.returncode,wall_seconds=time.time()-started)
                    dump(out/'command.json',config)
                    if proc.returncode:raise RuntimeError(f'failed: {out}: {proc.returncode}')
                    log=(out/'run.out').read_text()
                    assert 'Warming up index' not in log, 'unexpected native warmup'
                    lines=re.findall(r'^\s*'+str(l)+r'\s+4\s+(.+)$',log,re.M)
                    assert len(lines)==1,(out,lines)
                    fields=lines[0].split()
                    result={k:config[k] for k in ['engine','workload','mode','rep','queries']}
                    result.update(qps=float(fields[0]),recall=float(fields[-1]),wall_seconds=config['wall_seconds'],
                                  reported_ios_per_query=float(fields[4] if engine=='pipeann' else fields[3]))
                    nc=re.search(r'NeighborCache shared backend: (.+)',log)
                    result['cache_stats']={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',nc[1])} if nc else {}
                    if mode=='online':assert nc,'backend did not activate'
                    if mode=='off':assert not nc,'backend unexpectedly active'
                    dump(out/'measurement.json',result);records.append(result)
                    dump(a.root/'repeats.json',records);summarize(records,a.root)
                    print('DONE',out.name,'QPS',result['qps'],'recall',result['recall'],flush=True)
    print(json.dumps(summarize(records,a.root),indent=2),flush=True)

if __name__=='__main__':main()
