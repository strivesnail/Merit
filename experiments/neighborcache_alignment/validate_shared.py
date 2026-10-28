"""Full-record integration checks; verification overhead makes QPS unsuitable for reporting.
Uses the existing local immutable SIFT indices. Every run records the exact binary
and backend hashes. Old static-prefill fractions are deliberately not inherited.
"""
from pathlib import Path
import argparse, hashlib, json, os, struct, subprocess

REPO = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('/mnt/nvme2/nc_shared_backend_20261027/final'))
p.add_argument('--backend', type=Path, default=REPO/'neighborcache_backend/build/libneighborcache_backend.so')
p.add_argument('--engines', nargs='+', default=['pipeann','starling','margo','aisaq'])
p.add_argument('--pressure-runs', type=int, default=20)
a = p.parse_args(); a.root.mkdir(parents=True, exist_ok=True)
source = Path('/mnt/nvme_single/meeting_20261026_prefetch5_half_n/sift10m_pub')
ai_source = Path('/mnt/nvme2/aisaq_nmd_trial_20261027')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, obj): path.write_text(json.dumps(obj, indent=2)+'\n')
def run(name, cmd, settings):
    out = a.root/name; out.mkdir(exist_ok=True)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('PIPEANN_NMD','AISAQ_NMD','NEIGHBORCACHE_'))}
    env.update(settings)
    binary = next(x for x in cmd if x.endswith('/search_disk_index'))
    cfg = dict(command=cmd, environment=settings, binary_sha256=sha(binary),
               backend_sha256=sha(a.backend), purpose='correctness, not performance')
    write_json(out/'command.json', cfg)
    with (out/'run.out').open('w') as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    cfg['exit_code'] = r.returncode; write_json(out/'command.json',cfg)
    if r.returncode: raise RuntimeError(f'{name}: {r.returncode}; see {out}/run.out')
    print(name, 'PASS', flush=True)
    return out

def common(threads):
    return dict(LD_LIBRARY_PATH='/home/jianz/miniconda3/lib:/tmp/aisaq-nmd-deps/root/usr/lib/x86_64-linux-gnu',
                NEIGHBORCACHE_BACKEND_LIBRARY=str(a.backend), NEIGHBORCACHE_THREADS=str(threads),
                OMP_NUM_THREADS=str(threads), NEIGHBORCACHE_VERIFY='1')

def ids(path):
    b=path.read_bytes(); n,k=struct.unpack_from('<II',b)
    assert len(b)==8+4*n*k
    v=struct.unpack_from(f'<{n*k}I',b,8)
    return [v[i*k:(i+1)*k] for i in range(n)]
def compare(x,y):
    assert len(x)==len(y)
    return dict(identical_rows=sum(u==v for u,v in zip(x,y)), queries=len(x),
                equal_sets=sum(set(u)==set(v) for u,v in zip(x,y)),
                topk_overlap=sum(len(set(u)&set(v)) for u,v in zip(x,y))/(len(x)*len(x[0])))

query=a.root/'query1k.u8bin'; gt=a.root/'gt1k.bin'
b=Path('/home/jianz/workload/data/sift10m/query.public.10K.u8bin').read_bytes(); n,d=struct.unpack_from('<II',b)
query.write_bytes(struct.pack('<II',1000,d)+b[8:8+1000*d])
b=Path('/home/jianz/workload/data/sift10m/bigann-10M').read_bytes(); n,k=struct.unpack_from('<II',b)
assert len(b)==8+n*k*8
gt.write_bytes(struct.pack('<II',1000,k)+b[8:8+1000*k*4]+b[8+n*k*4:8+n*k*4+1000*k*4])
summary={}
for engine in a.engines:
    if engine=='aisaq':
        for workload in ['public','x5']:
            outs=[]
            for mode in ['off','online']:
                old=json.loads((ai_source/f'{workload}_nmd/command.json').read_text())
                cmd=old['command'].copy(); name=f'aisaq_{workload}_{mode}'; out=a.root/name
                cmd[cmd.index('--result_path')+1]=str(out/'result')
                s=common(8);s.update(AISAQ_NMD='1' if mode=='online' else '0', AISAQ_NMD_PREFILL='0',
                    AISAQ_NMD_NMB='1',AISAQ_NMD_PMB='1',AISAQ_NMD_VERIFY='1', AISAQ_NMD_PATH=str(out/'cache'))
                outs.append(run(name,cmd,s))
            same=all((outs[0]/f'result_64_{s}').read_bytes()==(outs[1]/f'result_64_{s}').read_bytes()
                     for s in ['idx_uint32.bin','dists_float.bin'])
            assert same, workload
            summary[f'aisaq_{workload}']={'identical_ids_and_distances':same}
        continue
    old=json.loads((source/engine/'command.json').read_text())
    result=[]
    # Two uncached runs quantify native scheduling variation independently.
    for mode in ['off0','off1','online']:
        name=f'{engine}_{mode}'; out=a.root/name; cmd=old['command'].copy(); s=common(20)
        s.update(OMP_PROC_BIND='close', OMP_PLACES='{0}:20',NEIGHBORCACHE_QUERY_CPUS='0-19')
        if engine=='pipeann':
            cmd[8]=str(query);cmd[9]=str(gt)
            s.update(PIPEANN_NMD='1' if mode=='online' else '0', PIPEANN_NMD_PREFILL='0',
                     PIPEANN_NMD_NMB='31',PIPEANN_NMD_PMB='16',PIPEANN_NMD_VERIFY='1',
                     PIPEANN_NMD_PATH=str(out/'cache'),PIPEANN_DUMP=str(out/'ids'))
        else:
            cmd[cmd.index('--query_file')+1]=str(query);cmd[cmd.index('--gt_file')+1]=str(gt)
            cmd[cmd.index('--result_path')+1]=str(out/'result')
            s.update(NEIGHBORCACHE_NCACHE_CAP_NODES='91842' if mode=='online' else '0',
                     NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES='4350',NEIGHBORCACHE_DCACHE_CAP='276551',
                     NEIGHBORCACHE_DCACHE_PREFILL='0',NEIGHBORCACHE_DCACHE_MAX_PAGES_PER_SEED='1')
        result.append(run(name,cmd,s))
    def load(out): return ids(out/'ids.L128' if engine=='pipeann' else next(out.glob('result*idx_uint32.bin')))
    baseline=load(result[0]);summary[engine]={'off_vs_off':compare(baseline,load(result[1])),
                                           'off_vs_online':compare(baseline,load(result[2]))}
    if engine=='pipeann':
        for i in range(a.pressure_runs):
            name=f'pipeann_pressure_{i:02d}'; out=a.root/name
            settings=dict(s);settings.update(PIPEANN_NMD_NMB='1',PIPEANN_NMD_PMB='1',
                NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES='4',NEIGHBORCACHE_DCACHE_CAP='128',
                NEIGHBORCACHE_MCACHE_CAP='512',PIPEANN_NMD_PATH=str(out/'cache'),PIPEANN_DUMP=str(out/'ids'))
            if i==0: settings['PIPEANN_NMD_PROFILE']=str(out/'visits')
            run(name,cmd,settings)
            log=(out/'run.out').read_text()
            import re
            assert int(re.search(r'D_reads=(\d+)',log)[1])>0
            if i==0:
                data=(out/'visits').read_bytes();np=struct.unpack_from('<Q',data)[0]
                assert len(data)==8+np*4 and any(data[8:]),'visit profile invalid'
        summary['pipeann_pressure']={'passes':a.pressure_runs, 'full_record_verification':True}
    write_json(a.root/'summary.json',summary)
write_json(a.root/'summary.json',summary)
print(json.dumps(summary,indent=2),flush=True)
