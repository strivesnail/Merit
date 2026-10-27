"""Replay the saved 5% prefetch experiment with half the N capacity."""
from pathlib import Path
import csv,hashlib,json,os,re,shutil,subprocess,time
OLD=Path('/mnt/nvme_single/meeting_20261026_prefetch5')
ROOT=Path('/mnt/nvme_single/meeting_20261026_prefetch5_half_n')
ARCHIVE=Path('/mnt/nvme2/merit_experiment_archive/meeting_20261026_prefetch5_half_n')
ROOT.mkdir(exist_ok=True)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    configs=sorted(OLD.glob('*/*/command.json'))
    assert len(configs)==30
    # Verify every binary before running any measurements.
    for path in configs:
        cfg=json.loads(path.read_text())
        exe=next(x for x in cfg['command'] if x.endswith('/search_disk_index'))
        if path.parent.name=='neighbor':exe='/mnt/nvme_single/sift100m/runs/n_prefetch_online_20261026/search_disk_index.before'
        assert digest(exe)==cfg['binary_sha256'],exe
    for path in configs:
        tag,method=path.parent.parent.name,path.parent.name
        out=ROOT/tag/method;out.mkdir(parents=True,exist_ok=True)
        if (out/'measurement.json').exists():continue
        cfg=json.loads(path.read_text());cmd=cfg['command'];settings=cfg['environment']
        cmd=[x.replace(str(OLD),str(ROOT)) for x in cmd]
        if method=='neighbor':
            i=next(i for i,x in enumerate(cmd) if x.endswith('/search_disk_index'))
            cmd[i]='/mnt/nvme_single/sift100m/runs/n_prefetch_online_20261026/search_disk_index.before'
            i=cmd.index('--neighborcache_memory_gb')+1
            cmd[i]=f'{float(cmd[i])/2:.9f}'
        capkey='PIPEANN_NMD_NMB' if method=='pipeann' else 'NEIGHBORCACHE_NCACHE_CAP_NODES'
        oldcap=int(settings[capkey]);settings[capkey]=str(oldcap//2)
        settings={k:v.replace(str(OLD),str(ROOT)) for k,v in settings.items()}
        cfg.update(command=cmd,environment=settings,n_fraction=0.1,original_n_capacity=oldcap,
                   new_n_capacity=oldcap//2,n_capacity_unit='MiB' if method=='pipeann' else 'nodes',
                   actual_capacity_ratio=(oldcap//2)/oldcap,reference_command=str(path))
        (out/'command.json').write_text(json.dumps(cfg,indent=2))
        paused=subprocess.run(['ps','-o','stat=','-p','2155989'],capture_output=True,text=True)
        assert paused.returncode!=0 or 'T' in paused.stdout
        assert int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])>6000000
        assert shutil.disk_usage(ROOT).free>8*1024**3
        env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','PIPEANN_NMD','MERIT_'))}
        env.update(settings)
        os.sync();print(f'START {tag} {method}: N {oldcap} -> {oldcap//2}',flush=True)
        start=time.time()
        with (out/'run.out').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        log=(out/'run.out').read_text();rows=re.findall(r'^\s*128\s+4\s+(.+)$',log,re.M);assert rows,out
        fields=rows[-1].split();qps=float(fields[0]);recall=float(fields[-1]);assert qps>0 and recall>90
        before=json.loads((path.parent/'measurement.json').read_text())
        result=dict(qps=qps,recall=recall,seconds=time.time()-start,row='128 4 '+rows[-1],
                    original_qps=before['qps'],qps_change_pct=100*(qps/before['qps']-1),
                    original_recall=before['recall'])
        (out/'measurement.json').write_text(json.dumps(result,indent=2))
        print(f'DONE {tag} {method}: {qps:.2f} QPS ({result["qps_change_pct"]:+.2f}%), recall={recall}',flush=True)
        # Keep generated cache files, offload them after timing to preserve SSD headroom.
        cache=out/'cache.nmd'
        if cache.exists() and not cache.is_symlink():
            dst=ARCHIVE/tag/method/'cache.nmd';dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(cache,dst);assert cache.stat().st_size==dst.stat().st_size
            cache.unlink();cache.symlink_to(dst)
    rows=[]
    for p in sorted(ROOT.glob('*/*/measurement.json')):
        rows.append(dict(dataset_workload=p.parent.parent.name,method=p.parent.name,**json.loads(p.read_text())))
    assert len(rows)==30
    with (ROOT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('ALLDONE 30 runs',flush=True)

if __name__=='__main__':main()
