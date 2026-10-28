"""Separate diagnostic runs; never include their QPS in the performance plots."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import time

ROOT = Path('/mnt/nvme2/nc_read_parent_20261027')
LIB = ROOT/'stage_profile_build/libneighborcache_backend.so'
OUT = ROOT/'stage_profile'
OUT.mkdir(exist_ok=True)
all_rows = []
for engine in ['pipeann','starling','margo','aisaq']:
    for mode in ['before','after']:
        cfg = json.loads((ROOT/f'{engine}_public_{mode}_0/command.json').read_text())
        target = OUT/f'{engine}_{mode}'
        target.mkdir(exist_ok=False)
        cmd = cfg['command'].copy()
        env = {k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_', 'PIPEANN_NMD', 'AISAQ_NMD'))}
        settings = cfg['environment'].copy()
        lib = Path('/mnt/nvme2/nc_hot20_20261027/stage_profile_build/libneighborcache_backend.so') if mode=='before' else LIB
        settings['NEIGHBORCACHE_BACKEND_LIBRARY'] = str(lib)
        if engine == 'pipeann': settings['PIPEANN_NMD_PATH'] = str(target/'cache')
        else:
            cmd[cmd.index('--result_path')+1] = str(target/'result')
            if engine == 'aisaq': settings['AISAQ_NMD_PATH'] = str(target/'cache')
        assert not any('VERIFY' in k for k in settings)
        env.update(settings)
        cfg.update(command=cmd, environment=settings, diagnostic_only=True,
                   backend_sha256=hashlib.sha256(lib.read_bytes()).hexdigest())
        (target/'command.json').write_text(json.dumps(cfg,indent=2)+'\n')
        print('START', target.name, flush=True)
        start=time.time()
        with (target/'run.out').open('w') as f:
            p=subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        cfg.update(exit_code=p.returncode, wall_seconds=time.time()-start)
        (target/'command.json').write_text(json.dumps(cfg,indent=2)+'\n')
        assert p.returncode==0, target
        log=(target/'run.out').read_text()
        stages = []
        for name,n,mean,total in re.findall(r'NC_STAGE stage=(\w+) samples=(\d+) mean_us=([\d.]+) estimated_thread_seconds=([\d.]+)',log):
            stages.append(dict(stage=name,samples=int(n),mean_us=float(mean),estimated_thread_seconds=float(total)))
        assert len(stages)==(12 if mode=='before' else 13), target
        row=dict(engine=engine, mode=mode, stages=stages)
        (target/'stages.json').write_text(json.dumps(row,indent=2)+'\n')
        all_rows.append(row)
        (OUT/'summary.json').write_text(json.dumps(all_rows,indent=2)+'\n')
        for f in target.glob('cache_neighborcache_dc.dyn.data'): f.unlink()
        print('DONE', target.name, flush=True)
