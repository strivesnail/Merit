from pathlib import Path
import json,os,re,subprocess,time,hashlib
root=Path('/mnt/nvme_single/pipeann_diagnosis_20261027')
old=Path('/mnt/nvme_single/meeting_20261026_prefetch5_half_n')
for work,mode in [('x5','no_rebuild'),('zipf','no_rebuild'),('x5','default_repeat')]:
 out=root/(work+'_'+mode);out.mkdir(exist_ok=True)
 cfg=json.loads((old/f'sift10m_{work}/pipeann/command.json').read_text())
 settings=cfg['environment'].copy();settings['PIPEANN_NMD_PATH']=str(out/'cache.nmd')
 if mode=='no_rebuild':settings['PIPEANN_NMD_HEAT']='4294967295'
 cmd=cfg['command'];exe=next(x for x in cmd if x.endswith('/search_disk_index'))
 assert hashlib.sha256(Path(exe).read_bytes()).hexdigest()==cfg['binary_sha256']
 (out/'command.json').write_text(json.dumps(dict(command=cmd,environment=settings),indent=2))
 env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','PIPEANN_NMD','MERIT_'))};env.update(settings)
 print('START',work,mode,flush=True);os.sync();start=time.time()
 with (out/'run.out').open('w') as f:subprocess.run(['/usr/bin/time','-v',*cmd],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 log=(out/'run.out').read_text();row=re.findall(r'^\s*128\s+4\s+(.+)$',log,re.M)[-1].split()
 result=dict(qps=float(row[0]),recall=float(row[-1]),seconds=time.time()-start,cache_bytes=(out/'cache.nmd').stat().st_size)
 (out/'result.json').write_text(json.dumps(result,indent=2));print('DONE',work,mode,result,flush=True)
 # Scratch D file belongs only to this diagnostic; all logs and commands retained.
 (out/'cache.nmd').unlink()
