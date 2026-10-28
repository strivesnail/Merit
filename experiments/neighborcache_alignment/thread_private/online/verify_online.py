from pathlib import Path
import os,json,subprocess,re,hashlib
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('/mnt/nvme2/nc_private_dynamic_20261028'))
R=parser.parse_args().root
S=Path('/mnt/nvme2/nc_shared_backend_20261027/final')
summary=[]
os.sched_setaffinity(0,{22,23})
for e in ['starling','margo','pipeann','aisaq']:
 name=e+'_online' if e!='aisaq' else 'aisaq_public_online'
 c=json.loads((S/name/'command.json').read_text());cmd=c['command'];env=c['environment'];out=R/'online_verify_completed'/e;out.mkdir(parents=True,exist_ok=False)
 threads=8 if e=='aisaq' else 20
 env.update(NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE='1',NEIGHBORCACHE_FREEZE_MAINTENANCE='0',NEIGHBORCACHE_BACKEND_LIBRARY=str(R/'build/libneighborcache_backend.so'),NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=str(threads*8),NEIGHBORCACHE_NCACHE_CAP_NODES='1024',NEIGHBORCACHE_DCACHE_CAP='512',NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES='512',NEIGHBORCACHE_MCACHE_CAP='2048',NEIGHBORCACHE_DCACHE_COMMIT_INLINE='1',NEIGHBORCACHE_BACKEND_REPORT='1',NEIGHBORCACHE_THREADS=str(threads),NEIGHBORCACHE_VERIFY='1',PIPEANN_NMD_VERIFY='1',AISAQ_NMD_VERIFY='1')
 if e=='pipeann':env.update(PIPEANN_NMD_PATH=str(out/'cache'),PIPEANN_DUMP=str(out/'ids'))
 else:
  cmd[cmd.index('--result_path')+1]=str(out/'result')
  if e=='aisaq':env['AISAQ_NMD_PATH']=str(out/'cache')
  else:
   disk=Path(cmd[cmd.index('--disk_file_path')+1]).resolve();alias=out/disk.name;alias.symlink_to(disk);cmd[cmd.index('--disk_file_path')+1]=str(alias)
 # Repeat 100 real query vectors 50 times to exercise warm dynamic D/P paths.
 qfile=cmd[8] if e=='pipeann' else cmd[cmd.index('--query_file')+1]
 gtfile=cmd[9] if e=='pipeann' else cmd[cmd.index('--gt_file')+1]
 with open(qfile,'rb') as f:
  n,d=map(int,np.fromfile(f,dtype='<u4',count=2));q=np.fromfile(f,dtype='u1',count=100*d).reshape(100,d)
 with open(gtfile,'rb') as f:
  n,k=map(int,np.fromfile(f,dtype='<u4',count=2));ids=np.fromfile(f,dtype='<u4',count=n*k).reshape(n,k)[:100];dist=np.fromfile(f,dtype='<f4',count=n*k).reshape(n,k)[:100]
 qout=out/'queries.u8bin';gout=out/'truth.bin'
 qout.write_bytes(np.array([5000,d],dtype='<u4').tobytes()+np.tile(q,(50,1)).tobytes())
 gout.write_bytes(np.array([5000,k],dtype='<u4').tobytes()+np.tile(ids,(50,1)).tobytes()+np.tile(dist,(50,1)).tobytes())
 if e=='pipeann':cmd[8]=str(qout);cmd[9]=str(gout)
 else:cmd[cmd.index('--query_file')+1]=str(qout);cmd[cmd.index('--gt_file')+1]=str(gout)
 cfg=dict(command=cmd,environment=env,purpose='Full-record validation only; timing excluded',backend_sha256=hashlib.sha256((R/'build/libneighborcache_backend.so').read_bytes()).hexdigest());(out/'command.json').write_text(json.dumps(cfg,indent=2))
 actual={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','NC_','PIPEANN_NMD','AISAQ_NMD'))};actual.update(env)
 print('START',e,flush=True)
 with (out/'run.out').open('w') as f:p=subprocess.run(cmd,env=actual,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 assert p.returncode==0,(e,p.returncode)
 log=(out/'run.out').read_text();assert 'mode=thread_private' in log
 m=re.search(r'NeighborCache shared backend: (.+)',log);assert m
 stats={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',m[1])}
 assert stats['M']>0 and stats['opens']>0 and stats['writes']>0 and stats['D_reads']>0 and stats['prefetch']>0,stats
 summary.append(dict(engine=e,full_record_verification=True,counters=stats));(R/'online_validation.json').write_text(json.dumps(summary,indent=2));print('PASS',e,stats,flush=True)
