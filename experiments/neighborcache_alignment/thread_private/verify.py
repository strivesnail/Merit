"""Full-record checking on 100 queries/adapter; never used for QPS reporting."""
import run as bench
from pathlib import Path
import json,subprocess,os
import numpy as np
R=bench.R
results=[]
for engine in ['starling','margo','pipeann','aisaq']:
 out=R/'verification'/engine;out.mkdir(parents=True,exist_ok=False)
 c=bench.config(engine,'private',out);cmd=c['command'];qpath=cmd[8] if engine=='pipeann' else cmd[cmd.index('--query_file')+1];gpath=c['gt']
 with open(qpath,'rb') as f:
  n,d=map(int,np.fromfile(f,dtype='<u4',count=2));body=f.read(100*d)
 (out/'query.u8bin').write_bytes(np.array([100,d],dtype='<u4').tobytes()+body)
 with open(gpath,'rb') as f:
  n,k=map(int,np.fromfile(f,dtype='<u4',count=2));ids=np.fromfile(f,dtype='<u4',count=n*k).reshape(n,k);dist=np.fromfile(f,dtype='<f4',count=n*k).reshape(n,k)
 (out/'gt.bin').write_bytes(np.array([100,k],dtype='<u4').tobytes()+ids[:100].tobytes()+dist[:100].tobytes())
 if engine=='pipeann':cmd[8]=str(out/'query.u8bin');cmd[9]=str(out/'gt.bin')
 else:cmd[cmd.index('--query_file')+1]=str(out/'query.u8bin');cmd[cmd.index('--gt_file')+1]=str(out/'gt.bin')
 c['queries']=100;c['purpose']='full-record verification only; QPS excluded';c['environment'].update(NEIGHBORCACHE_VERIFY='1',PIPEANN_NMD_VERIFY='1',AISAQ_NMD_VERIFY='1')
 bench.write(out/'command.json',c)
 env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','NC_','PIPEANN_NMD','AISAQ_NMD'))};env.update(c['environment'])
 print('VERIFY',engine,flush=True)
 with (out/'run.out').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 assert p.returncode==0,(engine,p.returncode)
 log=(out/'run.out').read_text();assert 'mode=thread_private' in log
 results.append(dict(engine=engine,queries=100,full_record_verification=True,exit_code=p.returncode))
 bench.write(R/'full_record_validation.json',results)
 if c['runtime_d'] and Path(c['runtime_d']).exists():Path(c['runtime_d']).unlink()
 print('PASS',engine,flush=True)
