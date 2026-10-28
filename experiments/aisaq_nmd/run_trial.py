"""Small SIFT trial; all NMD runs cold, no D prefill, no warmup."""
from pathlib import Path
import argparse,os,json,re,subprocess,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/mnt/nvme2/aisaq_nmd_trial_20261027'));p.add_argument('--binary',type=Path,default=Path('/tmp/nmd-candidates-aisaq/build/apps/search_disk_index'));a=p.parse_args()
r=a.root
with (r/'query.u8bin').open('rb') as f:
 n,d=np.fromfile(f,dtype='<u4',count=2);q=np.fromfile(f,dtype='u1').reshape(n,d)
with (r/'query_x5.u8bin').open('wb') as f:
 np.array([n*5,d],dtype='<u4').tofile(f);np.tile(q,(5,1)).tofile(f)
with (r/'gt.bin').open('rb') as f:
 gn,gk=np.fromfile(f,dtype='<u4',count=2);ids=np.fromfile(f,dtype='<u4',count=int(gn*gk)).reshape(gn,gk);dist=np.fromfile(f,dtype='<f4').reshape(gn,gk)
with (r/'gt_x5.bin').open('wb') as f:
 np.array([gn*5,gk],dtype='<u4').tofile(f);np.tile(ids,(5,1)).tofile(f);np.tile(dist,(5,1)).tofile(f)
results=[]
for workload in ['public','x5']:
 for mode in ['baseline','nmd_verify','nmd']:
  if workload=='x5' and mode=='nmd_verify':continue
  out=r/f'{workload}_{mode}';out.mkdir(exist_ok=True)
  env={k:v for k,v in os.environ.items() if not k.startswith(('AISAQ_NMD','PIPEANN_NMD','NEIGHBORCACHE_'))}
  settings={'LD_LIBRARY_PATH':'/home/jianz/miniconda3/lib:/tmp/aisaq-nmd-deps/root/usr/lib/x86_64-linux-gnu','OMP_NUM_THREADS':'8','AISAQ_NMD':'0' if mode=='baseline' else '1','AISAQ_NMD_NMB':'1','AISAQ_NMD_PMB':'1','AISAQ_NMD_PREFILL':'0','AISAQ_NMD_PATH':str(out/'cache.nmd')}
  if mode=='nmd_verify':settings['AISAQ_NMD_VERIFY']='1'
  env.update(settings)
  cmd=[str(a.binary),'--data_type','uint8','--dist_fn','l2','--index_path_prefix',str(r/'index'),'--query_file',str(r/('query.u8bin' if workload=='public' else 'query_x5.u8bin')),'--gt_file',str(r/('gt.bin' if workload=='public' else 'gt_x5.bin')),'--result_path',str(out/'result'),'-K','10','-L','64','-W','4','-T','8','--num_nodes_to_cache','0','--use_aisaq','-V','1']
  (out/'command.json').write_text(json.dumps(dict(command=cmd,environment=settings,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest()),indent=2))
  print('START',workload,mode,flush=True)
  with (out/'run.out').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  log=(out/'run.out').read_text()
  row=re.findall(r'^\s*64\s+4\s+(.+)$',log,re.M)[-1].split();count=int(n)*(1 if workload=='public' else 5)
  dp=re.search(r'NMD demand D (?:pread|read) requests=(\d+)',log)
  result=dict(workload=workload,mode=mode,queries=count,qps=float(row[0]),mean_base_reads=float(row[3]),recall=float(row[-1]),d_pread_requests=int(dp[1]) if dp else 0)
  result['mean_total_demand_reads']=result['mean_base_reads']+result['d_pread_requests']/count
  if mode!='baseline':
   for suffix in ['idx_uint32.bin','dists_float.bin']:
    assert (out/f'result_64_{suffix}').read_bytes()==(r/f'{workload}_baseline'/f'result_64_{suffix}').read_bytes(),(workload,mode,suffix)
   result['exact_results_match_baseline']=True
  results.append(result);(out/'result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
(r/'summary.json').write_text(json.dumps(results,indent=2))
