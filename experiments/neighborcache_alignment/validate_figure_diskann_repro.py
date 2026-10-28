from pathlib import Path
import hashlib,json,re
import numpy as np
R=Path('/mnt/nvme2/sift100m_figure_repro_20261027')
configs=json.loads((R/'configurations.json').read_text());out=[];binary_hashes=set()
def read(p):
 with p.open('rb') as f:
  n,k=np.fromfile(f,np.uint32,2);a=np.fromfile(f,np.uint32);assert a.size==int(n)*int(k);return a.reshape(int(n),int(k))
for c in configs:
 old_path=Path(c['source_log']).parent/'result_128_idx_uint32.bin';old=read(old_path)
 for rep in range(3):
  d=R/f"{c['workload']}_{c['mode']}_{rep}";cfg=json.loads((d/'command.json').read_text());assert cfg['exit_code']==0
  binary_hashes.add(cfg['binary_sha256']);assert cfg['environment']==c['environment']
  assert not any('VERIFY' in k or 'PROFILE' in k for k in cfg['environment'])
  arr=read(d/'result_128_idx_uint32.bin');assert arr.shape==old.shape
  out.append(dict(workload=c['workload'],mode=c['mode'],rep=rep,queries=len(arr),different_rows=int(np.any(arr!=old,axis=1).sum()),different_top10_sets=int(np.any(np.sort(arr,axis=1)!=np.sort(old,axis=1),axis=1).sum())))
assert len(binary_hashes)==1
sha=hashlib.sha256(Path('/home/jianz/workload/code/Merit/diskann/build/apps/search_disk_index').read_bytes()).hexdigest();assert binary_hashes=={sha}
report=dict(runs=27,binary_sha256=sha,identical_configuration=True,verification_and_profiling_disabled=True,output_comparison=out)
(R/'VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
for w in ['pub','x5','zipf']:
 for m in ['diskann','neighbor','bfs']:
  group=[r for r in out if (r['workload'],r['mode'])==(w,m)];print(w,m,'different rows', [r['different_rows'] for r in group],'different sets',[r['different_top10_sets'] for r in group])
