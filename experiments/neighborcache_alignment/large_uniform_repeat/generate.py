"""Distinct perturbed SIFT100M queries, each repeated ten times, global shuffle."""
from pathlib import Path
import json,os,struct,hashlib
import numpy as np
R=Path(os.environ.get('NC_LARGE_ROOT','/mnt/nvme3/nc_sift100m_uniform8mx10_20261101'));W=R/'workload';W.mkdir(parents=True,exist_ok=True)
N=int(os.environ.get('NC_LARGE_DISTINCT','8000000'));REPS=10;TOTAL=N*REPS;DIM=128
base=Path('/mnt/nvme_single/sift100m/sift100m_index_disk.index');assert TOTAL<=np.iinfo(np.uint32).max
def hashfile(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 return h.hexdigest()
marker=R/'workload_manifest.json'
if marker.exists():
 v=json.loads(marker.read_text());assert v['distinct_queries']==N and v['total_queries']==TOTAL;print('Workload already generated',flush=True);raise SystemExit
rng=np.random.default_rng(20261102)
rows=rng.choice(100000000,size=N,replace=False).astype(np.uint32);np.save(W/'source_node_ids.npy',rows)
raw=np.memmap(base,mode='r',dtype=np.uint8)
records=np.ndarray(shape=(10000000,10,128),dtype=np.uint8,buffer=raw,offset=4096,strides=(4096,388,1))
source=W/f'uniform{N}_synthetic.u8bin'
with source.open('wb') as f:f.write(struct.pack('<II',N,DIM));f.truncate(8+N*DIM)
queries=np.memmap(source,mode='r+',dtype=np.uint8,offset=8,shape=(N,DIM))
unchanged=0
for start in range(0,N,32768):
 ids=rows[start:start+32768];original=records[ids//10,ids%10,:]
 out=np.clip(np.rint(original.astype(np.float32)+rng.normal(0,1,original.shape)),0,255).astype(np.uint8)
 same=np.all(out==original,axis=1)
 # This branch is practically absent, but the construction explicitly rules it out.
 if np.any(same):
  which=np.flatnonzero(same)
  out[which,0]=np.where(original[which,0]<255,original[which,0]+1,original[which,0]-1)
 unchanged+=int(np.all(out==original,axis=1).sum());queries[start:start+len(ids)]=out
 if start%1048576==0:print('Generated',start,'/',N,flush=True)
queries.flush();assert unchanged==0
order=np.tile(np.arange(N,dtype=np.uint32),REPS);np.random.default_rng(20261103).shuffle(order);np.save(W/'workload_query_ids.npy',order)
counts=np.bincount(order,minlength=N);assert len(counts)==N and np.all(counts==REPS)
full=W/f'uniform{N}x10_shuffled.u8bin'
with full.open('wb') as f:
 f.write(struct.pack('<II',TOTAL,DIM))
 for start in range(0,TOTAL,65536):f.write(queries[order[start:start+65536]].tobytes())
for n in [20000,200000,2000000]:
 if n>TOTAL:continue
 with full.open('rb') as f:f.seek(8);(W/f'prefix{n}.u8bin').write_bytes(struct.pack('<II',n,DIM)+f.read(n*DIM))
# Exact duplicate detection in the generated source and membership against all
# indexed full vectors are performed by the existing independently tested C++
# verifier, before this experiment is allowed to start.
v=dict(dataset='SIFT100M',distinct_queries=N,total_queries=TOTAL,repetitions_each=10,globally_shuffled=True,zipf=False,source='SIFT100M indexed vectors sampled uniformly without replacement, then independent Gaussian std=1 perturbation rounded/clipped to uint8',query_file=str(full),source_query_file=str(source),source_node_ids=str(W/'source_node_ids.npy'),workload_ids=str(W/'workload_query_ids.npy'),generation_seed=20261102,shuffle_seed=20261103,unchanged_vs_sampled_source=unchanged,each_source_repeated_exactly_10=True,source_sha256=hashfile(source),query_sha256=hashfile(full),index_membership_status='pending exact full-index verification')
marker.write_text(json.dumps(v,indent=2)+'\n');print('Generation complete',flush=True)
