"""Build one full canonical D page per hottest seed, independently of query timing.
Ranks nonzero entries of an engine's visit profile; ties use ascending node ID.
Offline membership is seed + graph-order unique neighbors, padded with the seed.
This is a workload-profile oracle experiment, not a held-out prediction result.
"""
import argparse,hashlib,json,math,mmap,struct,time
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--engine',required=True,choices=['pipeann','starling','margo','aisaq']);p.add_argument('--profile',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--partition',type=Path);p.add_argument('--record',type=int,required=True);p.add_argument('--degree',type=int,required=True);p.add_argument('--prefix',type=Path,required=True);a=p.parse_args()
 a.prefix.parent.mkdir(parents=True,exist_ok=True);start=time.time()
 with a.profile.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];counts=np.fromfile(f,'<u4')
 assert len(counts)==n
 visited=np.flatnonzero(counts).astype(np.uint32);seeds=visited[np.lexsort((visited,-counts[visited].astype(np.int64)))][:math.ceil(len(visited)*.2)]
 assert len(seeds)>0
 rec=a.record;degree=a.degree;nps=4096//rec;vector=128;tail=vector+4+4*degree;coords=rec-4-4*degree
 countbytes=2 if a.engine=='pipeann' else 4
 offsets=np.empty(n,dtype=np.uint64)
 if a.partition:
  with a.partition.open('rb') as f:
   cap,pages,pn=struct.unpack('<QQQ',f.read(24));assert cap==nps and pn==n
   seen=0
   for page in range(pages):
    k=struct.unpack('<I',f.read(4))[0];ids=np.frombuffer(f.read(k*4),dtype='<u4')
    assert k<=nps and len(ids)==k
    offsets[ids]=(page+1)*4096+np.arange(k,dtype=np.uint64)*rec;seen+=k
   assert seen==n
 else:
  ids=np.arange(n,dtype=np.uint64);offsets=(1+ids//nps)*4096+(ids%nps)*rec
 with a.index.open('rb') as inf, mmap.mmap(inf.fileno(),0,access=mmap.ACCESS_READ) as disk, open(str(a.prefix)+'.pages','wb') as pages,open(str(a.prefix)+'.slots','wb') as slots:
  slots.write(struct.pack('<8s6Q',b'NCHOT20\0',n,rec,vector,degree,nps,len(seeds)))
  def raw(nid):
   off=int(offsets[nid]);r=disk[off:off+rec];assert len(r)==rec;return r
  def canonical(r):
   nn=int.from_bytes(r[128:128+countbytes],'little');assert nn<=degree
   if countbytes==2:assert r[130:132]==b'\0\0','dense PipeANN record unsupported'
   b=bytearray(rec);b[:128]=r[:128];b[128:coords]=r[tail:]
   struct.pack_into('<I',b,coords,nn);b[coords+4:coords+4+4*nn]=r[132:132+4*nn]
   return b
  distinct=0
  for pos,sv in enumerate(seeds):
   seed=int(sv);sr=raw(seed);nn=int.from_bytes(sr[128:128+countbytes],'little');assert nn<=degree
   members=[seed];present={seed}
   for nid in struct.unpack_from(f'<{nn}I',sr,132):
    if nid<n and nid not in present:members.append(nid);present.add(nid)
    if len(members)==nps:break
   distinct+=len(members)
   members += [seed]*(nps-len(members));page=bytearray(4096)
   for j,nid in enumerate(members):page[j*rec:(j+1)*rec]=canonical(sr if nid==seed else raw(nid))
   pages.write(page);slots.write(struct.pack(f'<{nps}I',*members))
   if pos and pos%50000==0:print(a.engine,pos,'/',len(seeds),flush=True)
 meta=dict(engine=a.engine,points=int(n),visited_nodes=len(visited),seed_fraction=.2,seeds=len(seeds),pages=len(seeds),bytes=len(seeds)*4096,record_bytes=rec,vector_bytes=128,degree=degree,nodes_per_page=nps,nonpadding_slots=distinct,profile=str(a.profile),profile_sha256=sha(a.profile),index=str(a.index),partition=str(a.partition) if a.partition else None,seed_order='descending visit count, ascending ID on ties',membership='seed + unique graph neighbors in adjacency order; seed padding',n_and_prefetch_start='empty',profile_scope='same workload profile; not held out',seconds=time.time()-start)
 for suffix in ['pages','slots']:meta[suffix+'_sha256']=sha(str(a.prefix)+'.'+suffix)
 np.asarray(seeds,dtype='<u4').tofile(str(a.prefix)+'.seed_ids')
 Path(str(a.prefix)+'.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
