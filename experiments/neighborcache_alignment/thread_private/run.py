"""One shared/private comparison per existing engine configuration; raw outputs retained."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,re,signal,subprocess,time
import numpy as np
P=argparse.ArgumentParser();P.add_argument('--root',type=Path,default=Path('/mnt/nvme2/nc_private_adapters_20261028'));P.add_argument('--engines',nargs='+',default=['diskann','starling','margo','pipeann','aisaq']);A=P.parse_args();R=A.root
R.mkdir(parents=True,exist_ok=True);(R/'runs').mkdir(exist_ok=True)
BACKEND=R/'build/libneighborcache_backend.so'
lib=ctypes.CDLL(None,use_errno=True)
lib.mmap.restype=ctypes.c_void_p;lib.mmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long]
lib.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];lib.munmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
def residency(path):
    fd=os.open(path,os.O_RDONLY);size=os.fstat(fd).st_size
    if not size:os.close(fd);return 0
    address=lib.mmap(None,size,1,1,fd,0)
    if address==ctypes.c_void_p(-1).value:raise OSError(ctypes.get_errno(),'mmap')
    vec=(ctypes.c_ubyte*((size+4095)//4096))()
    ret=lib.mincore(address,size,vec)
    if ret:raise OSError(ctypes.get_errno(),'mincore')
    count=int(np.count_nonzero(np.frombuffer(vec,dtype=np.uint8)&1))
    lib.munmap(address,size);os.close(fd)
    return count
def clear_files(files):
    start=time.time();os.sync();records=[]
    for p in sorted(set(str(Path(p).resolve()) for p in files)):
        before=residency(p)
        for attempt in range(3):
            fd=os.open(p,os.O_RDONLY)
            os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);os.close(fd)
            after=residency(p)
            if after==0:break
            os.sync()
        records.append(dict(path=p,bytes=Path(p).stat().st_size,before_pages=before,after_pages=after))
        assert after==0,f'File cache did not clear: {p}: {after} pages'
    return dict(method='sync + per-file POSIX_FADV_DONTNEED + mincore verification; not global drop_caches or device cache reset',seconds=time.time()-start,files=records)

def prefixed_files(prefix):
    p=Path(prefix)
    return [f for f in p.parent.glob(p.name+'*') if f.is_file() and '.nc.' not in f.name and 'neighborcache' not in f.name]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
os.sched_setaffinity(0,{22,23});active=None

def stop(sig,frame):
 if active and active.poll() is None:active.terminate();active.wait(timeout=30)
 raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def config(e,mode,out):
 if e in ('starling','margo'):
  dataset='SIFT100M';work='10K+Zipf'
 elif e=='diskann':
  dataset='SIFT100M';work='10K+Zipf'
 else:
  dataset='SIFT10M' if e=='pipeann' else 'SIFT100K';work='10K public' if e=='pipeann' else '1K public'
 src=Path(__file__).resolve().parent/'configurations'/f'{e}.json'
 c=json.loads(src.read_text());cmd=c['command'].copy();env=c['environment'].copy();threads=8 if e=='aisaq' else 20
 for k in list(env):
  if any(t in k for t in ['VERIFY','PROFILE','TIMELINE']) or k=='PIPEANN_DUMP':del env[k]
 env.update(NEIGHBORCACHE_BACKEND_LIBRARY=str(BACKEND),NEIGHBORCACHE_THREADS=str(threads),NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE=str(int(mode=='private')),NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY='0',NEIGHBORCACHE_FREEZE_MAINTENANCE='1',NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=str(threads*128),OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',NEIGHBORCACHE_BACKEND_REPORT='1')
 runtime=None
 if e=='pipeann':
  query,gt=cmd[8],cmd[9];env.update(PIPEANN_NMD_PATH=str(out/'cache'),PIPEANN_DUMP=str(out/'ids'));files=prefixed_files(cmd[5]);runtime=out/'cache_neighborcache_dc.dyn.data'
 else:
  query=cmd[cmd.index('--query_file')+1];gt=cmd[cmd.index('--gt_file')+1];cmd[cmd.index('--result_path')+1]=str(out/'result');files=prefixed_files(cmd[cmd.index('--index_path_prefix')+1])
  if e=='aisaq':env['AISAQ_NMD_PATH']=str(out/'cache');runtime=out/'cache_neighborcache_dc.dyn.data'
  if e in ('starling','margo'):
   disk=Path(cmd[cmd.index('--disk_file_path')+1]).resolve();alias=Path('/mnt/nvme_single/sift100m/runs/thread_private_20261028')/f'{e}_{mode}'/disk.name;alias.parent.mkdir(parents=True,exist_ok=True)
   if not alias.exists():alias.symlink_to(disk)
   cmd[cmd.index('--disk_file_path')+1]=str(alias);runtime=Path(str(alias)+'.nc.dyn_neighborcache_dc.dyn.data')
   files+=prefixed_files(cmd[cmd.index('--mem_index_path')+1])
 if '--neighborcache_disk_cache_reuse_prefix' in cmd:files+=prefixed_files(cmd[cmd.index('--neighborcache_disk_cache_reuse_prefix')+1])
 if 'NEIGHBORCACHE_HOT_PREFILL_PREFIX' in env:files += [Path(env['NEIGHBORCACHE_HOT_PREFILL_PREFIX']+x) for x in ['.pages','.slots']]
 files += [Path(query),Path(gt)];files=[str(p) for p in files if p.exists()]
 nq,dim=np.fromfile(query,dtype='<u4',count=2)
 binary=next(x for x in cmd if x.endswith('/search_disk_index'))
 return dict(engine=e,mode=mode,dataset=dataset,workload=work,threads=threads,queries=int(nq),total_prefetch_pages=threads*128,private_pages_per_thread=128,command=cmd,environment=env,source_configuration=str(src),binary_sha256=sha(binary),backend_sha256=sha(BACKEND),clear_files=files,gt=gt,runtime_d=str(runtime) if runtime else None)

def read_ids(p):
 b=np.fromfile(p,dtype='<u4');n,k=map(int,b[:2]);assert len(b)==2+n*k;return b[2:].reshape(n,k)
def recall(a,gt):
 with open(gt,'rb') as f:
  n,k=map(int,np.fromfile(f,dtype='<u4',count=2));truth=np.fromfile(f,dtype='<u4',count=n*k).reshape(n,k);dist=np.fromfile(f,dtype='<f4',count=n*k)
 assert n==len(a) and k>=10
 if len(dist)==n*k:eligible=dist.reshape(n,k)<=dist.reshape(n,k)[:,9:10]
 else:eligible=np.tile(np.arange(k)<10,(n,1))
 hit=0
 for start in range(0,n,500):hit+=int(((a[start:start+500,:,None]==truth[start:start+500,None,:])&eligible[start:start+500,None,:]).any(axis=2).sum())
 return hit/(n*10)*100

def main():
 global active
 rows=[]
 for engine in A.engines:
  for mode in ['shared','private']:
   out=R/'runs'/f'{engine}_{mode}';out.mkdir(exist_ok=False);c=config(engine,mode,out)
   write(out/'command.json',c);print('CLEAR',engine,mode,flush=True);write(out/'cache_clear.json',clear_files(c['clear_files']))
   env={k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_','NC_','AISAQ_NMD','PIPEANN_NMD','MERIT_','LD_PRELOAD'))};env.update(c['environment'])
   print('START',engine,mode,flush=True)
   with (out/'run.out').open('w') as f:
    active=subprocess.Popen(c['command'],env=env,stdout=f,stderr=subprocess.STDOUT);code=active.wait(timeout=1200);active=None
   c['exit_code']=code;write(out/'command.json',c);assert code==0,out
   log=(out/'run.out').read_text();L=64 if engine=='aisaq' else 128;m=re.findall(r'^\s*'+str(L)+r'\s+4\s+(.+)$',log,re.M);assert len(m)==1
   fields=m[0].split();idfile=out/'ids.L128' if engine=='pipeann' else next(out.glob('result*idx_uint32.bin'));ids=read_ids(idfile);assert ids.shape==(c['queries'],10)
   assert not np.any(np.diff(np.sort(ids,axis=1),axis=1)==0)
   row=dict(engine=engine,mode=mode,dataset=c['dataset'],workload=c['workload'],threads=c['threads'],queries=c['queries'],qps=float(fields[0]),recall_log=float(fields[-1]),recall_exact=recall(ids,c['gt']),reported_IOs_per_query=float(fields[4] if engine=='pipeann' else fields[3]),total_prefetch_pages=c['total_prefetch_pages'],idfile=str(idfile),result_sha256=sha(idfile))
   if engine!='diskann':
    m=re.search(r'NeighborCache prefetch policy: mode=(\w+) total_pages=(\d+) query_threads=(\d+)',log);assert m and m[1]==('thread_private' if mode=='private' else 'shared') and int(m[2])==c['total_prefetch_pages'] and int(m[3])==c['threads']
    m=re.search(r'NeighborCache shared backend: (.+)',log);assert m;stats={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',m[1])};assert all(stats[k]==0 for k in ['M','opens','appends','replaces','writes']);row['cache_stats']=stats
    row['D_reads_per_query']=stats['D_reads']/c['queries'];row['prefetch_hits_per_query']=stats['prefetch']/c['queries']
   else:
    assert ('thread-private LRUs' if mode=='private' else 'shared by all threads') in log
   rows.append(row);write(out/'measurement.json',row);write(R/'results_partial.json',rows)
   print('DONE',engine,mode,'QPS',row['qps'],'Recall',row['recall_exact'],flush=True)
   if c['runtime_d']:
    runtime=Path(c['runtime_d']);
    if runtime.exists():runtime.unlink() # regenerated per run; immutable source remains
 write(R/'results.json',rows);print('COMPLETE',flush=True)

if __name__=="__main__":main()
