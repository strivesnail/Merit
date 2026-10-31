from pathlib import Path
import ctypes,os,time
import numpy as np
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
