"""Archive an unused index tree; never relocate the active SIFT100M index/D."""
from pathlib import Path
import hashlib,json,os,shutil,stat,subprocess,time
R=Path(os.environ.get('NC_LARGE_ROOT','/mnt/nvme3/nc_sift100m_uniform8mx10_20261101'));R.mkdir(parents=True,exist_ok=True)
src=Path('/mnt/nvme_single/sift1b');dst=Path('/mnt/nvme3/archived_from_nvme_single_20261101/sift1b');marker=R/'storage_move.json'
def opened_under(path):
 found=[];prefix=str(path)+'/'
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   cwd=os.readlink(p/'cwd')
   if cwd==str(path) or cwd.startswith(prefix):found.append((int(p.name),'cwd: '+cwd))
   for line in (p/'maps').read_text().splitlines():
    fields=line.split(maxsplit=5)
    if len(fields)==6 and (fields[5]==str(path) or fields[5].startswith(prefix)):
     found.append((int(p.name),'mmap: '+fields[5]))
   for fd in (p/'fd').iterdir():
    try:
     target=os.readlink(fd)
     if target==str(path) or target.startswith(prefix):found.append((int(p.name),target))
    except OSError:pass
  except OSError:pass
 return found

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while data:=f.read(8*1024*1024):h.update(data)
 return h.hexdigest()
if marker.exists() and src.is_symlink() and src.resolve()==dst.resolve():print('Archive already verified',flush=True)
else:
 assert src.is_dir() and not src.is_symlink() and not opened_under(src)
 assert not dst.exists()
 (R/'storage_plan.json').write_text(json.dumps(dict(source=str(src),destination=str(dst),active_index='/mnt/nvme_single/sift100m/sift100m_index',D_root='/mnt/nvme_single/sift100m/runs/nc_uniform8mx10_20261101',operation='Copy, verify every regular file with SHA-256, retain old path as explicit symlink, remove verified original copy only'),indent=2)+'\n')
 size=sum(p.lstat().st_blocks*512 for p in src.rglob('*') if p.is_file() and not p.is_symlink())
 assert shutil.disk_usage('/mnt/nvme3').free>size+80*2**30
 tmp=dst.with_name(dst.name+'.copying');tmp.mkdir(parents=True,exist_ok=True)
 with (R/'storage_copy.out').open('w') as f:subprocess.run(['rsync','-aHS','--info=progress2',str(src)+'/',str(tmp)+'/'],stdout=f,stderr=subprocess.STDOUT,check=True)
 checks=[]
 for p in sorted(src.rglob('*')):
  q=tmp/p.relative_to(src);s=p.lstat()
  if stat.S_ISLNK(s.st_mode):assert q.is_symlink() and os.readlink(p)==os.readlink(q)
  elif stat.S_ISREG(s.st_mode):
   assert q.is_file() and p.stat().st_size==q.stat().st_size
   h=digest(p);assert h==digest(q),p
   again=p.stat();assert (s.st_ino,s.st_mtime_ns,s.st_size)==(again.st_ino,again.st_mtime_ns,again.st_size)
   checks.append(dict(relative_path=str(p.relative_to(src)),bytes=s.st_size,sha256=h))
 assert not opened_under(src)
 (R/'storage_verified.json').write_text(json.dumps(dict(source=str(src),destination=str(dst),files=checks),indent=2)+'\n')
 os.rename(tmp,dst)
 backup=src.with_name(src.name+'.verified_archive_removal')
 assert not backup.exists();os.rename(src,backup)
 try:src.symlink_to(dst,target_is_directory=True)
 except BaseException:os.rename(backup,src);raise
 marker.write_text(json.dumps(dict(source=str(src),destination=str(dst),old_path='Explicit symlink to verified archive',files=checks,active_SIFT100M='/mnt/nvme_single/sift100m/sift100m_index',D_root='/mnt/nvme_single/sift100m/runs/nc_uniform8mx10_20261101',time=time.time()),indent=2)+'\n')
 shutil.rmtree(backup)
print('Original SSD free GiB:',shutil.disk_usage('/mnt/nvme_single').free/2**30,flush=True)
base=Path('/mnt/nvme_single/sift100m/sift100m_index_disk.index');d=Path('/mnt/nvme_single/sift100m/runs/nc_uniform8mx10_20261101');d.mkdir(exist_ok=True)
assert base.stat().st_dev==d.stat().st_dev
assert shutil.disk_usage(d).free>390*2**30
