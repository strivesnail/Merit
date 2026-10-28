from pathlib import Path
import argparse,json,subprocess,shutil,hashlib,difflib
S=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--base',type=Path,default=Path('/mnt/nvme2/nc_query_reset_20261028'))
parser.add_argument('--output',type=Path,default=Path('/tmp/nc_fullseed_thread_private_build'))
args=parser.parse_args();B=args.base.resolve();R=args.output.resolve()
R.mkdir(parents=True,exist_ok=False)
# Historical experiment build requires the recorded query-reset snapshot.
for name in ['pq_flash_index.cpp','search_disk_index.cpp']:
 shutil.copy2(B/name,R/name)
shutil.copy2(S/'full_seed_prefetch.h',R/'full_seed_prefetch.h')
subprocess.run(['patch','-p1','-i',str(S/'search_disk_index.cpp.patch')],cwd=R,check=True)
for name,source,obj in [('compile_library.json','pq_flash_index.cpp','pq_flash_index.cpp.o'),('compile_driver.json','search_disk_index.cpp','search_disk_index.o')]:
 cmd=json.loads((B/name).read_text());cmd[cmd.index('-c')+1]=str(R/source);cmd[cmd.index('-o')+1]=str(R/obj)
 (R/name).write_text(json.dumps(cmd,indent=2));subprocess.run(cmd,check=True)
shutil.copy2(B/'libdiskann.a',R/'libdiskann.a');subprocess.run(['ar','r',str(R/'libdiskann.a'),str(R/'pq_flash_index.cpp.o')],check=True)
cmd=json.loads((B/'link.json').read_text());cmd=[str(R/'libdiskann.a') if v.endswith('/libdiskann.a') else str(R/'search_disk_index.o') if v.endswith('/search_disk_index.o') else v for v in cmd];cmd[cmd.index('-o')+1]=str(R/'search_disk_index')
(R/'link.json').write_text(json.dumps(cmd,indent=2));subprocess.run(cmd,check=True)
for name in ['full_seed_prefetch.h','search_disk_index.cpp']:
 (R/(name+'.patch')).write_text(''.join(difflib.unified_diff((B/name).read_text().splitlines(True),(R/name).read_text().splitlines(True),fromfile='previous/'+name,tofile='query_reset/'+name)))
(R/'source_manifest.json').write_text(json.dumps({name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['full_seed_prefetch.h','search_disk_index.cpp','pq_flash_index.cpp','search_disk_index','libdiskann.a']},indent=2))
print('BUILD COMPLETE',flush=True)
