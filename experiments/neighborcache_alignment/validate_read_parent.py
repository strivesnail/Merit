from pathlib import Path
import hashlib,json
R=Path('/mnt/nvme2/nc_read_parent_20261027')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
for engine in ['reference','starling','margo','aisaq']:
 a=R/'verify_reference_before' if engine=='reference' else R/f'verify_{engine}_hot_fixed'
 b=R/'verify_reference_after' if engine=='reference' else R/f'verified_static_{engine}'
 checks[engine]={}
 for p in a.glob('result*.bin'):
  q=b/p.name;assert q.exists()
  checks[engine][p.name]={'same':sha(p)==sha(q),'before_sha256':sha(p),'after_sha256':sha(q)}
  assert checks[engine][p.name]['same'],(engine,p.name)
records=[]
for p in R.glob('*_*/measurement.json'):
 if p.parent.name.startswith(('verify','verified')):continue
 d=json.loads(p.read_text());cfg=json.loads((p.parent/'command.json').read_text())
 assert not d['verification'];assert not any('VERIFY' in k or 'PROFILE' in k for k in cfg['environment'])
 if d['mode']=='after':
  assert d['cache_stats']['M']==d['recorded_parents']==0
  assert all(d['cache_stats'][k]==0 for k in ['opens','appends','replaces','writes'])
 records.append(str(p))
assert len(records)==72,len(records)
# Deterministic engines should return identical static results across the two builds.
static_ids={}
for engine,L in [('starling',128),('margo',128),('aisaq',64)]:
 for w in ['public','x5']:
  paths=[R/f'{engine}_{w}_{m}_{i}'/f'result_{L}_idx_uint32.bin' for m in ['before','after'] for i in range(3)]
  h=[sha(p) for p in paths];static_ids[f'{engine}_{w}']=len(set(h))==1
ctest=(R/'ctest_static.out').read_text();assert '100% tests passed' in ctest
v={'ctest':'8/8 passed','verified_static_full_records':['pipeann','starling','margo','aisaq'],
   'record_output_checks':checks,'static_output_ids_identical':static_ids,'timing_runs':len(records),
   'static_after_M_and_parents_zero':True,'static_D_mutations_zero':True,
   'default_backend_sha256':sha(Path('neighborcache_backend/build/libneighborcache_backend.so')),
   'measured_backend_sha256':sha(R/'libneighborcache_backend.so')}
assert v['default_backend_sha256']==v['measured_backend_sha256']
(R/'VALIDATION.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
