from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
HERE=Path(__file__).resolve().parent
R=Path(os.environ.get('NC_LARGE_ROOT','/mnt/nvme3/nc_sift100m_uniform8mx10_20261101'));R.mkdir(parents=True,exist_ok=True)
SMALL=Path('/mnt/nvme2/nc_unbounded_same_disk_20261101')
os.sched_setaffinity(0,{20,21,22,23})
def status(phase,**extra):
 (R/'status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),coordinator_pid=os.getpid(),distinct_queries=8000000,repetitions=10,total_queries=80000000,D_device='/dev/nvme4n1',D_same_drive_as_SIFT100M=True,**extra),indent=2)+'\n')
def run(script,log):
 with (R/log).open('w') as f:subprocess.run(['/home/jianz/miniconda3/bin/python',str(HERE/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
def interrupted(signum,frame):
 status('interrupted',signal=signum)
 raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
try:
 status('waiting_for_current_2M_N_only_match')
 while not (SMALL/'synthetic/matched.json').exists():time.sleep(10)
 binary=R/'build/search_disk_index';validation=R/'stream_smoke_validation.json'
 sources=[HERE/'build_stream.py',HERE/'stream_query_search.h']
 verified=binary.exists() and validation.exists() and max(p.stat().st_mtime for p in sources)<=binary.stat().st_mtime<=validation.stat().st_mtime
 if not verified:
  status('building_and_verifying_streaming_driver');run('build_stream.py','build.out');run('smoke_stream.py','smoke.out')
 status('archiving_unused_SIFT1B_to_make_space_on_original_SSD');run('prepare_storage.py','prepare_storage.out')
 status('generating_8M_distinct_perturbed_queries_and_80M_shuffled_stream');run('generate.py','generate.out')
 status('verifying_all_generated_vectors_absent_from_full_100M_index')
 manifest=json.loads((R/'workload_manifest.json').read_text())
 with (R/'index_membership.json').open('w') as f:subprocess.run(['/mnt/nvme2/nc_sift100m_uniform200kx10_hitmatch_20261101/verify_outside','/mnt/nvme_single/sift100m/sift100m_index_disk.index',manifest['source_query_file'],'100000000'],stdout=f,check=True)
 member=json.loads((R/'index_membership.json').read_text());assert member['query_vectors']==8000000 and member['matching_query_count']==0
 manifest['index_membership_status']='exact complete-index verification passed; no duplicate generated query vectors';manifest['index_membership']=member
 (R/'workload_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 assert (R/'stream_smoke_validation.json').exists()
 status('running_80M_NC_then_measured_hit_matched_N_only')
 run('run_large.py','experiment.out')
 status('complete',figure=str(R/'sift100m_hit_matched_cache_size.png'))
except BaseException as err:
 status('failed_or_interrupted',reason=str(err),traceback=traceback.format_exc());raise
