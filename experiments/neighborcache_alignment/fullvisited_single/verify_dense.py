"""Independent reruns to check slope/throughput stability after densification."""
from pathlib import Path
import json,os,statistics as st,subprocess,sys,time
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'experiments/neighborcache_alignment/n_capacity_match'))
import run as bench
import runpy
R=Path('/mnt/nvme2/nc_fullvisited_seed_single_20261031');D=R/'nonly_dense'
os.sched_setaffinity(0,{23})
recipe=dict(repeat_targets_mib=[80,240,360],runs_per_checked_capacity=2,
            nc_reference_control='one additional independent run, same 10 MiB P and N=2048',
            high_point_revision='Before any repeat result, use 360 instead of 400 MiB to independently verify the newly observed near-585-QPS crossing',
            purpose='Check repeatability of low/middle/high slopes; do not force curvature or smooth measured QPS')
bench.write(D/'verification_protocol.json',recipe)
while True:
 p=D/'grid_results.json';log=R/'nonly_dense.log'
 if p.exists() and len(json.loads(p.read_text()))==12 and log.exists() and log.read_text().splitlines()[-1]=='COMPLETE':break
 time.sleep(1)
plan=json.loads((D/'plan.json').read_text());base=json.loads((bench.DEFAULT_ROOT/'nonly_0_0/result.json').read_text())['extra_rss_mib']
checks=[]
for target in recipe['repeat_targets_mib']:
 spec=next(x for x in plan['points'] if x['target_memory_mib']==target)
 point=bench.mean_point(D,'nonly',spec['n_nodes'],2)
 point['target_memory_mib']=target;point['cache_rss_mib']=point['extra_rss_mib']-base
 point['qps_range_pct']=100*(max(point['qps_runs'])-min(point['qps_runs']))/point['qps']
 checks.append(point);bench.write(D/'verification_results.json',checks)
 print('REPEAT_CHECK',json.dumps(point),flush=True)
 subprocess.run(['/home/jianz/miniconda3/bin/python',str(REPO/'experiments/neighborcache_alignment/fullvisited_single/report.py')],stdout=subprocess.DEVNULL,check=True)
# Import the isolated full-D driver's functions without starting its default runs.
functions=runpy.run_path(str(REPO/'experiments/neighborcache_alignment/fullvisited_single/run.py'))
control=functions['run']('control_after_dense')
ref=json.loads((R/'results.json').read_text());reference_qps=st.mean(x['qps'] for x in ref)
bench.write(D/'nc_reference_control.json',dict(control=control,original_reference_qps=reference_qps,change_pct=100*(control['qps']/reference_qps-1)))
subprocess.run(['/home/jianz/miniconda3/bin/python',str(REPO/'experiments/neighborcache_alignment/fullvisited_single/report.py')],stdout=subprocess.DEVNULL,check=True)
print('COMPLETE',flush=True)
