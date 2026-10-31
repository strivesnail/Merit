"""Add measured N-only points at approximately uniform cache-RAM budgets.

Capacities are selected before reading the new QPS values, by inverting the
existing measured N-node-count versus additional-RSS curve. Actual RSS remains
an observation and is not replaced by the requested memory grid.
"""
from pathlib import Path
import json,os,statistics as st,subprocess,sys
import numpy as np
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'experiments/neighborcache_alignment/n_capacity_match'))
import run as bench
R=Path('/mnt/nvme2/nc_fullvisited_seed_single_20261031');D=R/'nonly_dense';D.mkdir(exist_ok=True)
os.sched_setaffinity(0,{23})
baseline=json.loads((bench.DEFAULT_ROOT/'nonly_0_0/result.json').read_text())['extra_rss_mib']
plan_path=D/'plan.json'
if plan_path.exists():plan=json.loads(plan_path.read_text())
else:
 groups={}
 for p in list(bench.DEFAULT_ROOT.glob('nonly_*/result.json'))+list((R/'nonly_match').glob('nonly_*/result.json')):
  row=json.loads(p.read_text());groups.setdefault(row['nodes'],[]).append(row['extra_rss_mib']-baseline)
 ref=sorted((st.mean(values),nodes) for nodes,values in groups.items())
 memory=np.array([x[0] for x in ref]);nodes=np.array([x[1] for x in ref])
 points=[]
 for target in range(40,481,40):
  if target>memory[-1]:requested=nodes[-1]+(target-memory[-1])*(nodes[-1]-nodes[-2])/(memory[-1]-memory[-2])
  else:requested=np.interp(target,memory,nodes)
  capacity=max(1024,int(round(requested/1024))*1024)
  points.append(dict(target_memory_mib=target,n_nodes=capacity))
 plan=dict(rule='40 MiB nominal spacing; N capacity from inverse prior measured RAM curve, rounded to 1024 nodes; actual RAM/QPS are measured, no synthetic points',repeats=1,query_threads=1,warmup_queries=20000,measured_queries=20000,reference_node_memory=ref,points=points,keep_original_points=True)
 bench.write(plan_path,plan)
print('PLAN',json.dumps(plan),flush=True)
results=[]
for spec in plan['points']:
 row=bench.mean_point(D,'nonly',spec['n_nodes'])
 row['target_memory_mib']=spec['target_memory_mib'];row['cache_rss_mib']=row['extra_rss_mib']-baseline
 results.append(row);bench.write(D/'grid_results.json',results)
 print('DENSE_POINT',json.dumps(row),flush=True)
 subprocess.run(['/home/jianz/miniconda3/bin/python',str(REPO/'experiments/neighborcache_alignment/fullvisited_single/report.py')],stdout=subprocess.DEVNULL,check=True)
print('COMPLETE',flush=True)
