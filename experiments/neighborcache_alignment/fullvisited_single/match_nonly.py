from pathlib import Path
import json,sys,os,statistics as st
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'experiments/neighborcache_alignment/n_capacity_match'))
import run as bench
R=Path('/mnt/nvme2/nc_fullvisited_seed_single_20261031');out=R/'nonly_match';out.mkdir(exist_ok=True)
os.sched_setaffinity(0,{23})
full=json.loads((R/'results.json').read_text());target=st.mean(x['qps'] for x in full)
bench.write(out/'protocol.json',dict(target_qps=target,queries=20000,warmup_queries=20000,query_threads=1,seed_scope='N-only remains online LRU; full-D comparison uses same-workload oracle',start_nodes=524288,maximum_bisections=2,selected_match_repeats=2))
low=65536;high=524288
while bench.mean_point(out,'nonly',high)['qps']<target:
 low=high;high*=2
 if high>8388608:raise RuntimeError('No crossing')
for _ in range(2):
 mid=(low+high)//2
 if bench.mean_point(out,'nonly',mid)['qps']>=target:high=mid
 else:low=mid
match=bench.mean_point(out,'nonly',high,2)
while match['qps']<target:
 high+=65536;match=bench.mean_point(out,'nonly',high,2)
baseline=json.loads((bench.DEFAULT_ROOT/'nonly_0_0/result.json').read_text())
full_rss=st.mean(x['extra_rss_mib'] for x in full)-baseline['extra_rss_mib']
n_rss=match['extra_rss_mib']-baseline['extra_rss_mib']
bench.write(R/'nonly_match.json',dict(target_qps=target,full_d_extra_rss_mib=full_rss,nonly_match=match,nonly_extra_rss_mib=n_rss,n_only_over_full_d_ram=n_rss/full_rss,lower_tested_nodes=low,upper_matched_nodes=high,resolution_nodes=high-low,baseline_common_query_rss_mib=baseline['extra_rss_mib'],scope='Finite warmed workload, offline full-visited D versus online N-only; no held-out/oracle-free claim'))
print('MATCH',json.dumps(json.loads((R/'nonly_match.json').read_text())),flush=True)
