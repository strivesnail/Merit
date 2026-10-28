from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
rows=json.loads((R/'results.json').read_text());base=next(x for x in rows if x['mode']=='shared');private=next(x for x in rows if x['mode']=='private_reset')
reference=np.genfromtxt('/mnt/nvme2/nc_query_reset_103pages_20261028/runs/reset_p103_0/query_stats.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
summary=[];checks={}
for x in rows:
 d=R/'runs'/f"{x['mode']}_p2060_0";q=np.genfromtxt(d/'query_stats.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
 c=json.loads((d/'cache_clear.json').read_text());assert all(f['after_pages']==0 for f in c['files'])
 checks[x['mode']]=dict(result_ids_match=x['result_ids_match_reference'],file_residency_zero_before_run=True,reset_calls=x['reset_calls'],query_threads=20)
 if x['mode']=='private_reset':
  matches={f:np.array_equal(q[f],reference[f]) for f in ['n_beam_lookups','n_neighborcache_prefetch_hits','n_disk_reads']};assert all(matches.values());checks[x['mode']]['same_per_query_counters_as_1thread']=matches
  assert int(q['n_disk_reads'].max())==103
  assert x['reset_calls']==40000
 else:assert x['reset_calls']==0
 summary.append(dict(mode=x['mode'],total_prefetch_KiB=8240,private_prefetch_KiB=412 if x['thread_private'] else None,qps=x['qps'],hit_pct=x['metrics']['memory_hit_pct'],D_reads_per_query=x['metrics']['disk_reads_per_query'],mean_latency_us=x['metrics']['latency_us_per_query'],mean_IO_us=x['metrics']['io_us_per_query'],reset_us_per_query=x['reset_us_per_query'],result_ids_match=x['result_ids_match_reference']))
qps_change=100*(private['qps']/base['qps']-1);read_change=100*(private['metrics']['disk_reads_per_query']/base['metrics']['disk_reads_per_query']-1)
result=dict(rows=summary,qps_change_pct=qps_change,D_read_change_pct=read_change,hit_change_percentage_points=private['metrics']['memory_hit_pct']-base['metrics']['memory_hit_pct'],validation=checks)
(R/'summary.json').write_text(json.dumps(result,indent=2))
with (R/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
(R/'validation.json').write_text(json.dumps(checks,indent=2))
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':13})
fig,ax=plt.subplots(figsize=(7,4.5));vals=[base['qps'],private['qps']]
ax.bar([0,1],vals,color=['#0072b2','#009e73'],width=.55)
ax.set_xticks([0,1],['Shared prefetch','Thread-private prefetch\nClear per query']);ax.set_ylabel('QPS');ax.set_title('SIFT100M · 20 threads');ax.set_ylim(0,max(vals)*1.18)
for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:,.0f}',ha='center')
ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(R/f'sift100m_thread_private_qps.{ext}',dpi=200)
lines=['# Thread-private prefetch experiment','', 'SIFT100M A→B 40K queries (10K uniform + 10K Zipf0.8 shuffled per phase), K=10 L=128 beam=4, 20 workers pinned to CPU 0–19. One run per mode, with sync + file-scoped POSIX_FADV_DONTNEED and mincore verification before each run. N=0 M=0. D is immutable, with all 2,632,133 profiled unique visited nodes as seeds.','', 'Both modes have exactly 2,060 pages = 8,240 KiB = 8.046875 MiB of prefetch payload. Shared mode uses 128 shard LRUs, 16 or 17 pages each, and retains pages across queries. Private mode has one lock-free cache and LRU per worker, exactly 103 pages = 412 KiB per worker, and clears only that worker’s cache before each query. Resident-node owner directories are private; the immutable D data and seed map remain shared. Metadata and the rest of the index are outside this payload budget.','', '| Mode | QPS | Hit % | D reads/query | Mean latency µs | IO µs/query | Reset µs/query |','|---|---:|---:|---:|---:|---:|---:|']
for x in summary:lines.append(f"| {x['mode']} | {x['qps']:.2f} | {x['hit_pct']:.2f} | {x['D_reads_per_query']:.2f} | {x['mean_latency_us']:.2f} | {x['mean_IO_us']:.2f} | {x['reset_us_per_query']:.2f} |")
lines += ['',f'Private-reset QPS change: {qps_change:+.2f}%; D reads/query change: {read_change:+.2f}%.', '', 'Overall wall-clock QPS includes query resets. Per-query search latency excludes reset. Reset time is summed across workers; it must not be divided by wall time and interpreted as a serial overhead fraction.', '', 'Private-reset per-query visits, hits and reads exactly match the single-thread 103-page reset run. Maximum loads/query = 103. All 40K×10 result IDs match the existing reference in both modes; this is not a new ground-truth recall measurement.', '', 'This comparison changes cache ownership, number of LRUs, locking and query-boundary retention together. It evaluates the requested combined design; it cannot isolate a single cause such as lock contention. Dynamic query scheduling also changes which worker receives each query. One run per configuration does not provide confidence intervals.', '', 'Unit tests: thread isolation even for identical record IDs, clearing one worker while another retains its page, replica cleanup during LRU eviction, reinsert after reset, exact capacity allocation/remainder, 20-worker stress, insufficient capacity rejection. Existing shared-cache LRU, remainder, capacity-clamp and concurrent tests also pass.', '', 'Artifacts: method.json, summary.json, validation.json, comparison.csv; raw command, cache-clear evidence, output, per-query statistics, reference result IDs and timeline under runs/.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n');(R/'STATUS.md').write_text('Complete: shared and private-reset 20-thread runs; all result and per-query counter validations passed. No benchmark remains running.\n')
print(json.dumps(result,indent=2))
