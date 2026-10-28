from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
R=Path(__file__).resolve().parent
runs=json.loads((R/'results.json').read_text());rows=[];checks={}
assert len(runs)==10
ref=np.genfromtxt('/mnt/nvme2/nc_query_reset_103pages_20261028/runs/reset_p103_0/query_stats.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
for x in runs:
 p=x['prefetch_pages'];d=R/'runs'/f'private_retain_p{p}_0';q=np.genfromtxt(d/'query_stats.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
 c=json.loads((d/'cache_clear.json').read_text());assert all(f['after_pages']==0 for f in c['files'])
 assert x['reset_calls']==0 and not x['reset_each_query'] and x['thread_private']
 assert x['result_ids_match_reference'] and np.array_equal(q['n_beam_lookups'],ref['n_beam_lookups'])
 assert len(q)==40000 and np.all(q['n_mem_hits']==0)
 assert np.all(q['n_neighborcache_prefetch_hits']+q['n_disk_reads']==q['n_beam_lookups'])
 checks[str(p)]=dict(result_ids_match=True,per_query_visits_match=True,query_reset_calls=0,file_residency_zero_before_run=True,hit_read_accounting_valid=True)
 rows.append(dict(per_thread_pages=p//20,per_thread_node_record_slots=p//20*10,per_thread_MiB=p/20/256,total_prefetch_MiB=p/256,qps=x['qps'],prefetch_hit_pct=x['metrics']['memory_hit_pct'],D_reads_per_query=x['metrics']['disk_reads_per_query'],IO_us_per_query=x['metrics']['io_us_per_query'],latency_us_per_query=x['metrics']['latency_us_per_query'],phase_A_qps=x['phases'][0]['qps'],phase_B_qps=x['phases'][1]['qps'],phase_A_hit_pct=x['phases'][0]['memory_hit_pct'],phase_B_hit_pct=x['phases'][1]['memory_hit_pct']))
rows.sort(key=lambda x:x['per_thread_pages'])
with (R/'capacity_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
summary=dict(capacities=rows,qps_change_first_to_last_pct=100*(rows[-1]['qps']/rows[0]['qps']-1),hit_change_first_to_last_percentage_points=rows[-1]['prefetch_hit_pct']-rows[0]['prefetch_hit_pct'],best_measured_qps=max(rows,key=lambda x:x['qps']),repetitions=1,hit_definition='Number of full target-node accesses served from this worker prefetch / all full target-node accesses; insertion-causing demand is a miss. N=0; D disk reads are not memory hits.',x_axis='Node-record capacity per thread = page capacity times 10. K = 1000. Nominal slots, not distinct resident nodes; partially populated pages and replicated nodes can reduce occupancy. Total slot capacity across workers is 20 times this value.')
(R/'summary.json').write_text(json.dumps(summary,indent=2));(R/'validation.json').write_text(json.dumps(checks,indent=2))
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':13})
fig,(ax,bx)=plt.subplots(2,1,figsize=(14,7),sharex=True)
x=[r['per_thread_node_record_slots'] for r in rows]
ax.plot(x,[r['qps'] for r in rows],'-o',color='#009e73',linewidth=2,markersize=5)
bx.plot(x,[r['prefetch_hit_pct'] for r in rows],'-o',color='#0072b2',linewidth=2,markersize=5)
ax.set_title('SIFT100M · 20 threads · Private prefetch');ax.set_ylabel('QPS');ax.set_ylim(0,max(r['qps'] for r in rows)*1.12)
bx.set_ylabel('Prefetch hit ratio (%)');bx.set_ylim(0,100);bx.set_xlabel('Prefetch capacity per thread (nodes)');bx.set_xlim(0,168000);bx.set_xticks(x);bx.set_xticklabels([f'{v/1000:g}K' for v in x],rotation=90,ha='center',fontsize=9);bx.tick_params(axis='x',pad=5)
ax.annotate('1,030 nodes',xy=(x[0],rows[0]['qps']),xycoords='data',xytext=(.12,.92),textcoords='axes fraction',ha='center',fontsize=12,arrowprops=dict(arrowstyle='->',color='#555555',lw=1))
bx.annotate('1,030 nodes',xy=(x[0],rows[0]['prefetch_hit_pct']),xycoords='data',xytext=(.12,.82),textcoords='axes fraction',ha='center',fontsize=12,arrowprops=dict(arrowstyle='->',color='#555555',lw=1))
for a in [ax,bx]:a.grid(axis='y',alpha=.25);a.spines[['top','right']].set_visible(False)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(R/f'sift100m_private_prefetch_capacity_qps_hit.{ext}',dpi=200)
lines=['# Thread-private prefetch capacity sweep','', '20 query workers, CPUs 0–19. Each worker owns one LRU, page payloads and resident-node replica directory. Pages are retained across queries and the A→B boundary; there are zero query resets. Immutable D and seed map are shared. Each run starts with empty prefetch; sync + file-scoped POSIX_FADV_DONTNEED and mincore verification precede every run. One run per capacity.','', 'SIFT100M A→B 40K queries; each phase is 10K uniform + corresponding 10K Zipf0.8, shuffled. K=10 L=128 beam=4, N=0 M=0, full profiled visited-seed immutable D (2,632,133 seeds, one 4KiB page each). Same executable and query order at every capacity. OpenMP dynamic scheduling may assign queries to different workers across runs.', '', 'Memory capacities below are page payload, excluding metadata and other index memory. Total payload = 20 × per-thread payload. The x-axis shows nominal node-record capacity per thread (pages × 10), using actual numeric spacing and K = 1000. This is capacity, not measured distinct-node occupancy; partial pages and replicated records can reduce occupancy.','', '| Per-thread MiB | Total MiB | QPS | Prefetch hit % | D reads/query |','|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['per_thread_MiB']:.6g} | {r['total_prefetch_MiB']:.6g} | {r['qps']:.2f} | {r['prefetch_hit_pct']:.2f} | {r['D_reads_per_query']:.2f} |")
lines += ['',f"First to last capacity: QPS {summary['qps_change_first_to_last_pct']:+.2f}%; hit ratio {summary['hit_change_first_to_last_percentage_points']:+.2f} percentage points.", '', summary['hit_definition'], '', 'All 40K×10 returned IDs match the existing reference for every run. Per-query full target counts are unchanged, and hit + D-read counts equal target accesses. These are reference-ID checks, not a new ground-truth recall measurement. Clear logs confirm zero file residency before each run. Private-LRU count and exact capacity per worker are asserted from the run log.', '', 'Curves use raw measurements without smoothing, interpolation-generated points or timing adjustment. One run per point provides no uncertainty interval; small local QPS variations cannot establish a causal effect. A larger retained cache can provide cross-query reuse even after the within-query working set fits.', '', 'Source executable: /mnt/nvme2/nc_thread_private_20261028/search_disk_index. See method.json and runs/*/command.json for full settings, hashes and affinity observations. Raw query stats, timeline and returned IDs are retained for audit.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n');(R/'STATUS.md').write_text('Complete: 10 private-retain capacities, one run each, all checks passed; no benchmark remains running.\n')
print(json.dumps(summary,indent=2))
