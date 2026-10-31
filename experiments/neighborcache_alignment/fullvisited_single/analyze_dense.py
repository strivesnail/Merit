"""Summarize measured slopes and independent checks; never generate plot points."""
from pathlib import Path
import csv,json,statistics as st
import numpy as np
R=Path('/mnt/nvme2/nc_fullvisited_seed_single_20261031');D=R/'nonly_dense'
plan=json.loads((D/'plan.json').read_text())
baseline_path=Path('/mnt/nvme2/nc_n_capacity_match_20261031/nonly_0_0/result.json')
base=json.loads(baseline_path.read_text())['extra_rss_mib']
points=[]
for spec in plan['points']:
 rows=[json.loads(p.read_text()) for p in D.glob(f"nonly_{spec['n_nodes']}_*/result.json")]
 assert rows,spec
 p=dict(target_memory_mib=spec['target_memory_mib'],nodes=spec['n_nodes'],runs=len(rows),
        memory_mib=st.mean(x['extra_rss_mib'] for x in rows)-base,
        qps=st.mean(x['qps'] for x in rows),qps_runs=[x['qps'] for x in rows],
        all_access_counts_equal=all(x['counts']==rows[0]['counts'] for x in rows),
        all_output_checks_passed=all(all(x['ids_and_distances_equal'].values()) for x in rows))
 points.append(p)
source=Path('/mnt/nvme2/nc_n_capacity_match_20261031/nonly_0_0/result.json');b=json.loads(source.read_text())
series=[dict(target_memory_mib=0,memory_mib=0,qps=b['qps'])]+points
slopes=[]
for a,b in zip(series,series[1:]):
 slopes.append(dict(from_target_mib=a['target_memory_mib'],to_target_mib=b['target_memory_mib'],
                    actual_memory_delta_mib=b['memory_mib']-a['memory_mib'],
                    qps_delta=b['qps']-a['qps'],
                    qps_per_mib=(b['qps']-a['qps'])/(b['memory_mib']-a['memory_mib'])))
x=np.array([p['memory_mib'] for p in series]);y=np.array([p['qps'] for p in series]);fit=np.polyfit(x,y,1)
residual=y-np.polyval(fit,x);r2=1-float(np.sum(residual**2)/np.sum((y-y.mean())**2))
report=dict(measured_grid=points,adjacent_measured_slopes=slopes,
            linear_fit_diagnostic_only=dict(slope=float(fit[0]),intercept=float(fit[1]),r_squared=r2,
                note='Diagnostic only; the figure connects measured point means, without plotting this fitted model'),
            all_counts_repeatable=all(p['all_access_counts_equal'] for p in points),
            all_output_checks_passed=all(p['all_output_checks_passed'] for p in points))
for file,key in [('verification_results.json','independent_nonly_reruns'),('nc_reference_control.json','nc_reference_rerun')]:
 if (D/file).exists():report[key]=json.loads((D/file).read_text())
near=min(points,key=lambda p:abs(p['qps']-585.055))
report['closest_measured_grid_point']=dict(near,reference_qps=585.055,relative_qps_difference_pct=100*(near['qps']/585.055-1),memory_ratio_to_nc=near['memory_mib']/17.54296875)
(D/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
with (D/'adjacent_slopes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(slopes[0]),lineterminator='\n');w.writeheader();w.writerows(slopes)
print(json.dumps({'closest_point':report['closest_measured_grid_point'],'r_squared_diagnostic':r2,'all_counts_repeatable':report['all_counts_repeatable']},indent=2))
