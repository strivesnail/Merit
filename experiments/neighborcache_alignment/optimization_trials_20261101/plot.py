"""Plot every measured directory-update run; do not normalize or discard runs."""
from pathlib import Path
import argparse,csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=here)
out=p.parse_args().output_dir;out.mkdir(parents=True,exist_ok=True)
rows=json.loads((here/'results.json').read_text())['prefetch_directory_update']
assert len(rows)==6
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10,4.8));x=list(range(6))
io=[a['mean_io_us']/1000 for a in rows]
other=[(a['mean_latency_us']-a['mean_io_us'])/1000 for a in rows]
ax.bar(x,io,width=.62,color='#0072b2',label='Measured I/O')
ax.bar(x,other,width=.62,bottom=io,color='#e69f00',label='Remaining query time')
for i,a in enumerate(rows):ax.text(i,a['mean_latency_us']/1000+.04,f"{a['mean_latency_us']/1000:.3f}",ha='center',fontsize=11)
ax.set_xticks(x,['Before 1','After 1','After 2','Before 2','Before\nrepeat','After\nrepeat'])
ax.set_ylabel('Latency (ms/query)');ax.set_ylim(0,max(a['mean_latency_us']/1000 for a in rows)*1.17)
ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.04));ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'sift1b_400k_directory_latency_all_runs.{ext}',dpi=200,bbox_inches='tight')
with (out/'directory_all_runs.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
