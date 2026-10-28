"""Plot only static hot20 stage samples; no benchmark rerun or synthetic timings."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('/mnt/nvme2/nc_hot20_20261027'))
root = p.parse_args().root
raw = json.loads((root/'stage_profile/summary.json').read_text())
engines = ['pipeann', 'starling', 'margo', 'aisaq']
names = ['PipeANN', 'Starling', 'MARGO', 'AiSAQ']
categories = [
    ('Parent-edge tracking', ['visit'], '#D55E00'),
    ('N admission / LRU', ['admit_N_and_seed'], '#E69F00'),
    ('Cache lookup', ['resolve'], '#0072B2'),
    ('D completion processing', ['complete'], '#56B4E9'),
    ('Query setup / reset', ['begin'], '#009E73'),
    ('Query cleanup', ['query_end'], '#CC79A7'),
    ('Other sampled hooks', ['selected_M', 'note_base_stash', 'contains'], '#888888'),
]
rows = []
for engine in engines:
    matches = [r for r in raw if r['engine'] == engine and r['mode'] == 'hot_fixed']
    assert len(matches) == 1
    command = json.loads((root/'stage_profile'/f'{engine}_hot_fixed/command.json').read_text())
    assert command['environment']['NEIGHBORCACHE_FREEZE_MAINTENANCE'] == '1'
    n = json.loads((root/f'{engine}_public_hot_fixed_0/measurement.json').read_text())['queries']
    stages = {s['stage']: s for s in matches[0]['stages']}
    # Exclude nested child scopes; these are also zero under static D.
    assert all(stages[k]['samples'] == 0 for k in ['seed_note', 'n_evict', 'global_dyn_write_wait'])
    costs = [sum(stages[k]['estimated_thread_seconds'] for k in keys)*1000/n
             for _, keys, _ in categories]
    total = sum(costs)
    for (label, keys, _), cost in zip(categories, costs):
        rows.append(dict(engine=engine, queries=n, category=label,
                         ms_per_query=cost, share_of_measured_stages_pct=100*cost/total,
                         sampled_stage_total_ms_per_query=total,
                         source_stages=';'.join(keys)))

with (root/'static_hot20_cost.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
fig.subplots_adjust(left=.19, right=.97, top=.89, bottom=.13, wspace=.72, hspace=.48)
for engine, name, ax in zip(engines, names, axes.flat):
    data = [r for r in rows if r['engine'] == engine]
    costs = [r['ms_per_query'] for r in data]
    bars = ax.barh(np.arange(len(categories)), costs,
                   color=[c[2] for c in categories], height=.66)
    ax.set_yticks(np.arange(len(categories)), [c[0] for c in categories])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.04)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel('Sampled thread time (ms/query)')
    ax.set_axisbelow(True); ax.grid(axis='x', alpha=.2)
    dataset = 'SIFT100K / 1K queries' if engine == 'aisaq' else 'SIFT10M / 10K queries'
    ax.set_title(f'{name} · {dataset}', loc='left', fontsize=12, pad=10)
    ax.bar_label(bars, labels=[f'{v:.3f}' for v in costs], padding=4, fontsize=9)
fig.suptitle('Static visited hot 20%: cache cost breakdown', fontsize=16, y=.97)
fig.text(.5, .045,
         'One diagnostic run per engine; sampled accumulated thread time / query count.\n'
         'Excludes host I/O waiting, distance computation and unsampled adapter work; not total query latency.',
         ha='center', fontsize=10, color='#444444')
for ext in ['png', 'pdf']:
    fig.savefig(root/f'static_hot20_cost_bar.{ext}', dpi=200)
plt.close(fig)

start = '<!-- static-cost-bar:start -->'
end = '<!-- static-cost-bar:end -->'
section = [start, '## 静态开销柱状图', '',
           '![静态 hot20 缓存阶段开销](static_hot20_cost_bar.png)', '',
           '[PDF](static_hot20_cost_bar.pdf) · [柱状图原始数值 CSV](static_hot20_cost.csv)', '',
           '四格使用相同横轴。仅采用 `hot_fixed` 的独立诊断运行；每个引擎一次采样运行，没有重复采样误差条。',
           '以下占比的分母仅为图中已采样缓存阶段之和，不是完整查询耗时，也不是相对无 NC 基线的净新增耗时。', '',
           '| 引擎 | 已采样阶段合计 ms/query | 最大项 | 最大项占比 |',
           '|---|---:|---|---:|']
for engine, name in zip(engines, names):
    data = [r for r in rows if r['engine'] == engine]
    biggest = max(data, key=lambda r:r['ms_per_query'])
    section.append(f"| {name} | {biggest['sampled_stage_total_ms_per_query']:.3f} | {biggest['category']} | {biggest['share_of_measured_stages_pct']:.1f}% |")
section += ['', 'Starling/MARGO 首先应检查静态路径仍执行的父边哈希记录，其次是 N 插入、复制和 LRU。PipeANN/AiSAQ 的已测最大项为缓存查找；D completion 是完成后的处理，不代表磁盘等待。阶段时间不能直接换算为删掉该项后的 QPS 提升。', end, '']
report = root/'COST.md'
text = report.read_text()
if start in text:
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    text = before+'\n'.join(section)+after.lstrip('\n')
else:
    title, body = text.split('\n', 1)
    text = title+'\n\n'+'\n'.join(section)+'\n'+body.lstrip('\n')
report.write_text(text)
print(root/'static_hot20_cost_bar.png')
for engine in engines:
    data = [r for r in rows if r['engine'] == engine]
    print(engine, [(r['category'], round(r['ms_per_query'], 4), round(r['share_of_measured_stages_pct'], 1)) for r in data])
