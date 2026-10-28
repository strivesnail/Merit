"""Summarize measured shared/private pairs; never alter measured QPS."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/mnt/nvme2/nc_private_adapters_20261028'))
args = parser.parse_args()
root = args.root
results = json.loads((root / 'results.json').read_text())

def ids(path):
    raw = np.fromfile(path, dtype='<u4')
    n, k = map(int, raw[:2])
    return raw[2:].reshape(n, k)

rows = []
for engine in dict.fromkeys(r['engine'] for r in results):
    a, b = [next(r for r in results if r['engine'] == engine and r['mode'] == mode)
            for mode in ['shared', 'private']]
    x, y = ids(a['idfile']), ids(b['idfile'])
    row = dict(engine=engine, dataset=a['dataset'], workload=a['workload'], threads=a['threads'],
               shared_qps=a['qps'], private_qps=b['qps'], change_percent=100*(b['qps']/a['qps']-1),
               shared_recall=a['recall_exact'], private_recall=b['recall_exact'],
               identical_rows=int(np.all(x == y, axis=1).sum()), queries=len(x),
               top10_overlap_percent=100*float((x[:, :, None] == y[:, None, :]).any(axis=2).mean()))
    rows.append(row)
with (root / 'comparison.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
(root / 'comparison.json').write_text(json.dumps(rows, indent=2)+'\n')

lines = ['# Thread-private immutable prefetch: adapter comparison', '',
         'One timed run per mode, 2026-10-28. Compare modes within each row; datasets differ across rows.', '',
         '| Engine | Dataset | Shared QPS | Private QPS | Change | Recall@10 (%) shared → private | Identical top-10 rows |',
         '|---|---|---:|---:|---:|---|---:|']
for r in rows:
    lines.append(f"| {r['engine']} | {r['dataset']} | {r['shared_qps']:.2f} | {r['private_qps']:.2f} | {r['change_percent']:+.2f}% | {r['shared_recall']:.4f} → {r['private_recall']:.4f} | {r['identical_rows']}/{r['queries']} |")
lines += ['', '## Interpretation', '',
          'The general shared-backend integration does not reproduce a universal private-cache speedup. '
          'DiskANN and PipeANN change by less than 1%; one run cannot distinguish these changes from measurement noise. '
          'Starling and MARGO slow down despite slightly fewer D reads; AiSAQ also issues more D reads. '
          'These counters alone do not locate the CPU cost. No lock-time or allocator-cost attribution is established here.', '',
          'This experiment retains each engine’s existing N cache, partial static D prefill, and search policy. '
          'It is not the earlier full-seed, N=0/M=0 standalone experiment. The general cache wrapper uses '
          'different lookup and page-lifetime machinery. Do not compare its absolute QPS directly with that earlier sweep.', '',
          'Recall was recomputed from returned IDs and ground truth with ties at the tenth distance. '
          'DiskANN and AiSAQ outputs are identical across modes. Other engines have output differences; '
          'equal rounded recall is not proof of identical results. Starling/MARGO have prefetch-page expansion enabled, '
          'whose work can change with residency. PipeANN uses asynchronous candidate processing; '
          'the cause of its small recall difference is not isolated by this experiment.', '',
          '## Configuration and validation', '',
          '- Private caches retain pages between queries. Each worker gets 128 × 4 KiB = 512 KiB. '
          'Shared mode has exactly the same total resident payload capacity: 10 MiB for 20 threads; 4 MiB for AiSAQ’s 8 threads.',
          '- DiskANN/Starling/MARGO: SIFT100M, shuffled 10K+Zipf (20K queries), T=20, L=128, W=4, K=10. '
          'N capacity is 212,371 nodes. Static D is reused from each engine’s existing configuration.',
          '- PipeANN: SIFT10M public 10K, T=20, L=128, W=4, K=10, N=31 MiB. '
          'AiSAQ: SIFT100K public 1K, T=8, L=64, W=4, K=10, N=1 MiB.',
          '- Query CPUs pinned to 0–19 (AiSAQ 0–7). File cache cleared before every run using sync + '
          'POSIX_FADV_DONTNEED, checked with mincore. This is not a device-cache reset.',
          '- Fixed D, no M maintenance or writes. Adapter counters confirm M/opens/appends/replaces/writes = 0. '
          'Shared/private pairs use identical prefill and search parameters. Query scratch and metadata are outside the payload budget.',
          '- 17/17 backend CTests passed, including isolation, LRU eviction, query reset, pointer lifetime, '
          'immutable-page guards, and concurrent prefetch expansion. Native DiskANN rebuilt successfully.',
          '- Separate first-100-query runs passed original-index full-record verification on Starling, MARGO, PipeANN, and AiSAQ. '
          'Verification timing is excluded. This does not establish complete search equivalence.', '',
          'Raw commands, binary hashes, cache-clear checks, output IDs and logs: '
          '`/mnt/nvme2/nc_private_adapters_20261028/runs/`.',
          'Independent verification: `/mnt/nvme2/nc_private_adapters_20261028/verification/`.', '']
(root / 'REPORT.md').write_text('\n'.join(lines))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4.6))
x = np.arange(len(rows))
ax.bar(x-.19, [r['shared_qps'] for r in rows], .38, label='Shared prefetch', color='#0072B2')
ax.bar(x+.19, [r['private_qps'] for r in rows], .38, label='Thread-private prefetch', color='#009E73')
ax.set_xticks(x, [r['engine']+'\n'+r['dataset'] for r in rows])
ax.set_ylabel('QPS')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis='y', alpha=.2)
fig.tight_layout()
for suffix in ['png', 'pdf']:
    fig.savefig(root / f'private_prefetch_comparison.{suffix}', dpi=180)
print('\n'.join(lines[:11]))
