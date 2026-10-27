"""Rerun only NC bars with 5% prefetch; retain N=20%, D seeds=30%."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

ROOT = Path('/mnt/nvme_single/meeting_20261026_prefetch5')
OLD = Path('/mnt/nvme_single/meeting_20261026')
PRE = Path('/mnt/nvme_single/meeting_20261026_prefill')
SIZED = Path('/mnt/nvme_single/meeting_20261026_sized')
REPO = Path('/home/jianz/workload/code/Merit')
BASE = Path('/home/jianz/workload/code/baselines')
ROOT.mkdir(exist_ok=True)

def visited(path):
    with open(path, 'rb') as f:
        n = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
        a = np.fromfile(f, dtype=np.uint32, count=n)
    assert len(a) == n
    return int(np.count_nonzero(a))

def run(tag, method, binary, args, env):
    out = ROOT / tag / method
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'measurement.json').exists():
        return
    stat = subprocess.check_output(['ps', '-o', 'stat=', '-p', '2155989'], text=True)
    assert 'T' in stat, 'Background builder must remain paused'
    available = int(re.search(r'MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text())[1])
    assert available > 6000000
    fullenv = {k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_', 'PIPEANN_NMD'))}
    config = dict(LD_LIBRARY_PATH='/home/jianz/miniconda3/lib', OMP_NUM_THREADS='20',
                  OMP_PROC_BIND='close', OMP_PLACES='{0}:20', NEIGHBORCACHE_QUERY_CPUS='0-19')
    config.update({k:str(v) for k,v in env.items()})
    fullenv.update(config)
    command = ['taskset', '-c', '0-19']
    if method in ('starling', 'margo'):
        command += ['nice', '-n', '10']
    command += [str(binary)] + list(map(str, args))
    metadata = dict(command=command, environment=config, binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                    prefetch_fraction=0.05, n_fraction=0.2, seed_fraction=0.3)
    (out / 'command.json').write_text(json.dumps(metadata, indent=2))
    os.sync()
    print(f'START {tag} {method}', flush=True)
    start = time.time()
    with (out / 'run.out').open('w') as log:
        subprocess.run(command, env=fullenv, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (out / 'run.out').read_text()
    rows = re.findall(r'^\s*128\s+4\s+(.+)$', log, re.M)
    assert rows, f'Missing result: {out}'
    fields = rows[-1].split()
    result = dict(qps=float(fields[0]), recall=float(fields[-1]), seconds=time.time()-start,
                  row='128 4 '+rows[-1])
    assert result['qps'] > 0 and result['recall'] > 90, result
    (out / 'measurement.json').write_text(json.dumps(result, indent=2))
    print(f'DONE {tag} {method}: {result}', flush=True)

for dataset, tagbase, ncap, pages, memgb in [
    ('sift100m', 'sift100m', 212371, 5030, '0.076740'),
    ('sift10m', 'sift10m', 183685, 4350, '0.066383'),
    ('deep10m', 'deep', 193480, 7606, '0.116089'),
]:
    deep = dataset == 'deep10m'
    disk = Path('/mnt/nvme1/deep10m') if deep else Path('/mnt/nvme_single') / dataset
    dtype = 'float' if deep else 'uint8'
    for work in ('pub', 'zipf', 'x5'):
        tag = f'{tagbase}_{work}'
        if deep:
            q = disk / ('query.public.10K.fbin' if work == 'pub' else 'workloads/'+('pub10k_zipf0p8_10k_shuffled.fbin' if work == 'zipf' else 'pub10kx5_shuffled.fbin'))
            gt = disk / ('deep10m_gt100' if work == 'pub' else 'workloads/'+('pub10k_zipf0p8_10k_shuffled_gt100.bin' if work == 'zipf' else 'pub10kx5_shuffled_gt100.bin'))
        elif work == 'pub':
            q = Path('/home/jianz/workload/data') / dataset / 'query.public.10K.u8bin'
            gt = q.parent / ('bigann-100M' if dataset == 'sift100m' else 'bigann-10M')
        elif dataset == 'sift100m':
            h = Path('/home/jianz/workload/data/sift100m/workloads/cluster_hotspot')
            stem = 'pub10k_zipf0p8_10k_shuffled' if work == 'zipf' else 'pub10kx5_shuffled'
            q, gt = h / (stem+'.u8bin'), h / (stem+'_gt10.bin')
        else:
            q, gt = OLD/'workloads'/f'sift10m_{work}.u8bin', OLD/'workloads'/f'sift10m_{work}_gt.bin'
        out = ROOT/tag/'neighbor'
        run(tag, 'neighbor', REPO/'diskann/build/apps/search_disk_index', [
            '--data_type', dtype, '--dist_fn', 'l2', '--index_path_prefix', disk/f'{dataset}_index',
            '--query_file', q, '--gt_file', gt, '--recall_at', 10, '--search_list', 128,
            '--beamwidth', 4, '--num_threads', 20, '--result_path', out/'result', '--num_nodes_to_cache', 0,
            '--neighborcache_memory_gb', memgb, '--neighborcache_profile_prefix', PRE/'profile'/dataset,
            '--neighborcache_memory_runtime_admit', 'true', '--enable_query_sector_cache', 'true',
            '--neighborcache_disk_cache_ratio', 0.15, '--neighborcache_disk_cache_exclude_memory', 'false',
            '--neighborcache_disk_cache_reuse_prefix', PRE/'dc'/tag/'dc'],
            dict(NEIGHBORCACHE_DYNAMIC_3CACHE=0, NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=pages,
                 NEIGHBORCACHE_NCACHE_CAP_NODES=ncap, NEIGHBORCACHE_NCACHE_COLD=1))
        for method in ('starling', 'margo'):
            visits = (SIZED if deep else OLD)/tag/f'{method}.visits'
            dp = math.ceil(visited(visits)*0.3)
            prefix = disk/method/dataset
            run(tag, method, BASE/('MARGO' if method == 'margo' else method)/'build/tests/search_disk_index', [
                '--data_type', dtype, '--dist_fn', 'l2', '--index_path_prefix', prefix,
                '--query_file', q, '--gt_file', gt, '-K', 10, '-L', 128, '-W', 4, '-T', 20,
                '--num_nodes_to_cache', 0, '--mem_L', 10, '--mem_index_path', disk/'starling/mem/_index',
                '--use_page_search', 1, '--use_ratio', 1.0, '--disk_file_path', str(prefix)+'_disk.index',
                '--result_path', ROOT/tag/method/'result'],
                dict(NEIGHBORCACHE_NCACHE_CAP_NODES=ncap, NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=dp,
                     NEIGHBORCACHE_DCACHE_CAP=dp+1024, NEIGHBORCACHE_DCACHE_PREFILL=1,
                     NEIGHBORCACHE_DCACHE_READ_PAGES=pages, NEIGHBORCACHE_VISIT_SEED_FRAC=0.3,
                     NEIGHBORCACHE_VISIT_RANK=visits, NEIGHBORCACHE_DCACHE_MAX_PAGES_PER_SEED=1))
        if dataset == 'sift10m':
            rank = PRE/'pipeann'/f'{work}.visits'
            v = visited(rank)
            nmb = math.ceil(math.ceil(0.2*v)*260/1048576)
            pref_bytes = math.ceil(0.05*v)*260
            pmb = math.ceil(pref_bytes/1048576)
            while (pmb*1048576//3900)*3900 < pref_bytes:
                pmb += 1
            run(tag, 'pipeann', '/tmp/PipeANN/build/tests/search_disk_index', [
                'uint8', '/mnt/nvme_single/pipeann_nmd/sift10m', 20, 4, q, gt, 10, 'l2', 'pq', 2, 0, 128],
                dict(PIPEANN_NMD=1, PIPEANN_NMD_PREFILL=1, PIPEANN_NMD_SEED_FRAC=0.3,
                     PIPEANN_NMD_VISIT_RANK=rank, PIPEANN_NMD_NMB=nmb, PIPEANN_NMD_PMB=pmb,
                     PIPEANN_NMD_PATH=ROOT/tag/'pipeann/cache.nmd'))
print('ALLDONE', flush=True)
