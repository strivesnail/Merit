"""Cold online D-cache counterpart of n_prefetch_sweep_20261026.

39 measured runs: 7 prefetch budgets plus 6 additional N budgets per workload.
All run commands and per-query counters are retained. Only this run's temporary
dynamic data files are removed after successful extraction (disk space limit).
"""
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path('/mnt/nvme_single/sift100m/runs/n_prefetch_online_probe_20261026')
BIN = ROOT / 'search_disk_index'
INDEX = '/mnt/nvme_single/sift100m/sift100m_index'
HS = Path('/home/jianz/workload/data/sift100m/workloads/cluster_hotspot')
WORK = {
    'pub10k': ('/home/jianz/workload/data/sift100m/query.public.10K.u8bin', '/home/jianz/workload/data/sift100m/bigann-100M'),
    'zipf': (str(HS/'pub10k_zipf0p8_10k_shuffled.u8bin'), str(HS/'pub10k_zipf0p8_10k_shuffled_gt10.bin')),
    'x5': (str(HS/'pub10kx5_shuffled.u8bin'), str(HS/'pub10kx5_shuffled_gt10.bin')),
}
PREFETCH = [0, 5, 10, 25, 50, 100, 200]
NCACHE = {4:10810, 16:43242, 32:86484, 75:202688, 100:270251, 200:540502, 300:810754}

def measure(work, nmb, pmb):
    name = f'pf{pmb}_n75' if nmb == 75 else f'n{nmb}_pf25'
    out = ROOT/work/name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'measurement.json').exists():
        return
    avail = int(re.search(r'MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text())[1])
    assert avail > 6000000
    assert shutil.disk_usage(ROOT).free > 8*1024**3
    builder = subprocess.run(['ps','-o','stat=','-p','2155989'],capture_output=True,text=True)
    assert builder.returncode != 0 or 'T' in builder.stdout, 'Existing builder is running'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('NEIGHBORCACHE_', 'MERIT_'))}
    settings = {
        'LD_LIBRARY_PATH': '/home/jianz/miniconda3/lib:/tmp/merit-link-deps/root/usr/lib/x86_64-linux-gnu',
        'OMP_NUM_THREADS': '15', 'NEIGHBORCACHE_QUERY_CPUS':'0-14',
        'NEIGHBORCACHE_USE_RAMFS':'0', 'NEIGHBORCACHE_DCACHE_PROFILE_GRAPH':'1',
        'NEIGHBORCACHE_SEED_REPLICA_GRANDCHILD':'1', 'NEIGHBORCACHE_SEED_REPLICA_MAX_PAGES':'2',
        'NEIGHBORCACHE_SEED_REPLICA_ALL_CHILDREN':'1', 'NEIGHBORCACHE_SEED_REPLICA_SEED_FRAC':'0.5',
        'NEIGHBORCACHE_CACHE_REPLICA_FALLBACK':'1',
        'NEIGHBORCACHE_DYNAMIC_3CACHE':'1', 'NEIGHBORCACHE_SHARED_CACHE_DYN':'1',
        'NEIGHBORCACHE_DYNAMIC_PREFILL_STATIC':'0', 'NEIGHBORCACHE_REQUIRE_FULL_DCACHE_WARMUP':'0',
        'NEIGHBORCACHE_DCACHE_CAP':'1500000', 'NEIGHBORCACHE_DCACHE_WRITER_THREADS':'1',
        'NEIGHBORCACHE_DCACHE_COMMIT_THREADS':'1', 'NEIGHBORCACHE_DCACHE_REPACK_ON_READ':'0',
        'NEIGHBORCACHE_NCACHE_CAP_NODES':str(NCACHE[nmb]), 'NEIGHBORCACHE_NCACHE_COLD':'1',
        'NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES':str(pmb*256),
    }
    if nmb == 75 and pmb == 25:
        settings['NEIGHBORCACHE_DUMP_DYNAMIC_PAGES'] = str(out/'dynamic_pages.bin')
    env.update(settings)
    q, gt = WORK[work]
    command = [str(BIN), '--data_type','uint8','--dist_fn','l2','--index_path_prefix',INDEX,
               '--query_file',q,'--gt_file',gt,'--recall_at','10','--search_list','128',
               '--beamwidth','4','--num_threads','15','--num_nodes_to_cache','0',
               '--result_path',str(out/'result'), '--neighborcache_memory_gb',f'{NCACHE[nmb]*388/1073741824:.6f}',
               '--neighborcache_memory_runtime_admit','true', '--neighborcache_disk_cache_exclude_memory','false',
               '--enable_query_sector_cache','true', '--dump_query_stats',str(out/'query_stats.csv')]
    (out/'command.json').write_text(json.dumps(dict(command=command,environment=settings),indent=2))
    os.sync()
    print(f'START {work} N={nmb}MB Prefetch={pmb}MB', flush=True)
    start=time.time()
    with (out/'run.out').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    text=(out/'run.out').read_text()
    matches=re.findall(r'^\s*128\s+4\s+(.+)$',text,re.M)
    assert matches, out
    fields=matches[-1].split()
    assert 'dynamic 3-cache enabled' in text and 'cold start' in text
    assert 'disk-cache loaded (reuse)' not in text
    sums={}
    queries=0
    with (out/'query_stats.csv').open() as f:
        for row in csv.DictReader(f):
            queries+=1
            row_probes=int(row['n_neighborcache_dcache_demand_probes'])
            row_hits=int(row['n_neighborcache_dcache_demand_hits'])
            assert 0<=row_hits<=row_probes<=int(row['n_beam_lookups'])-int(row['n_mem_hits'])
            for key in ('n_beam_lookups','n_mem_hits','n_neighborcache_prefetch_hits','n_neighborcache_dcache_seed_hits',
                        'n_neighborcache_dcache_demand_probes','n_neighborcache_dcache_demand_hits'):
                sums[key]=sums.get(key,0)+int(row[key])
    misses=sums['n_beam_lookups']-sums['n_mem_hits']
    # seed_prefetch's direct resident-page path bypasses the per-query D counters.
    # The dyn probe miss counter increments only after that path also misses.
    probe=re.search(r'dyn probe: ncache_hit=(\d+) ncache_miss=(\d+)',text)
    assert probe and int(probe[1])==sums['n_mem_hits']
    fast_prefetch_hits=misses-int(probe[2])
    assert fast_prefetch_hits>=0
    hits=fast_prefetch_hits+sums['n_neighborcache_prefetch_hits']+sums['n_neighborcache_dcache_seed_hits']
    assert misses>0 and 0<=hits<=misses
    result=dict(workload=work,n_mb=nmb,prefetch_mb=pmb,queries=queries,qps=float(fields[0]),recall=float(fields[-1]),
                d_hit_after_n_miss_pct=100*hits/misses,dcache_served=hits,n_misses=misses,counters=sums,
                fast_prefetch_hits=fast_prefetch_hits,
                duration_seconds=time.time()-start,mode='online cold N+D; no prefill or warmup')
    demand_probes=sums['n_neighborcache_dcache_demand_probes']
    demand_hits=sums['n_neighborcache_dcache_demand_hits']
    assert demand_probes>0
    result.update(d_probe_hit_pct=100*demand_hits/demand_probes,
                  d_demand_probes=demand_probes,d_demand_hits=demand_hits,d_demand_misses=demand_probes-demand_hits)
    assert result['recall']>95
    if 'NEIGHBORCACHE_DUMP_DYNAMIC_PAGES' in settings:
        assert (out/'dynamic_pages.bin').stat().st_size>12
    (out/'measurement.json').write_text(json.dumps(result,indent=2))
    # These files belong only to this just-completed run. Logs, counters and snapshots stay.
    for path in out.glob('result*.dyn.data'):
        path.unlink()
    print(f'DONE {work} N={nmb} PF={pmb}: QPS={result["qps"]} D hit/probe={result["d_probe_hit_pct"]:.2f}%',flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    args=parser.parse_args()
    ROOT=args.output_dir
    BIN=ROOT/'search_disk_index'
    ROOT.mkdir(exist_ok=True,parents=True)
    if not BIN.exists():
        shutil.copy2('/home/jianz/workload/code/Merit/diskann/build/apps/search_disk_index', BIN)
    (ROOT/'binary.sha256').write_text(hashlib.sha256(BIN.read_bytes()).hexdigest()+'\n')
    # First run also validates the optional snapshot before the full sweep.
    measure('pub10k',75,25)
    for work in WORK:
        for pmb in PREFETCH:
            measure(work,75,pmb)
        for nmb in NCACHE:
            measure(work,nmb,25)
    print('ALLDONE 39 runs',flush=True)
