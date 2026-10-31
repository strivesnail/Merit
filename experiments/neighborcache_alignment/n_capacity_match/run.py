"""Adaptive single-worker NMD N sweep, followed by a same-QPS N-only search.

Each process executes --search_list 128 128: first the identical 20K queries
warm both schemes, then the second pass is measured. The final per-query CSV
and result files are the second pass. No cache/search-policy changes are made.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import statistics as st
import subprocess
import time
from cache_control import clear_files, prefixed_files

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path('/mnt/nvme2/nc_n_capacity_match_20261031')
BASE = REPO / 'experiments/neighborcache_alignment/compact_memory/measured_configuration.json'
REFERENCE = Path('/mnt/nvme2/nc_private_prefetch_fix_20261031/cost_final/bfs_1')


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root, mode, nodes, rep, prefetch_pages=None):
    tag=f'{mode}_{nodes}_{rep}'
    if prefetch_pages is not None:tag+=f'_p{prefetch_pages}'
    out = root / tag
    if (out / 'result.json').exists():
        return json.loads((out / 'result.json').read_text())
    out.mkdir(exist_ok=False)
    config = json.loads(BASE.read_text())
    cmd = config['command']
    cmd[2] = '0,20-23'
    cmd[3] = str(REPO / 'diskann/build/apps/search_disk_index')
    for key, value in {'--num_threads':'1', '--result_path':str(out/'result'),
                       '--dump_query_stats':str(out/'query_stats.csv'),
                       '--neighborcache_memory_gb':'2.0' if nodes else '0'}.items():
        cmd[cmd.index(key)+1] = value
    at = cmd.index('--search_list') + 1
    cmd.insert(at, '128')
    env = {k:v for k,v in config['environment'].items()
           if not k.startswith(('NC_', 'NEIGHBORCACHE_', 'OMP_'))}
    env.update(OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='close',
               OMP_PLACES='{0}', NEIGHBORCACHE_QUERY_CPUS='0',
               NEIGHBORCACHE_THREADS='1', NEIGHBORCACHE_QUERY_SCHEDULE='static',
               NEIGHBORCACHE_HEAT_SAMPLE_SEED='20261031')
    if nodes:
        env.update(NEIGHBORCACHE_NCACHE_CAP_NODES=str(nodes), NEIGHBORCACHE_NCACHE_COLD='1')
    if mode == 'nmd':
        env.update({k:v for k,v in config['environment'].items() if k.startswith('NEIGHBORCACHE_')})
        env.update(NEIGHBORCACHE_NCACHE_CAP_NODES=str(nodes), NEIGHBORCACHE_THREADS='1',
                   NEIGHBORCACHE_QUERY_CPUS='0', NEIGHBORCACHE_BG_CPUS='20-23',
                   NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=str(2560 if prefetch_pages is None else prefetch_pages))
    elif nodes:
        env['NEIGHBORCACHE_NCACHE_ONLY'] = '1'
    # Query-local original-page reuse is enabled for BOTH schemes. Only NMD
    # receives a cross-query prefetch page budget, M, D, writer and stash.
    cfg = dict(command=cmd, environment=env, binary_sha256=sha(cmd[3]),
               warmup_queries=20000, measured_queries=20000, threads=1,
               input_sha256=sha(cmd[cmd.index('--query_file')+1]))
    write(out/'command.json', cfg)
    actual = {k:v for k,v in os.environ.items()
              if not k.startswith(('NEIGHBORCACHE_', 'NC_', 'OMP_', 'PIPEANN_', 'AISAQ_'))}
    actual.update(env)
    files = prefixed_files(cmd[cmd.index('--index_path_prefix')+1]) + [cmd[cmd.index('--query_file')+1]]
    write(out/'cache_clear.json', clear_files(files))
    print('START', mode, nodes, rep, flush=True)
    start = time.monotonic()
    with (out/'run.out').open('w') as log:
        subprocess.run(cmd, env=actual, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    log = (out/'run.out').read_text()
    rows = re.findall(r'^\s*128\s+4\s+([\d.]+).*$', log, re.M)
    assert len(rows) == 2, (out, rows)
    with (out/'query_stats.csv').open() as source:
        stats = list(csv.DictReader(source))
    assert len(stats) == 20000
    counts = {key:sum(int(row[key]) for row in stats) for key in
              ('n_beam_lookups','n_mem_hits','n_neighborcache_prefetch_hits','n_disk_reads',
               'n_neighborcache_dyn_disk_reads','n_neighborcache_dcache_demand_probes',
               'n_neighborcache_dcache_demand_hits')}
    assert counts['n_beam_lookups'] == 2907400, counts
    assert all(int(row['query_worker']) == 0 for row in stats)
    if nodes:
        assert f'cold start, capacity {nodes} nodes, 0 preloaded' in log
    if mode == 'nmd':
        assert 'seed_prefetch=on' in log and 'thread-private LRUs' in log
    else:
        assert counts['n_neighborcache_prefetch_hits'] == counts['n_neighborcache_dyn_disk_reads'] == 0
        assert 'dynamic 3-cache enabled' not in log
        assert 'shared D-cache page cache:' not in log
    equality = {}
    for suffix in ('idx_uint32.bin', 'dists_float.bin'):
        target = out / ('result_128_' + suffix)
        ref = REFERENCE / ('result_128_' + suffix)
        equality[suffix] = target.read_bytes() == ref.read_bytes()
        assert equality[suffix], target
    rss = {stage:int(kib) for stage,kib in re.findall(r'MEM_STAGE (\w+) rss=\s*(\d+) kB', log)}
    accesses, n_hits, p_hits = (counts[k] for k in ('n_beam_lookups','n_mem_hits','n_neighborcache_prefetch_hits'))
    result = dict(mode=mode, nodes=nodes, rep=rep, qps=float(rows[-1]),
                  prefetch_pages=(2560 if prefetch_pages is None else prefetch_pages) if mode=='nmd' else 0,
                  cold_first_pass_qps=float(rows[0]), counts=counts,
                  n_hit_pct=100*n_hits/accesses, p_hit_pct=100*p_hits/accesses,
                  p_after_n_miss_pct=100*p_hits/(accesses-n_hits),
                  disk_reads_per_query=counts['n_disk_reads']/20000,
                  d_reads_per_query=counts['n_neighborcache_dyn_disk_reads']/20000,
                  extra_rss_mib=(rss['after_search']-rss['after_load'])/1024,
                  rss_kib=rss, ids_and_distances_equal=equality,
                  elapsed_seconds=time.monotonic()-start)
    write(out/'result.json', result)
    all_rows = [json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))]
    write(root/'results.json', all_rows)
    print('DONE', json.dumps(result), flush=True)
    return result


def mean_point(root, mode, nodes, reps=1):
    rows = [run(root, mode, nodes, rep) for rep in range(reps)]
    result = dict(mode=mode, nodes=nodes, runs=len(rows),
                  qps_runs=[row['qps'] for row in rows],
                  extra_rss_runs=[row['extra_rss_mib'] for row in rows])
    for key in ('qps','extra_rss_mib','n_hit_pct','p_hit_pct','p_after_n_miss_pct',
                'disk_reads_per_query','d_reads_per_query'):
        result[key] = st.mean(row[key] for row in rows)
    return result


def experiment(root):
    # Written before observing this experiment's results. A flat/decreasing
    # all-access P curve alone can simply mean N intercepts P hits: require
    # no >0.5pp improvement in the N-miss-conditional P ratio as well.
    write(root/'protocol.json', dict(
        threads=1, prefetch_mib=10, m_entries=131072, d_ssd_mib=256,
        stash_capacity_nodes=424742, n_start=1024, n_step='double',
        warmup_queries=20000, measured_queries=20000, no_offline_prefill=True,
        screening_repeats=1, selected_and_match_repeats=2,
        plateau_rule='After at least 4 N levels, first of latest 3 points whose '
                     'next 2 doublings improve neither P/all-accesses nor P/N-misses by more than 0.5 percentage points.',
        match_rule='First tested N-only mean QPS >= selected NMD mean; refine bracket up to 3 times. '
                   'Observed finite-workload crossing, not a proof of global minimal memory.',
        memory='RSS after_search minus RSS after_load; includes all additional resident allocations, excludes common PQ/index and SSD D payload'))
    series=[]
    selected=None
    for exponent in range(11):
        point=mean_point(root,'nmd',1024*2**exponent)
        series.append(point)
        write(root/'nmd_series.json', series)
        if len(series)>=4:
            first,*later=series[-3:]
            if all(max(p[k] for p in later)-first[k] <= 0.5
                   for k in ('p_hit_pct','p_after_n_miss_pct')):
                selected=first
                break
    if selected is None:
        write(root/'selection.json',dict(plateau_found=False,series=series))
        raise RuntimeError('No supported P-hit plateau in the tested range; inspect saved data.')
    # Repeat the plateau point and its two confirming doublings.
    for point in series[-3:]:
        point.update(mean_point(root,'nmd',point['nodes'],2))
    first,*later=series[-3:]
    stable=all(max(p[k] for p in later)-first[k] <= 0.5
               for k in ('p_hit_pct','p_after_n_miss_pct'))
    if not stable:
        write(root/'selection.json',dict(plateau_found=False,series=series,reason='confirmation failed'))
        raise RuntimeError('Plateau did not survive second-run confirmation; continue sweep before matching.')
    selected=first
    write(root/'selection.json',dict(plateau_found=True,selected=selected,series=series))
    print('SELECTED',json.dumps(selected),flush=True)
    target=selected['qps']
    nseries=[]
    low=0
    high=None
    for nodes in [0]+[1024*2**i for i in range(14)]:
        point=mean_point(root,'nonly',nodes)
        nseries.append(point)
        write(root/'nonly_series.json',nseries)
        if point['qps']>=target:
            confirmed=mean_point(root,'nonly',nodes,2)
            nseries[-1]=confirmed
            if confirmed['qps']>=target:
                high=nodes
                break
        low=nodes
    if high is None:
        raise RuntimeError('N-only did not reach target within test range.')
    if high:
        mean_point(root,'nonly',low,2)
        for _ in range(3):
            mid=((low+high)//2)//256*256
            if mid<=low:break
            point=mean_point(root,'nonly',mid,2)
            if point['qps']>=target:high=mid
            else:low=mid
    matching=mean_point(root,'nonly',high,2)
    below=mean_point(root,'nonly',low,2) if high else None
    baseline=mean_point(root,'nonly',0,2)
    nmd_memory=selected['extra_rss_mib']-baseline['extra_rss_mib']
    n_memory=matching['extra_rss_mib']-baseline['extra_rss_mib']
    write(root/'match.json',dict(selected_nmd=selected,nonly_match=matching,
                                lower_tested_point=below,
                                no_cache_baseline=baseline,
                                nmd_cache_rss_mib=nmd_memory,nonly_cache_rss_mib=n_memory,
                                n_only_over_nmd_ram=n_memory/nmd_memory,
                                target_qps=target))
    print('MATCH',json.dumps(json.loads((root/'match.json').read_text())),flush=True)


def match_only(root):
    """Use a coarse bracket instead of running many far-below-target capacities."""
    selected=json.loads((root/'selection.json').read_text())['selected']
    target=selected['qps']
    protocol=json.loads((root/'protocol.json').read_text())
    protocol['nonly_search_revision']='After NMD selection, bracket starting at 65,536 nodes; '
    protocol['nonly_search_revision']+='double if needed, then 3 single-run bisections and two-run boundary confirmation. '
    protocol['nonly_search_revision']+='Chosen before observing any cached N-only result to avoid unnecessary tiny-capacity runs.'
    write(root/'protocol.json',protocol)
    baseline=mean_point(root,'nonly',0)
    low=0
    high=0 if baseline['qps']>=target else 65536
    while high and mean_point(root,'nonly',high)['qps']<target:
        low=high;high*=2
        if high>8388608:raise RuntimeError('No N-only crossing found.')
    for _ in range(3):
        if not high:break
        mid=((low+high)//2)//256*256
        if mid<=low:break
        if mean_point(root,'nonly',mid)['qps']>=target:high=mid
        else:low=mid
    matching=mean_point(root,'nonly',high,2)
    below=mean_point(root,'nonly',low,2) if high else None
    # A second run can move a close crossing. Keep all observations and require
    # a confirmed point above target before reporting a match.
    while matching['qps']<target:
        high+=max(8192,high//8)
        matching=mean_point(root,'nonly',high,2)
    baseline=mean_point(root,'nonly',0,2)
    nmd_memory=selected['extra_rss_mib']-baseline['extra_rss_mib']
    n_memory=matching['extra_rss_mib']-baseline['extra_rss_mib']
    write(root/'match.json',dict(selected_nmd=selected,nonly_match=matching,
                                lower_tested_point=below,no_cache_baseline=baseline,
                                nmd_cache_rss_mib=nmd_memory,nonly_cache_rss_mib=n_memory,
                                n_only_over_nmd_ram=n_memory/nmd_memory,target_qps=target,
                                crossing_resolution_nodes=8192))
    print('MATCH',json.dumps(json.loads((root/'match.json').read_text())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--point',nargs=3,metavar=('MODE','NODES','REP'))
    parser.add_argument('--match-only',action='store_true')
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,{23})
    if args.point:run(args.root,args.point[0],int(args.point[1]),int(args.point[2]))
    elif args.match_only:match_only(args.root)
    else:experiment(args.root)
