"""Check all host adapters against their original index records, not QPS.

Command manifests supply local binary/index/query paths. Each variant starts a
new process/cache. Existing outputs are never overwritten. The small resident
budget intentionally exercises private LRU eviction and asynchronous D writes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


def sha(path):
    with open(path, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commands', type=Path, required=True,
                        help='Directory containing starling_online/command.json, '
                             'margo_online, pipeann_online and aisaq_public_online')
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engines', nargs='+', default=['starling', 'margo', 'pipeann', 'aisaq'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for engine in args.engines:
        name = 'aisaq_public_online' if engine == 'aisaq' else engine + '_online'
        template = json.loads((args.commands / name / 'command.json').read_text())
        for trim in (0, 1):
            cfg = json.loads(json.dumps(template))
            cmd, env = cfg['command'], cfg['environment']
            out = args.output / f'{engine}_trim{trim}'
            out.mkdir()
            threads = 8 if engine == 'aisaq' else 20
            if 'taskset' in cmd:
                cmd[cmd.index('-c') + 1] = '0-23'
            env.update(NEIGHBORCACHE_BACKEND_LIBRARY=str(args.backend.resolve()),
                       NEIGHBORCACHE_PREFETCH_THREAD_PRIVATE='1',
                       NEIGHBORCACHE_PREFETCH_RESET_EACH_QUERY='0',
                       NEIGHBORCACHE_PREFETCH_TRIM_ZERO_TAIL=str(trim),
                       NEIGHBORCACHE_FREEZE_MAINTENANCE='0',
                       NEIGHBORCACHE_NCACHE_CAP_NODES='1024',
                       NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES=str(threads),
                       NEIGHBORCACHE_DCACHE_CAP='512',
                       NEIGHBORCACHE_DCACHE_SPACE_CAP_PAGES='512',
                       NEIGHBORCACHE_MCACHE_CAP='2048',
                       NEIGHBORCACHE_DCACHE_COMMIT_INLINE='0',
                       NEIGHBORCACHE_DCACHE_WRITER_THREADS='1',
                       NEIGHBORCACHE_DCACHE_COMMIT_THREADS='1',
                       NEIGHBORCACHE_DCACHE_SECOND_PAGE='0',
                       NEIGHBORCACHE_DCACHE_MAX_PAGES='1',
                       NEIGHBORCACHE_BG_CPUS='20-23',
                       NEIGHBORCACHE_HEAT_SAMPLE_SEED='20261031',
                       NEIGHBORCACHE_BACKEND_REPORT='1',
                       NEIGHBORCACHE_THREADS=str(threads),
                       NEIGHBORCACHE_VERIFY='1', PIPEANN_NMD_VERIFY='1', AISAQ_NMD_VERIFY='1')
            qi = 8 if engine == 'pipeann' else cmd.index('--query_file') + 1
            gi = 9 if engine == 'pipeann' else cmd.index('--gt_file') + 1
            with open(cmd[qi], 'rb') as source:
                n, dim = map(int, np.fromfile(source, dtype='<u4', count=2))
                assert n >= 100
                query = np.fromfile(source, dtype='u1', count=100 * dim)
            with open(cmd[gi], 'rb') as source:
                n, k = map(int, np.fromfile(source, dtype='<u4', count=2))
                assert n >= 100
                ids = np.fromfile(source, dtype='<u4', count=n * k).reshape(n, k)[:100]
                distances = np.fromfile(source, dtype='<f4', count=n * k).reshape(n, k)[:100]
            query_path, gt_path = out / 'query.u8bin', out / 'truth.bin'
            query_path.write_bytes(np.array([5000, dim], dtype='<u4').tobytes() + np.tile(query, 50).tobytes())
            gt_path.write_bytes(np.array([5000, k], dtype='<u4').tobytes() +
                                np.tile(ids, (50, 1)).tobytes() + np.tile(distances, (50, 1)).tobytes())
            cmd[qi], cmd[gi] = str(query_path.resolve()), str(gt_path.resolve())
            if engine == 'pipeann':
                env.update(PIPEANN_NMD_PATH=str(out / 'cache'), PIPEANN_DUMP=str(out / 'ids'))
            else:
                cmd[cmd.index('--result_path') + 1] = str(out / 'result')
                if engine == 'aisaq':
                    env['AISAQ_NMD_PATH'] = str(out / 'cache')
                else:
                    at = cmd.index('--disk_file_path') + 1
                    disk = Path(cmd[at]).resolve()
                    alias = out / disk.name
                    alias.symlink_to(disk)
                    cmd[at] = str(alias.absolute())
            cfg.update(purpose='Full-record correctness only; verification I/O invalidates timings',
                       backend_sha256=sha(args.backend))
            binary = next(x for x in cmd if x.endswith('/search_disk_index'))
            cfg['binary_sha256'] = sha(binary)
            write(out / 'command.json', cfg)
            actual = {key: value for key, value in os.environ.items()
                      if not key.startswith(('NEIGHBORCACHE_', 'NC_', 'PIPEANN_NMD', 'AISAQ_NMD'))}
            actual.update(env)
            print('START', engine, 'trim', trim, flush=True)
            with (out / 'run.out').open('w') as log:
                subprocess.run(cmd, env=actual, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            log = (out / 'run.out').read_text()
            assert 'mode=thread_private' in log
            row = re.search(r'NeighborCache shared backend: (.+)', log)
            assert row, out
            stats = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', row[1])}
            assert all(stats[key] > 0 for key in ('M', 'opens', 'writes', 'D_reads', 'prefetch')), stats
            summary.append(dict(engine=engine, trim_zero_tail=bool(trim), queries=5000,
                                full_record_verification=True, counters=stats))
            write(args.output / 'validation.json', summary)
            print('PASS', engine, 'trim', trim, stats, flush=True)


if __name__ == '__main__':
    main()
