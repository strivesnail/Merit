"""Fail closed before accepting cache-only native-search benchmark results.

Applicable when caching must preserve the search, not across different algorithms.
The reference must use the same index, L, W, metric and query vectors.
"""
import hashlib
from pathlib import Path

import numpy as np


# Preserve the old binary for provenance, but never silently benchmark it again.
KNOWN_UNSAFE_BINARIES = {
    "8a7d186ad93c19ea5d04aa4b09c1190546e9b0a1e792a841ce2b815594eac5b7":
        "Prefetch vectors lack aligned zero padding (SPACEV100D regression).",
}

# Saved command manifests still contain these options.
LEGACY_MCACHE_HEAT_OPTIONS = frozenset({
    "NEIGHBORCACHE_MCACHE_NODE_HEAT_PROBABILITY",
    "NEIGHBORCACHE_MCACHE_ADAPTIVE_UPDATE",
    "NEIGHBORCACHE_MCACHE_EXPAND_SAMPLE",
    "NEIGHBORCACHE_HEAT_SAMPLE_SEED",
    "MERIT_MCACHE_ADAPTIVE_UPDATE",
    "MERIT_MCACHE_EXPAND_SAMPLE",
    "NC_DIAG_SAMPLING_MODE",
    "NC_DIAG_HEAT_MULTIPLIER",
})


def checked_binary_digest(path, *, full_mcache_heat=False):
    binary = Path(path).read_bytes()
    digest = hashlib.sha256(binary).hexdigest()
    if digest in KNOWN_UNSAFE_BINARIES:
        raise RuntimeError(f"Rejected known-unsafe binary {path}: {KNOWN_UNSAFE_BINARIES[digest]}")
    if full_mcache_heat and b"NEIGHBORCACHE mcache heat: update=every_expansion" not in binary:
        raise RuntimeError(f"{path} predates unconditional M-cache heat updates. Rebuild it from current source.")
    return digest


def matrix(path, dtype):
    path = Path(path)
    with path.open('rb') as stream:
        shape = np.fromfile(stream, dtype='<u4', count=2)
    if len(shape) != 2 or not all(shape):
        raise ValueError(f"Invalid matrix header: {path}")
    shape = tuple(map(int, shape))
    expected = 8 + shape[0] * shape[1] * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"Truncated/trailing matrix data: {path}")
    return np.memmap(path, mode='r', dtype=dtype, offset=8, shape=shape)


def validate_cache_outputs(reference_prefix, candidate_prefix, L,
                           reference_queries, candidate_queries, query_dtype):
    rq = matrix(reference_queries, query_dtype)
    cq = matrix(candidate_queries, query_dtype)
    ri = matrix(f'{reference_prefix}_{L}_idx_uint32.bin', '<u4')
    ci = matrix(f'{candidate_prefix}_{L}_idx_uint32.bin', '<u4')
    rd = matrix(f'{reference_prefix}_{L}_dists_float.bin', '<u4')
    cd = matrix(f'{candidate_prefix}_{L}_dists_float.bin', '<u4')
    if (rq.shape[1] != cq.shape[1] or ri.shape != rd.shape or ci.shape != cd.shape
            or len(ri) != len(rq) or len(ci) != len(cq) or ri.shape[1] != ci.shape[1]):
        raise ValueError('Query/result dimensions do not match')
    if not np.isfinite(rd.view('<f4')).all() or not np.isfinite(cd.view('<f4')).all():
        raise ValueError('Non-finite result distances')
    lookup = {}
    for row, query in enumerate(rq):
        key = query.tobytes()
        previous = lookup.get(key)
        if previous is not None and (not np.array_equal(ri[row], ri[previous])
                                     or not np.array_equal(rd[row], rd[previous])):
            raise ValueError('Reference has inconsistent outputs for identical queries')
        lookup[key] = row
    try:
        rows = np.array([lookup[q.tobytes()] for q in cq])
    except KeyError as error:
        raise ValueError('Candidate query is absent from the reference workload') from error
    bad_ids = np.any(ci != ri[rows], axis=1)
    bad_distances = np.any(cd != rd[rows], axis=1)
    report = dict(queries=len(ci), different_ID_queries=int(bad_ids.sum()),
                  different_distance_queries=int(bad_distances.sum()))
    if bad_ids.any() or bad_distances.any():
        first = np.flatnonzero(bad_ids | bad_distances)[:10].tolist()
        raise ValueError(f'Cache changed search outputs: {report}; first query rows={first}')
    return report
