import tempfile
import unittest
from pathlib import Path
import numpy as np
from cache_validation import matrix, validate_cache_outputs, checked_binary_digest, KNOWN_UNSAFE_BINARIES


class CacheValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.q = np.array([[1, 2], [3, 4]], dtype='i1')
        self.ids = np.array([[1, 2], [3, 4]], dtype='<u4')
        self.dist = np.array([[1., 2.], [3., 4.]], dtype='<f4')
        self.write('rq', self.q)
        self.write('cq', self.q[[1, 0, 1]])
        self.write('ref_64_idx_uint32.bin', self.ids)
        self.write('ref_64_dists_float.bin', self.dist)
        self.write('nc_64_idx_uint32.bin', self.ids[[1, 0, 1]])
        self.write('nc_64_dists_float.bin', self.dist[[1, 0, 1]])

    def write(self, name, values):
        with (self.root/name).open('wb') as stream:
            np.array(values.shape, dtype='<u4').tofile(stream)
            values.tofile(stream)

    def check(self):
        return validate_cache_outputs(self.root/'ref', self.root/'nc', 64,
                                      self.root/'rq', self.root/'cq', 'i1')

    def test_repeated_shuffled_queries(self):
        self.assertEqual(self.check()['different_ID_queries'], 0)

    def test_same_recall_different_output_rejected(self):
        self.write('nc_64_idx_uint32.bin', self.ids[[1, 0, 1]][:, ::-1].copy())
        with self.assertRaisesRegex(ValueError, 'changed search'): self.check()

    def test_one_bit_distance_error_rejected(self):
        dist = self.dist[[1, 0, 1]].copy(); dist.view('<u4')[1, 0] += 1
        self.write('nc_64_dists_float.bin', dist)
        with self.assertRaisesRegex(ValueError, 'changed search'): self.check()

    def test_wrong_workload_rejected(self):
        self.write('cq', np.array([[9, 9], [1, 2], [3, 4]], dtype='i1'))
        with self.assertRaisesRegex(ValueError, 'absent'): self.check()

    def test_truncated_file_rejected(self):
        p = self.root/'nc_64_idx_uint32.bin'; p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'Truncated'): self.check()

    def test_known_bad_binary_rejected(self):
        p = self.root/'binary'; p.write_bytes(b'test fixture')
        digest = checked_binary_digest(p); KNOWN_UNSAFE_BINARIES[digest] = 'test'
        try:
            with self.assertRaisesRegex(RuntimeError, 'known-unsafe'): checked_binary_digest(p)
        finally: del KNOWN_UNSAFE_BINARIES[digest]


if __name__ == '__main__': unittest.main()
