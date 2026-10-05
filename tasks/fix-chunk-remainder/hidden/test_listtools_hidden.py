import unittest

from listtools import chunk


class ChunkHiddenTest(unittest.TestCase):
    def test_keeps_the_remainder(self):
        self.assertEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

    def test_bad_size(self):
        with self.assertRaises(ValueError):
            chunk([1, 2], 0)
