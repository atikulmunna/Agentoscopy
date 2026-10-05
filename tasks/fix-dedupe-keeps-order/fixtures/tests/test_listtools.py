import unittest

from listtools import dedupe


class DedupeTest(unittest.TestCase):
    def test_removes_duplicates(self):
        self.assertEqual(sorted(dedupe([3, 1, 3])), [1, 3])
