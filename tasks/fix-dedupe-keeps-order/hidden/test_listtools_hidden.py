import unittest

from listtools import dedupe


class DedupeHiddenTest(unittest.TestCase):
    def test_keeps_first_occurrence_order(self):
        self.assertEqual(dedupe([3, 1, 3, 2, 1]), [3, 1, 2])
        self.assertEqual(dedupe(["b", "a", "b"]), ["b", "a"])
