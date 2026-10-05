import unittest

from intervals import merge_intervals


class MergeIntervalsHiddenTest(unittest.TestCase):
    def test_overlapping(self):
        self.assertEqual(merge_intervals([(1, 3), (2, 6), (8, 10)]), [(1, 6), (8, 10)])

    def test_unsorted_and_touching(self):
        self.assertEqual(merge_intervals([(5, 6), (1, 2), (2, 4)]), [(1, 4), (5, 6)])

    def test_contained_and_empty(self):
        self.assertEqual(merge_intervals([(1, 10), (2, 3)]), [(1, 10)])
        self.assertEqual(merge_intervals([]), [])
