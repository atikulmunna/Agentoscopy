import unittest

from stats import median


class MedianHiddenTest(unittest.TestCase):
    def test_even_length(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_single_value(self):
        self.assertEqual(median([7]), 7)

    def test_empty(self):
        with self.assertRaises(ValueError):
            median([])
