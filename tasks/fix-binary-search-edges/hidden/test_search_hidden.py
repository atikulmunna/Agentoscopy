import unittest

from search import binary_search


class BinarySearchHiddenTest(unittest.TestCase):
    def test_ends(self):
        items = [1, 3, 5, 7, 9]
        self.assertEqual(binary_search(items, 1), 0)
        self.assertEqual(binary_search(items, 9), 4)

    def test_missing_and_empty(self):
        self.assertEqual(binary_search([1, 3, 5], 4), -1)
        self.assertEqual(binary_search([1, 3, 5], 10), -1)
        self.assertEqual(binary_search([], 1), -1)
