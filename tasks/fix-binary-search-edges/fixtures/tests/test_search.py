import unittest

from search import binary_search


class BinarySearchTest(unittest.TestCase):
    def test_middle(self):
        self.assertEqual(binary_search([1, 3, 5, 7, 9], 5), 2)
