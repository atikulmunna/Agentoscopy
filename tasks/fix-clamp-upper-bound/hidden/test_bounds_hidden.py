import unittest

from bounds import clamp


class ClampHiddenTest(unittest.TestCase):
    def test_above(self):
        self.assertEqual(clamp(15, 0, 10), 10)

    def test_inverted_bounds(self):
        with self.assertRaises(ValueError):
            clamp(1, 10, 0)
