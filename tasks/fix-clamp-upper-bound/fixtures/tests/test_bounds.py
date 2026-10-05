import unittest

from bounds import clamp


class ClampTest(unittest.TestCase):
    def test_inside_and_below(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-1, 0, 10), 0)
