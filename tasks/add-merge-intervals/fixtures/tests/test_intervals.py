import unittest

from intervals import length


class LengthTest(unittest.TestCase):
    def test_length(self):
        self.assertEqual(length((2, 7)), 5)
