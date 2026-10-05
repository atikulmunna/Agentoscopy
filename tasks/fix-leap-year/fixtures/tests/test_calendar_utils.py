import unittest

from calendar_utils import is_leap


class LeapTest(unittest.TestCase):
    def test_ordinary_years(self):
        self.assertTrue(is_leap(2024))
        self.assertFalse(is_leap(2023))
