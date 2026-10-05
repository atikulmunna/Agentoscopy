import unittest

from calendar_utils import is_leap


class LeapHiddenTest(unittest.TestCase):
    def test_centuries(self):
        self.assertFalse(is_leap(1900))
        self.assertFalse(is_leap(2100))
        self.assertTrue(is_leap(2000))
