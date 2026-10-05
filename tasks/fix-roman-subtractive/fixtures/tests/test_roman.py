import unittest

from roman import to_roman


class ToRomanTest(unittest.TestCase):
    def test_additive(self):
        self.assertEqual(to_roman(3), "III")
        self.assertEqual(to_roman(2026), "MMXXVI")
