import unittest

from roman import to_roman


class ToRomanHiddenTest(unittest.TestCase):
    def test_subtractive(self):
        self.assertEqual(to_roman(4), "IV")
        self.assertEqual(to_roman(9), "IX")
        self.assertEqual(to_roman(40), "XL")
        self.assertEqual(to_roman(1994), "MCMXCIV")

    def test_out_of_range(self):
        for number in (0, 4000, -3):
            with self.assertRaises(ValueError):
                to_roman(number)
