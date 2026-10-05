import unittest

from temperature import to_fahrenheit


class ToFahrenheitTest(unittest.TestCase):
    def test_boiling(self):
        self.assertEqual(to_fahrenheit(100), 212)
