import unittest

from temperature import to_celsius


class ToCelsiusTest(unittest.TestCase):
    def test_known_points(self):
        self.assertAlmostEqual(to_celsius(212), 100)
        self.assertAlmostEqual(to_celsius(32), 0)
        self.assertAlmostEqual(to_celsius(-40), -40)
