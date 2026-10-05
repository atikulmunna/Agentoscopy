import unittest

from durations import parse_duration


class ParseDurationHiddenTest(unittest.TestCase):
    def test_minutes(self):
        self.assertEqual(parse_duration("10m"), 600)
        self.assertEqual(parse_duration("1h30m"), 5400)

    def test_not_a_duration(self):
        for text in ("abc", "", "5x"):
            with self.assertRaises(ValueError):
                parse_duration(text)
