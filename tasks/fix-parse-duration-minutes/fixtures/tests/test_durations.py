import unittest

from durations import parse_duration


class ParseDurationTest(unittest.TestCase):
    def test_hours_and_seconds(self):
        self.assertEqual(parse_duration("2h"), 7200)
        self.assertEqual(parse_duration("45s"), 45)
