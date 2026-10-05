import unittest
from datetime import date

from dates import date_range


class DateRangeTest(unittest.TestCase):
    def test_starts_at_start(self):
        self.assertEqual(date_range(date(2026, 1, 1), date(2026, 1, 5))[0], date(2026, 1, 1))
