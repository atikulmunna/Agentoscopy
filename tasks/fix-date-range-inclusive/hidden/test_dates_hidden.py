import unittest
from datetime import date

from dates import date_range


class DateRangeHiddenTest(unittest.TestCase):
    def test_includes_end(self):
        days = date_range(date(2026, 2, 27), date(2026, 3, 1))
        self.assertEqual(days, [date(2026, 2, 27), date(2026, 2, 28), date(2026, 3, 1)])

    def test_single_day(self):
        self.assertEqual(date_range(date(2026, 1, 1), date(2026, 1, 1)), [date(2026, 1, 1)])

    def test_backwards(self):
        with self.assertRaises(ValueError):
            date_range(date(2026, 1, 2), date(2026, 1, 1))
