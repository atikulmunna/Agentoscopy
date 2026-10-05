import unittest

from series import moving_average


class MovingAverageHiddenTest(unittest.TestCase):
    def test_windows(self):
        self.assertEqual(moving_average([1, 2, 3, 4, 5], 2), [1.5, 2.5, 3.5, 4.5])
        self.assertEqual(moving_average([2, 4, 6], 3), [4])

    def test_window_longer_than_data(self):
        self.assertEqual(moving_average([1, 2], 3), [])

    def test_bad_window(self):
        with self.assertRaises(ValueError):
            moving_average([1, 2], 0)
