import unittest

from versions import compare_versions


class CompareVersionsHiddenTest(unittest.TestCase):
    def test_numeric_parts(self):
        self.assertEqual(compare_versions("1.2.10", "1.2.9"), 1)
        self.assertEqual(compare_versions("1.9", "1.10"), -1)

    def test_equal_and_padded(self):
        self.assertEqual(compare_versions("2.0", "2.0"), 0)
        self.assertEqual(compare_versions("1.0", "1.0.0"), 0)
        self.assertEqual(compare_versions("1.0.1", "1"), 1)
