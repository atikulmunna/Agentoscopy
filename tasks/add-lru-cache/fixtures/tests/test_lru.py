import unittest

from lru import describe


class DescribeTest(unittest.TestCase):
    def test_describe(self):
        self.assertEqual(describe("pages", 3), "pages (holds 3)")
