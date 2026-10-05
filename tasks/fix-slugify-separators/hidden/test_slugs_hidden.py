import unittest

from slugs import slugify


class SlugifyHiddenTest(unittest.TestCase):
    def test_collapses_separators(self):
        self.assertEqual(slugify("Hello,  World!"), "hello-world")

    def test_strips_the_ends(self):
        self.assertEqual(slugify("  --Top 10 Tips--  "), "top-10-tips")
