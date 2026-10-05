import unittest

from slugs import slugify


class SlugifyTest(unittest.TestCase):
    def test_two_words(self):
        self.assertEqual(slugify("hello world"), "hello-world")
