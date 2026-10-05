import unittest

from textstats import word_count


class WordCountTest(unittest.TestCase):
    def test_simple_sentence(self):
        self.assertEqual(word_count("one two three"), 3)
