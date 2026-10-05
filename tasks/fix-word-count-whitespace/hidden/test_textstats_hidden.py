import unittest

from textstats import word_count


class WordCountHiddenTest(unittest.TestCase):
    def test_repeated_spaces(self):
        self.assertEqual(word_count("one  two   three"), 3)

    def test_other_whitespace(self):
        self.assertEqual(word_count("one\ttwo\nthree "), 3)

    def test_empty(self):
        self.assertEqual(word_count(""), 0)
        self.assertEqual(word_count("   "), 0)
