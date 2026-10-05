import unittest

from csvline import parse_line


class ParseLineHiddenTest(unittest.TestCase):
    def test_quoted_comma(self):
        self.assertEqual(parse_line('a,"b,c",d'), ["a", "b,c", "d"])

    def test_escaped_quote(self):
        self.assertEqual(parse_line('"say ""hi""",x'), ['say "hi"', "x"])
