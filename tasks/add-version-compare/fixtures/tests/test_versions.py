import unittest

from versions import parse


class ParseTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse("1.2.10"), [1, 2, 10])
