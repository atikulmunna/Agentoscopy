import unittest

from config_merge import flatten_keys


class FlattenKeysTest(unittest.TestCase):
    def test_nested(self):
        self.assertEqual(flatten_keys({"a": {"b": 1}, "c": 2}), ["a.b", "c"])
