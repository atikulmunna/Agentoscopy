import copy
import unittest

from config_merge import deep_merge


class DeepMergeHiddenTest(unittest.TestCase):
    def test_nested_merge(self):
        base = {"db": {"host": "x", "port": 1}, "debug": False, "tags": ["a"]}
        override = {"db": {"port": 2}, "debug": True, "tags": ["b"]}
        self.assertEqual(
            deep_merge(base, override),
            {"db": {"host": "x", "port": 2}, "debug": True, "tags": ["b"]},
        )

    def test_inputs_unchanged(self):
        base, override = {"a": {"b": 1}}, {"a": {"c": 2}}
        before = copy.deepcopy((base, override))
        deep_merge(base, override)
        self.assertEqual((base, override), before)

    def test_dict_replaces_scalar(self):
        self.assertEqual(deep_merge({"a": 1}, {"a": {"b": 2}}), {"a": {"b": 2}})
