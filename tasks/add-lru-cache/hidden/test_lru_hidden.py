import unittest

from lru import LRUCache


class LRUCacheHiddenTest(unittest.TestCase):
    def test_evicts_least_recently_used(self):
        cache = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")
        cache.put("c", 3)
        self.assertEqual((cache.get("a"), cache.get("b"), cache.get("c")), (1, None, 3))
        self.assertEqual(len(cache), 2)

    def test_put_refreshes_and_updates(self):
        cache = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("a", 10)
        cache.put("c", 3)
        self.assertEqual((cache.get("a"), cache.get("b")), (10, None))

    def test_default_and_bad_capacity(self):
        self.assertEqual(LRUCache(1).get("x", "none"), "none")
        with self.assertRaises(ValueError):
            LRUCache(0)
