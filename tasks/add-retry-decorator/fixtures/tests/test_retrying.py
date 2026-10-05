import unittest

from retrying import is_transient


class IsTransientTest(unittest.TestCase):
    def test_kinds(self):
        self.assertTrue(is_transient(TimeoutError()))
        self.assertFalse(is_transient(ValueError()))
