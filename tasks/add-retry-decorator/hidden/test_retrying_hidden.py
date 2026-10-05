import unittest

from retrying import retry


class RetryHiddenTest(unittest.TestCase):
    def test_succeeds_after_failures(self):
        calls = []

        @retry(3, (ConnectionError,))
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError
            return "ok"

        self.assertEqual(flaky(), "ok")
        self.assertEqual(len(calls), 3)

    def test_gives_up_with_the_last_error(self):
        calls = []

        @retry(2, (ConnectionError,))
        def broken():
            calls.append(1)
            raise ConnectionError(len(calls))

        with self.assertRaises(ConnectionError) as error:
            broken()
        self.assertEqual(error.exception.args, (2,))

    def test_other_errors_are_not_retried(self):
        calls = []

        @retry(5, (ConnectionError,))
        def wrong():
            calls.append(1)
            raise KeyError

        with self.assertRaises(KeyError):
            wrong()
        self.assertEqual(len(calls), 1)

    def test_keeps_metadata(self):
        @retry(2, (ConnectionError,))
        def named():
            """Docs."""

        self.assertEqual((named.__name__, named.__doc__), ("named", "Docs."))
