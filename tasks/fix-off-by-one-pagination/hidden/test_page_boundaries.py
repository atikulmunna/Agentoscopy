import unittest

from shop.pagination import paginate


class PageBoundaryTest(unittest.TestCase):
    def test_second_page_starts_after_first(self):
        self.assertEqual(paginate(list(range(10)), page=2, per_page=3), [3, 4, 5])

    def test_last_partial_page(self):
        self.assertEqual(paginate(list(range(10)), page=4, per_page=3), [9])


if __name__ == "__main__":
    unittest.main()
