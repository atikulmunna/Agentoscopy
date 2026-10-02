import unittest

from shop.pagination import paginate


class FirstPageTest(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(paginate(list(range(10)), page=1, per_page=3), [0, 1, 2])

    def test_empty_list(self):
        self.assertEqual(paginate([], page=1, per_page=3), [])


if __name__ == "__main__":
    unittest.main()
