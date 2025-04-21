import unittest
import math
from pcalib.py.algorithms import Matrix, handle_missing_values, add_noise_and_compare, mean_by_column


class TestUtils(unittest.TestCase):
    # Проверяет заполнение пропущенных значений средними по столбцу
    def test_handle_missing_values(self):
        X = Matrix([[1, math.nan], [3, 4]])
        X_filled = handle_missing_values(X)

        self.assertAlmostEqual(X_filled.data[0][1], 4.0)

    # Проверяет вычисление средних по столбцам
    def test_mean_by_column(self):
        X = Matrix([[1, 2], [3, 4]])
        means = mean_by_column(X)

        self.assertEqual(means, [2, 3])

    # Проверяет, что после добавления шума проекция данных меняется
    def test_add_noise_and_compare(self):
        X = Matrix([[1, 2], [3, 4], [5, 6]])
        result = add_noise_and_compare(X, noise_level=0.5)
        before = result['X_proj'].data
        after = result['X_proj_noisy'].data

        self.assertNotEqual(before, after)


if __name__ == '__main__':
    unittest.main()
