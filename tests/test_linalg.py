import unittest
from pcalib.py.algorithms import Matrix, gauss_solver, center_data, covariance_matrix, find_eigenvalues, \
    find_eigenvectors


class TestLinalg(unittest.TestCase):
    # Проверяет решение СЛАУ методом Гаусса
    def test_gauss_solver(self):
        A = Matrix([[2, 1], [1, 3]])
        b = [5, 10]
        x = gauss_solver(A, b)

        self.assertAlmostEqual(x[0], 1.0, places=5)
        self.assertAlmostEqual(x[1], 3.0, places=5)

    # Проверяет корректность центрирования матрицы
    def test_center_data(self):
        X = Matrix([[1, 2], [3, 4]])
        Xc = center_data(X)

        self.assertAlmostEqual(Xc.data[0][0], -1.0)
        self.assertAlmostEqual(Xc.data[0][1], -1.0)
        self.assertAlmostEqual(Xc.data[1][0], 1.0)
        self.assertAlmostEqual(Xc.data[1][1], 1.0)

    # Проверяет вычисление матрицы ковариаций
    def test_covariance_matrix(self):
        X = Matrix([[1, 2], [3, 4]])
        Xc = center_data(X)
        C = covariance_matrix(Xc)

        self.assertAlmostEqual(C.data[0][0], 2.0)
        self.assertAlmostEqual(C.data[0][1], 2.0)
        self.assertAlmostEqual(C.data[1][0], 2.0)
        self.assertAlmostEqual(C.data[1][1], 2.0)

    # Проверяет поиск собственных значений и векторов
    def test_find_eigenvalues_and_vectors(self):
        C = Matrix([[2, 0], [0, 1]])
        eigenvalues = find_eigenvalues(C)
        eigenvalues_sorted = sorted(eigenvalues)

        self.assertAlmostEqual(eigenvalues_sorted[0], 1.0, places=5)
        self.assertAlmostEqual(eigenvalues_sorted[1], 2.0, places=5)

        eigenvectors = find_eigenvectors(C, eigenvalues_sorted)
        for i, val in enumerate(eigenvalues_sorted):
            v = [eigenvectors[i].data[0][0], eigenvectors[i].data[1][0]]
            Av = [C.data[0][0] * v[0] + C.data[0][1] * v[1], C.data[1][0] * v[0] + C.data[1][1] * v[1]]
            lv = [val * v[0], val * v[1]]

            self.assertAlmostEqual(Av[0], lv[0], places=5)
            self.assertAlmostEqual(Av[1], lv[1], places=5)


if __name__ == '__main__':
    unittest.main()
