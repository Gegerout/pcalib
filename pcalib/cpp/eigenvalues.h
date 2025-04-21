#ifndef EIGENVALUES_H
#define EIGENVALUES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Вычисляет собственные значения симметричной матрицы
 * Находит все собственные значения вещественной симметричной матрицы C размером m×m
 * методом бисекции с использованием разложения LDL^T
 *
 * @param C Указатель на массив double
 * @param m Размер матрицы (число строк/столбцов)
 * @param tol Допустимая погрешность для поиска корней
 * @return double* Указатель на массив собственных значений (размер m)
 */
double *find_eigenvalues(double *C, int m, double tol);

#ifdef __cplusplus
}
#endif

#endif // EIGENVALUES_H
