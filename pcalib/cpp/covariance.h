#ifndef COVARIANCE_H
#define COVARIANCE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Вычисляет ковариационную матрицу для центрированной матрицы X (n x m).
 * На выходе X_covariance[i * m + j] = ковариация между признаками i и j.
 * Формула: C = (X^T X) / (n-1)
 *
 * @param X_in         Входная центрированная матрица (размер n x m, row-major)
 * @param X_covariance Выходная ковариационная матрица (размер m x m, row-major)
 * @param n            Количество строк (объектов)
 * @param m            Количество столбцов (признаков)
 */
void covariance_matrix(const double *X_in, double *X_covariance, int n, int m);

#ifdef __cplusplus
}
#endif

#endif // COVARIANCE_H
