#ifndef CENTERING_H
#define CENTERING_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Центрирует матрицу данных X (n x m) по каждому столбцу
 * Если means == nullptr, вычисляет средние по X
 * Если means != nullptr, использует их для центрирования
 * На выходе X_centered[i * m + j] = X[i * m + j] - means[j]
 *
 * @param X           Входная матрица
 * @param X_centered  Выходная матрица
 * @param n           Количество строк
 * @param m           Количество столбцов
 * @param means       Массив средних (размер m) или nullptr
 */
void center_data(const double *X, double *X_centered, int n, int m, const double *means);

#ifdef __cplusplus
}
#endif

#endif // CENTERING_H
