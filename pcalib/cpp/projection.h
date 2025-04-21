#ifndef PROJECTION_H
#define PROJECTION_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Проецирует матрицу X (n x m) на матрицу главных компонент Vk (m x k)
 * На выходе X_proj[i * k + j] = sum_l X[i * m + l] * Vk[l * k + j]
 *
 * @param X        Входная матрица
 * @param Vk       Матрица главных компонент
 * @param X_proj   Выходная матрица
 * @param n        Количество строк во входной матрице X
 * @param m        Количество столбцов во входной матрице X (и строк в Vk)
 * @param k        Количество главных компонент (столбцов в Vk)
 */
void project_data(const double *X, const double *Vk, double *X_proj, int n, int m, int k);

#ifdef __cplusplus
}
#endif

#endif //PROJECTION_H
