#ifndef CENTERING_H
#define CENTERING_H

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Центрирует матрицу данных X (n x m) по каждому столбцу.
     * На выходе X_centered[i * m + j] = X[i * m + j] - mean_j,
     * где mean_j — среднее по j-му столбцу.
     *
     * @param X           Входная матрица (размер n x m, row-major)
     * @param X_centered  Выходная матрица (размер n x m, row-major)
     * @param n           Количество строк
     * @param m           Количество столбцов
     */
    void center_data(const double* X, double* X_centered, int n, int m);

#ifdef __cplusplus
}
#endif

#endif // CENTERING_H
