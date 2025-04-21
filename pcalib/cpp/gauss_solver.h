#ifndef GAUSS_SOLVER_H
#define GAUSS_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Решает систему линейных уравнений Ax = b методом Гаусса
 *
 * @param A_in Входная матрица коэффициентов
 * @param b_in Вектор правых частей (размер n)
 * @param x    Вектор решения (размер n), результат записывается сюда
 * @param n    Размер системы (число уравнений)
 * @return 0 если успешно, -1 если система несовместна или матрица вырождена
 */
int gauss_solver(const double *A_in, const double *b_in, double *x, int n);

#ifdef __cplusplus
}
#endif

#endif // GAUSS_SOLVER_H
