#ifndef EIGENVECTORS_H
#define EIGENVECTORS_H

#include <vector>

namespace Eigenvectors {
    /**
     * Находит собственные векторы матрицы C для заданных собственных значений
     * 
     * @param C Матрица ковариаций (m×m)
     * @param eigenvalues Список собственных значений
     * @return std::vector<std::vector<double>> Список собственных векторов
     */
    std::vector<std::vector<double> > find_eigenvectors(
        const std::vector<std::vector<double> > &C,
        const std::vector<double> &eigenvalues);

    /**
     * Решает систему уравнений (C-λI)v = 0 для заданного собственного значения λ
     * 
     * @param C Матрица ковариаций
     * @param lambda Собственное значение
     * @return std::vector<double> Собственный вектор
     */
    std::vector<double> solve_eigenvector_equation(
        const std::vector<std::vector<double> > &C,
        double lambda);

    /**
     * Создает матрицу C-λI для заданного значения λ
     * 
     * @param C Исходная матрица
     * @param lambda Значение λ
     * @return std::vector<std::vector<double>> Матрица C-λI
     */
    std::vector<std::vector<double> > create_characteristic_matrix(
        const std::vector<std::vector<double> > &C,
        double lambda);

    /**
     * Нормализует вектор (приводит к единичной длине)
     * 
     * @param v Вектор для нормализации
     * @return std::vector<double> Нормализованный вектор
     */
    std::vector<double> normalize_vector(const std::vector<double> &v);
}

extern "C" {
double *find_eigenvectors(double *C, double *eigenvalues, int m, int n_eigenvalues);
}

#endif // EIGENVECTORS_H
