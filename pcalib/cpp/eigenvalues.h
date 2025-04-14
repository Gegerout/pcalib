#ifndef EIGENVALUES_H
#define EIGENVALUES_H

#include <vector>
#include <utility>
#include <stdexcept>

namespace Eigenvalues {

    /**
     * @brief Находит собственные значения матрицы C.
     * 
     * @param C Матрица ковариаций (m×m).
     * @param tol Допустимая погрешность для определения сходимости.
     * @return std::vector<double> Список собственных значений.
     */
    std::vector<double> find_eigenvalues(const std::vector<std::vector<double>>& C, double tol);
}

extern "C" {

/**
 * @brief Находит собственные значения матрицы C.
 * 
 * @param C Указатель на одномерный массив, представляющий матрицу (m×m).
 * @param m Размер матрицы.
 * @param tol Допустимая погрешность для определения сходимости.
 * @return double* Указатель на массив собственных значений.
 *                 Вызывающий код должен освободить память с помощью delete[].
 */
double* find_eigenvalues(double* C, int m, double tol);

}

#endif // EIGENVALUES_H
