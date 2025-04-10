#ifndef EIGENVALUES_H
#define EIGENVALUES_H

#include <vector>
#include <utility>
#include <stdexcept>

// Все функции и реализации находятся в пространстве имён Eigenvalues.
namespace Eigenvalues {

    // Функция для нахождения собственных значений симметричной матрицы C с заданной точностью tol.
    // Возвращает вектор собственных значений, отсортированный по убыванию.
    std::vector<double> find_eigenvalues(const std::vector<std::vector<double>>& C, double tol);
}

extern "C" {
    // C-style интерфейс для вызова из других языков.
    // C – указатель на плоский массив (размер m x m), tol – требуемая точность.
    // Возвращается динамически выделенный массив длины m (освобождение памяти производится вызывающей стороной).
    double* find_eigenvalues(double* C, int m, double tol);
}

#endif // EIGENVALUES_H
