#include "eigenvalues.h"
#include "matrix_ops.h"   // Используется для обмена строк, если потребуется в дальнейших доработках
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstdlib>        // Для std::abs (при необходимости)

// Вспомогательная функция для вычисления детерминанта квадратной матрицы методом Гаусса.
// Предназначена для малых размеров (подматриц ведущих миноров).
static double determinant(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    if(n == 0 || A[0].size() != n)
        throw std::invalid_argument("Матрица должна быть квадратной");
    
    std::vector<std::vector<double>> mat = A; // копия матрицы
    double det = 1.0;
    const double eps = 1e-12;
    
    for (int i = 0; i < n; i++) {
        // Поиск опорного элемента
        int pivot = i;
        for (int j = i; j < n; j++) {
            if (std::abs(mat[j][i]) > std::abs(mat[pivot][i]))
                pivot = j;
        }
        if (std::abs(mat[pivot][i]) < eps)
            return 0.0; // матрица вырождена
        
        if (pivot != i) {
            std::swap(mat[i], mat[pivot]);
            det *= -1.0;
        }
        det *= mat[i][i];
        double invPivot = 1.0 / mat[i][i];
        for (int j = i + 1; j < n; j++) {
            double factor = mat[j][i] * invPivot;
            for (int k = i; k < n; k++) {
                mat[j][k] -= factor * mat[i][k];
            }
        }
    }
    return det;
}

// Все внутренние функции помещены в безымянное пространство имён для локальности.
namespace {

    // Вычисляет детерминант ведущей главной подматрицы размера k x k матрицы (C - lambda I)
    double submatrix_determinant(const std::vector<std::vector<double>>& C, double lambda, int k) {
        std::vector<std::vector<double>> sub(k, std::vector<double>(k, 0.0));
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                sub[i][j] = C[i][j];
                if (i == j)
                    sub[i][j] -= lambda;
            }
        }
        return determinant(sub);
    }

    // Возвращает знак x с учётом порогового значения tol_sign:
    // если x > tol_sign, то +1, иначе (включая ноль) -1.
    int sign_val(double x, double tol_sign = 1e-12) {
        return (x > tol_sign) ? 1 : -1;
    }

    // По числу изменений знака в последовательности детерминантов ведущих главных миноров
    // матрицы (C - lambda I) определяется число собственных значений, меньших lambda.
    // Согласно теореме Стурма это число равно количеству изменений знака в последовательности:
    // p0 = 1, p1, p2, …, pn, где p_i — детерминант i x i подматрицы.
    int count_eigenvalues_less_than(const std::vector<std::vector<double>>& C, double lambda) {
        int n = C.size();
        int sign_changes = 0;
        int prev_sign = 1; // p0 считается положительным

        for (int i = 1; i <= n; i++) {
            double p_i = submatrix_determinant(C, lambda, i);
            int cur_sign = sign_val(p_i);
            if (cur_sign != prev_sign)
                sign_changes++;
            prev_sign = cur_sign;
        }
        return sign_changes;
    }

    // Определяет границы собственных значений с помощью теоремы Гершгорина.
    std::pair<double, double> find_eigenvalue_bounds(const std::vector<std::vector<double>>& C) {
        int n = C.size();
        double min_bound = std::numeric_limits<double>::infinity();
        double max_bound = -std::numeric_limits<double>::infinity();
        
        for (int i = 0; i < n; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i != j)
                    row_sum += std::abs(C[i][j]);
            }
            double lower = C[i][i] - row_sum;
            double upper = C[i][i] + row_sum;
            min_bound = std::min(min_bound, lower);
            max_bound = std::max(max_bound, upper);
        }
        double margin = 0.1 * (max_bound - min_bound);
        return {min_bound - margin, max_bound + margin};
    }

} // end anonymous namespace

namespace Eigenvalues {

    // Основная функция поиска собственных значений.
    // Для каждого собственного числа (в порядке возрастания) осуществляется двоичный поиск по интервалу,
    // в котором число собственных значений, меньших mid, равно целевому индексу.
    // Результат сортируется по убыванию.
    std::vector<double> find_eigenvalues(const std::vector<std::vector<double>>& C, double tol) {
        int n = C.size();
        if (n == 0 || C[0].size() != static_cast<size_t>(n))
            throw std::invalid_argument("Матрица должна быть квадратной");

        auto bounds = find_eigenvalue_bounds(C);
        double lower_bound = bounds.first;
        double upper_bound = bounds.second;

        std::vector<double> eigenvalues_asc(n, 0.0);

        // Ищем i-е собственное значение (индекс j от 1 до n — по возрастанию).
        for (int j = 1; j <= n; j++) {
            double left = lower_bound;
            double right = upper_bound;
            while (right - left > tol) {
                double mid = (left + right) / 2.0;
                int count = count_eigenvalues_less_than(C, mid);
                if (count < j)
                    left = mid;
                else
                    right = mid;
            }
            eigenvalues_asc[j - 1] = (left + right) / 2.0;
        }
        std::sort(eigenvalues_asc.begin(), eigenvalues_asc.end(), std::greater<double>());
        return eigenvalues_asc;
    }

} // namespace Eigenvalues

extern "C" {

    double* find_eigenvalues(double* C, int m, double tol) {
        std::vector<std::vector<double>> matrix(m, std::vector<double>(m, 0.0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                matrix[i][j] = C[i * m + j];
            }
        }
        std::vector<double> eigen = Eigenvalues::find_eigenvalues(matrix, tol);
        double* result = new double[m];
        for (int i = 0; i < m; i++)
            result[i] = eigen[i];
        return result;
    }
}
