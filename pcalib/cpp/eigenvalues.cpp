#include "eigenvalues.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>

// Возвращает количество собственных значений матрицы C, меньших x, через разложение LDL^T
int sturm_count_full(const std::vector<std::vector<double> > &C, double x, double eps = 1e-12) {
    int n = C.size();
    std::vector<std::vector<double> > L(n, std::vector<double>(n, 0.0));
    std::vector<double> D(n, 0.0);
    for (int i = 0; i < n; ++i) L[i][i] = 1.0;
    for (int k = 0; k < n; ++k) {
        double sum_LDL = 0.0;
        for (int j = 0; j < k; ++j)
            sum_LDL += L[k][j] * L[k][j] * D[j];
        double Dk = C[k][k] - x - sum_LDL;
        if (std::abs(Dk) < eps) Dk = (Dk < 0 ? -eps : eps);
        D[k] = Dk;
        for (int i = k + 1; i < n; ++i) {
            double sum_LDL2 = 0.0;
            for (int j = 0; j < k; ++j)
                sum_LDL2 += L[i][j] * L[k][j] * D[j];
            L[i][k] = (C[i][k] - sum_LDL2) / Dk;
        }
    }
    int cnt = 0;
    for (int k = 0; k < n; ++k)
        if (D[k] < 0) ++cnt;
    return cnt;
}

// Рекурсивно делит отрезок и ищет собственные значения методом бисекции
void bisect_interval(const std::vector<std::vector<double> > &C,
                     double a, double b,
                     int cnt_a, int cnt_b,
                     double tol,
                     std::vector<double> &out) {
    int roots = cnt_b - cnt_a;
    if (roots == 0) return;
    if ((b - a) < tol) {
        double mid = 0.5 * (a + b);
        for (int i = 0; i < roots; ++i) out.push_back(mid);
        return;
    }
    double m = 0.5 * (a + b);
    int cnt_m = sturm_count_full(C, m);
    bisect_interval(C, a, m, cnt_a, cnt_m, tol, out);
    bisect_interval(C, m, b, cnt_m, cnt_b, tol, out);
}

// Находит все собственные значения симметричной матрицы C методом бисекции и разложения LDL^T
std::vector<double> find_eigenvalues(const std::vector<std::vector<double> > &C, double tol) {
    int n = C.size();
    if (n == 0 || C[0].size() != static_cast<size_t>(n))
        throw std::invalid_argument("Матрица должна быть квадратной");
    double L = 0.0;
    double U = 0.0;
    for (int i = 0; i < n; ++i) U += C[i][i];
    U += tol;
    int cnt_L = sturm_count_full(C, L);
    int cnt_U = sturm_count_full(C, U);
    if (cnt_U - cnt_L != n)
        throw std::runtime_error(
            "Ожидалось " + std::to_string(n) + " корней, а найдено " + std::to_string(cnt_U - cnt_L));
    std::vector<double> eigenvals;
    bisect_interval(C, L, U, cnt_L, cnt_U, tol, eigenvals);
    std::sort(eigenvals.begin(), eigenvals.end(), std::greater<double>());
    return eigenvals;
}

extern "C" {
double *find_eigenvalues(double *C, int m, double tol) {
    std::vector<std::vector<double> > matrix(m, std::vector<double>(m, 0.0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i][j] = C[i * m + j];
        }
    }
    std::vector<double> eigen = find_eigenvalues(matrix, tol);
    double *result = new double[m];
    for (int i = 0; i < m; i++)
        result[i] = eigen[i];
    return result;
}
}
