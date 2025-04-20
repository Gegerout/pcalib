#include "eigenvectors.h"
#include "gauss_solver.h"
#include <cmath>
#include <stdexcept>
#include <random>
#include <vector>

namespace Eigenvectors {
    std::vector<std::vector<double> > find_eigenvectors(
        const std::vector<std::vector<double> > &C,
        const std::vector<double> &eigenvalues) {
        int m = C.size();
        if (m == 0 || C[0].size() != static_cast<size_t>(m)) {
            throw std::invalid_argument("Матрица должна быть квадратной");
        }

        std::vector<std::vector<double> > eigenvectors;

        for (double lambda: eigenvalues) {
            std::vector<double> eigenvector = solve_eigenvector_equation(C, lambda);
            eigenvectors.push_back(eigenvector);
        }

        return eigenvectors;
    }

    std::vector<double> solve_eigenvector_equation(
        const std::vector<std::vector<double> > &C,
        double lambda) {
        int m = C.size();
        const double eps = 1e-10;

        std::vector<std::vector<double> > A = C;
        for (int i = 0; i < m; i++) {
            A[i][i] -= lambda;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        std::vector<double> b(m);
        for (int i = 0; i < m; i++) {
            b[i] = dis(gen);
        }

        std::vector<double> A_flat(m * m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                A_flat[i * m + j] = A[i][j];
            }
        }

        std::vector<double> x(m, 0.0);
        int ret = gauss_solver(A_flat.data(), b.data(), x.data(), m);
        if (ret != 0) {
            return b;
        }

        x = normalize_vector(x);
        return x;
    }

    std::vector<double> normalize_vector(const std::vector<double> &v) {
        double norm = 0.0;
        for (double val: v) {
            norm += val * val;
        }
        norm = std::sqrt(norm);

        if (norm < 1e-10) {
            return v;
        }

        std::vector<double> normalized(v.size());
        for (size_t i = 0; i < v.size(); i++) {
            normalized[i] = v[i] / norm;
        }

        if (normalized[0] < 0) {
            for (double &val: normalized) {
                val = -val;
            }
        }

        return normalized;
    }
}

extern "C" {
double *find_eigenvectors(double *C, double *eigenvalues, int m, int n_eigenvalues) {
    std::vector<std::vector<double> > matrix(m, std::vector<double>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i][j] = C[i * m + j];
        }
    }

    std::vector<double> eigenvals(n_eigenvalues);
    for (int i = 0; i < n_eigenvalues; i++) {
        eigenvals[i] = eigenvalues[i];
    }

    std::vector<std::vector<double> > eigenvectors = Eigenvectors::find_eigenvectors(matrix, eigenvals);

    double *result = new double[m * n_eigenvalues];
    for (int i = 0; i < n_eigenvalues; i++) {
        for (int j = 0; j < m; j++) {
            result[i * m + j] = eigenvectors[i][j];
        }
    }

    return result;
}
}
