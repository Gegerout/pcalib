#include <cmath>
#include "gauss_solver.h"
#include "matrix_ops.h"
#include <vector>

extern "C" {
int gauss_solver(const double *A_in, const double *b_in, double *x, int n) {
    std::vector<std::vector<double> > A(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = A_in[i * n + j];
        }
    }
    std::vector<double> b(n, 0.0);
    for (int i = 0; i < n; i++) {
        b[i] = b_in[i];
    }

    for (int k = 0; k < n; k++) {
        if (std::fabs(A[k][k]) < 1e-12) {
            return -1;
        }
        double pivot = A[k][k];
        MatrixOps::scaleRow(A, k, 1.0 / pivot);
        b[k] /= pivot;

        for (int i = k + 1; i < n; i++) {
            double factor = A[i][k];

            MatrixOps::addRows(A, i, k, -factor);
            b[i] -= factor * b[k];
        }
    }

    std::vector<double> sol(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * sol[j];
        }
        sol[i] = b[i] - sum;
    }

    for (int i = 0; i < n; i++) {
        x[i] = sol[i];
    }
    return 0;
}
}
