#include "covariance.h"
#include "matrix_ops.h"
#include <vector>

extern "C" {
void covariance_matrix(const double *X_in, double *X_covariance, int n, int m) {
    std::vector<std::vector<double> > X(n, std::vector<double>(m, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            X[i][j] = X_in[i * m + j];
        }
    }

    std::vector<std::vector<double> > XT = MatrixOps::transpose(X);

    std::vector<std::vector<double> > C = MatrixOps::multiply(XT, X);

    double scale = 1.0 / (n - 1);
    for (int i = 0; i < m; i++) {
        MatrixOps::scaleRow(C, i, scale);
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            X_covariance[i * m + j] = C[i][j];
        }
    }
}
}
