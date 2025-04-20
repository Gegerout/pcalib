#include "projection.h"

void project_data(const double *X, const double *Vk, double *X_proj, int n, int m, int k) {
    // X: n x m, Vk: m x k, X_proj: n x k
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int l = 0; l < m; ++l) {
                sum += X[i * m + l] * Vk[l * k + j];
            }
            X_proj[i * k + j] = sum;
        }
    }
}
