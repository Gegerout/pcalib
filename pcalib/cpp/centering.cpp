#include "centering.h"
#include <vector>

extern "C" {
void center_data(const double *X, double *X_centered, int n, int m, const double *means) {
    std::vector<double> local_means;
    const double *use_means = means;
    if (!means) {
        // Вычисляем средние по столбцам
        local_means.resize(m, 0.0);
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += X[i * m + j];
            }
            local_means[j] = sum / n;
        }
        use_means = local_means.data();
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            X_centered[i * m + j] = X[i * m + j] - use_means[j];
        }
    }
}
}
