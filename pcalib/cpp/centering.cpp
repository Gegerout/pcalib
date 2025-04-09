#include "centering.h"
#include <vector>

extern "C" {

    void center_data(const double* X, double* X_centered, int n, int m) {
        std::vector<double> means(m, 0.0);
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += X[i * m + j];
            }
            means[j] = sum / n;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                X_centered[i * m + j] = X[i * m + j] - means[j];
            }
        }
    }

}
