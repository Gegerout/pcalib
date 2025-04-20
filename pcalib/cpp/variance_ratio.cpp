#include "variance_ratio.h"
#include <stdexcept>
#include <numeric>

namespace VarianceRatio {
    double explained_variance_ratio(const std::vector<double> &eigenvalues, int k) {
        int m = eigenvalues.size();
        if (k < 1 || k > m) {
            throw std::invalid_argument("k должно быть в диапазоне от 1 до числа собственных значений");
        }
        double total = std::accumulate(eigenvalues.begin(), eigenvalues.end(), 0.0);
        double explained = std::accumulate(eigenvalues.begin(), eigenvalues.begin() + k, 0.0);
        return (total != 0.0) ? (explained / total) : 0.0;
    }
}

extern "C" {
double explained_variance_ratio(double *eigenvalues, int m, int k) {
    std::vector<double> vals(eigenvalues, eigenvalues + m);
    return VarianceRatio::explained_variance_ratio(vals, k);
}
}
