#ifndef EIGENVALUES_H
#define EIGENVALUES_H

#include <vector>
#include <utility>
#include <stdexcept>

namespace Eigenvalues {

    std::vector<double> find_eigenvalues(const std::vector<std::vector<double>>& C, double tol);
}

extern "C" {
    double* find_eigenvalues(double* C, int m, double tol);
}

#endif // EIGENVALUES_H
