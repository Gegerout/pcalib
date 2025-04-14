#include "eigenvectors.h"
#include "matrix_ops.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <random>

namespace Eigenvectors {

std::vector<std::vector<double>> find_eigenvectors(
    const std::vector<std::vector<double>>& C, 
    const std::vector<double>& eigenvalues) {
    
    int m = C.size();
    if (m == 0 || C[0].size() != static_cast<size_t>(m)) {
        throw std::invalid_argument("Матрица должна быть квадратной");
    }
    
    std::vector<std::vector<double>> eigenvectors;
    
    for (double lambda : eigenvalues) {
        std::vector<double> eigenvector = solve_eigenvector_equation(C, lambda);
        eigenvectors.push_back(eigenvector);
    }
    
    return eigenvectors;
}

std::vector<double> solve_eigenvector_equation(
    const std::vector<std::vector<double>>& C, 
    double lambda) {
    
    int m = C.size();
    const double eps = 1e-10;
    
    std::vector<std::vector<double>> A = create_characteristic_matrix(C, lambda);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    std::vector<double> v(m);
    for (int i = 0; i < m; i++) {
        v[i] = dis(gen);
    }
    v = normalize_vector(v);
    
    const int max_iter = 100;
    for (int iter = 0; iter < max_iter; iter++) {
        std::vector<std::vector<double>> augmented(m, std::vector<double>(m + 1));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                augmented[i][j] = A[i][j];
                if (i == j) augmented[i][j] += eps;
            }
            augmented[i][m] = v[i];
        }
        
        for (int i = 0; i < m; i++) {
            int max_row = i;
            double max_val = std::abs(augmented[i][i]);
            for (int j = i + 1; j < m; j++) {
                if (std::abs(augmented[j][i]) > max_val) {
                    max_val = std::abs(augmented[j][i]);
                    max_row = j;
                }
            }
            
            if (max_row != i) {
                std::swap(augmented[i], augmented[max_row]);
            }
            
            if (std::abs(augmented[i][i]) < eps) {
                continue;
            }
            
            double pivot = augmented[i][i];
            for (int j = i; j <= m; j++) {
                augmented[i][j] /= pivot;
            }
            
            for (int j = 0; j < m; j++) {
                if (j != i && std::abs(augmented[j][i]) > eps) {
                    double factor = augmented[j][i];
                    for (int k = i; k <= m; k++) {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }
        
        std::vector<double> new_v(m, 0.0);
        for (int i = m - 1; i >= 0; i--) {
            new_v[i] = augmented[i][m];
            for (int j = i + 1; j < m; j++) {
                new_v[i] -= augmented[i][j] * new_v[j];
            }
        }
        
        new_v = normalize_vector(new_v);
        
        // Проверка сходимости
        double diff = 0.0;
        for (int i = 0; i < m; i++) {
            diff += std::abs(new_v[i] - v[i]);
        }
        
        if (diff < eps) {
            return new_v;
        }
        
        v = new_v;
    }
    
    return v;
}

std::vector<std::vector<double>> create_characteristic_matrix(
    const std::vector<std::vector<double>>& C, 
    double lambda) {
    
    int m = C.size();
    std::vector<std::vector<double>> result = C;
    
    for (int i = 0; i < m; i++) {
        result[i][i] -= lambda;
    }
    
    return result;
}

std::vector<double> normalize_vector(const std::vector<double>& v) {
    double norm = 0.0;
    for (double val : v) {
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
        for (double& val : normalized) {
            val = -val;
        }
    }
    
    return normalized;
}

}

extern "C" {

double* find_eigenvectors(double* C, double* eigenvalues, int m, int n_eigenvalues) {
    std::vector<std::vector<double>> matrix(m, std::vector<double>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i][j] = C[i * m + j];
        }
    }
    
    std::vector<double> eigenvals(n_eigenvalues);
    for (int i = 0; i < n_eigenvalues; i++) {
        eigenvals[i] = eigenvalues[i];
    }
    
    std::vector<std::vector<double>> eigenvectors = Eigenvectors::find_eigenvectors(matrix, eigenvals);
    
    double* result = new double[m * n_eigenvalues];
    for (int i = 0; i < n_eigenvalues; i++) {
        for (int j = 0; j < m; j++) {
            result[i * m + j] = eigenvectors[i][j];
        }
    }
    
    return result;
}

}
