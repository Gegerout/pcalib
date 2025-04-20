#include "matrix_ops.h"
#include <vector>
#include <stdexcept>

namespace MatrixOps {
    std::vector<std::vector<double> > add(const std::vector<std::vector<double> > &A,
                                          const std::vector<std::vector<double> > &B) {
        if (A.size() != B.size() || (A.size() > 0 && A[0].size() != B[0].size())) {
            throw std::invalid_argument("Размеры матриц должны совпадать для сложения.");
        }

        std::vector<std::vector<double> > result = A;
        for (size_t i = 0; i < A.size(); i++) {
            for (size_t j = 0; j < A[i].size(); j++) {
                result[i][j] += B[i][j];
            }
        }
        return result;
    }

    void scaleRow(std::vector<std::vector<double> > &matrix, int rowIndex, double factor) {
        if (rowIndex < 0 || rowIndex >= static_cast<int>(matrix.size())) {
            throw std::out_of_range("Индекс строки вне допустимого диапазона.");
        }
        for (size_t j = 0; j < matrix[rowIndex].size(); j++) {
            matrix[rowIndex][j] *= factor;
        }
    }

    void addRows(std::vector<std::vector<double> > &matrix, int targetRow, int sourceRow, double factor) {
        if (targetRow < 0 || targetRow >= static_cast<int>(matrix.size()) ||
            sourceRow < 0 || sourceRow >= static_cast<int>(matrix.size())) {
            throw std::out_of_range("Индекс строки вне допустимого диапазона.");
        }
        if (matrix[targetRow].size() != matrix[sourceRow].size()) {
            throw std::invalid_argument("Строки должны быть одинаковой длины.");
        }
        for (size_t j = 0; j < matrix[targetRow].size(); j++) {
            matrix[targetRow][j] += factor * matrix[sourceRow][j];
        }
    }

    void swapRows(std::vector<std::vector<double> > &matrix, int row1, int row2) {
        if (row1 < 0 || row1 >= static_cast<int>(matrix.size()) ||
            row2 < 0 || row2 >= static_cast<int>(matrix.size())) {
            throw std::out_of_range("Индекс строки вне допустимого диапазона.");
        }
        std::swap(matrix[row1], matrix[row2]);
    }

    std::vector<std::vector<double> > multiply(const std::vector<std::vector<double> > &A,
                                               const std::vector<std::vector<double> > &B) {
        if (A.empty() || B.empty() || A[0].size() != B.size()) {
            throw std::invalid_argument("Неверные размеры матриц для перемножения.");
        }
        int n = A.size();
        int p = A[0].size();
        int m = B[0].size();

        std::vector<std::vector<double> > result(n, std::vector<double>(m, 0.0));
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < p; k++) {
                for (int j = 0; j < m; j++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    std::vector<std::vector<double> > transpose(const std::vector<std::vector<double> > &A) {
        if (A.empty()) return {};
        int n = A.size();
        int m = A[0].size();
        std::vector<std::vector<double> > result(m, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[j][i] = A[i][j];
            }
        }
        return result;
    }
}
