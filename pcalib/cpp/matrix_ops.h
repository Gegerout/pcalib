#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <vector>

namespace MatrixOps {
    /**
     * @brief Складывает две матрицы одинаковых размеров.
     *
     * @param A Первая матрица (двумерный вектор).
     * @param B Вторая матрица (двумерный вектор).
     * @return std::vector<std::vector<double>> Результирующая матрица.
     * @throws std::invalid_argument, если размеры матриц не совпадают.
     */
    std::vector<std::vector<double> > add(const std::vector<std::vector<double> > &A,
                                          const std::vector<std::vector<double> > &B);

    /**
     * @brief Умножает строку матрицы на скаляр.
     *
     * @param matrix Ссылка на матрицу (двумерный вектор).
     * @param rowIndex Номер строки, которую нужно умножить.
     * @param factor Скаляр, на который умножается строка.
     * @throws std::out_of_range, если индекс строки вне допустимого диапазона.
     */
    void scaleRow(std::vector<std::vector<double> > &matrix, int rowIndex, double factor);

    /**
     * @brief Прибавляет к целевой строке матрицы скалярное кратное другой строки.
     *
     * Реализует операцию: targetRow = targetRow + factor * sourceRow.
     *
     * @param matrix Ссылка на матрицу.
     * @param targetRow Номер строки, к которой прибавляется значение.
     * @param sourceRow Номер строки, которая умножается на factor.
     * @param factor Скаляр множитель.
     * @throws std::out_of_range, если номера строк выходят за границы.
     * @throws std::invalid_argument, если длины строк не совпадают.
     */
    void addRows(std::vector<std::vector<double> > &matrix, int targetRow, int sourceRow, double factor);

    /**
     * @brief Обменивает местами две строки матрицы.
     *
     * @param matrix Ссылка на матрицу.
     * @param row1 Номер первой строки.
     * @param row2 Номер второй строки.
     * @throws std::out_of_range, если номера строк выходят за границы.
     */
    void swapRows(std::vector<std::vector<double> > &matrix, int row1, int row2);

    /**
     * @brief Перемножает две матрицы.
     *
     * @param A Первая матрица размерами (n x p).
     * @param B Вторая матрица размерами (p x m).
     * @return std::vector<std::vector<double>> Результирующая матрица размерами (n x m).
     * @throws std::invalid_argument, если число столбцов A не равно числу строк B.
     */
    std::vector<std::vector<double> > multiply(const std::vector<std::vector<double> > &A,
                                               const std::vector<std::vector<double> > &B);

    /**
     * @brief Вычисляет транспонированную матрицу.
     *
     * @param A Исходная матрица.
     * @return std::vector<std::vector<double>> Транспонированная матрица.
     */
    std::vector<std::vector<double> > transpose(const std::vector<std::vector<double> > &A);
}

#endif // MATRIX_OPS_H
