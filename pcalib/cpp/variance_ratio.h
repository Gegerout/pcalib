#ifndef VARIANCE_RATIO_H
#define VARIANCE_RATIO_H

#include <vector>

namespace VarianceRatio {

/**
 * @brief Вычисляет долю объяснённой дисперсии.
 * 
 * @param eigenvalues Вектор собственных значений (размер m).
 * @param k Число компонент (k <= m).
 * @return double Доля объяснённой дисперсии.
 */
double explained_variance_ratio(const std::vector<double>& eigenvalues, int k);

}

extern "C" {
/**
 * @brief C-интерфейс для вычисления доли объяснённой дисперсии.
 * 
 * @param eigenvalues Указатель на массив собственных значений (размер m).
 * @param m Количество собственных значений.
 * @param k Число компонент.
 * @return double Доля объяснённой дисперсии.
 */
double explained_variance_ratio(double* eigenvalues, int m, int k);
}

#endif // VARIANCE_RATIO_H
