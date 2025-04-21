#ifndef VARIANCE_RATIO_H
#define VARIANCE_RATIO_H

extern "C" {
/**
 * C-интерфейс для вычисления доли объяснённой дисперсии
 * 
 * @param eigenvalues Указатель на массив собственных значений (размер m)
 * @param m Количество собственных значений
 * @param k Число компонент
 * @return double Доля объяснённой дисперсии
 */
double explained_variance_ratio(double *eigenvalues, int m, int k);
}

#endif // VARIANCE_RATIO_H
