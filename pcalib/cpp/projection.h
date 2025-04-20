#ifndef PROJECTION_H
#define PROJECTION_H

#ifdef __cplusplus
extern "C" {
#endif

// Проецирует матрицу X (n x m) на Vk (m x k), результат X_proj (n x k)
void project_data(const double *X, const double *Vk, double *X_proj, int n, int m, int k);

#ifdef __cplusplus
}
#endif

#endif //PROJECTION_H
