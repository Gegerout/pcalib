#ifndef COVARIANCE_H
#define COVARIANCE_H

#ifdef __cplusplus
extern "C" {
#endif

    void covariance_matrix(const double* X_in, double* X_covariance, int n, int m);

#ifdef __cplusplus
}
#endif

#endif //COVARIANCE_H
