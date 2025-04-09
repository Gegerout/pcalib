#ifndef GAUSS_SOLVER_H
#define GAUSS_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

    int gauss_solver(const double* A_in, const double* b_in, double* x, int n);

#ifdef __cplusplus
}
#endif

#endif // GAUSS_SOLVER_H
