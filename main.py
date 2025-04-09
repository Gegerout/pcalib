from pcalib import Matrix, gauss_solver, center_data, covariance_matrix

def main():
    A = [
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
    ]
    b = [1, 0, 1]
    mat = Matrix(A)
    try:
        solution = gauss_solver(mat, b)
        print("Решение системы:", solution)
    except ValueError as e:
        print("Ошибка:", e)

    X = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    mat2 = Matrix(X)
    centered_mat = center_data(mat2)
    print("Центрированная матрица:")
    for row in centered_mat.data:
        print(row)

    cov_mat = covariance_matrix(centered_mat)
    print("Матрица ковариаций:")
    for row in cov_mat.data:
        print(row)

if __name__ == "__main__":
    main()
