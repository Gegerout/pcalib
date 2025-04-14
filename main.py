from pcalib import Matrix, gauss_solver, center_data, covariance_matrix, find_eigenvalues, find_eigenvectors

def verify_eigenvector(C: Matrix, v: Matrix, lambda_val: float, tol: float = 1e-6) -> bool:
    """
    Проверяет, является ли v собственным вектором матрицы C с собственным значением lambda.
    Проверяет уравнение (C-λI)v = 0.
    """
    m = C.n
    result = []
    for i in range(m):
        val = 0.0
        for j in range(m):
            if i == j:
                val += (C.data[i][j] - lambda_val) * v.data[j][0]
            else:
                val += C.data[i][j] * v.data[j][0]
        result.append(abs(val))
    
    return all(x < tol for x in result)

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

    eigenvalues = find_eigenvalues(cov_mat, tol=1e-8)
    print("Собственные значения:")
    for val in eigenvalues:
        print(val)
        
    eigenvectors = find_eigenvectors(cov_mat, eigenvalues)
    print("Собственные векторы:")
    for i, vec in enumerate(eigenvectors):
        print(f"Вектор {i+1}:")
        for row in vec.data:
            print(row)
            
    print("\nПроверка собственных векторов:")
    for i, (vec, val) in enumerate(zip(eigenvectors, eigenvalues)):
        is_valid = verify_eigenvector(cov_mat, vec, val)
        print(f"Вектор {i+1} {'является' if is_valid else 'не является'} собственным вектором")


if __name__ == "__main__":
    main()
