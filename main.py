from pcalib import Matrix, gauss_solver, center_data, covariance_matrix, find_eigenvalues, find_eigenvectors, explained_variance_ratio, pca, plot_pca_projection, reconstruction_error, reconstruct_from_pca, auto_select_k, handle_missing_values, add_noise_and_compare, apply_pca_to_dataset
import os

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
    
    # Вычисление доли объяснённой дисперсии для первых k компонент
    for k in range(1, len(eigenvalues)+1):
        gamma = explained_variance_ratio(eigenvalues, k)
        print(f"Доля объяснённой дисперсии для первых {k} компонент: {gamma:.6f}")
        
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

        print("\n=== Проверка на другой матрице ===")

    X2 = [
        [10, 1, 0],
        [8, 2, 1],
        [6, 1, 2],
        [4, 3, 3],
        [2, 4, 4]
    ]
    mat3 = Matrix(X2)
    centered_mat2 = center_data(mat3)
    print("Центрированная матрица:")
    for row in centered_mat2.data:
        print(row)

    cov_mat2 = covariance_matrix(centered_mat2)
    print("Матрица ковариаций:")
    for row in cov_mat2.data:
        print(row)

    eigenvalues2 = find_eigenvalues(cov_mat2, tol=1e-8)
    print("Собственные значения:")
    for val in eigenvalues2:
        print(val)

    for k in range(1, len(eigenvalues2)+1):
        gamma = explained_variance_ratio(eigenvalues2, k)
        print(f"Доля объяснённой дисперсии для первых {k} компонент: {gamma:.6f}")

    X = Matrix([
        [10, 1, 0],
        [8, 2, 1],
        [6, 1, 2],
        [4, 3, 3],
        [2, 4, 4]
    ])
    X_proj, gamma = pca(X, k=2)
    print("Проекция данных:")
    for row in X_proj.data:
        print(row)
    print("Доля объяснённой дисперсии:", gamma)

    X_recon = reconstruct_from_pca(X_proj, X, k=2)
    mse = reconstruction_error(X, X_recon)
    print("MSE:", mse)

    fig = plot_pca_projection(X_proj)
    fig.savefig('results/pca_projection.png')

    k_opt = auto_select_k(eigenvalues, threshold=0.95)
    print("Оптимальное число компонент:", k_opt)

    X_filled = handle_missing_values(X)

    results = add_noise_and_compare(X, noise_level=0.2)
    os.makedirs('results', exist_ok=True)
    results['fig_before'].savefig('results/pca_noise_before.png')
    results['fig_after'].savefig('results/pca_noise_after.png')

    print("\n=== PCA на датасете wine ===")
    X_proj_wine, acc_wine = apply_pca_to_dataset('wine', k=5)
    print("Accuracy after PCA on wine:", acc_wine)

    print("\n=== PCA на датасете iris ===")
    X_proj_iris, acc_iris = apply_pca_to_dataset('iris', k=2)
    print("Accuracy after PCA on iris:", acc_iris)
    os.makedirs('results', exist_ok=True)
    fig_iris = plot_pca_projection(X_proj_iris)
    fig_iris.savefig('results/pca_iris.png')


if __name__ == "__main__":
    main()
