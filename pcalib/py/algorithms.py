import os
import math
import ctypes
import platform
import random
from typing import List, Tuple
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from .matrix import Matrix

# --- C++ библиотека ---
if platform.system() == 'Windows':
    lib_ext = '.dll'
elif platform.system() == 'Darwin':
    lib_ext = '.dylib'
else:
    lib_ext = '.so'

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, 'libpca_lib' + lib_ext)
pca_lib = ctypes.CDLL(lib_path)

# --- Аргументы C++ функций ---
pca_lib.gauss_solver.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
]
pca_lib.gauss_solver.restype = ctypes.c_int

pca_lib.center_data.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]
pca_lib.center_data.restype = None

pca_lib.covariance_matrix.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]
pca_lib.covariance_matrix.restype = None

pca_lib.find_eigenvalues.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double
]
pca_lib.find_eigenvalues.restype = ctypes.POINTER(ctypes.c_double)

pca_lib.find_eigenvectors.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]
pca_lib.find_eigenvectors.restype = ctypes.POINTER(ctypes.c_double)

pca_lib.explained_variance_ratio.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]
pca_lib.explained_variance_ratio.restype = ctypes.c_double


def accuracy_score_manual(y_true, y_pred):
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / len(y_true)


def mean_by_column(X: Matrix) -> List[float]:
    n, m = X.n, X.m
    return [sum(X.data[i][j] for i in range(n)) / n for j in range(m)]


def gauss_solver(matrix: Matrix, b: List[float], ndigits: int = 6) -> List[float]:
    """
    Решает систему Ax = b методом Гаусса с использованием C++ функции.
    """
    A_dense = matrix.data
    n = matrix.n
    if any(len(row) != n for row in A_dense):
        raise ValueError("Матрица A должна быть квадратной")
    flat_A = [elem for row in A_dense for elem in row]
    ArrayAType = ctypes.c_double * (n * n)
    A_ctypes = ArrayAType(*flat_A)
    ArrayBType = ctypes.c_double * n
    b_ctypes = ArrayBType(*b)
    XArrayType = ctypes.c_double * n
    x_ctypes = XArrayType()
    ret = pca_lib.gauss_solver(A_ctypes, b_ctypes, x_ctypes, n)
    if ret != 0:
        raise ValueError("Система несовместна или матрица вырождена")
    raw_solution = [x_ctypes[i] for i in range(n)]
    rounded_solution = [round(val, ndigits) for val in raw_solution]
    return rounded_solution


def center_data(matrix: Matrix) -> Matrix:
    """
    Центрирует данные матрицы с использованием C++ функции.
    """
    dense = matrix.data
    n, m = matrix.n, matrix.m
    flat_X = [elem for row in dense for elem in row]
    ArrayXType = ctypes.c_double * (n * m)
    X_ctypes = ArrayXType(*flat_X)
    X_centered = ArrayXType()
    pca_lib.center_data(X_ctypes, X_centered, n, m)
    result = [[X_centered[i * m + j] for j in range(m)] for i in range(n)]
    return Matrix(result)


def covariance_matrix(matrix: Matrix) -> Matrix:
    """
    Вычисляет матрицу ковариаций для центрированной матрицы X.
    """
    dense = matrix.data
    n, m = matrix.n, matrix.m
    flat_X = [elem for row in dense for elem in row]
    ArrayXType = ctypes.c_double * (n * m)
    X_ctypes = ArrayXType(*flat_X)
    ArrayCovType = ctypes.c_double * (m * m)
    cov_ctypes = ArrayCovType()
    pca_lib.covariance_matrix(X_ctypes, cov_ctypes, n, m)
    result = [[cov_ctypes[i * m + j] for j in range(m)] for i in range(m)]
    return Matrix(result)


def find_eigenvalues(C: Matrix, tol: float = 1e-6) -> List[float]:
    """
    Находит собственные значения матрицы методом бисекции.
    """
    if C.n != C.m:
        raise ValueError("Матрица должна быть квадратной")
    dense = C.data
    m = C.m
    flat_C = [elem for row in dense for elem in row]
    ArrayCType = ctypes.c_double * (m * m)
    C_ctypes = ArrayCType(*flat_C)
    result_ptr = pca_lib.find_eigenvalues(C_ctypes, m, tol)
    return [result_ptr[i] for i in range(m)]


def find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]:
    """
    Находит собственные векторы матрицы C для заданных собственных значений.
    """
    if C.n != C.m:
        raise ValueError("Матрица должна быть квадратной")
    dense = C.data
    m = C.m
    n_eigenvalues = len(eigenvalues)
    flat_C = [elem for row in dense for elem in row]
    ArrayCType = ctypes.c_double * (m * m)
    C_ctypes = ArrayCType(*flat_C)
    ArrayEigenvaluesType = ctypes.c_double * n_eigenvalues
    eigenvalues_ctypes = ArrayEigenvaluesType(*eigenvalues)
    result_ptr = pca_lib.find_eigenvectors(C_ctypes, eigenvalues_ctypes, m, n_eigenvalues)
    return [Matrix([[result_ptr[i * m + j]] for j in range(m)]) for i in range(n_eigenvalues)]


def handle_missing_values(X: 'Matrix') -> 'Matrix':
    """
    Заполняет пропущенные значения средними по столбцу.
    """
    n, m = X.n, X.m
    means = []
    for j in range(m):
        col = [X.data[i][j] for i in range(n) if not math.isnan(X.data[i][j])]
        mean = sum(col) / len(col) if col else 0.0
        means.append(mean)
    filled = []
    for i in range(n):
        row = []
        for j in range(m):
            val = X.data[i][j]
            if math.isnan(val):
                row.append(means[j])
            else:
                row.append(val)
        filled.append(row)
    return type(X)(filled)


def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """
    Вычисляет долю объяснённой дисперсии.
    """
    m = len(eigenvalues)
    ArrayType = ctypes.c_double * m
    eigenvalues_ctypes = ArrayType(*eigenvalues)
    return pca_lib.explained_variance_ratio(eigenvalues_ctypes, m, k)


def pca(X: 'Matrix', k: int) -> Tuple['Matrix', float]:
    """
    Реализует алгоритм PCA.
    """
    n, m = X.n, X.m
    means = mean_by_column(X)
    X_centered_data = [[X.data[i][j] - means[j] for j in range(m)] for i in range(n)]
    X_centered = type(X)(X_centered_data)
    C = covariance_matrix(X_centered)
    eigenvalues = find_eigenvalues(C)
    eigenvectors = find_eigenvectors(C, eigenvalues)
    Vk_data = [[eigenvectors[j].data[i][0] for i in range(m)] for j in range(k)]
    Vk = type(X)([list(col) for col in zip(*Vk_data)])
    X_proj_data = []
    for i in range(n):
        row = []
        for j in range(k):
            val = sum(X_centered.data[i][l] * Vk.data[l][j] for l in range(m))
            row.append(val)
        X_proj_data.append(row)
    X_proj = type(X)(X_proj_data)
    gamma = explained_variance_ratio(eigenvalues, k)
    return X_proj, gamma


def plot_pca_projection(X_proj: 'Matrix', y=None, class_names=None, title=None) -> Figure:
    """
    Визуализирует проекцию данных на первые две главные компоненты.
    """
    if X_proj.m != 2:
        raise ValueError("Для визуализации требуется проекция на 2 компоненты (n x 2)")
    x = [row[0] for row in X_proj.data]
    y_proj = [row[1] for row in X_proj.data]
    fig, ax = plt.subplots(figsize=(7, 5))
    if y is not None:
        import numpy as np
        y = list(y)
        scatter = ax.scatter(x, y_proj, c=y, cmap='viridis', edgecolor='k', s=50, alpha=0.8)
        if class_names is not None:
            # Для дискретных классов
            handles = []
            unique = sorted(set(y))
            for i, cl in enumerate(unique):
                handles.append(
                    plt.Line2D([], [], marker='o', color='w',
                               markerfacecolor=plt.cm.viridis(i / max(1, len(unique) - 1)),
                               markeredgecolor='k', markersize=8,
                               label=str(class_names[cl] if cl < len(class_names) else cl))
                )
            ax.legend(handles=handles, title="Класс")
        else:
            fig.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(x, y_proj, c='blue', edgecolor='k', s=50, alpha=0.8)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('PCA Projection onto First Two Components')
    ax.grid(True)
    return fig


def reconstruction_error(X_orig: 'Matrix', X_recon: 'Matrix') -> float:
    """
    Вычисляет среднеквадратическую ошибку восстановления данных.
    """
    if X_orig.n != X_recon.n or X_orig.m != X_recon.m:
        raise ValueError("Размеры матриц должны совпадать")
    n, m = X_orig.n, X_orig.m
    mse = 0.0
    for i in range(n):
        for j in range(m):
            diff = X_orig.data[i][j] - X_recon.data[i][j]
            mse += diff * diff
    mse /= (n * m)
    return mse


def reconstruct_from_pca(X_proj: Matrix, X: Matrix, k: int) -> Matrix:
    """
    Восстанавливает данные из проекции PCA.
    """
    X_centered = center_data(X)
    n, m = X.n, X.m
    C = covariance_matrix(X_centered)
    eigenvalues = find_eigenvalues(C)
    eigenvectors = find_eigenvectors(C, eigenvalues)
    Vk_data = [[eigenvectors[j].data[i][0] for i in range(m)] for j in range(k)]
    Vk = Matrix([list(col) for col in zip(*Vk_data)])  # m x k
    X_recon_centered_data = []
    for i in range(n):
        row = []
        for j in range(m):
            val = sum(X_proj.data[i][l] * Vk.data[j][l] for l in range(k))
            row.append(val)
        X_recon_centered_data.append(row)
    X_recon_centered = Matrix(X_recon_centered_data)
    means = mean_by_column(X)
    X_recon_data = []
    for i in range(n):
        row = [X_recon_centered.data[i][j] + means[j] for j in range(m)]
        X_recon_data.append(row)
    X_recon = Matrix(X_recon_data)
    return X_recon


def auto_select_k(eigenvalues: list[float], threshold: float = 0.95) -> int:
    """
    Автоматический выбор числа главных компонент по порогу объяснённой дисперсии.
    """
    total = sum(eigenvalues)
    explained = 0.0
    for k, val in enumerate(eigenvalues, 1):
        explained += val
        if explained / total >= threshold:
            return k
    return len(eigenvalues)


def add_noise_and_compare(X: 'Matrix', noise_level: float = 0.1):
    """
    Добавляет шум к данным и сравнивает результаты PCA до и после.
    """
    n, m = X.n, X.m
    means = []
    stds = []
    for j in range(m):
        col = [X.data[i][j] for i in range(n)]
        mean = sum(col) / n
        means.append(mean)
        variance = sum((x - mean) ** 2 for x in col) / n
        stds.append(variance ** 0.5)
    X_noisy_data = []
    for i in range(n):
        row = []
        for j in range(m):
            noise = random.gauss(0, stds[j] * noise_level)
            row.append(X.data[i][j] + noise)
        X_noisy_data.append(row)
    X_noisy = type(X)(X_noisy_data)
    X_proj, gamma = pca(X, k=2)
    X_proj_noisy, gamma_noisy = pca(X_noisy, k=2)
    fig1 = plot_pca_projection(X_proj)
    fig2 = plot_pca_projection(X_proj_noisy)
    print(f"Доля объяснённой дисперсии до шума: {gamma:.4f}")
    print(f"Доля объяснённой дисперсии после шума: {gamma_noisy:.4f}")
    return {
        'X_proj': X_proj,
        'gamma': gamma,
        'fig_before': fig1,
        'X_proj_noisy': X_proj_noisy,
        'gamma_noisy': gamma_noisy,
        'fig_after': fig2
    }


def apply_pca_and_visualize(X, y, class_names, k, fig_path, title=None):
    """
    Применяет PCA к данным, строит график и возвращает проекцию и точность классификации.
    """
    from .matrix import Matrix
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_mat = Matrix([list(row) for row in X_train])
    X_test_mat = Matrix([list(row) for row in X_test])

    # PCA по train
    from .algorithms import pca, mean_by_column
    X_train_proj, _ = pca(X_train_mat, k)
    # Получаем Vk и means
    means = mean_by_column(X_train_mat)
    X_train_centered = [[X_train_mat.data[i][j] - means[j] for j in range(X_train_mat.m)] for i in range(X_train_mat.n)]
    X_test_centered = [[X_test_mat.data[i][j] - means[j] for j in range(X_test_mat.m)] for i in range(X_test_mat.n)]
    # Получаем Vk
    C = covariance_matrix(Matrix(X_train_centered))
    eigenvalues = find_eigenvalues(C)
    eigenvectors = find_eigenvectors(C, eigenvalues)
    Vk_data = [[eigenvectors[j].data[i][0] for i in range(C.m)] for j in range(k)]
    Vk = Matrix([list(col) for col in zip(*Vk_data)])  # m x k

    # Проекция test
    X_test_proj_data = []
    for i in range(len(X_test_centered)):
        row = []
        for j in range(k):
            val = sum(X_test_centered[i][l] * Vk.data[l][j] for l in range(len(X_test_centered[0])))
            row.append(val)
        X_test_proj_data.append(row)
    X_test_proj = Matrix(X_test_proj_data)

    # Классификация KNN вручную
    def knn_predict(X_train, y_train, X_test, k_neighbors=3):
        def euclidean(a, b):
            return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

        preds = []
        for test_vec in X_test:
            dists = [(euclidean(test_vec, train_vec), y) for train_vec, y in zip(X_train, y_train)]
            dists.sort()
            top_k = [y for _, y in dists[:k_neighbors]]
            preds.append(max(set(top_k), key=top_k.count))
        return preds

    y_pred_pca = knn_predict(X_train_proj.data, y_train, X_test_proj.data)
    acc_pca = sum(yt == yp for yt, yp in zip(y_test, y_pred_pca)) / len(y_test)

    # Визуализация по всему X (а не только по test!)
    X_full_mat = Matrix([list(row) for row in X])
    X_full_proj, _ = pca(X_full_mat, k)
    fig = plot_pca_projection(X_full_proj, y=y, class_names=class_names, title=title)
    fig.savefig(fig_path)

    return X_full_proj, acc_pca
