import os
import math
import ctypes
import platform
import random
from typing import List, Optional
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .matrix import Matrix

# C++ библиотека
if platform.system() == 'Windows':
    lib_ext = '.dll'
elif platform.system() == 'Darwin':
    lib_ext = '.dylib'
else:
    lib_ext = '.so'

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, 'libpca_lib' + lib_ext)
pca_lib = ctypes.CDLL(lib_path)

# Аргументы C++ функций
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
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
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

pca_lib.project_data.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
pca_lib.project_data.restype = None


def project_data_py(X: Matrix, Vk: Matrix) -> Matrix:
    """
    Проецирует матрицу X (n x m) на Vk (m x k) с использованием C++ функции
    """
    n, m = X.n, X.m
    m2, k = Vk.n, Vk.m
    assert m == m2, f"Несоответсвие размеров: {m} != {m2}"
    flat_X = [elem for row in X.data for elem in row]
    flat_Vk = [elem for row in Vk.data for elem in row]
    ArrayXType = ctypes.c_double * (n * m)
    ArrayVkType = ctypes.c_double * (m * k)
    ArrayProjType = ctypes.c_double * (n * k)
    X_ctypes = ArrayXType(*flat_X)
    Vk_ctypes = ArrayVkType(*flat_Vk)
    X_proj_ctypes = ArrayProjType()
    pca_lib.project_data(X_ctypes, Vk_ctypes, X_proj_ctypes, n, m, k)
    result = [[X_proj_ctypes[i * k + j] for j in range(k)] for i in range(n)]
    return Matrix(result)


def mean_by_column(X: Matrix) -> List[float]:
    """
    Считает среднее по столбцам
    """
    n, m = X.n, X.m
    return [sum(X.data[i][j] for i in range(n)) / n for j in range(m)]


def gauss_solver(matrix: Matrix, b: List[float], ndigits: int = 6) -> List[float]:
    """
    Решает систему Ax = b методом Гаусса с использованием C++ функции
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


def center_data(matrix: Matrix, means: Optional[list] = None) -> Matrix:
    """
    Центрирует данные матрицы с использованием C++ функции
    Если means=None, центрирует по средним самой матрицы
    Если means задан, центрирует по ним
    """
    dense = matrix.data
    n, m = matrix.n, matrix.m
    flat_X = [elem for row in dense for elem in row]
    ArrayXType = ctypes.c_double * (n * m)
    X_ctypes = ArrayXType(*flat_X)
    X_centered = ArrayXType()
    if means is not None:
        ArrayMeansType = ctypes.c_double * m
        means_ctypes = ArrayMeansType(*means)
        means_ptr = means_ctypes
    else:
        means_ptr = None
    pca_lib.center_data(X_ctypes, X_centered, n, m, means_ptr)
    result = [[X_centered[i * m + j] for j in range(m)] for i in range(n)]
    return Matrix(result)


def covariance_matrix(matrix: Matrix) -> Matrix:
    """
    Вычисляет матрицу ковариаций для центрированной матрицы X
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
    Находит собственные значения матрицы методом бисекции
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
    Находит собственные векторы матрицы C для заданных собственных значений
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
    Заполняет пропущенные значения средними по столбцу
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
    Вычисляет долю объяснённой дисперсии
    """
    m = len(eigenvalues)
    ArrayType = ctypes.c_double * m
    eigenvalues_ctypes = ArrayType(*eigenvalues)
    return pca_lib.explained_variance_ratio(eigenvalues_ctypes, m, k)


def auto_select_k(eigenvalues: list[float], threshold: float = 0.95) -> int:
    """
    Автоматический выбор числа главных компонент по порогу объяснённой дисперсии
    """
    total = sum(eigenvalues)
    explained = 0.0
    for k, val in enumerate(eigenvalues, 1):
        explained += val
        if explained / total >= threshold:
            return k
    return len(eigenvalues)


def pca(X: 'Matrix', k: Optional[int] = None, threshold: float = 0.95):
    """
    Реализует алгоритм PCA. Если k=None, выбирает оптимальное k по порогу explained variance
    Возвращает (X_proj, gamma, k_used, Vk, means), где:
      - X_proj: проекция X на k компонент
      - gamma: доля объяснённой дисперсии
      - k_used: выбранное число компонент
      - Vk: матрица главных компонент (m x k)
      - means: средние по столбцам исходной матрицы X
    """
    n, m = X.n, X.m
    # Вычисляем средние по столбцам
    means = [sum(X.data[i][j] for i in range(n)) / n for j in range(m)]
    # Центрируем X по этим средним
    X_centered_data = [[X.data[i][j] - means[j] for j in range(m)] for i in range(n)]
    X_centered = type(X)(X_centered_data)
    C = covariance_matrix(X_centered)
    eigenvalues = find_eigenvalues(C)
    if k is None:
        k_used = auto_select_k(eigenvalues, threshold=threshold)
    else:
        k_used = k
    eigenvectors = find_eigenvectors(C, eigenvalues)
    Vk_data = [[eigenvectors[j].data[i][0] for i in range(m)] for j in range(k_used)]
    Vk = type(X)([list(col) for col in zip(*Vk_data)])
    X_proj = project_data_py(X_centered, Vk)
    gamma = explained_variance_ratio(eigenvalues, k_used)
    return X_proj, gamma, k_used, Vk, means


def plot_pca_projection(X_proj: 'Matrix', y=None, class_names=None, title=None) -> Figure:
    """
    Визуализирует проекцию данных на первые k главных компонент.
    Если k == 2: обычный scatter plot.
    Если k > 2: матрица парных scatter plot (pairplot) для первых min(k, 5) компонент средствами только matplotlib.
    Если k == 1: гистограмма/stripplot по первой компоненте.
    """
    k = X_proj.m
    n = X_proj.n
    data = X_proj.data  # data: list of lists, shape n x k
    if k < 1:
        raise ValueError("Для визуализации требуется хотя бы одна компонента (n x k, k >= 1)")
    if k == 1:
        fig, ax = plt.subplots(figsize=(7, 2))
        x = [row[0] for row in data]
        y_zeros = [0 for _ in range(n)]
        if y is not None:
            scatter = ax.scatter(x, y_zeros, c=y, cmap='viridis', edgecolor='k', s=50, alpha=0.8)
            if class_names is not None:
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
            ax.scatter(x, y_zeros, c='blue', edgecolor='k', s=50, alpha=0.8)
        ax.set_xlabel('PC1')
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('PCA Projection onto First Component')
        ax.grid(True)
        return fig
    elif k == 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = [row[0] for row in data]
        y2 = [row[1] for row in data]
        if y is not None:
            y_list = list(y)
            scatter = ax.scatter(x, y2, c=y_list, cmap='viridis', edgecolor='k', s=50, alpha=0.8)
            if class_names is not None:
                handles = []
                unique = sorted(set(y_list))
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
            ax.scatter(x, y2, c='blue', edgecolor='k', s=50, alpha=0.8)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('PCA Projection onto First Two Components')
        ax.grid(True)
        return fig
    else:
        # k > 2: pairwise scatter plot средствами только matplotlib
        max_dim = min(k, 5)
        fig, axes = plt.subplots(max_dim, max_dim, figsize=(2.5 * max_dim, 2.5 * max_dim))
        # data: n x k, хотим брать data[:, j] и data[:, i]
        for i in range(max_dim):
            for j in range(max_dim):
                ax = axes[i, j]
                if i == j:
                    ax.text(0.5, 0.5, f'PC{i + 1}', fontsize=10, ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    x = [row[j] for row in data]
                    y_ = [row[i] for row in data]
                    if y is not None:
                        scatter = ax.scatter(x, y_, c=y, cmap='viridis', edgecolor='k', s=10, alpha=0.8)
                    else:
                        ax.scatter(x, y_, c='blue', edgecolor='k', s=10, alpha=0.8)
                    if i == max_dim - 1:
                        ax.set_xlabel(f'PC{j + 1}')
                    else:
                        ax.set_xticks([])
                    if j == 0:
                        ax.set_ylabel(f'PC{i + 1}')
                    else:
                        ax.set_yticks([])
        if y is not None and class_names is not None:
            unique = sorted(set(y))
            colors = [plt.cm.viridis(i / max(1, len(unique) - 1)) for i in range(len(unique))]
            handles = [mpatches.Patch(color=colors[i], label=str(class_names[cl] if cl < len(class_names) else cl)) for
                       i, cl in enumerate(unique)]
            fig.legend(handles=handles, title="Класс", bbox_to_anchor=(1.05, 1), loc='upper left')
        if title is not None:
            fig.suptitle(title, y=1.02)
        else:
            fig.suptitle(f'PCA Pairplot (первые {max_dim} компонент)', y=1.02)
        plt.tight_layout()
        return fig


def reconstruction_error(X_orig: 'Matrix', X_recon: 'Matrix') -> float:
    """
    Вычисляет среднеквадратическую ошибку восстановления данных
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
    Восстанавливает данные из проекции PCA
    """
    X_centered = center_data(X)
    n, m = X.n, X.m
    C = covariance_matrix(X_centered)
    eigenvalues = find_eigenvalues(C)
    eigenvectors = find_eigenvectors(C, eigenvalues)
    Vk_data = [[eigenvectors[j].data[i][0] for i in range(m)] for j in range(k)]
    Vk = Matrix([list(col) for col in zip(*Vk_data)])
    Vk_T = Matrix([list(row) for row in zip(*Vk.data)])
    X_recon_centered = project_data_py(X_proj, Vk_T)
    means = mean_by_column(X)
    X_recon_data = []
    for i in range(n):
        row = [X_recon_centered.data[i][j] + means[j] for j in range(m)]
        X_recon_data.append(row)
    X_recon = Matrix(X_recon_data)
    return X_recon


def add_noise_and_compare(X: 'Matrix', noise_level: float = 0.1, k: int = None, threshold: float = 0.95):
    """
    Добавляет шум к данным и сравнивает результаты PCA до и после
    Если k=None, используется auto_select_k
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
    X_proj, gamma, k_used, _, _ = pca(X, k=k, threshold=threshold)
    X_proj_noisy, gamma_noisy, _, _, _ = pca(X_noisy, k=k_used, threshold=threshold)
    fig1 = plot_pca_projection(X_proj)
    fig2 = plot_pca_projection(X_proj_noisy)

    return {
        'X_proj': X_proj,
        'gamma': gamma,
        'fig_before': fig1,
        'X_proj_noisy': X_proj_noisy,
        'gamma_noisy': gamma_noisy,
        'fig_after': fig2,
        'k_used': k_used
    }
