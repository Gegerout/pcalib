import os
import ctypes
import platform
from typing import List
from .matrix import Matrix

if platform.system() == 'Windows':
    lib_ext = '.dll'
elif platform.system() == 'Darwin':
    lib_ext = '.dylib'
else:
    lib_ext = '.so'

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, 'libpca_lib' + lib_ext)

pca_lib = ctypes.CDLL(lib_path)

pca_lib.gauss_solver.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
]
pca_lib.gauss_solver.restype = ctypes.c_int

def gauss_solver(matrix: Matrix, b: List[float], ndigits: int = 6) -> List[float]:
    """
    Решает систему Ax = b методом Гаусса с использованием C++ функции.
    :param matrix: объект Matrix, содержащий плотную матрицу A (n x n)
    :param b: список чисел (вектор правых частей, длины n)
    :return: список чисел – решение системы
    :raises ValueError: если матрица не квадратная или система несовместна
    """
    A_dense = matrix.data
    n = matrix.n

    if any(len(row) != n for row in A_dense):
        raise ValueError("Матрица A должна быть квадратной")

    flat_A = []

    for row in A_dense:
        flat_A.extend(row)

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

pca_lib.center_data.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]
pca_lib.center_data.restype = None

def center_data(matrix: Matrix) -> Matrix:
    """
    Центрирует данные матрицы с использованием C++ функции.
    :param matrix: объект Matrix (n x m)
    :return: новый объект Matrix с центрированными данными
    """
    dense = matrix.data
    n, m = matrix.n, matrix.m
    flat_X = []

    for row in dense:
        flat_X.extend(row)

    ArrayXType = ctypes.c_double * (n * m)
    X_ctypes = ArrayXType(*flat_X)
    X_centered = ArrayXType()
    pca_lib.center_data(X_ctypes, X_centered, n, m)

    result = []

    for i in range(n):
        row = [X_centered[i * m + j] for j in range(m)]
        result.append(row)

    return Matrix(result)


pca_lib.covariance_matrix.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]
pca_lib.covariance_matrix.restype = None

def covariance_matrix(matrix: Matrix) -> Matrix:
    """
    Вычисляет матрицу ковариаций для центрированной матрицы X.
    :param matrix: объект Matrix, содержащий данные X (n x m),
                   предполагается, что данные уже центрированы.
    :return: новый объект Matrix, представляющий матрицу ковариаций (m x m)
    """
    dense = matrix.data
    n, m = matrix.n, matrix.m
    flat_X = []

    for row in dense:
        flat_X.extend(row)

    ArrayXType = ctypes.c_double * (n * m)
    X_ctypes = ArrayXType(*flat_X)
    ArrayCovType = ctypes.c_double * (m * m)
    cov_ctypes = ArrayCovType()

    pca_lib.covariance_matrix(X_ctypes, cov_ctypes, n, m)

    result = []

    for i in range(m):
        row = [cov_ctypes[i * m + j] for j in range(m)]
        result.append(row)

    return Matrix(result)

pca_lib.find_eigenvalues.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double
]
pca_lib.find_eigenvalues.restype = ctypes.POINTER(ctypes.c_double)

def find_eigenvalues(C: Matrix, tol: float = 1e-6) -> List[float]:
    """
    Находит собственные значения матрицы методом бисекции.
    
    Вход:
    C: матрица ковариаций (m×m)
    tol: допустимая погрешность
    
    Выход: список вещественных собственных значений
    """
    if C.n != C.m:
        raise ValueError("Матрица должна быть квадратной")
    
    dense = C.data
    m = C.m
    flat_C = []
    
    for row in dense:
        flat_C.extend(row)
    
    ArrayCType = ctypes.c_double * (m * m)
    C_ctypes = ArrayCType(*flat_C)
    
    result_ptr = pca_lib.find_eigenvalues(C_ctypes, m, tol)
    
    result = []
    for i in range(m):
        result.append(result_ptr[i])
    
    return result

pca_lib.find_eigenvectors.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]
pca_lib.find_eigenvectors.restype = ctypes.POINTER(ctypes.c_double)

def find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]:
    """
    Находит собственные векторы матрицы C для заданных собственных значений.
    
    Вход:
    C: матрица ковариаций (m×m)
    eigenvalues: список собственных значений
    
    Выход: список собственных векторов (каждый вектор - объект Matrix)
    """
    if C.n != C.m:
        raise ValueError("Матрица должна быть квадратной")
    
    dense = C.data
    m = C.m
    n_eigenvalues = len(eigenvalues)
    
    flat_C = []
    for row in dense:
        flat_C.extend(row)
    
    ArrayCType = ctypes.c_double * (m * m)
    C_ctypes = ArrayCType(*flat_C)
    
    ArrayEigenvaluesType = ctypes.c_double * n_eigenvalues
    eigenvalues_ctypes = ArrayEigenvaluesType(*eigenvalues)
    
    result_ptr = pca_lib.find_eigenvectors(C_ctypes, eigenvalues_ctypes, m, n_eigenvalues)
    
    result = []
    for i in range(n_eigenvalues):
        vector_data = [[result_ptr[i * m + j]] for j in range(m)]
        result.append(Matrix(vector_data))
    
    return result
