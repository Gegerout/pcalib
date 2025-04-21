# PCALib

**PCALib** — это библиотека для анализа главных компонент (PCA), в которой все численные методы реализованы на C++ с удобной обёрткой для Python. Библиотека предназначена для обучения, исследований и практического применения в анализе данных и снижении размерности.

## Возможности

- **Чистое C++ ядро**: Все основные алгоритмы линейной алгебры и PCA реализованы вручную на C++ для максимальной скорости и прозрачности.
- **Python-интерфейс**: Удобная обёртка через `ctypes`.
- **Без внешних математических зависимостей**: Не используется numpy.linalg или аналогичные библиотеки — все методы (Гаусс, собственные значения и др.) реализованы самостоятельно.
- **Полный пайплайн PCA**: От центрирования и ковариации до поиска собственных значений/векторов и проекции.
- **Устойчивость к пропущенным значениям и шуму**: Включены утилиты для заполнения пропусков и анализа влияния шума.
- **Тесты**: В комплекте есть набор unit-тестов.

## Установка

1. **Соберите C++ библиотеку**
   - Из корня:
     ```sh
     g++ -O2 -fPIC -shared -o libpca_lib.so *.cpp
     # или для Mac:
     g++ -O2 -dynamiclib -o libpca_lib.dylib *.cpp
     # или для Windows (MSVC):
     cl /LD *.cpp
     ```

2. **Установите зависимости Python**
   - Для визуализации требуется только `matplotlib`:
     ```sh
     pip install matplotlib
     ```

## Пример использования

```python
from pcalib import Matrix, center_data, covariance_matrix, find_eigenvalues, find_eigenvectors, explained_variance_ratio

# Создаём матрицу данных
X = Matrix([[1, 2], [3, 4], [5, 6]])

# Центрируем данные
Xc = center_data(X)

# Вычисляем матрицу ковариаций
C = covariance_matrix(Xc)

# Находим собственные значения и векторы
eigenvalues = find_eigenvalues(C)
eigenvectors = find_eigenvectors(C, eigenvalues)

# Доля объяснённой дисперсии для первых k компонент
gamma = explained_variance_ratio(eigenvalues, k=1)

print("Собственные значения:", eigenvalues)
print("Доля объяснённой дисперсии:", gamma)
```

## Основные функции

- `gauss_solver(A: Matrix, b: List[float]) -> List[float]` — Решение СЛАУ методом Гаусса
- `center_data(X: Matrix) -> Matrix` — Центрирование данных по столбцам
- `covariance_matrix(X: Matrix) -> Matrix` — Вычисление матрицы ковариаций
- `find_eigenvalues(C: Matrix, tol=1e-6) -> List[float]` — Поиск собственных значений методом бисекции
- `find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]` — Поиск собственных векторов по значениям
- `explained_variance_ratio(eigenvalues: List[float], k: int) -> float` — Доля объяснённой дисперсии
- `pca(X: Matrix, k: int)` — Полный алгоритм PCA
- `plot_pca_projection(X_proj: Matrix, ...)` — Визуализация проекций (matplotlib)
- `handle_missing_values(X: Matrix) -> Matrix` — Заполнение пропущенных значений
- `add_noise_and_compare(X: Matrix, noise_level=0.1)` — Анализ устойчивости PCA к шуму

## Тестирование

Запустить все тесты можно так:
```sh
python -m unittest discover tests
```

## Лицензия

Этот проект распространяется под лицензией [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).  
Вы можете использовать, модифицировать и распространять проект в некоммерческих целях при условии указания авторства. Для коммерческого использования необходимо получить разрешение от автора.