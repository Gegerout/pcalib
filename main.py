from pcalib import add_noise_and_compare, Matrix, pca, project_data_py, plot_pca_projection, center_data, gauss_solver, \
    covariance_matrix, find_eigenvalues, find_eigenvectors, explained_variance_ratio
from sklearn.model_selection import train_test_split

results = {}


def input_matrix(name="матрица"):
    print(f"Введите размер {name}: n")
    n = int(input())
    print(f"Введите {n} строк по {n} чисел через пробел:")
    data = []
    for i in range(n):
        row = list(map(float, input().split()))
        if len(row) != n:
            print(f"Ошибка: ожидалось {n} чисел!")
            return None
        data.append(row)
    return Matrix(data)


def input_vector(name="вектор", n=None):
    if n is None:
        print(f"Введите длину {name}:")
        n = int(input())
    print(f"Введите {n} чисел через пробел:")
    row = list(map(float, input().split()))
    if len(row) != n:
        print(f"Ошибка: ожидалось {n} чисел!")
        return None
    return Matrix([[x] for x in row])


def print_matrix(mat, name="Результат"):
    print(f"{name} (размер {mat.n}x{mat.m}):")
    for row in mat.data:
        print(" ".join(f"{x:.6g}" for x in row))


def print_vector(vec, name="Вектор"):
    print(f"{name}:")
    print(" ".join(f"{x[0]:.6g}" for x in vec.data))


def menu():
    print("\nВыберите режим:")
    print("1. Решить СЛАУ методом Гаусса")
    print("2. Центрировать матрицу")
    print("3. Вычислить матрицу ковариаций")
    print("4. Найти собственные значения")
    print("5. Найти собственные векторы")
    print("6. Доля объяснённой дисперсии")
    print("0. Выйти")
    return input("Ваш выбор: ").strip()


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


def accuracy_score(X, y, k_neighbors=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_list = [list(row) for row in X_train]
    X_test_list = [list(row) for row in X_test]
    y_pred = knn_predict(X_train_list, y_train, X_test_list, k_neighbors=k_neighbors)
    acc = sum(yt == yp for yt, yp in zip(y_test, y_pred)) / len(y_test)
    dim = len(X_train_list[0])
    return acc, dim


def main():
    print("\n=== Консольное приложение PCA Lab ===")
    while True:
        choice = menu()
        if choice == "0":
            print("Выход.")
            break
        elif choice == "1":
            print("\n--- Решение СЛАУ методом Гаусса ---")
            A = input_matrix("A")
            if A is None:
                continue
            b = input_vector("b", n=A.n)
            if b is None:
                continue
            try:
                sol = gauss_solver(A, [row[0] for row in b.data])
                print("Решение:")
                print(" ".join(f"{x:.6g}" for x in sol))
                results['last_solution'] = sol
            except Exception as e:
                print(f"Ошибка: {e}")
        elif choice == "2":
            print("\n--- Центрирование матрицы ---")
            X = input_matrix("X")
            if X is None:
                continue
            Xc = center_data(X)
            print_matrix(Xc, "Центрированная матрица")
            results['last_centered'] = Xc
        elif choice == "3":
            print("\n--- Матрица ковариаций ---")
            print("Использовать последнюю центрированную матрицу? (y/n)")
            use_last = input().strip().lower() == 'y'
            if use_last and 'last_centered' in results:
                Xc = results['last_centered']
            else:
                Xc = input_matrix("центрированная матрица")
                if Xc is None:
                    continue
            C = covariance_matrix(Xc)
            print_matrix(C, "Матрица ковариаций")
            results['last_cov'] = C
        elif choice == "4":
            print("\n--- Собственные значения ---")
            print("Использовать последнюю матрицу ковариаций? (y/n)")
            use_last = input().strip().lower() == 'y'
            if use_last and 'last_cov' in results:
                C = results['last_cov']
            else:
                C = input_matrix("матрица ковариаций")
                if C is None:
                    continue
            try:
                eigs = find_eigenvalues(C)
                print("Собственные значения:")
                print(" ".join(f"{x:.6g}" for x in eigs))
                results['last_eigenvalues'] = eigs
                results['last_cov'] = C
            except Exception as e:
                print(f"Ошибка: {e}")
        elif choice == "5":
            print("\n--- Собственные векторы ---")
            print("Использовать последние матрицу ковариаций и собственные значения? (y/n)")
            use_last = input().strip().lower() == 'y'
            if use_last and 'last_cov' in results and 'last_eigenvalues' in results:
                C = results['last_cov']
                eigs = results['last_eigenvalues']
            else:
                C = input_matrix("матрица ковариаций")
                if C is None:
                    continue
                print("Введите собственные значения через пробел:")
                eigs = list(map(float, input().split()))
            try:
                vecs = find_eigenvectors(C, eigs)
                for idx, v in enumerate(vecs):
                    print_vector(v, f"Собственный вектор {idx + 1}")
                results['last_eigenvectors'] = vecs
            except Exception as e:
                print(f"Ошибка: {e}")
        elif choice == "6":
            print("\n--- Доля объяснённой дисперсии ---")
            print("Использовать последние собственные значения? (y/n)")
            use_last = input().strip().lower() == 'y'
            if use_last and 'last_eigenvalues' in results:
                eigs = results['last_eigenvalues']
            else:
                print("Введите собственные значения через пробел:")
                eigs = list(map(float, input().split()))
            print("Введите k (число компонент):")
            k = int(input())
            try:
                gamma = explained_variance_ratio(eigs, k)
                print(f"Доля объяснённой дисперсии для k={k}: {gamma:.6g}")
            except Exception as e:
                print(f"Ошибка: {e}")
        else:
            print("Неизвестный режим. Попробуйте снова.")


if __name__ == "__main__":
    main()
