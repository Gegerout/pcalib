from pcalib import add_noise_and_compare, Matrix, pca, project_data_py, plot_pca_projection, center_data
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split


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
    # IRIS
    iris = load_iris()
    print("\n=== PCA на датасете iris ===")
    acc_before_iris, dim_iris = accuracy_score(iris.data, iris.target)
    print(f"Точность классификации на исходных данных (iris): {acc_before_iris:.4f}")
    print(f"Размерность исходных данных: {dim_iris}")

    X = iris.data
    y = iris.target
    class_names = iris.target_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_mat = Matrix([list(row) for row in X_train])
    X_test_mat = Matrix([list(row) for row in X_test])
    X_train_proj, gamma, k_used, Vk, means = pca(X_train_mat, k=None, threshold=0.95)
    X_test_centered = center_data(X_test_mat, means)
    X_test_proj = project_data_py(X_test_centered, Vk)
    y_pred_pca = knn_predict(X_train_proj.data, y_train, X_test_proj.data)
    acc_pca = sum(yt == yp for yt, yp in zip(y_test, y_pred_pca)) / len(y_test)
    print(f"Размерность после PCA: {X_train_proj.m}")
    print(f"Оптимальное число компонент (k): {k_used}")
    print(f"Точность после применения PCA (iris): {acc_pca:.4f}")
    X_full_mat = Matrix([list(row) for row in X])
    X_full_proj, _, _, _, _ = pca(X_full_mat, k=k_used, threshold=0.95)
    fig = plot_pca_projection(X_full_proj, y=y, class_names=class_names, title="PCA Projection (iris)")
    fig.savefig('results/pca_iris.png')
    print(f"См. визуализацию: results/pca_iris.png")

    # WINE
    wine = load_wine()
    print("\n=== PCA на датасете wine ===")
    acc_before_wine, dim_wine = accuracy_score(wine.data, wine.target)
    print(f"Точность классификации на исходных данных (wine): {acc_before_wine:.4f}")
    print(f"Размерность исходных данных: {dim_wine}")
    X = wine.data
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_mat = Matrix([list(row) for row in X_train])
    X_test_mat = Matrix([list(row) for row in X_test])
    X_train_proj, gamma, k_used, Vk, means = pca(X_train_mat, k=1, threshold=0.95)
    X_test_centered = center_data(X_test_mat, means)
    X_test_proj = project_data_py(X_test_centered, Vk)
    y_pred_pca = knn_predict(X_train_proj.data, y_train, X_test_proj.data)
    acc_pca = sum(yt == yp for yt, yp in zip(y_test, y_pred_pca)) / len(y_test)
    print(f"Размерность после PCA: {X_train_proj.m}")
    print(f"Оптимальное число компонент (k): {k_used}")
    print(f"Точность после применения PCA (wine): {acc_pca:.4f}")

    # Проверка влияния шума на PCA (wine)
    print("\n--- Влияние шума на PCA (wine) ---")
    X_mat = Matrix([list(row) for row in wine.data])
    res = add_noise_and_compare(X_mat, noise_level=0.5)
    print(f"Использовано k = {res['k_used']}")
    print(f"Доля объяснённой дисперсии до шума: {res['gamma']:.4f}")
    print(f"Доля объяснённой дисперсии после шума: {res['gamma_noisy']:.4f}")


if __name__ == "__main__":
    main()
