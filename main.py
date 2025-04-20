from pcalib import apply_pca_and_visualize, add_noise_and_compare, Matrix
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def count_accuracy(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Точность классификации на исходных данных ({dataset_name}): {acc:.4f}")
    print(f"Размерность исходных данных: {X.shape[1]}")
    return acc


def main():
    # IRIS
    iris = load_iris()
    print("\n=== PCA на датасете iris ===")
    acc_before_iris = count_accuracy(iris.data, iris.target, "iris")
    X_proj_iris, acc_iris, k_iris = apply_pca_and_visualize(
        X=iris.data,
        y=iris.target,
        class_names=iris.target_names,
        k=None,
        fig_path='results/pca_iris.png',
        title="PCA Projection (iris)"
    )
    print(f"Размерность после PCA: {X_proj_iris.m}")
    print(f"Оптимальное число компонент (k): {k_iris}")
    print(f"Точность после применения PCA (iris): {acc_iris:.4f}")
    print(f"См. визуализацию: results/pca_iris.png")

    # WINE
    wine = load_wine()
    print("\n=== PCA на датасете wine ===")
    acc_before_wine = count_accuracy(wine.data, wine.target, "wine")
    X_proj_wine, acc_wine, k_wine = apply_pca_and_visualize(
        X=wine.data,
        y=wine.target,
        class_names=wine.target_names,
        k=None,
        fig_path=None,
        title=None
    )
    print(f"Размерность после PCA: {X_proj_wine.m}")
    print(f"Оптимальное число компонент (k): {k_wine}")
    print(f"Точность после применения PCA (wine): {acc_wine:.4f}")

    # Проверка влияния шума на PCA (теперь для wine)
    print("\n--- Влияние шума на PCA (wine) ---")
    X_mat = Matrix([list(row) for row in wine.data])
    res = add_noise_and_compare(X_mat, noise_level=0.5)
    print(f"Использовано k = {res['k_used']}")
    print(f"Доля объяснённой дисперсии до шума: {res['gamma']:.4f}")
    print(f"Доля объяснённой дисперсии после шума: {res['gamma_noisy']:.4f}")


if __name__ == "__main__":
    main()
