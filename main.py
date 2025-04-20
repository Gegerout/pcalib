from pcalib import apply_pca_and_visualize
import os
from sklearn.datasets import load_iris, load_wine


def main():
    os.makedirs('results', exist_ok=True)

    # IRIS
    iris = load_iris()
    print("\n=== PCA на датасете iris ===")
    X_proj_iris, acc_iris = apply_pca_and_visualize(
        X=iris.data,
        y=iris.target,
        class_names=iris.target_names,
        k=2,
        fig_path='results/pca_iris.png',
        title="PCA Projection (iris)"
    )
    print("Accuracy after PCA on iris:", acc_iris)

    # WINE
    wine = load_wine()
    print("\n=== PCA на датасете wine ===")
    X_proj_wine, acc_wine = apply_pca_and_visualize(
        X=wine.data,
        y=wine.target,
        class_names=wine.target_names,
        k=2,
        fig_path='results/pca_wine.png',
        title="PCA Projection (wine)"
    )
    print("Accuracy after PCA on wine:", acc_wine)


if __name__ == "__main__":
    main()
