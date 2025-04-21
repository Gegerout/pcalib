import unittest
from pcalib.py.algorithms import Matrix, pca, explained_variance_ratio, auto_select_k, reconstruction_error, \
    reconstruct_from_pca, plot_pca_projection


class TestPCA(unittest.TestCase):
    def setUp(self):
        self.X = Matrix([[1, 2], [3, 4], [5, 6]])

    def test_pca_projection(self):
        X_proj, gamma, *_ = pca(self.X, k=1)
        mean_proj = sum(row[0] for row in X_proj.data) / len(X_proj.data)

        self.assertAlmostEqual(mean_proj, 0.0, places=5)
        self.assertTrue(0 <= gamma <= 1)

    def test_explained_variance_ratio(self):
        vals = [3, 1]

        self.assertAlmostEqual(explained_variance_ratio(vals, 1), 0.75, places=5)

    def test_auto_select_k(self):
        vals = [3, 1, 0.5]
        k = auto_select_k(vals, threshold=0.8)

        self.assertEqual(k, 2)

    def test_reconstruction_error(self):
        X_proj, *_ = pca(self.X, k=1)
        X_recon = reconstruct_from_pca(X_proj, self.X, k=1)
        err = reconstruction_error(self.X, X_recon)

        self.assertTrue(err > 0)
        self.assertTrue(err < 5)

    def test_plot_pca_projection(self):
        X_proj, *_ = pca(self.X, k=2)
        fig = plot_pca_projection(X_proj)

        self.assertTrue(hasattr(fig, 'savefig'))


if __name__ == '__main__':
    unittest.main()
