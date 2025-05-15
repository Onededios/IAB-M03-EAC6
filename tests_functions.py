"""Unit tests for functions of file functions.py"""

import unittest

from functions import create_dataset, transform_pca, model_kmeans, predict_clusters


class TestsFunctions(unittest.TestCase):
    """Unit tests class for functions of file functions.py"""

    def test_when_create_dataset_success(self):
        """
        Unit test to check that the created data is correct
        """
        features = 4
        x, _ = create_dataset(features)
        self.assertEqual(x.shape[0], 250)
        self.assertEqual(x.shape[1], features)

    def test_when_transform_pca_success(self):
        """
        Unit test to check that the transformed data to PCA is correct
        """
        x, _ = create_dataset(4)
        features = 2
        x_pca = transform_pca(x, features)

        self.assertEqual(x_pca.shape[0], 250)
        self.assertEqual(x_pca.shape[1], features)

    def test_when_same_cluster_assignation(self):
        """
        Unit test to check that the KMeans cluster assignation is the same for both PCA and None
        """
        x, _ = create_dataset(4)
        x_pca = transform_pca(x, 2)

        km = model_kmeans(3)
        _, y_km = predict_clusters(km, x)
        _, pca_y_km = predict_clusters(km, x_pca)

        self.assertTrue((y_km == pca_y_km).all())


if __name__ == "__main__":
    unittest.main()
