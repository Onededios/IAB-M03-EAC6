import unittest

from functions import create_dataset, transform_pca


class TestsFunctions(unittest.TestCase):
    def test_create_dataset_success(self):
        features = 4
        x, y = create_dataset(features)
        print(x.shape, y.shape)
        self.assertEqual(x.shape[0], 250)
        self.assertEqual(x.shape[1], 4)

    def test_transform_pca_success(self):
        features = 2
        x_pca = transform_pca(features)

        self.assertEqual(x_pca.shape[0], 250)
        self.assertEqual(x_pca.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
