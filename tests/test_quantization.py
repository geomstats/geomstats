import geomstats.tests

from sklearn.utils.testing import assert_allclose
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.quantization import Quantization


SPHERE = Hypersphere(dimension=2)
METRIC = SPHERE.metric
N_SAMPLES = 1000
N_CLUSTERS = 1


class TestQuantizationMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)

        self.dimension = 2
        self.space = Hypersphere(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 1000
        self.n_clusters = 1
        self.data = self.space.random_von_mises_fisher(
            kappa=10, n_samples=self.n_samples)

        def test_fit(self):
            X = self.data
            clustering = Quantization(self.metric, self.n_clusters)
            clustering.fit(X)

        def test_predict(self):
            point = self.data[0, :]
            clustering = Quantization(self.metric, self.n_clusters)

            prediction = clustering.predict(point)
            result = prediction
            expected = clustering.labels[0]
            self.assertAllClose(expected, result)


if __name__ == '__main__':
        geomstats.tests.main()
