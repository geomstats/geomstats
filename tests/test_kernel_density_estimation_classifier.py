"""Unit tests for the KNN classifier."""

import math

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.kernel_density_estimation_classifier \
    import KernelDensityEstimationClassifier
from geomstats.learning.radial_kernel_functions import triangular_radial_kernel

TOLERANCE = 1e-4


class TestKernelDensityEstimationClassifier(geomstats.tests.TestCase):
    """Class defining the Kernel Density Estimation Classifier tests."""

    def setUp(self):
        """Define the parameters to test."""
        gs.random.seed(1234)
        self.radius = math.inf
        self.dim = 1
        self.space = Euclidean(dim=self.dim)
        self.distance = self.space.metric.dist

    @geomstats.tests.np_only
    def test_predict(self):
        """Test the 'predict' class method."""
        training_dataset = gs.array([[0], [1], [2], [3]])
        labels = [0, 0, 1, 1]

        kde = KernelDensityEstimationClassifier(
            radius=self.radius,
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict([[1.1]])
        expected = gs.array([0])
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_predict_proba_uniform_kernel(self):
        """Test the 'predict_proba' class method using the 'uniform' kernel."""
        training_dataset = gs.array([[0], [1], [2], [3]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            radius=self.radius,
            kernel='uniform',
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict_proba([[0.9]])
        expected = gs.array([[1 / 2, 1 / 2]])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_predict_proba_distance_kernel(self):
        """Test the 'predict_proba' class method using 'distance' kernel."""
        training_dataset = gs.array([[0], [1], [2], [3]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            radius=self.radius,
            kernel='distance',
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict_proba([[1]])
        expected = gs.array([[1, 0]])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_predict_proba_triangular_kernel(self):
        """Test the 'predict_proba' class method using a triangular kernel."""
        training_dataset = gs.array([[0], [1], [2], [3]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            radius=self.radius,
            kernel=triangular_radial_kernel,
            bandwidth=2.0,
            p=2,
            distance='minkowski')
        kde.fit(training_dataset, labels)
        result = kde.predict_proba([[1]])
        expected = gs.array([[3 / 4, 1 / 4]])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_predict_proba_triangular_kernel_callable_distance(self):
        """Test the 'predict_proba' class method using a triangular kernel."""
        training_dataset = gs.array([[0], [1], [2], [3]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            radius=self.radius,
            kernel=triangular_radial_kernel,
            bandwidth=2.0,
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict_proba([[1]])
        expected = gs.array([[3 / 4, 1 / 4]])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_fit_hypersphere_distance(self):
        """Test the 'fit' class method using the hypersphere distance."""
        dim = 2
        space = Hypersphere(dim=dim)
        distance = space.metric.dist
        training_dataset = gs.array(
            [[1, 0, 0],
             [3 ** (1 / 2) / 2, 1 / 2, 0],
             [3 ** (1 / 2) / 2, - 1 / 2, 0],
             [0, 0, 1],
             [0, 1 / 2, 3 ** (1 / 2) / 2],
             [0, - 1 / 2, 3 ** (1 / 2) / 2]])
        labels = [0, 0, 0, 1, 1, 1]
        kde = KernelDensityEstimationClassifier(
            radius=self.radius,
            distance=distance)
        kde.fit(training_dataset, labels)
        target_dataset = gs.array(
            [[2 ** (1 / 2) / 2, 2 ** (1 / 2) / 2, 0],
             [0, 1 / 2, - 3 ** (1 / 2) / 2],
             [0, - 1 / 2, - 3 ** (1 / 2) / 2],
             [- 3 ** (1 / 2) / 2, 1 / 2, 0],
             [- 3 ** (1 / 2) / 2, - 1 / 2, 0],
             [0, 2 ** (1 / 2) / 2, 2 ** (1 / 2) / 2]])
        result = kde.predict(target_dataset)
        expected = [0, 0, 0, 1, 1, 1]
        self.assertAllClose(expected, result)
