"""Unit tests for the KDE classifier."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.kernel_density_estimation_classifier \
    import KernelDensityEstimationClassifier
from geomstats.learning.radial_kernel_functions import triangular_radial_kernel

TOLERANCE = 1e-4


class TestKernelDensityEstimationClassifier(geomstats.tests.TestCase):
    """Class defining the Kernel Density Estimation Classifier tests."""

    def setUp(self):
        """Define the parameters to test."""
        gs.random.seed(1234)
        self.dim = 2
        self.space = Euclidean(dim=self.dim)
        self.distance = self.space.metric.dist

    def test_predict(self):
        """Test the 'predict' class method."""
        training_dataset = gs.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [2.0, 0.0],
             [3.0, 0.0]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict(gs.array([[1.1, 0.0]]))
        expected = gs.array([0])
        self.assertAllClose(expected, result)

    def test_predict_one_dimensional_data(self):
        """Test the 'predict' class method."""
        training_dataset = gs.array(
            [[0.0],
             [1.0],
             [2.0],
             [3.0]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            distance='minkowski')
        kde.fit(training_dataset, labels)
        result = kde.predict(gs.array([1.1]))
        expected = gs.array([0])
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_predict_one_dimensional_data_callable_distance(self):
        """Test the 'predict' class method on one dimensional data."""
        training_dataset = gs.array([0, 1, 2, 3])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict(gs.array([1.1]))
        expected = gs.array([0])
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_predict_proba_uniform_kernel_one_dimensional_data(self):
        """Test the 'predict_proba' class method using the 'uniform' kernel.

        Test the 'predict_proba' class method using the 'uniform' kernel on
        one-dimensional date of shape [n_samples,].
        """
        training_dataset = gs.array([0, 1, 2, 3])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            kernel='uniform',
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict_proba(gs.array([0.9]))
        expected = gs.array([[1 / 2, 1 / 2]])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_predict_proba_uniform_kernel(self):
        """Test the 'predict_proba' class method using the 'uniform' kernel."""
        training_dataset = gs.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [2.0, 0.0],
             [3.0, 0.0]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            kernel='uniform',
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict_proba(gs.array([[0.9, 0.0]]))
        expected = gs.array([[1 / 2, 1 / 2]])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_predict_proba_distance_kernel(self):
        """Test the 'predict_proba' class method using 'distance' kernel."""
        training_dataset = gs.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [2.0, 0.0],
             [3.0, 0.0]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            kernel='distance',
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict_proba(gs.array([[1.0, 0.0]]))
        expected = gs.array([[1, 0]])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_and_pytorch_only
    def test_predict_proba_triangular_kernel(self):
        """Test the 'predict_proba' class method using a triangular kernel."""
        training_dataset = gs.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [2.0, 0.0],
             [3.0, 0.0]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            kernel=triangular_radial_kernel,
            bandwidth=2.0,
            p=2,
            distance='minkowski')
        kde.fit(training_dataset, labels)
        result = kde.predict_proba(gs.array([[1.0, 0.0]]))
        expected = gs.array([[3 / 4, 1 / 4]])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_and_pytorch_only
    def test_predict_proba_triangular_kernel_callable_distance(self):
        """Test the 'predict_proba' class method using a triangular kernel."""
        training_dataset = gs.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [2.0, 0.0],
             [3.0, 0.0]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            kernel=triangular_radial_kernel,
            bandwidth=2.0,
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict_proba(gs.array([[1.0, 0.0]]))
        expected = gs.array([[3 / 4, 1 / 4]])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_and_pytorch_only
    def test_predict_triangular_kernel_callable_distance(self):
        """Test the 'predict' class method using a triangular kernel."""
        training_dataset = gs.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [2.0, 0.0],
             [3.0, 0.0]])
        labels = [0, 0, 1, 1]
        kde = KernelDensityEstimationClassifier(
            kernel=triangular_radial_kernel,
            bandwidth=2.0,
            distance=self.distance)
        kde.fit(training_dataset, labels)
        result = kde.predict(gs.array([[1.0, 0.0], [1.0, 0.0]]))
        expected = gs.array([0, 0])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_predict_hypersphere_distance(self):
        """Test the 'predict' class method using the hypersphere distance."""
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

    def test_predict_poincare_ball_distance(self):
        """Test the 'predict' class method using the Poincare ball distance."""
        dim = 2
        space = PoincareBall(dim=dim)
        distance = space.metric.dist
        training_dataset = gs.array(
            [[1 / 2, 1 / 4],
             [1 / 2, 0],
             [1 / 2, - 1 / 4],
             [- 1 / 2, 1 / 4],
             [- 1 / 2, 0],
             [- 1 / 2, - 1 / 4]])
        labels = [0, 0, 0, 1, 1, 1]
        kde = KernelDensityEstimationClassifier(
            distance=distance,
            kernel='distance')
        kde.fit(training_dataset, labels)
        target_dataset = gs.array(
            [[1 / 2, 1 / 5],
             [1 / 2, 0],
             [1 / 2, - 1 / 5],
             [- 1 / 2, 1 / 5],
             [- 1 / 2, 0],
             [- 1 / 2, - 1 / 5]])
        result = kde.predict(target_dataset)
        expected = [0, 0, 0, 1, 1, 1]
        self.assertAllClose(expected, result)

    def test_predict_hyperboloid_distance(self):
        """Test the 'predict' class method using the hyperboloid distance."""
        dim = 2
        space = Hyperboloid(dim=dim)
        distance = space.metric.dist
        training_dataset_intrinsic = gs.array(
            [[1 / 2, 1 / 4],
             [1 / 2, 0],
             [1 / 2, - 1 / 4],
             [- 1 / 2, 1 / 4],
             [- 1 / 2, 0],
             [- 1 / 2, - 1 / 4]])
        training_dataset = space.change_coordinates_system(
            training_dataset_intrinsic,
            from_coordinates_system='intrinsic',
            to_coordinates_system='extrinsic')
        labels = [0, 0, 0, 1, 1, 1]
        kde = KernelDensityEstimationClassifier(
            distance=distance,
            kernel='distance')
        kde.fit(training_dataset, labels)
        target_dataset_intrinsic = gs.array(
            [[1 / 2, 1 / 5],
             [1 / 2, 0],
             [1 / 2, - 1 / 5],
             [- 1 / 2, 1 / 5],
             [- 1 / 2, 0],
             [- 1 / 2, - 1 / 5]])
        target_dataset = space.change_coordinates_system(
            target_dataset_intrinsic,
            from_coordinates_system='intrinsic',
            to_coordinates_system='extrinsic')
        result = kde.predict(target_dataset)
        expected = [0, 0, 0, 1, 1, 1]
        self.assertAllClose(expected, result)
