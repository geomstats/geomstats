"""Unit tests for the beta manifold."""

import random
import warnings

import pytest
from scipy.stats import beta

import geomstats.backend as gs
import geomstats.tests
from geomstats.information_geometry.beta import BetaDistributions, BetaMetric
from tests.conftest import Parametrizer
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestBetaDistributions(OpenSetTestCase, metaclass=Parametrizer):
    space = BetaDistributions

    class BetaDistributionsTestsData(_OpenSetTestData):
        space = BetaDistributions
        space_args_list = [()]
        shape_list = [(2,)]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(1, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_test_data(self):
            smoke_data = [
                dict(dim=3, vec=[0.1, 1.0, 0.3], expected=True),
                dict(dim=3, vec=[0.1, 1.0], expected=False),
                dict(dim=3, vec=[0.0, 1.0, 0.3], expected=False),
                dict(dim=2, vec=[-1.0, 0.3], expected=False),
            ]
            return self.generate_tests(smoke_data)

        def random_point_test_data(self):
            random_data = [
                dict(point=self.space(2).random_point(1), expected=(2,)),
                dict(point=self.space(3).random_point(5), expected=(5, 3)),
            ]
            return self.generate_tests([], random_data)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(), ()]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                self.space,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def to_tangent_is_tangent_in_ambient_space_test_data(self):
            return self._to_tangent_is_tangent_in_ambient_space_test_data(
                self.space,
                self.space_args_list,
                self.shape_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                self.space,
                self.space_args_list,
                self.n_vecs_list,
                is_tangent_atol=gs.atol,
            )

        def point_to_pdf_test_data(self):
            x = gs.linspace(0.0, 1.0, 10)
            point = self.space().random_point(2)
            pdf1 = beta.pdf(x, a=point[0, 0], b=point[0, 1])
            pdf2 = beta.pdf(x, a=point[1, 0], b=point[1, 1])
            expected = gs.stack([gs.array(pdf1), gs.array(pdf2)], axis=1)

            random_data = [
                dict(point=point, x=x, expected=expected),
                dict(point=point[0], x=x, expected=expected[0]),
            ]
            return self.generate_tests([], random_data)

    testing_data = BetaDistributionsTestsData()

    def test_point_to_pdf(self, point, x, expected):
        point = self.space().random_point()
        pdf = self.space().point_to_pdf(point)
        result = pdf(x)
        self.assertAllClose(result, expected)


class TestBetaMetric(RiemannianMetricTestCase, metaclass=Parametrizer):

    space = BetaMetric
    connection = metric = BetaMetric
    skip_test_exp_shape = True  # because several base points for one vector
    skip_test_log_shape = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_exp_belongs = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_log_is_tangent = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_is_symmetric = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_is_positive = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_squared_dist_is_symmetric = True
    skip_test_squared_dist_is_positive = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_is_norm_of_log = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_point_to_itself_is_zero = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_log_after_exp = True
    skip_test_exp_after_log = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True

    class BetaMetricTestData(_RiemannianMetricTestData):
        space = BetaDistributions
        metric = BetaMetric
        metric_args_list = [()]
        shape_list = [(2,)]
        space_list = [BetaDistributions()]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(1, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def exp_shape_test_data(self):
            return self._exp_shape_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(
                self.metric_args_list,
                self.space_list,
            )

        def exp_belongs_test_data(self):
            return self._exp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_samples_list,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
            )

        def log_after_exp_test_data(self):
            return self._log_after_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                rtol=0.1,
                atol=0.0,
            )

        def exp_after_log_test_data(self):
            return self._exp_after_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                self.n_vecs_list,
                rtol=0.1,
                atol=0.0,
            )

        def squared_dist_is_symmetric_test_data(self):
            return self._squared_dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                self.n_points_list,
                0.1,
                0.1,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                self.n_points_list,
                is_positive_atol=gs.atol,
            )

        def dist_is_symmetric_test_data(self):
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                self.n_points_list,
                rtol=0.1,
                atol=gs.atol,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                self.n_points_list,
                is_positive_atol=gs.atol,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                self.n_points_list,
                rtol=0.1,
                atol=gs.atol,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol,
                atol=1e-5,
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_vecs_list,
                rtol=gs.rtol,
                atol=gs.atol,
            )

        def metric_matrix_test_data(self):
            smoke_data = [
                dict(
                    point=gs.array([1.0, 1.0]),
                    expected=gs.array([[1.0, -0.644934066], [-0.644934066, 1.0]]),
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = BetaMetricTestData()

    def test_metric_matrix(self, point, expected):
        result = self.metric().metric_matrix(point)
        self.assertAllClose(result, expected)


class TestBetaDistributions(geomstats.tests.TestCase):
    """Class defining the beta distributions tests."""

    def setup_method(self):
        """Define the parameters of the tests."""
        warnings.simplefilter("ignore", category=UserWarning)
        self.beta = BetaDistributions()
        self.metric = BetaMetric()
        self.n_samples = 10
        self.dim = self.beta.dim

    def test_random_uniform_and_belongs(self):
        """Test random_uniform and belongs.

        Test that the random uniform method samples
        on the beta distribution space.
        """
        point = self.beta.random_point()
        result = self.beta.belongs(point)
        expected = True
        self.assertAllClose(expected, result)

    def test_random_uniform_and_belongs_vectorization(self):
        """Test random_uniform and belongs.

        Test that the random uniform method samples
        on the beta distribution space.
        """
        n_samples = self.n_samples
        point = self.beta.random_point(n_samples)
        result = self.beta.belongs(point)
        expected = gs.array([True] * n_samples)
        self.assertAllClose(expected, result)

    def test_random_uniform(self):
        """Test random_uniform.

        Test that the random uniform method samples points of the right shape
        """
        point = self.beta.random_point(self.n_samples)
        self.assertAllClose(gs.shape(point), (self.n_samples, self.dim))

    @geomstats.tests.np_and_autograd_only
    def test_exp(self):
        """Test Exp.

        Test that the Riemannian exponential at points on the first
        bisector computed in the direction of the first bisector stays
        on the first bisector.
        """
        gs.random.seed(123)
        n_samples = self.n_samples
        points = self.beta.random_point(n_samples)
        vectors = self.beta.random_point(n_samples)
        initial_vectors = gs.array([[vec_x, vec_x] for vec_x in vectors[:, 0]])
        points = gs.array([[param_a, param_a] for param_a in points[:, 0]])
        result_points = self.metric.exp(initial_vectors, points)
        result = gs.isclose(result_points[:, 0], result_points[:, 1]).all()
        expected = gs.array([True] * n_samples)
        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_autograd_only
    def test_log_and_exp(self):
        """Test Log and Exp.

        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        n_samples = self.n_samples
        gs.random.seed(123)
        base_point = self.beta.random_point(n_samples=n_samples, bound=5)
        point = self.beta.random_point(n_samples=n_samples, bound=5)
        log = self.metric.log(point, base_point, n_steps=500)
        expected = point
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        self.assertAllClose(result, expected, rtol=1e-2)

    @geomstats.tests.np_and_autograd_only
    def test_log_vectorization(self):
        """Test vectorization of Log.

        Test the case with several base points and one end point.
        """
        n_points = 10
        base_points = self.beta.random_point(n_samples=n_points)
        point = self.beta.random_point()
        tangent_vecs = self.metric.log(base_point=base_points, point=point)
        result = tangent_vecs.shape
        expected = (n_points, 2)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_christoffels_vectorization(self):
        """Test Christoffel synbols.

        Check vectorization of Christoffel symbols.
        """
        points = self.beta.random_point(self.n_samples)
        christoffel = self.metric.christoffels(points)
        result = christoffel.shape
        expected = gs.array([self.n_samples, self.dim, self.dim, self.dim])
        self.assertAllClose(result, expected)

    def test_metric_matrix(self):
        """Test metric matrix.

        Check the value of the metric matrix for a particular
        point in the space of beta distributions.
        """
        point = gs.array([1.0, 1.0])
        result = self.beta.metric.metric_matrix(point)
        expected = gs.array([[1.0, -0.644934066], [-0.644934066, 1.0]])
        self.assertAllClose(result, expected)

        with pytest.raises(ValueError):
            self.beta.metric.metric_matrix()

    def test_point_to_pdf(self):
        """Test point_to_pdf.

        Check the computation of the pdf.
        """
        point = self.beta.random_point()
        pdf = self.beta.point_to_pdf(point)
        x = gs.linspace(0.0, 1.0, 10)
        result = pdf(x)
        expected = beta.pdf(x, a=point[0], b=point[1])
        self.assertAllClose(result, expected)

    def test_point_to_pdf_vectorization(self):
        """Test point_to_pdf.

        Check vectorization of the computation of the pdf.
        """
        point = self.beta.random_point(n_samples=2)
        pdf = self.beta.point_to_pdf(point)
        x = gs.linspace(0.0, 1.0, 10)
        result = pdf(x)
        pdf1 = beta.pdf(x, a=point[0, 0], b=point[0, 1])
        pdf2 = beta.pdf(x, a=point[1, 0], b=point[1, 1])
        expected = gs.stack([gs.array(pdf1), gs.array(pdf2)], axis=1)
        self.assertAllClose(result, expected)
