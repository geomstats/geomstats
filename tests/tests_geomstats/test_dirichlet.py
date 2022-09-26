"""Unit tests for the Dirichlet manifold."""

from scipy.stats import dirichlet

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import Parametrizer
from tests.data.dirichlet_data import DirichletMetricTestData, DirichletTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase

TF_OR_PYTORCH_BACKEND = tests.conftest.tf_backend() or tests.conftest.pytorch_backend()


class TestDirichlet(OpenSetTestCase, metaclass=Parametrizer):
    testing_data = DirichletTestData()

    def test_belongs(self, dim, vec, expected):
        self.assertAllClose(self.Space(dim).belongs(gs.array(vec)), expected)

    def test_random_point(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, dim, point, n_samples, expected):
        self.assertAllClose(self.Space(dim).sample(point, n_samples).shape, expected)

    @tests.conftest.np_and_autograd_only
    def test_sample_belongs(self, dim, point, n_samples, expected):
        samples = self.Space(dim).sample(point, n_samples)
        self.assertAllClose(gs.sum(samples, axis=-1), expected)

    @tests.conftest.np_and_autograd_only
    def test_point_to_pdf(self, dim, point, n_samples):
        point = gs.to_ndarray(point, 2)
        n_points = point.shape[0]
        pdf = self.Space(dim).point_to_pdf(point)
        alpha = gs.ones(dim)
        samples = self.Space(dim).sample(alpha, n_samples)
        result = pdf(samples)
        pdf = []
        for i in range(n_points):
            pdf.append(gs.array([dirichlet.pdf(x, point[i, :]) for x in samples]))
        expected = gs.squeeze(gs.stack(pdf, axis=0))
        self.assertAllClose(result, expected)


class TestDirichletMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_shape = True  # because several base points for one vector
    skip_test_log_shape = TF_OR_PYTORCH_BACKEND
    skip_test_exp_belongs = TF_OR_PYTORCH_BACKEND
    skip_test_log_is_tangent = TF_OR_PYTORCH_BACKEND
    skip_test_dist_is_symmetric = TF_OR_PYTORCH_BACKEND
    skip_test_dist_is_positive = TF_OR_PYTORCH_BACKEND
    skip_test_squared_dist_is_symmetric = True
    skip_test_squared_dist_is_positive = TF_OR_PYTORCH_BACKEND
    skip_test_dist_is_norm_of_log = TF_OR_PYTORCH_BACKEND
    skip_test_dist_point_to_itself_is_zero = TF_OR_PYTORCH_BACKEND
    skip_test_log_after_exp = True
    skip_test_exp_after_log = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_triangle_inequality_of_dist = (
        tests.conftest.tf_backend() or tests.conftest.pytorch_backend()
    )

    testing_data = DirichletMetricTestData()
    Space = testing_data.Space

    @tests.conftest.np_autograd_and_torch_only
    def test_metric_matrix_shape(self, dim, point, expected):
        return self.assertAllClose(
            self.Metric(dim).metric_matrix(point).shape, expected
        )

    @tests.conftest.np_autograd_and_torch_only
    def test_metric_matrix_dim_2(self, point):
        param_a = point[..., 0]
        param_b = point[..., 1]
        vector = gs.stack(
            [
                gs.polygamma(1, param_a) - gs.polygamma(1, param_a + param_b),
                -gs.polygamma(1, param_a + param_b),
                gs.polygamma(1, param_b) - gs.polygamma(1, param_a + param_b),
            ],
            axis=-1,
        )
        expected = SymmetricMatrices.from_vector(vector)
        return self.assertAllClose(self.Metric(2).metric_matrix(point), expected)

    @tests.conftest.np_autograd_and_tf_only
    def test_christoffels_vectorization(self, dim, point, expected):
        return self.assertAllClose(self.Metric(dim).christoffels(point), expected)

    @tests.conftest.np_autograd_and_tf_only
    def test_christoffels_shape(self, dim, point, expected):
        return self.assertAllClose(self.Metric(dim).christoffels(point).shape, expected)

    @tests.conftest.np_autograd_and_tf_only
    def test_christoffels_dim_2(self, point, expected):
        return self.assertAllClose(self.Metric(2).christoffels(point), expected)

    @tests.conftest.np_and_autograd_only
    def test_exp_diagonal(self, dim, param, param_list):
        """Check that the diagonal x1 = ... = xn is totally geodesic."""
        base_point = param * gs.ones(dim)
        initial_vectors = gs.transpose(gs.tile(param_list, (dim, 1)))
        result = self.Metric(dim).exp(initial_vectors, base_point)
        expected = gs.squeeze(gs.transpose(gs.tile(result[..., 0], (dim, 1))))
        return self.assertAllClose(expected, result)

    @tests.conftest.np_and_autograd_only
    def test_exp_subspace(self, dim, vec, point, expected, atol):
        """Check that subspaces xi1 = ... = xik are totally geodesic."""
        end_point = self.Metric(dim).exp(vec, point)
        result = gs.isclose(end_point - end_point[0], 0.0, atol=atol)
        return self.assertAllClose(expected, result)

    @tests.conftest.np_and_autograd_only
    def test_exp_vectorization(self, dim, point, tangent_vecs):
        """Test the case with one initial point and several tangent vectors."""
        end_points = self.Metric(dim).exp(tangent_vec=tangent_vecs, base_point=point)
        result = end_points.shape
        expected = (tangent_vecs.shape[0], dim)
        self.assertAllClose(result, expected)

    @tests.conftest.np_and_autograd_only
    def test_exp_after_log(self, dim, base_point, point):
        log = self.Metric(dim).log(point, base_point, n_steps=500)
        expected = point
        result = self.Metric(dim).exp(tangent_vec=log, base_point=base_point)
        self.assertAllClose(result, expected, rtol=1e-2)

    @tests.conftest.np_and_autograd_only
    def test_geodesic_ivp_shape(self, dim, point, vec, n_steps, expected):
        t = gs.linspace(0.0, 1.0, n_steps)
        geodesic = self.Metric(dim)._geodesic_ivp(point, vec)
        geodesic_at_t = geodesic(t)
        result = geodesic_at_t.shape
        return self.assertAllClose(result, expected)

    @tests.conftest.np_and_autograd_only
    def test_geodesic_bvp_shape(self, dim, point_a, point_b, n_steps, expected):
        t = gs.linspace(0.0, 1.0, n_steps)
        geodesic = self.Metric(dim)._geodesic_bvp(point_a, point_b)
        geodesic_at_t = geodesic(t)
        result = geodesic_at_t.shape
        return self.assertAllClose(result, expected)

    @tests.conftest.np_and_autograd_only
    def test_geodesic(self, dim, point_a, point_b):
        """Check that the norm of the geodesic velocity is constant."""
        n_steps = 10000
        geod = self.Metric(dim).geodesic(initial_point=point_a, end_point=point_b)
        t = gs.linspace(0.0, 1.0, n_steps)
        geod_at_t = geod(t)
        velocity = n_steps * (geod_at_t[1:, :] - geod_at_t[:-1, :])
        velocity_norm = self.Metric(dim).norm(velocity, geod_at_t[:-1, :])
        result = 1 / velocity_norm.min() * (velocity_norm.max() - velocity_norm.min())
        expected = 0.0
        return self.assertAllClose(expected, result, rtol=1.0)

    @tests.conftest.np_and_autograd_only
    def test_geodesic_shape(self, dim, point, vec, time, expected):
        geod = self.Metric(dim).geodesic(initial_point=point, initial_tangent_vec=vec)
        result = geod(time).shape
        self.assertAllClose(expected, result)

    @tests.conftest.autograd_and_torch_only
    def test_jacobian_christoffels(self, dim, point):
        result = self.Metric(dim).jacobian_christoffels(point[0, :])
        self.assertAllClose((dim, dim, dim, dim), result.shape)

        expected = gs.autodiff.jacobian(self.Metric(dim).christoffels)(point[0, :])
        self.assertAllClose(expected, result)

        result = self.Metric(dim).jacobian_christoffels(point)
        expected = [
            self.Metric(dim).jacobian_christoffels(point[0, :]),
            self.Metric(dim).jacobian_christoffels(point[1, :]),
        ]
        expected = gs.stack(expected, 0)
        self.assertAllClose(expected, result)

    @tests.conftest.np_and_autograd_only
    def test_jacobian_in_geodesic_bvp(self, dim, point_a, point_b):
        result = self.Metric(dim).dist(point_a, point_b, jacobian=True)
        expected = self.Metric(dim).dist(point_a, point_b)
        self.assertAllClose(expected, result)

    @tests.conftest.np_and_autograd_only
    def test_approx_geodesic_bvp(self, dim, point_a, point_b):
        res = self.Metric(dim)._approx_geodesic_bvp(point_a, point_b)
        result = res[0]
        expected = self.Metric(dim).dist(point_a, point_b)
        self.assertAllClose(expected, result, atol=0, rtol=1e-1)

    @tests.conftest.np_and_autograd_only
    def test_polynomial_init(self, dim, point_a, point_b, expected):
        result = self.Metric(dim).dist(point_a, point_b, init="polynomial")
        self.assertAllClose(expected, result, atol=0, rtol=1e-1)
