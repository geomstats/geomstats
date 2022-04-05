"""Unit tests for the Dirichlet manifold."""

import random

from scipy.stats import dirichlet

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.information_geometry.dirichlet import (
    DirichletDistributions,
    DirichletMetric,
)
from tests.conftest import Parametrizer
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestDirichlet(OpenSetTestCase, metaclass=Parametrizer):
    space = DirichletDistributions

    class TestDataDirichlet(_OpenSetTestData):
        space = DirichletDistributions
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = [(n,) for n in n_list]
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
            smoke_space_args_list = [(2,), (3,)]
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

        def sample_test_data(self):
            smoke_data = [
                dict(dim=2, point=gs.array([1.0, 1.0]), n_samples=1, expected=(1, 2)),
                dict(
                    dim=3, point=gs.array([0.1, 0.2, 0.3]), n_samples=2, expected=(2, 3)
                ),
            ]
            return self.generate_tests(smoke_data)

        def sample_belongs_test_data(self):
            random_data = [
                dict(
                    dim=2,
                    point=self.space(2).random_point(3),
                    n_samples=4,
                    expected=gs.ones((3, 4)),
                ),
                dict(
                    dim=3,
                    point=self.space(3).random_point(1),
                    n_samples=2,
                    expected=gs.ones(2),
                ),
                dict(
                    dim=4,
                    point=self.space(4).random_point(2),
                    n_samples=3,
                    expected=gs.ones((2, 3)),
                ),
            ]
            return self.generate_tests([], random_data)

        def point_to_pdf_test_data(self):
            random_data = [
                dict(
                    dim=2,
                    point=self.space(2).random_point(2),
                    n_samples=10,
                ),
                dict(
                    dim=3,
                    point=self.space(3).random_point(4),
                    n_samples=10,
                ),
                dict(
                    dim=4,
                    point=self.space(4).random_point(1),
                    n_samples=10,
                ),
            ]
            return self.generate_tests([], random_data)

    testing_data = TestDataDirichlet()

    def test_belongs(self, dim, vec, expected):
        self.assertAllClose(self.space(dim).belongs(gs.array(vec)), expected)

    def test_random_point(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, dim, point, n_samples, expected):
        self.assertAllClose(self.space(dim).sample(point, n_samples).shape, expected)

    @geomstats.tests.np_and_autograd_only
    def test_sample_belongs(self, dim, point, n_samples, expected):
        samples = self.space(dim).sample(point, n_samples)
        self.assertAllClose(gs.sum(samples, axis=-1), expected)

    @geomstats.tests.np_and_autograd_only
    def test_point_to_pdf(self, dim, point, n_samples):
        point = gs.to_ndarray(point, 2)
        n_points = point.shape[0]
        pdf = self.space(dim).point_to_pdf(point)
        alpha = gs.ones(dim)
        samples = self.space(dim).sample(alpha, n_samples)
        result = pdf(samples)
        pdf = []
        for i in range(n_points):
            pdf.append(gs.array([dirichlet.pdf(x, point[i, :]) for x in samples]))
        expected = gs.squeeze(gs.stack(pdf, axis=0))
        self.assertAllClose(result, expected)


class TestDirichletMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    space = DirichletDistributions
    connection = metric = DirichletMetric
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
    skip_test_squared_dist_is_symmetric = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_log_then_exp = True
    skip_test_exp_then_log = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True

    class TestDataDirichletMetric(_RiemannianMetricTestData):
        space = DirichletDistributions
        metric = DirichletMetric
        n_list = random.sample(range(2, 5), 2)
        metric_args_list = list(
            zip(
                n_list,
            )
        )
        space_list = [DirichletDistributions(n) for n in n_list]
        space_args_list = [(n,) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        shape_list = [(n,) for n in n_list]
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

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
                rtol=0.1,
                atol=0.0,
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_samples_list,
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

        def metric_matrix_shape_test_data(self):
            random_data = [
                dict(dim=2, point=self.space(2).random_point(1), expected=(2, 2)),
                dict(dim=2, point=self.space(2).random_point(3), expected=(3, 2, 2)),
                dict(dim=3, points=self.space(3).random_point(2), expected=(2, 3, 3)),
            ]
            return self.generate_tests([], random_data)

        def metric_matrix_dim_2_test_data(self):
            random_data = [
                dict(point=self.space(2).random_point(n_points))
                for n_points in self.n_points_list
            ]
            return self.generate_tests([], random_data)

        def christoffels_vectorization_test_data(self):
            n_points = 2
            dim = 3
            points = self.space(dim).random_point(n_points)
            christoffel_1 = self.metric(dim).christoffels(points[0, :])
            christoffel_2 = self.metric(dim).christoffels(points[1, :])
            expected = gs.stack((christoffel_1, christoffel_2), axis=0)
            random_data = [dict(dim=dim, point=points, expected=expected)]
            return self.generate_tests([], random_data)

        def christoffels_shape_test_data(self):
            random_data = [
                dict(dim=2, point=self.space(2).random_point(1), expected=(2, 2, 2)),
                dict(dim=2, point=self.space(2).random_point(3), expected=(3, 2, 2, 2)),
                dict(dim=3, point=self.space(3).random_point(2), expected=(2, 3, 3, 3)),
            ]
            return self.generate_tests([], random_data)

        def christoffels_dim_2_test_data(self):
            def coefficients(param_a, param_b):
                """Christoffel coefficients for the beta distributions."""
                poly1a = gs.polygamma(1, param_a)
                poly2a = gs.polygamma(2, param_a)
                poly1b = gs.polygamma(1, param_b)
                poly2b = gs.polygamma(2, param_b)
                poly1ab = gs.polygamma(1, param_a + param_b)
                poly2ab = gs.polygamma(2, param_a + param_b)
                metric_det = 2 * (poly1a * poly1b - poly1ab * (poly1a + poly1b))

                c1 = (poly2a * (poly1b - poly1ab) - poly1b * poly2ab) / metric_det
                c2 = -poly1b * poly2ab / metric_det
                c3 = (poly2b * poly1ab - poly1b * poly2ab) / metric_det
                return c1, c2, c3

            gs.random.seed(123)
            n_points = 3
            points = self.space(2).random_point(n_points)
            param_a, param_b = points[:, 0], points[:, 1]
            c1, c2, c3 = coefficients(param_a, param_b)
            c4, c5, c6 = coefficients(param_b, param_a)
            vector_0 = gs.stack([c1, c2, c3], axis=-1)
            vector_1 = gs.stack([c6, c5, c4], axis=-1)
            gamma_0 = SymmetricMatrices.from_vector(vector_0)
            gamma_1 = SymmetricMatrices.from_vector(vector_1)
            random_data = [
                dict(point=points, expected=gs.stack([gamma_0, gamma_1], axis=-3))
            ]
            return self.generate_tests([], random_data)

        def exp_vectorization_test_data(self):
            dim = 3
            point = self.space(dim).random_point()
            tangent_vec = gs.array([1.0, 0.5, 2.0])
            n_tangent_vecs = 10
            t = gs.linspace(0.0, 1.0, n_tangent_vecs)
            tangent_vecs = gs.einsum("i,...k->...ik", t, tangent_vec)
            random_data = [dict(dim=dim, point=point, tangent_vecs=tangent_vecs)]
            return self.generate_tests([], random_data)

        def exp_diagonal_test_data(self):
            param_list = [0.8, 1.2, 2.5]
            smoke_data = [
                dict(dim=dim, param=param, param_list=param_list)
                for dim in self.n_list
                for param in param_list
            ]
            return self.generate_tests(smoke_data)

        def exp_subspace_test_data(self):
            smoke_data = [
                dict(
                    dim=3,
                    point=[0.1, 0.1, 0.5],
                    vec=[1.3, 1.3, 2.2],
                    expected=[True, True, False],
                ),
                dict(
                    dim=3,
                    point=[3.5, 0.1, 3.5],
                    vec=[0.8, 0.1, 0.8],
                    expected=[True, False, True],
                ),
                dict(
                    dim=4,
                    point=[1.1, 1.1, 2.3, 1.1],
                    vec=[0.6, 0.6, 2.1, 0.6],
                    expected=[True, True, False, True],
                ),
            ]
            return self.generate_tests(smoke_data)

        # def exp_then_log_test_data(self):
        #     random_data = [
        #         dict(
        #             dim=3,
        #             base_point=self.space(3).random_point(),
        #             point=self.space(3).random_point(),
        #         )
        #     ]
        #     return self.generate_tests([], random_data)

        def geodesic_ivp_shape_test_data(self):
            random_data = [
                dict(
                    dim=2,
                    point=self.space(2).random_point(1),
                    vec=self.space(2).random_point(1),
                    n_steps=50,
                    expected=(50, 2),
                ),
                dict(
                    dim=2,
                    point=self.space(2).random_point(3),
                    vec=self.space(2).random_point(3),
                    n_steps=50,
                    expected=(3, 50, 2),
                ),
                dict(
                    dim=3,
                    point=self.space(3).random_point(4),
                    vec=self.space(3).random_point(4),
                    n_steps=50,
                    expected=(4, 50, 3),
                ),
            ]
            return self.generate_tests([], random_data)

        def geodesic_bvp_shape_test_data(self):
            random_data = [
                dict(
                    dim=2,
                    point_a=self.space(2).random_point(1),
                    point_b=self.space(2).random_point(1),
                    n_steps=50,
                    expected=(50, 2),
                ),
                dict(
                    dim=2,
                    point_a=self.space(2).random_point(3),
                    point_b=self.space(2).random_point(3),
                    n_steps=50,
                    expected=(3, 50, 2),
                ),
                dict(
                    dim=3,
                    point_a=self.space(3).random_point(4),
                    point_b=self.space(3).random_point(4),
                    n_steps=50,
                    expected=(4, 50, 3),
                ),
            ]
            return self.generate_tests([], random_data)

        def geodesic_test_data(self):
            random_data = [
                dict(
                    dim=2,
                    point_a=self.space(2).random_point(),
                    point_b=self.space(2).random_point(),
                ),
                dict(
                    dim=4,
                    point_a=self.space(4).random_point(),
                    point_b=self.space(4).random_point(),
                ),
            ]
            return self.generate_tests([], random_data)

        def geodesic_shape_test_data(self):
            random_data = [
                dict(
                    dim=2,
                    point=self.space(2).random_point(),
                    vec=self.space(2).random_point(),
                    time=0.5,
                    expected=(2,),
                ),
                dict(
                    dim=3,
                    point=self.space(3).random_point(),
                    vec=self.space(3).random_point(4),
                    time=0.5,
                    expected=(4, 3),
                ),
                dict(
                    dim=3,
                    point=self.space(3).random_point(),
                    vec=self.space(3).random_point(4),
                    time=gs.linspace(0.0, 1.0, 10),
                    expected=(4, 10, 3),
                ),
            ]
            return self.generate_tests([], random_data)

        def jacobian_christoffels_test_data(self):
            random_data = [
                dict(dim=2, point=self.space(2).random_point(2)),
                dict(dim=4, point=self.space(4).random_point(2)),
            ]
            return self.generate_tests([], random_data)

        def jacobian_in_geodesic_bvp_test_data(self):
            random_data = [
                dict(
                    dim=2,
                    point_a=self.space(2).random_point(),
                    point_b=self.space(2).random_point(),
                ),
                dict(
                    dim=3,
                    point_a=self.space(3).random_point(),
                    point_b=self.space(3).random_point(),
                ),
            ]
            return self.generate_tests([], random_data)

        def approx_geodesic_bvp_test_data(self):
            random_data = [
                dict(
                    dim=2,
                    point_a=self.space(2).random_point(),
                    point_b=self.space(2).random_point(),
                ),
                dict(
                    dim=3,
                    point_a=self.space(3).random_point(),
                    point_b=self.space(3).random_point(),
                ),
            ]
            return self.generate_tests([], random_data)

        def polynomial_init_test_data(self):
            smoke_data = [
                dict(
                    dim=3,
                    point_a=[100.0, 1.0, 1.0],
                    point_b=[1.0, 1.0, 100.0],
                    expected=8.5,
                ),
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataDirichletMetric()

    @geomstats.tests.np_autograd_and_torch_only
    def test_metric_matrix_shape(self, dim, point, expected):
        return self.assertAllClose(
            self.metric(dim).metric_matrix(point).shape, expected
        )

    @geomstats.tests.np_autograd_and_torch_only
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
        return self.assertAllClose(self.metric(2).metric_matrix(point), expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_christoffels_vectorization(self, dim, point, expected):
        return self.assertAllClose(self.metric(dim).christoffels(point), expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_christoffels_shape(self, dim, point, expected):
        return self.assertAllClose(self.metric(dim).christoffels(point).shape, expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_christoffels_dim_2(self, point, expected):
        return self.assertAllClose(self.metric(2).christoffels(point), expected)

    @geomstats.tests.np_and_autograd_only
    def test_exp_diagonal(self, dim, param, param_list):
        """Check that the diagonal x1 = ... = xn is totally geodesic."""
        base_point = param * gs.ones(dim)
        initial_vectors = gs.transpose(gs.tile(param_list, (dim, 1)))
        result = self.metric(dim).exp(initial_vectors, base_point)
        expected = gs.squeeze(gs.transpose(gs.tile(result[..., 0], (dim, 1))))
        return self.assertAllClose(expected, result)

    @geomstats.tests.np_and_autograd_only
    def test_exp_subspace(self, dim, vec, point, expected):
        """Check that subspaces xi1 = ... = xik are totally geodesic."""
        end_point = self.metric(dim).exp(vec, point)
        result = gs.isclose(end_point - end_point[0], 0)
        return self.assertAllClose(expected, result)

    @geomstats.tests.np_and_autograd_only
    def test_exp_vectorization(self, dim, point, tangent_vecs):
        """Test the case with one initial point and several tangent vectors."""
        end_points = self.metric(dim).exp(tangent_vec=tangent_vecs, base_point=point)
        result = end_points.shape
        expected = (tangent_vecs.shape[0], dim)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_autograd_only
    def test_exp_then_log(self, dim, base_point, point):
        log = self.metric(dim).log(point, base_point, n_steps=500)
        expected = point
        result = self.metric(dim).exp(tangent_vec=log, base_point=base_point)
        self.assertAllClose(result, expected, rtol=1e-2)

    @geomstats.tests.np_and_autograd_only
    def test_geodesic_ivp_shape(self, dim, point, vec, n_steps, expected):
        t = gs.linspace(0.0, 1.0, n_steps)
        geodesic = self.metric(dim)._geodesic_ivp(point, vec)
        geodesic_at_t = geodesic(t)
        result = geodesic_at_t.shape
        return self.assertAllClose(result, expected)

    @geomstats.tests.np_and_autograd_only
    def test_geodesic_bvp_shape(self, dim, point_a, point_b, n_steps, expected):
        t = gs.linspace(0.0, 1.0, n_steps)
        geodesic = self.metric(dim)._geodesic_bvp(point_a, point_b)
        geodesic_at_t = geodesic(t)
        result = geodesic_at_t.shape
        return self.assertAllClose(result, expected)

    @geomstats.tests.np_and_autograd_only
    def test_geodesic(self, dim, point_a, point_b):
        """Check that the norm of the geodesic velocity is constant."""
        n_steps = 10000
        geod = self.metric(dim).geodesic(initial_point=point_a, end_point=point_b)
        t = gs.linspace(0.0, 1.0, n_steps)
        geod_at_t = geod(t)
        velocity = n_steps * (geod_at_t[1:, :] - geod_at_t[:-1, :])
        velocity_norm = self.metric(dim).norm(velocity, geod_at_t[:-1, :])
        result = 1 / velocity_norm.min() * (velocity_norm.max() - velocity_norm.min())
        expected = 0.0
        return self.assertAllClose(expected, result, rtol=1.0)

    @geomstats.tests.np_and_autograd_only
    def test_geodesic_shape(self, dim, point, vec, time, expected):
        geod = self.metric(dim).geodesic(initial_point=point, initial_tangent_vec=vec)
        result = geod(time).shape
        self.assertAllClose(expected, result)

    @geomstats.tests.autograd_and_torch_only
    def test_jacobian_christoffels(self, dim, point):
        result = self.metric(dim).jacobian_christoffels(point[0, :])
        self.assertAllClose((dim, dim, dim, dim), result.shape)

        expected = gs.autodiff.jacobian(self.metric(dim).christoffels)(point[0, :])
        self.assertAllClose(expected, result)

        result = self.metric(dim).jacobian_christoffels(point)
        expected = [
            self.metric(dim).jacobian_christoffels(point[0, :]),
            self.metric(dim).jacobian_christoffels(point[1, :]),
        ]
        expected = gs.stack(expected, 0)
        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_autograd_only
    def test_jacobian_in_geodesic_bvp(self, dim, point_a, point_b):
        result = self.metric(dim).dist(point_a, point_b, jacobian=True)
        expected = self.metric(dim).dist(point_a, point_b)
        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_autograd_only
    def test_approx_geodesic_bvp(self, dim, point_a, point_b):
        res = self.metric(dim)._approx_geodesic_bvp(point_a, point_b)
        result = res[0]
        expected = self.metric(dim).dist(point_a, point_b)
        self.assertAllClose(expected, result, atol=0, rtol=1e-1)

    @geomstats.tests.np_and_autograd_only
    def test_polynomial_init(self, dim, point_a, point_b, expected):
        result = self.metric(dim).dist(point_a, point_b, init="polynomial")
        self.assertAllClose(expected, result, atol=0, rtol=1e-1)
