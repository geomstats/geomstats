"""Unit tests for parameterized manifolds."""

import random

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
import geomstats.tests
from geomstats.geometry.discrete_curves import (
    ClosedDiscreteCurves,
    ClosedSRVMetric,
    DiscreteCurves,
    ElasticMetric,
    L2CurvesMetric,
    SRVMetric,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from tests.conftest import Parametrizer
from tests.data_generation import TestData, _ManifoldTestData, _RiemannianMetricTestData
from tests.geometry_test_cases import (
    ManifoldTestCase,
    RiemannianMetricTestCase,
    TestCase,
)

s2 = Hypersphere(dim=2)
r2 = Euclidean(dim=2)
r3 = s2.embedding_space

initial_point = [0.0, 0.0, 1.0]
initial_tangent_vec_a = [1.0, 0.0, 0.0]
initial_tangent_vec_b = [0.0, 1.0, 0.0]
initial_tangent_vec_c = [-1.0, 0.0, 0.0]

curve_fun_a = s2.metric.geodesic(
    initial_point=initial_point, initial_tangent_vec=initial_tangent_vec_a
)
curve_fun_b = s2.metric.geodesic(
    initial_point=initial_point, initial_tangent_vec=initial_tangent_vec_b
)
curve_fun_c = s2.metric.geodesic(
    initial_point=initial_point, initial_tangent_vec=initial_tangent_vec_c
)


n_sampling_points = 10
sampling_times = gs.linspace(0.0, 1.0, n_sampling_points)
curve_a = curve_fun_a(sampling_times)
curve_b = curve_fun_b(sampling_times)
curve_c = curve_fun_c(sampling_times)


n_discretized_curves = 5
times = gs.linspace(0.0, 1.0, n_discretized_curves)


class TestDiscreteCurves(ManifoldTestCase, metaclass=Parametrizer):
    space = DiscreteCurves
    skip_test_projection_belongs = True
    skip_test_random_tangent_vec_is_tangent = True

    class DiscreteCurvesTestData(_ManifoldTestData):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = Euclidean(dim=3)
        space_args_list = [(r2,), (r3,)]
        shape_list = [(10, 2), (10, 3)]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(self.s2,), (self.r2,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_points_list
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                DiscreteCurves,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                DiscreteCurves, self.space_args_list, self.n_vecs_list
            )

    testing_data = DiscreteCurvesTestData()


class TestL2CurvesMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = L2CurvesMetric
    skip_test_exp_belongs = True
    skip_test_exp_shape = True
    skip_test_log_shape = True
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True

    class L2CurvesMetricTestData(_RiemannianMetricTestData):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = Euclidean(dim=3)
        ambient_manifolds_list = [r2, r3]
        metric_args_list = [
            (ambient_manifolds,) for ambient_manifolds in ambient_manifolds_list
        ]
        shape_list = [(10, 2), (10, 3)]
        space_list = [
            DiscreteCurves(ambient_manifolds)
            for ambient_manifolds in ambient_manifolds_list
        ]
        n_points_list = random.sample(range(2, 5), 2)
        n_tangent_vecs_list = random.sample(range(2, 5), 2)
        n_points_a_list = random.sample(range(2, 5), 2)
        n_points_b_list = [1]
        batch_size_list = random.sample(range(2, 5), 2)
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(self.metric_args_list, self.space_list)

        def squared_dist_is_symmetric_test_data(self):
            return self._squared_dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_test_data(self):
            return self._exp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def exp_after_log_test_data(self):
            return self._exp_after_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def log_after_exp_test_data(self):
            return self._log_after_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_ladder_parallel_transport_test_data(self):
            return self._exp_ladder_parallel_transport_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_test_data(self):
            return self._exp_geodesic_ivp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_points_list,
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def dist_is_symmetric_test_data(self):
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list, self.space_list, self.n_points_list
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
            )

    testing_data = L2CurvesMetricTestData()


class TestSRVMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = SRVMetric
    skip_test_exp_shape = True
    skip_test_log_shape = True
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_geodesic_ivp_belongs = True

    class SRVMetricTestData(_RiemannianMetricTestData):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = Euclidean(dim=3)
        ambient_manifolds_list = [r2, r3]
        metric_args_list = [
            (ambient_manifolds,) for ambient_manifolds in ambient_manifolds_list
        ]
        shape_list = [(10, 2), (10, 3)]
        space_list = [
            DiscreteCurves(ambient_manifolds)
            for ambient_manifolds in ambient_manifolds_list
        ]
        n_points_list = random.sample(range(2, 5), 2)
        n_tangent_vecs_list = random.sample(range(2, 5), 2)
        n_points_a_list = [1, 2]
        n_points_b_list = [1, 2]
        batch_size_list = random.sample(range(2, 5), 2)
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(self.metric_args_list, self.space_list)

        def squared_dist_is_symmetric_test_data(self):
            return self._squared_dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_test_data(self):
            return self._exp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def exp_after_log_test_data(self):
            return self._exp_after_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def log_after_exp_test_data(self):
            return self._log_after_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_ladder_parallel_transport_test_data(self):
            return self._exp_ladder_parallel_transport_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_test_data(self):
            return self._exp_geodesic_ivp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_points_list,
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def dist_is_symmetric_test_data(self):
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list, self.space_list, self.n_points_list
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
            )

        def aux_differential_srv_transform_test_data(self):
            smoke_data = [
                dict(
                    dim=3,
                    n_sampling_points=2000,
                    n_curves=2000,
                    curve_fun_a=curve_fun_a,
                )
            ]
            return self.generate_tests(smoke_data)

        def aux_differential_srv_transform_inverse_test_data(self):
            smoke_data = [
                dict(dim=3, n_sampling_points=n_sampling_points, curve_a=curve_a)
            ]
            return self.generate_tests(smoke_data)

        def aux_differential_srv_transform_vectorization_test_data(self):
            smoke_data = [
                dict(
                    dim=3,
                    n_sampling_points=n_sampling_points,
                    curve_a=curve_a,
                    curve_b=curve_b,
                )
            ]
            return self.generate_tests(smoke_data)

        def srv_inner_product_elastic_test_data(self):
            smoke_data = [dict(dim=3, n_sampling_points=n_sampling_points)]
            return self.generate_tests(smoke_data)

        def srv_inner_product_and_dist_test_data(self):
            smoke_data = [dict(dim=3, curve_a=curve_a, curve_b=curve_b)]
            return self.generate_tests(smoke_data)

        def srv_inner_product_vectorization_test_data(self):
            smoke_data = [
                dict(
                    dim=3,
                    n_sampling_points=n_sampling_points,
                    curve_a=curve_a,
                    curve_b=curve_b,
                )
            ]
            return self.generate_tests(smoke_data)

        def split_horizontal_vertical_test_data(self):
            smoke_data = [dict(times=times, n_discretized_curves=n_discretized_curves)]
            return self.generate_tests(smoke_data)

        def space_derivative_test_data(self):
            smoke_data = [
                dict(
                    dim=3,
                    n_points=3,
                    n_discretized_curves=n_discretized_curves,
                    n_sampling_points=n_sampling_points,
                )
            ]
            return self.generate_tests(smoke_data)

        def srv_inner_product_test_data(self):
            smoke_data = [
                dict(curve_a=curve_a, curve_b=curve_b, curve_c=curve_c, times=times)
            ]
            return self.generate_tests(smoke_data)

        def srv_norm_test_data(self):
            smoke_data = [dict(curve_a=curve_a, curve_b=curve_b, times=times)]
            return self.generate_tests(smoke_data)

    testing_data = SRVMetricTestData()

    def test_srv_inner_product(self, curve_a, curve_b, curve_c, times):
        l2_metric_s2 = L2CurvesMetric(ambient_manifold=s2)
        srv_metric_r3 = SRVMetric(ambient_manifold=r3)
        curves_ab = l2_metric_s2.geodesic(curve_a, curve_b)
        curves_bc = l2_metric_s2.geodesic(curve_b, curve_c)
        curves_ab = curves_ab(times)
        curves_bc = curves_bc(times)
        srvs_ab = srv_metric_r3.srv_transform(curves_ab)
        srvs_bc = srv_metric_r3.srv_transform(curves_bc)

        result = srv_metric_r3.l2_metric.inner_product(srvs_ab, srvs_bc)
        products = srvs_ab * srvs_bc
        expected = [gs.sum(product) for product in products]
        expected = gs.array(expected) / (srvs_ab.shape[-2] + 1)
        self.assertAllClose(result, expected)

        result = result.shape
        expected = [srvs_ab.shape[0]]
        self.assertAllClose(result, expected)

    def test_srv_norm(self, curve_a, curve_b, times):
        l2_metric_s2 = L2CurvesMetric(ambient_manifold=s2)
        srv_metric_r3 = SRVMetric(ambient_manifold=r3)
        curves_ab = l2_metric_s2.geodesic(curve_a, curve_b)
        curves_ab = curves_ab(times)
        srvs_ab = srv_metric_r3.srv_transform(curves_ab)

        result = srv_metric_r3.l2_metric.norm(srvs_ab)
        products = srvs_ab * srvs_ab
        sums = [gs.sum(product) for product in products]
        squared_norm = gs.array(sums) / (srvs_ab.shape[-2] + 1)
        expected = gs.sqrt(squared_norm)
        self.assertAllClose(result, expected)

        result = result.shape
        expected = [srvs_ab.shape[0]]
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_autograd_only
    def test_aux_differential_srv_transform(
        self, dim, n_sampling_points, n_curves, curve_fun_a
    ):
        """Test differential of square root velocity transform.
        Check that its value at (curve, tangent_vec) coincides
        with the derivative at zero of the square root velocity
        transform of a path of curves starting at curve with
        initial derivative tangent_vec.
        """
        srv_metric_r3 = SRVMetric(r3)
        sampling_times = gs.linspace(0.0, 1.0, n_sampling_points)
        curve_a = curve_fun_a(sampling_times)
        tangent_vec = gs.transpose(
            gs.tile(gs.linspace(1.0, 2.0, n_sampling_points), (dim, 1))
        )
        result = srv_metric_r3.aux_differential_srv_transform(tangent_vec, curve_a)

        times = gs.linspace(0.0, 1.0, n_curves)
        path_of_curves = curve_a + gs.einsum("i,jk->ijk", times, tangent_vec)
        srv_path = srv_metric_r3.srv_transform(path_of_curves)
        expected = n_curves * (srv_path[1] - srv_path[0])
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)

    @geomstats.tests.np_and_autograd_only
    def test_aux_differential_srv_transform_inverse(
        self, dim, n_sampling_points, curve_a
    ):
        """Test inverse of differential of square root velocity transform.
        Check that it is the inverse of aux_differential_srv_transform.
        """
        tangent_vec = gs.transpose(
            gs.tile(gs.linspace(0.0, 1.0, n_sampling_points), (dim, 1))
        )
        srv_metric_r3 = SRVMetric(r3)
        d_srv = srv_metric_r3.aux_differential_srv_transform(tangent_vec, curve_a)
        result = srv_metric_r3.aux_differential_srv_transform_inverse(d_srv, curve_a)
        expected = tangent_vec
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)

    def test_aux_differential_srv_transform_vectorization(
        self, dim, n_sampling_points, curve_a, curve_b
    ):
        """Test differential of square root velocity transform.
        Check vectorization.
        """
        dim = 3
        curves = gs.stack((curve_a, curve_b))
        tangent_vecs = gs.random.rand(2, n_sampling_points, dim)
        srv_metric_r3 = SRVMetric(r3)
        result = srv_metric_r3.aux_differential_srv_transform(tangent_vecs, curves)

        res_a = srv_metric_r3.aux_differential_srv_transform(tangent_vecs[0], curve_a)
        res_b = srv_metric_r3.aux_differential_srv_transform(tangent_vecs[1], curve_b)
        expected = gs.stack([res_a, res_b])
        self.assertAllClose(result, expected)

    def test_srv_inner_product_elastic(self, dim, n_sampling_points):
        """Test inner product of SRVMetric.
        Check that the pullback metric gives an elastic metric
        with parameters a=1, b=1/2.
        """
        tangent_vec_a = gs.random.rand(n_sampling_points, dim)
        tangent_vec_b = gs.random.rand(n_sampling_points, dim)
        r3 = Euclidean(dim)
        srv_metric_r3 = SRVMetric(r3)
        result = srv_metric_r3.inner_product(tangent_vec_a, tangent_vec_b, curve_a)

        d_vec_a = (n_sampling_points - 1) * (
            tangent_vec_a[1:, :] - tangent_vec_a[:-1, :]
        )
        d_vec_b = (n_sampling_points - 1) * (
            tangent_vec_b[1:, :] - tangent_vec_b[:-1, :]
        )
        velocity_vec = (n_sampling_points - 1) * (curve_a[1:, :] - curve_a[:-1, :])
        velocity_norm = r3.metric.norm(velocity_vec)
        unit_velocity_vec = gs.einsum("ij,i->ij", velocity_vec, 1 / velocity_norm)
        a_param = 1
        b_param = 1 / 2
        integrand = (
            a_param**2 * gs.sum(d_vec_a * d_vec_b, axis=1)
            - (a_param**2 - b_param**2)
            * gs.sum(d_vec_a * unit_velocity_vec, axis=1)
            * gs.sum(d_vec_b * unit_velocity_vec, axis=1)
        ) / velocity_norm
        expected = gs.sum(integrand) / n_sampling_points
        self.assertAllClose(result, expected)

    def test_srv_inner_product_and_dist(self, dim, curve_a, curve_b):
        """Test that norm of log and dist coincide
        for curves with same / different starting points, and for
        the translation invariant / non invariant SRV metric.
        """
        r3 = Euclidean(dim=dim)
        curve_b_transl = curve_b + gs.array([1.0, 0.0, 0.0])
        curve_b = [curve_b, curve_b_transl]
        translation_invariant = [True, False]
        for curve in curve_b:
            for param in translation_invariant:
                srv_metric = SRVMetric(ambient_manifold=r3, translation_invariant=param)
                log = srv_metric.log(point=curve, base_point=curve_a)
                result = srv_metric.norm(vector=log, base_point=curve_a)
                expected = srv_metric.dist(curve_a, curve)
                self.assertAllClose(result, expected)

    def test_srv_inner_product_vectorization(
        self, dim, n_sampling_points, curve_a, curve_b
    ):
        """Test inner product of SRVMetric.
        Check vectorization.
        """
        curves = gs.stack((curve_a, curve_b))
        tangent_vecs_1 = gs.random.rand(2, n_sampling_points, dim)
        tangent_vecs_2 = gs.random.rand(2, n_sampling_points, dim)
        srv_metric_r3 = SRVMetric(r3)
        result = srv_metric_r3.inner_product(tangent_vecs_1, tangent_vecs_2, curves)

        res_a = srv_metric_r3.inner_product(
            tangent_vecs_1[0], tangent_vecs_2[0], curve_a
        )
        res_b = srv_metric_r3.inner_product(
            tangent_vecs_1[1], tangent_vecs_2[1], curve_b
        )
        expected = gs.stack((res_a, res_b))
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_split_horizontal_vertical(self, times, n_discretized_curves):
        """Test split horizontal vertical.
        Check that horizontal and vertical parts of any tangent
        vector are othogonal with respect to the SRVMetric inner
        product, and check vectorization.
        """
        srv_metric_r3 = SRVMetric(r3)
        quotient_srv_metric_r3 = DiscreteCurves(
            ambient_manifold=r3
        ).quotient_square_root_velocity_metric
        geod = srv_metric_r3.geodesic(initial_curve=curve_a, end_curve=curve_b)
        geod = geod(times)
        tangent_vec = n_discretized_curves * (geod[1, :, :] - geod[0, :, :])
        (
            tangent_vec_hor,
            tangent_vec_ver,
            _,
        ) = quotient_srv_metric_r3.split_horizontal_vertical(tangent_vec, curve_a)
        result = srv_metric_r3.inner_product(tangent_vec_hor, tangent_vec_ver, curve_a)
        expected = 0.0
        self.assertAllClose(result, expected, atol=1e-4)

        tangent_vecs = n_discretized_curves * (geod[1:] - geod[:-1])
        _, _, result = quotient_srv_metric_r3.split_horizontal_vertical(
            tangent_vecs, geod[:-1]
        )
        expected = []
        for i in range(n_discretized_curves - 1):
            _, _, res = quotient_srv_metric_r3.split_horizontal_vertical(
                tangent_vecs[i], geod[i]
            )
            expected.append(res)
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)

    def test_space_derivative(
        self, dim, n_points, n_discretized_curves, n_sampling_points
    ):
        """Test space derivative.
        Check result on an example and vectorization.
        """
        n_points = 3
        dim = 3
        srv_metric_r3 = SRVMetric(Euclidean(dim))
        curve = gs.random.rand(n_points, dim)
        result = srv_metric_r3.space_derivative(curve)
        delta = 1 / n_points
        d_curve_1 = (curve[1] - curve[0]) / delta
        d_curve_2 = (curve[2] - curve[0]) / (2 * delta)
        d_curve_3 = (curve[2] - curve[1]) / delta
        expected = gs.squeeze(
            gs.vstack(
                (
                    gs.to_ndarray(d_curve_1, 2),
                    gs.to_ndarray(d_curve_2, 2),
                    gs.to_ndarray(d_curve_3, 2),
                )
            )
        )
        self.assertAllClose(result, expected)

        path_of_curves = gs.random.rand(n_discretized_curves, n_sampling_points, dim)
        result = srv_metric_r3.space_derivative(path_of_curves)
        expected = []
        for i in range(n_discretized_curves):
            expected.append(srv_metric_r3.space_derivative(path_of_curves[i]))
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)


class TestClosedDiscreteCurves(ManifoldTestCase, metaclass=Parametrizer):
    # closed discrete curves doesn't have random point
    space = ClosedDiscreteCurves
    skip_test_projection_belongs = True
    skip_test_random_point_belongs = True
    skip_test_random_tangent_vec_is_tangent = True
    skip_test_to_tangent_is_tangent = True

    class DiscreteCurvesTestData(_ManifoldTestData):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = Euclidean(dim=3)
        space_args_list = [(r2,), (r3,)]
        shape_list = [(10, 2), (10, 3)]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(self.s2,), (self.r2,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_points_list
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                DiscreteCurves,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                DiscreteCurves, self.space_args_list, self.n_vecs_list
            )

        def projection_closed_curves_test_data(self):
            cells, _, _ = data_utils.load_cells()
            curves = [cell[:-10] for cell in cells[:5]]
            ambient_manifold = Euclidean(dim=2)
            smoke_data = []
            for curve in curves:
                smoke_data += [dict(ambient_manifold=ambient_manifold, curve=curve)]

            return self.generate_tests(smoke_data)

    testing_data = DiscreteCurvesTestData()

    @geomstats.tests.np_and_autograd_only
    def test_projection_closed_curves(self, ambient_manifold, curve):
        planar_closed_curve = ClosedDiscreteCurves(ambient_manifold)
        proj = planar_closed_curve.project(curve)
        expected = proj
        result = planar_closed_curve.project(proj)
        self.assertAllClose(result, expected)

        result = proj[-1, :]
        expected = proj[0, :]
        self.assertAllClose(result, expected, rtol=10 * gs.rtol)


class ClosedSRVMetric(RiemannianMetricTestCase, metaclass=Parametrizer):

    metric = connection = ClosedSRVMetric
    skip_test_exp_shape = True
    skip_test_log_shape = True
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_exp_belongs = True
    skip_test_squared_dist_is_symmetric = True
    skip_test_log_is_tangent = True
    skip_test_dist_is_positive = True
    skip_test_dist_is_symmetric = True
    skip_test_squared_dist_is_symmetric = True
    skip_test_squared_dist_is_positive = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_inner_product_is_symmetric = True
    skip_test_dist_is_norm_of_log = True
    skip_test_exp_after_log = True
    skip_test_log_after_exp = True
    skip_test_exp_ladder_parallel_transport = True

    class ClosedSRVMetricTestData(_RiemannianMetricTestData):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = Euclidean(dim=3)
        ambient_manifolds_list = [r2, r3]
        metric_args_list = [
            (ambient_manifolds,) for ambient_manifolds in ambient_manifolds_list
        ]
        shape_list = [(10, 2), (10, 3)]
        space_list = [
            ClosedDiscreteCurves(ambient_manifolds)
            for ambient_manifolds in ambient_manifolds_list
        ]
        n_points_list = random.sample(range(2, 5), 2)
        n_tangent_vecs_list = random.sample(range(2, 5), 2)
        n_points_a_list = [1, 2]
        n_points_b_list = [1, 2]
        batch_size_list = random.sample(range(2, 5), 2)
        alpha_list = [1] * 2
        n_rungs_list = [1] * 2
        scheme_list = ["pole"] * 2

        def exp_shape_test_data(self):
            return self._exp_shape_test_data(
                self.metric_args_list, self.space_list, self.shape_list
            )

        def log_shape_test_data(self):
            return self._log_shape_test_data(self.metric_args_list, self.space_list)

        def squared_dist_is_symmetric_test_data(self):
            return self._squared_dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
                atol=gs.atol * 1000,
            )

        def exp_belongs_test_data(self):
            return self._exp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                belongs_atol=gs.atol * 1000,
            )

        def log_is_tangent_test_data(self):
            return self._log_is_tangent_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                is_tangent_atol=gs.atol * 1000,
            )

        def geodesic_ivp_belongs_test_data(self):
            return self._geodesic_ivp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def geodesic_bvp_belongs_test_data(self):
            return self._geodesic_bvp_belongs_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                belongs_atol=gs.atol * 1000,
            )

        def exp_after_log_test_data(self):
            return self._exp_after_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def log_after_exp_test_data(self):
            return self._log_after_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_ladder_parallel_transport_test_data(self):
            return self._exp_ladder_parallel_transport_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_rungs_list,
                self.alpha_list,
                self.scheme_list,
            )

        def exp_geodesic_ivp_test_data(self):
            return self._exp_geodesic_ivp_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                self.n_points_list,
                rtol=gs.rtol * 100000,
                atol=gs.atol * 100000,
            )

        def parallel_transport_ivp_is_isometry_test_data(self):
            return self._parallel_transport_ivp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def parallel_transport_bvp_is_isometry_test_data(self):
            return self._parallel_transport_bvp_is_isometry_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
                is_tangent_atol=gs.atol * 1000,
                atol=gs.atol * 1000,
            )

        def dist_is_symmetric_test_data(self):
            return self._dist_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_positive_test_data(self):
            return self._dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def squared_dist_is_positive_test_data(self):
            return self._squared_dist_is_positive_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_is_norm_of_log_test_data(self):
            return self._dist_is_norm_of_log_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_a_list,
                self.n_points_b_list,
            )

        def dist_point_to_itself_is_zero_test_data(self):
            return self._dist_point_to_itself_is_zero_test_data(
                self.metric_args_list, self.space_list, self.n_points_list
            )

        def inner_product_is_symmetric_test_data(self):
            return self._inner_product_is_symmetric_test_data(
                self.metric_args_list,
                self.space_list,
                self.shape_list,
                self.n_tangent_vecs_list,
            )

    testing_data = ClosedSRVMetricTestData()


class TestElasticMetric(TestCase, metaclass=Parametrizer):
    class ElasticMetricTestData(TestData):
        def f_transform_test_data(self):
            smoke_data = [
                dict(
                    a=1.0,
                    b=0.5,
                    curve_a_projected=gs.stack((curve_a[:, 0], curve_a[:, 2]), axis=-1),
                )
            ]
            return self.generate_tests(smoke_data)

        def f_transform_and_inverse_test_data(self):
            cells, _, _ = data_utils.load_cells()
            curve = cells[0]
            smoke_data = [dict(a=1.0, b=0.5, curve=curve)]
            return self.generate_tests(smoke_data)

        def elastic_dist_test_data(self):
            cells, _, _ = data_utils.load_cells()
            curve_1, curve_2 = cells[0][:10], cells[1][:10]
            smoke_data = [dict(a=1.0, b=0.5, curve_1=curve_1, curve_2=curve_2)]
            return self.generate_tests(smoke_data)

        def elastic_and_srv_dist_test_data(self):
            smoke_data = [dict(a=1.0, b=0.5, curve_a=curve_a, curve_b=curve_b)]
            return self.generate_tests(smoke_data)

        def cartesian_to_polar_and_inverse_test_data(self):
            cells, _, _ = data_utils.load_cells()
            curve = cells[0]
            smoke_data = [dict(a=1.0, b=1.0, curve=curve)]
            return self.generate_tests(smoke_data)

    testing_data = ElasticMetricTestData()

    def test_f_transform(self, a, b, curve_a_projected):
        """Test that the f transform coincides with the SRVF.
        With the parameters: a=1, b=1/2.
        """
        elastic_metric = ElasticMetric(a, b)
        curves_r2 = DiscreteCurves(ambient_manifold=r2)
        curve_a_projected = gs.stack((curve_a[:, 0], curve_a[:, 2]), axis=-1)

        result = elastic_metric.f_transform(curve_a_projected)
        expected = gs.squeeze(
            curves_r2.square_root_velocity_metric.srv_transform(curve_a_projected)
        )
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_f_transform_and_inverse(self, a, b, curve):
        """Test that the inverse is right."""
        elastic_metric = ElasticMetric(a, b)
        f = elastic_metric.f_transform(curve)
        f_inverse = elastic_metric.f_transform_inverse(f, curve[0])

        result = f.shape
        expected = (curve.shape[0] - 1, 2)
        self.assertAllClose(result, expected)

        result = f_inverse
        expected = curve
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_elastic_dist(self, a, b, curve_1, curve_2):
        """Test shape and positivity."""
        elastic_metric = ElasticMetric(a, b)
        dist = elastic_metric.dist(curve_1, curve_2)
        result = dist.shape
        expected = ()
        self.assertAllClose(result, expected)

        result = dist > 0
        self.assertTrue(result)

    @geomstats.tests.np_autograd_and_torch_only
    def test_elastic_and_srv_dist(self, a, b, curve_a, curve_b):
        """Test that SRV dist and elastic dist coincide.
        For a=1 and b=1/2.
        """
        elastic_metric = ElasticMetric(a=1.0, b=0.5)
        curves_r2 = DiscreteCurves(ambient_manifold=r2)
        curve_a_projected = gs.stack((curve_a[:, 0], curve_a[:, 2]), axis=-1)
        curve_b_projected = gs.stack((curve_b[:, 0], curve_b[:, 2]), axis=-1)
        result = elastic_metric.dist(curve_a_projected, curve_b_projected)
        expected = curves_r2.square_root_velocity_metric.dist(
            curve_a_projected, curve_b_projected
        )
        self.assertAllClose(result, expected)

    def test_cartesian_to_polar_and_inverse(
        self,
        a,
        b,
        curve,
    ):
        """Test that going back to cartesian works."""
        elastic_metric = ElasticMetric(a, b)
        norms, args = elastic_metric.cartesian_to_polar(curve)

        result = elastic_metric.polar_to_cartesian(norms, args)
        expected = curve
        self.assertAllClose(result, expected, rtol=10000 * gs.rtol)


class TestQuotientSRVMetric(TestCase, metaclass=Parametrizer):
    class QuotientSRVMetricTestData(TestData):
        def horizontal_geodesic_test_data(self):
            smoke_data = [
                dict(n_sampling_points=n_sampling_points, curve_a=curve_a, n_times=20)
            ]
            return self.generate_tests(smoke_data)

        def quotient_dist_test_data(self):
            smoke_data = [
                dict(
                    sampling_times=sampling_times,
                    curve_fun_a=curve_fun_a,
                    curve_a=curve_a,
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = QuotientSRVMetricTestData()

    @geomstats.tests.np_autograd_and_torch_only
    def test_horizontal_geodesic(self, n_sampling_points, curve_a, n_times):
        """Test horizontal geodesic.
        Check that the time derivative of the geodesic is
        horizontal at all time.
        """
        curve_b = gs.transpose(
            gs.stack(
                (
                    gs.zeros(n_sampling_points),
                    gs.zeros(n_sampling_points),
                    gs.linspace(1.0, 0.5, n_sampling_points),
                )
            )
        )
        quotient_srv_metric_r3 = DiscreteCurves(
            ambient_manifold=r3
        ).quotient_square_root_velocity_metric
        horizontal_geod_fun = quotient_srv_metric_r3.horizontal_geodesic(
            curve_a, curve_b
        )
        times = gs.linspace(0.0, 1.0, n_times)
        horizontal_geod = horizontal_geod_fun(times)
        velocity_vec = n_times * (horizontal_geod[1:] - horizontal_geod[:-1])
        _, _, vertical_norms = quotient_srv_metric_r3.split_horizontal_vertical(
            velocity_vec, horizontal_geod[:-1]
        )
        result = gs.sum(vertical_norms**2, axis=1) ** (1 / 2)
        expected = gs.zeros(n_times - 1)
        self.assertAllClose(result, expected, atol=1e-3)

    @geomstats.tests.np_autograd_and_torch_only
    def test_quotient_dist(self, sampling_times, curve_fun_a, curve_a):
        """Test quotient distance.
        Check that the quotient distance is the same as the distance
        between the end points of the horizontal geodesic.
        """
        curve_a_resampled = curve_fun_a(sampling_times**2)
        curve_b = gs.transpose(
            gs.stack(
                (
                    gs.zeros(n_sampling_points),
                    gs.zeros(n_sampling_points),
                    gs.linspace(1.0, 0.5, n_sampling_points),
                )
            )
        )
        quotient_srv_metric_r3 = DiscreteCurves(
            ambient_manifold=r3
        ).quotient_square_root_velocity_metric
        result = quotient_srv_metric_r3.dist(curve_a_resampled, curve_b)
        expected = quotient_srv_metric_r3.dist(curve_a, curve_b)
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)
