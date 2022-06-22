import random

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.discrete_curves import ClosedDiscreteCurves, DiscreteCurves
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from tests.data_generation import TestData, _ManifoldTestData, _RiemannianMetricTestData

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


class DiscreteCurvesTestData(_ManifoldTestData):
    space_args_list = [(r2,), (r3,)]
    shape_list = [(10, 2), (10, 3)]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    def random_point_belongs_test_data(self):
        smoke_space_args_list = [(s2,), (r2,)]
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


class L2CurvesMetricTestData(_RiemannianMetricTestData):

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

    def triangle_inequality_of_dist_test_data(self):
        return self._triangle_inequality_of_dist_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def l2_metric_geodesic_test_data(self):
        smoke_data = [
            dict(
                ambient_manfold=s2,
                curve_a=curve_a,
                curve_b=curve_b,
                times=times,
                n_sampling_points=n_sampling_points,
            )
        ]
        return self.generate_tests(smoke_data)


class SRVMetricTestData(_RiemannianMetricTestData):
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

    def triangle_inequality_of_dist_test_data(self):
        return self._triangle_inequality_of_dist_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def srv_transform_and_srv_transform_inverse_test_data(self):
        smoke_data = [dict(rtol=gs.rtol, atol=gs.atol)]
        return self.generate_tests(smoke_data)

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
        smoke_data = [dict(dim=3, n_sampling_points=n_sampling_points, curve_a=curve_a)]
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
        smoke_data = [dict(dim=3, n_sampling_points=n_sampling_points, curve_a=curve_a)]
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
        smoke_data = [
            dict(
                times=times,
                n_discretized_curves=n_discretized_curves,
                curve_a=curve_a,
                curve_b=curve_b,
            )
        ]
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

    def srv_metric_pointwise_inner_products_test_data(self):
        smoke_data = [
            dict(
                times=times,
                curve_a=curve_a,
                curve_b=curve_b,
                curve_c=curve_c,
                n_discretized_curves=n_discretized_curves,
                n_sampling_points=n_sampling_points,
            )
        ]
        return self.generate_tests(smoke_data)

    def srv_transform_and_inverse_test_data(self):
        smoke_data = [dict(times=times, curve_a=curve_a, curve_b=curve_b)]
        return self.generate_tests(smoke_data)


class ElasticMetricTestData(TestData):
    a_b_list = [(1, 1)]

    def cartesian_to_polar_and_polar_to_cartesian_test_data(self):
        smoke_data = [
            dict(a=a, b=b, rtol=10 * gs.rtol, atol=10 * gs.atol)
            for a, b in self.a_b_list
        ]
        return self.generate_tests(smoke_data)

    def cartesian_to_polar_and_polar_to_cartesian_vectorization_test_data(self):
        smoke_data = [
            dict(a=a, b=b, rtol=10 * gs.rtol, atol=10 * gs.atol)
            for a, b in self.a_b_list
        ]
        return self.generate_tests(smoke_data)

    def f_transform_and_srv_transform_test_data(self):
        smoke_data = [
            dict(
                curve=gs.stack([curve_a[:, 0], curve_a[:, 2]], axis=-1),
                rtol=gs.rtol,
                atol=gs.atol,
            )
        ]
        return self.generate_tests(smoke_data)

    def f_transform_inverse_and_srv_transform_inverse_test_data(self):
        smoke_data = [
            dict(
                curve=gs.stack([curve_a[:, 0], curve_a[:, 2]], axis=-1),
                rtol=gs.rtol,
                atol=gs.atol,
            )
        ]
        return self.generate_tests(smoke_data)

    def f_transform_and_srv_transform_vectorization_test_data(self):
        smoke_data = [
            dict(
                rtol=gs.rtol,
                atol=gs.atol,
            )
        ]
        return self.generate_tests(smoke_data)

    def f_transform_and_inverse_test_data(self):
        # cells, _, _ = data_utils.load_cells()
        smoke_data = [dict(a=a, b=b) for a, b in self.a_b_list]
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
                n_sampling_points=n_sampling_points,
            )
        ]
        return self.generate_tests(smoke_data)


class ClosedDiscreteCurvesTestData(_ManifoldTestData):
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
            ClosedDiscreteCurves,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
        )

    def random_tangent_vec_is_tangent_test_data(self):
        return self._random_tangent_vec_is_tangent_test_data(
            ClosedDiscreteCurves, self.space_args_list, self.n_vecs_list
        )

    def projection_closed_curves_test_data(self):
        cells, _, _ = data_utils.load_cells()
        curves = [cell[:-10] for cell in cells[:5]]
        ambient_manifold = Euclidean(dim=2)
        smoke_data = []
        for curve in curves:
            smoke_data += [dict(ambient_manifold=ambient_manifold, curve=curve)]

        return self.generate_tests(smoke_data)
