import random

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.discrete_curves import (
    ClosedDiscreteCurves,
    DiscreteCurves,
    L2CurvesMetric,
    SRVMetric,
)
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

    space = DiscreteCurves


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

    Metric = L2CurvesMetric

    def l2_metric_geodesic_test_data(self):
        smoke_data = [
            dict(
                ambient_manifold=s2,
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

    Metric = SRVMetric

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

    space = ClosedDiscreteCurves

    def projection_closed_curves_test_data(self):
        cells, _, _ = data_utils.load_cells()
        curves = [cell[:-10] for cell in cells[:5]]
        ambient_manifold = Euclidean(dim=2)
        smoke_data = []
        for curve in curves:
            smoke_data += [dict(ambient_manifold=ambient_manifold, curve=curve)]

        return self.generate_tests(smoke_data)
