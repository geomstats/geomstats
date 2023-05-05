import random

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.discrete_curves import (
    ClosedDiscreteCurves,
    DiscreteCurves,
    ElasticMetric,
    L2CurvesMetric,
    SRVMetric,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from tests.data_generation import TestData, _ManifoldTestData, _RiemannianMetricTestData

s2 = Hypersphere(dim=2)
r2 = Euclidean(dim=2)
r3 = s2.embedding_space

point = gs.array([0.0, 0.0, 1.0])
vec_a = gs.array([1.0, 0.0, 0.0])
vec_b = gs.array([0.0, 1.0, 0.0])
vec_c = gs.array([-1.0, 0.0, 0.0])

spherical_curve_fun_a = s2.metric.geodesic(point, initial_tangent_vec=vec_a)
spherical_curve_fun_b = s2.metric.geodesic(point, initial_tangent_vec=vec_b)
spherical_curve_fun_c = s2.metric.geodesic(point, initial_tangent_vec=vec_c)


def curve_fun_a(times):
    return spherical_curve_fun_a(times) - point


def curve_fun_b(times):
    return spherical_curve_fun_b(times) - point


def curve_fun_c(times):
    return spherical_curve_fun_c(times) - point


k_sampling_points = 10
sampling_times = gs.linspace(0.0, 1.0, k_sampling_points)
curve_a = curve_fun_a(sampling_times)
curve_b = curve_fun_b(sampling_times)
curve_c = curve_fun_c(sampling_times)
spherical_curve_a = spherical_curve_fun_a(sampling_times)
spherical_curve_b = spherical_curve_fun_b(sampling_times)
spherical_curve_c = spherical_curve_fun_c(sampling_times)

n_discretized_curves = 5
times = gs.linspace(0.0, 1.0, n_discretized_curves)

l2metric_s2 = L2CurvesMetric(DiscreteCurves(ambient_manifold=s2))
spherical_curves_fun_ab = l2metric_s2.geodesic(spherical_curve_a, spherical_curve_b)
spherical_curves_fun_bc = l2metric_s2.geodesic(spherical_curve_b, spherical_curve_c)
spherical_curves_ab = spherical_curves_fun_ab(times)
spherical_curves_bc = spherical_curves_fun_bc(times)
curves_ab = spherical_curves_ab - point
curves_bc = spherical_curves_bc - point
vec_a = gs.transpose(gs.tile(gs.linspace(0.0, 1.0, k_sampling_points), (3, 1)))
vec_b = gs.transpose(gs.tile(gs.linspace(0.0, 2.0, k_sampling_points), (3, 1)))


class DiscreteCurvesTestData(_ManifoldTestData):
    space_args_list = [(r2,), (r3,)]
    shape_list = [(10, 2), (10, 3)]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = DiscreteCurves


class L2CurvesMetricTestData(_RiemannianMetricTestData):
    ambient_manifolds_list = [r2, r3]

    shape_list = [(10, 2), (10, 3)]
    space_list = [
        DiscreteCurves(ambient_manifold, equip=False)
        for ambient_manifold in ambient_manifolds_list
    ]
    metric_args_list = [{} for _ in shape_list]

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
                space=DiscreteCurves(s2, equip=False),
                curve_a=spherical_curve_a,
                curve_b=spherical_curve_b,
                times=times,
                k_sampling_points=k_sampling_points,
            )
        ]
        return self.generate_tests(smoke_data)


class SRVMetricTestData(_RiemannianMetricTestData):
    ambient_manifolds_list = [r2, r3]

    shape_list = [(10, 2), (10, 3)]
    space_list = [
        DiscreteCurves(ambient_manifold, equip=False)
        for ambient_manifold in ambient_manifolds_list
    ]
    metric_args_list = [{} for _ in shape_list]

    n_points_list = random.sample(range(2, 5), 2)
    n_tangent_vecs_list = random.sample(range(2, 5), 2)
    n_points_a_list = [1, 2]
    n_points_b_list = [1, 2]
    batch_size_list = random.sample(range(2, 5), 2)
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = SRVMetric

    def srv_transform_and_srv_transform_inverse_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(ambient_manifold=r3, equip=False),
                rtol=gs.rtol,
                atol=gs.atol,
            )
        ]
        return self.generate_tests(smoke_data)

    def diffeomorphism_and_inverse_diffeomorphism_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(ambient_manifold=r3, equip=False),
                rtol=gs.rtol,
                atol=gs.atol,
            )
        ]
        return self.generate_tests(smoke_data)

    def tangent_diffeomorphism_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r3, k_sampling_points=2000, equip=False
                ),
                n_curves=2000,
                curve_fun_a=curve_fun_a,
            )
        ]
        return self.generate_tests(smoke_data)

    def inverse_tangent_diffeomorphism_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r3,
                    k_sampling_points=k_sampling_points,
                    equip=False,
                ),
                curve_a=curve_a,
            )
        ]
        return self.generate_tests(smoke_data)

    def tangent_diffeomorphism_vectorization_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r3,
                    k_sampling_points=k_sampling_points,
                    equip=False,
                ),
                curve_a=curve_a,
                curve_b=curve_b,
            )
        ]
        return self.generate_tests(smoke_data)

    def tangent_diffeomorphism_and_inverse_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(ambient_manifold=r3, equip=False),
                curve=curve_a,
                tangent_vec=vec_a,
            )
        ]
        return self.generate_tests(smoke_data)

    def srv_inner_product_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r3,
                    k_sampling_points=k_sampling_points,
                    equip=False,
                ),
                curve=curve_a,
                vec_a=vec_a,
                vec_b=vec_b,
                n_vecs=3,
            )
        ]
        return self.generate_tests(smoke_data)

    def srv_inner_product_elastic_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r3,
                    k_sampling_points=k_sampling_points,
                    equip=False,
                ),
                curve=curve_a,
                vec_a=vec_a,
                vec_b=vec_b,
            )
        ]
        return self.generate_tests(smoke_data)

    def srv_inner_product_and_dist_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r3,
                    k_sampling_points=k_sampling_points,
                    equip=False,
                ),
                curve_a=curve_a,
                curve_b=curve_b,
            )
        ]
        return self.generate_tests(smoke_data)

    def space_derivative_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r3,
                    k_sampling_points=k_sampling_points,
                    equip=False,
                ),
                n_points=3,
                n_discretized_curves=n_discretized_curves,
            )
        ]
        return self.generate_tests(smoke_data)

    def srv_metric_pointwise_inner_products_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r3,
                    k_sampling_points=k_sampling_points,
                    equip=False,
                ),
                curves_ab=curves_ab,
                curves_bc=curves_bc,
                n_discretized_curves=n_discretized_curves,
            )
        ]
        return self.generate_tests(smoke_data)

    def srv_transform_and_inverse_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r3,
                    k_sampling_points=k_sampling_points,
                    equip=False,
                ),
                curves=curves_ab,
            )
        ]
        return self.generate_tests(smoke_data)


class ElasticMetricTestData(_RiemannianMetricTestData):

    n_samples_list = [1, 5]
    a_list = [1, 2]
    b_list = [2, 5]

    ambient_manifolds_list = [r2]
    connection_args_list = metric_args_list = [(1, 1)]
    shape_list = [(10, 2), (10, 3)]
    space_list = [
        DiscreteCurves(ambient_manifold, equip=False)
        for ambient_manifold in ambient_manifolds_list
    ]
    connection_args_list = metric_args_list = [{"a": a, "b": b} for a, b in a_b_list]

    n_points_list = random.sample(range(2, 5), 2)
    n_tangent_vecs_list = random.sample(range(2, 5), 2)
    n_points_a_list = [1, 2]
    n_points_b_list = [1, 2]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = ElasticMetric

    def cartesian_to_polar_and_polar_to_cartesian_test_data(self):
        smoke_data = [
            dict(a=a, b=b, n_samples=n_samples, rtol=10 * gs.rtol, atol=10 * gs.atol)
            for a in self.a_list
            for b in self.b_list
            for n_samples in self.n_samples_list
        ]
        return self.generate_tests(smoke_data)

    def f_transform_and_srv_transform_test_data(self):
        smoke_data = [
            dict(n_samples=n_samples, rtol=gs.rtol, atol=gs.atol)
            for n_samples in self.n_samples_list
        ]
        return self.generate_tests(smoke_data)

    def f_transform_inverse_and_srv_transform_inverse_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r2, start_at_the_origin=True, equip=False
                ),
                curve=gs.stack([curve_a[:, 0], curve_a[:, 2]], axis=-1),
                rtol=10 * gs.rtol,
                atol=10 * gs.atol,
            )
        ]
        return self.generate_tests(smoke_data)

    def f_transform_and_f_transform_inverse_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteCurves(
                    ambient_manifold=r2, start_at_the_origin=True, equip=False
                ),
                curve=gs.stack([curve_a[:, 0], curve_a[:, 2]], axis=-1),
                a=a,
                b=b,
                rtol=10 * gs.rtol,
                atol=10 * gs.atol,
            )
            for a in self.a_list
            for b in self.b_list
        ]
        return self.generate_tests(smoke_data)

    def f_transform_and_diffeomorphism_test_data(self):
        smoke_data = [
            dict(a=a, b=b, n_samples=n_samples, rtol=10 * gs.rtol, atol=10 * gs.atol)
            for a in self.a_list
            for b in self.b_list
            for n_samples in self.n_samples_list
        ]
        return self.generate_tests(smoke_data)

    def f_transform_inverse_and_inverse_diffeomorphism_test_data(self):
        smoke_data = [
            dict(
                curve=gs.stack(
                    [curve_a[:, 0], curve_a[:, 2]],
                    axis=-1,
                ),
                a=a,
                b=b,
                rtol=10 * gs.rtol,
                atol=10 * gs.atol,
            )
            for a in self.a_list
            for b in self.b_list
            for n_samples in self.n_samples_list
        ]
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


class SRVShapeBundleTestData(TestData):
    def horizontal_and_vertical_projections_test_data(self):
        smoke_data = [
            dict(
                times=times,
                n_discretized_curves=n_discretized_curves,
                curve_a=curve_a,
                curve_b=curve_b,
            )
        ]
        return self.generate_tests(smoke_data)

    def horizontal_geodesic_test_data(self):
        smoke_data = [
            dict(
                k_sampling_points=k_sampling_points,
                curve_a=spherical_curve_a,
                n_times=20,
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

    Space = ClosedDiscreteCurves

    def projection_closed_curves_test_data(self):
        cells, _, _ = data_utils.load_cells()
        curves = [cell[:-10] for cell in cells[:5]]
        ambient_manifold = Euclidean(dim=2)
        smoke_data = []
        for curve in curves:
            smoke_data += [dict(ambient_manifold=ambient_manifold, curve=curve)]

        return self.generate_tests(smoke_data)

    def projection_belongs_test_data(self, belongs_atol=gs.atol):
        space_args_list = [(self.r2,)]
        n_points_list = [1, 2]
        random_data = [
            dict(
                space_args=space_args,
                point=gs.random.normal(size=(n_points,) + shape),
                belongs_atol=belongs_atol,
            )
            for space_args, shape, n_points in zip(
                space_args_list, self.shape_list, n_points_list
            )
        ]
        return self.generate_tests([], random_data)


class SRVQuotientMetricTestData(TestData):
    def dist_test_data(self):
        smoke_data = [
            dict(
                sampling_times=sampling_times,
                curve_fun_a=curve_fun_a,
                curve_a=curve_a,
                k_sampling_points=k_sampling_points,
            )
        ]
        return self.generate_tests(smoke_data)
