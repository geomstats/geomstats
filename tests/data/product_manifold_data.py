import random

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.product_manifold import (
    NFoldManifold,
    NFoldMetric,
    ProductManifold,
)
from geomstats.geometry.product_riemannian_metric import ProductRiemannianMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data_generation import _ManifoldTestData, _RiemannianMetricTestData

smoke_manifolds_1 = [Hypersphere(dim=2), Hyperboloid(dim=2)]
smoke_metrics_1 = [Hypersphere(dim=2).metric, Hyperboloid(dim=2).metric]

smoke_manifolds_2 = [Euclidean(3), Minkowski(3)]
smoke_metrics_2 = [Euclidean(3).metric, Minkowski(3).metric]


class ProductManifoldTestData(_ManifoldTestData):
    n_list = random.sample(range(2, 4), 2)
    default_point_list = ["vector", "matrix"]
    manifolds_list = [[Hypersphere(dim=n), Hyperboloid(dim=n)] for n in n_list]
    space_args_list = [
        (manifold, None, default_point)
        for manifold, default_point in zip(manifolds_list, default_point_list)
    ]
    shape_list = [
        (2, n + 1) if default_point == "matrix" else (2 * (n + 1),)
        for n, default_point in zip(n_list, default_point_list)
    ]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = ProductManifold

    def dimension_test_data(self):
        smoke_data = [
            dict(
                manifolds=smoke_manifolds_1,
                default_point_type="vector",
                expected=4,
            ),
            dict(
                manifolds=smoke_manifolds_1,
                default_point_type="matrix",
                expected=4,
            ),
        ]
        return self.generate_tests(smoke_data)

    def regularize_test_data(self):
        smoke_data = [
            dict(
                manifolds=smoke_manifolds_1,
                default_point_type="vector",
                point=self.Space(
                    smoke_manifolds_1, default_point_type="vector"
                ).random_point(5),
            ),
            dict(
                manifolds=smoke_manifolds_1,
                default_point_type="matrix",
                point=self.Space(
                    smoke_manifolds_1, default_point_type="matrix"
                ).random_point(5),
            ),
        ]
        return self.generate_tests(smoke_data)


class ProductRiemannianMetricTestData(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 3), 1)
    default_point_list = ["vector", "matrix"]
    manifolds_list = [[Hypersphere(dim=n), Hyperboloid(dim=n)] for n in n_list]
    metrics_list = [
        [Hypersphere(dim=n).metric, Hyperboloid(dim=n).metric] for n in n_list
    ]
    metric_args_list = list(zip(metrics_list, default_point_list))
    shape_list = [
        (n + 1, n + 1) if default_point == "matrix" else (2 * (n + 1),)
        for n, default_point in zip(n_list, default_point_list)
    ]
    space_list = [
        ProductManifold(manifolds, None, default_point_type)
        for manifolds, default_point_type in zip(manifolds_list, default_point_list)
    ]
    n_points_list = random.sample(range(2, 5), 1)
    n_tangent_vecs_list = random.sample(range(2, 5), 1)
    n_points_a_list = random.sample(range(2, 5), 1)
    n_points_b_list = [1]
    alpha_list = [1] * 1
    n_rungs_list = [1] * 1
    scheme_list = ["pole"] * 1

    Metric = ProductRiemannianMetric

    def inner_product_matrix_test_data(self):
        smoke_data = [
            dict(
                manifolds=smoke_metrics_2,
                default_point_type="vector",
                point=ProductManifold(
                    smoke_manifolds_1, default_point_type="vector").random_point(5),
                base_point=ProductManifold(
                    smoke_manifolds_1, default_point_type="vector").random_point(5),
            ),
            dict(
                manifolds=smoke_metrics_2,
                default_point_type="matrix",
                point=ProductManifold(
                    smoke_manifolds_2, default_point_type="matrix").random_point(5),
                base_point=ProductManifold(
                    smoke_manifolds_2, default_point_type="matrix").random_point(5),
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_matrix_vector_test_data(self):
        random_data = [
            dict(default_point_type="matrix"),
            dict(default_point_type="vector"),
        ]
        return self.generate_tests([], random_data)

    def dist_exp_after_log_norm_test_data(self):
        smoke_data = [
            dict(
                manifolds=smoke_manifolds_1,
                default_point_type="vector",
                n_samples=10,
                einsum_str="..., ...j->...j",
                expected=gs.ones(10),
            ),
            dict(
                manifolds=smoke_manifolds_1,
                default_point_type="matrix",
                n_samples=10,
                einsum_str="..., ...jl->...jl",
                expected=gs.ones(
                    10,
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)


class NFoldManifoldTestData(_ManifoldTestData):
    n_list = random.sample(range(2, 4), 2)
    base_list = [SpecialOrthogonal(n) for n in n_list]
    power_list = random.sample(range(2, 4), 2)
    space_args_list = list(zip(base_list, power_list))
    shape_list = [(power, n, n) for n, power in zip(n_list, power_list)]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = NFoldManifold

    tolerances = {
        "projection_belongs": {"atol": 1e-8},
    }

    def belongs_test_data(self):
        smoke_data = [
            dict(
                base=SpecialOrthogonal(3),
                power=2,
                point=gs.stack([gs.eye(3) + 1.0, gs.eye(3)])[None],
                expected=gs.array(False),
            ),
            dict(
                base=SpecialOrthogonal(3),
                power=2,
                point=gs.array([gs.eye(3), gs.eye(3)]),
                expected=gs.array(True),
            ),
        ]
        return self.generate_tests(smoke_data)

    def shape_test_data(self):
        smoke_data = [dict(base=SpecialOrthogonal(3), power=2, expected=(2, 3, 3))]
        return self.generate_tests(smoke_data)


class NFoldMetricTestData(_RiemannianMetricTestData):

    n_list = random.sample(range(3, 5), 2)
    power_list = random.sample(range(2, 5), 2)
    base_list = [SpecialOrthogonal(n) for n in n_list]
    metric_args_list = [
        (base.metric, power) for base, power in zip(base_list, power_list)
    ]
    shape_list = [(power, n, n) for n, power in zip(n_list, power_list)]
    space_list = [
        NFoldManifold(base, power) for base, power in zip(base_list, power_list)
    ]
    n_points_list = random.sample(range(2, 5), 2)
    n_tangent_vecs_list = random.sample(range(2, 5), 2)
    n_points_a_list = random.sample(range(2, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = NFoldMetric

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)

    def inner_product_shape_test_data(self):
        space = NFoldManifold(SpecialOrthogonal(3), 2)
        n_samples = 4
        point = gs.stack([gs.eye(3)] * space.n_copies * n_samples)
        point = gs.reshape(point, (n_samples, *space.shape))
        tangent_vec = space.to_tangent(gs.zeros((n_samples, *space.shape)), point)
        smoke_data = [
            dict(space=space, n_samples=4, point=point, tangent_vec=tangent_vec)
        ]
        return self.generate_tests(smoke_data)
