import random

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.product_manifold import ProductManifold, ProductRiemannianMetric
from geomstats.geometry.siegel import Siegel
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data_generation import _ManifoldTestData, _RiemannianMetricTestData

smoke_manifolds_1 = [Hypersphere(dim=2), Hyperboloid(dim=2)]
smoke_manifolds_2 = [Euclidean(3), Minkowski(3)]
smoke_manifolds_3 = [Siegel(2), Siegel(2), Siegel(2)]


class ProductManifoldTestData(_ManifoldTestData):
    manifolds_list = [
        [Hypersphere(dim=3), Hyperboloid(dim=3)],
        [Hypersphere(dim=3), Hyperboloid(dim=3)],
        [Hypersphere(dim=3), Hyperboloid(dim=4)],
        [Hypersphere(dim=1), Euclidean(dim=1)],
        [SpecialOrthogonal(n=2), SpecialOrthogonal(n=3)],
        [SpecialOrthogonal(n=2), Euclidean(dim=3)],
        [Euclidean(dim=2), Euclidean(dim=1), Euclidean(dim=4)],
        [Siegel(2), Siegel(2), Siegel(2)],
    ]
    default_point_list = ["matrix"] + ["vector"] * 6 + ["other"]
    default_coords_type_list = ["extrinsic"] * 6 + ["intrinsic"] + ["extrinsic"]
    shape_list = [
        (2, 3 + 1),
        (2 * (3 + 1),),
        ((3 + 1) + (4 + 1),),
        (2 + 1,),
        (4 + 6,),
        (4 + 3,),
        (7,),
        (3, 2, 2),
    ]

    if len(manifolds_list) != len(default_point_list) or len(manifolds_list) != len(
        default_coords_type_list
    ):
        raise Exception("One of the lists is incomplete.")

    space_args_list = [
        (manifolds, default_point)
        for manifolds, default_point in zip(manifolds_list, default_point_list)
    ]

    n_points_list = [1] + random.sample(range(2, 5), 2)
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
            dict(
                manifolds=smoke_manifolds_3,
                default_point_type="other",
                expected=12,
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

    def default_coords_type_test_data(self):
        smoke_data = [
            dict(space_args=space_args, expected=default_coords_type)
            for space_args, default_coords_type in zip(
                self.space_args_list, self.default_coords_type_list
            )
        ]

        return self.generate_tests(smoke_data)

    def embed_to_after_project_from_test_data(self):
        random_data = []
        for space_args in self.space_args_list:
            for n_points in [1, 2]:
                random_data.append(dict(space_args=space_args, n_points=n_points))

        return self.generate_tests([], random_data)


class ProductRiemannianMetricTestData(_RiemannianMetricTestData):
    manifolds_list = [
        [Hypersphere(dim=3), Hyperboloid(dim=3)],
        [Hypersphere(dim=3), Hyperboloid(dim=3)],
        [Hypersphere(dim=3), Hyperboloid(dim=4)],
        [Hypersphere(dim=1), Euclidean(dim=1)],
        [SpecialOrthogonal(n=2), SpecialOrthogonal(n=3)],
        [SpecialOrthogonal(n=2), Euclidean(dim=3)],
        [Euclidean(dim=2), Euclidean(dim=1), Euclidean(dim=4)],
        [Siegel(2), Siegel(2), Siegel(2)],
    ]
    default_point_list = ["matrix"] + ["vector"] * 6 + ["other"]
    default_coords_type_list = ["extrinsic"] * 6 + ["intrinsic"] + ["extrinsic"]
    shape_list = [
        (2, 3 + 1),
        (2 * (3 + 1),),
        ((3 + 1) + (4 + 1),),
        (2 + 1,),
        (4 + 6,),
        (4 + 3,),
        (7,),
        (
            3,
            2,
            2,
        ),
    ]

    if len(manifolds_list) != len(default_point_list) or len(manifolds_list) != len(
        default_coords_type_list
    ):
        raise Exception("One of the lists is incomplete.")

    n_list = random.sample(range(2, 3), 1)

    space_list = [
        ProductManifold(manifolds, default_point_type=default_point_type)
        for manifolds, default_point_type in zip(manifolds_list, default_point_list)
    ]
    connection_args_list = metric_args_list = [{} for _ in space_list]

    n_points_list = random.sample(range(2, 5), 1)
    n_tangent_vecs_list = random.sample(range(2, 5), 1)
    n_points_a_list = random.sample(range(2, 5), 1)
    n_points_b_list = [1]
    alpha_list = [1] * 1
    n_rungs_list = [1] * 1
    scheme_list = ["pole"] * 1

    Metric = ProductRiemannianMetric

    tolerances = {
        "dist_point_to_itself_is_zero": {"atol": 1e-6},
    }

    def inner_product_matrix_test_data(self):
        smoke_data = [
            dict(
                space=ProductManifold(
                    smoke_manifolds_2,
                    default_point_type="vector",
                    equip=False,
                ),
                n_points=5,
            ),
            dict(
                space=ProductManifold(
                    smoke_manifolds_2,
                    default_point_type="matrix",
                    equip=False,
                ),
                n_points=5,
            ),
            dict(
                space=ProductManifold(
                    smoke_manifolds_3,
                    default_point_type="other",
                    equip=False,
                ),
                n_points=5,
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
                space=ProductManifold(
                    smoke_manifolds_1,
                    default_point_type="vector",
                    equip=False,
                ),
                n_samples=10,
                einsum_str="..., ...j->...j",
                expected=gs.ones(10),
            ),
            dict(
                space=ProductManifold(
                    smoke_manifolds_1,
                    default_point_type="matrix",
                    equip=False,
                ),
                n_samples=10,
                einsum_str="..., ...jl->...jl",
                expected=gs.ones(10),
            ),
            dict(
                space=ProductManifold(
                    smoke_manifolds_3,
                    default_point_type="other",
                    equip=False,
                ),
                n_samples=10,
                einsum_str="..., ...jkl->...jkl",
                expected=gs.ones(10),
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)

    def exp_shape_test_data(self):
        smoke_data = [dict(space=space) for space in self.space_list]
        return self.generate_tests(smoke_data)

    def exp_vectorization_test_data(self):
        smoke_data = [
            dict(space=space, n_samples=n_samples)
            for space in self.space_list
            for n_samples in self.n_list
        ]
        return self.generate_tests(smoke_data)

    def log_shape_test_data(self):
        smoke_data = [dict(space=space) for space in self.space_list]
        return self.generate_tests(smoke_data)

    def log_vectorization_test_data(self):
        smoke_data = [
            dict(space=space, n_samples=n_samples)
            for space in self.space_list
            for n_samples in self.n_list
        ]
        return self.generate_tests(smoke_data)

    def dist_vectorization_test_data(self):
        smoke_data = [
            dict(space=space, n_samples=n_samples)
            for space in self.space_list
            for n_samples in self.n_list
        ]
        return self.generate_tests(smoke_data)

    def geodesic_test_data(self):
        smoke_data = [dict(space=space) for space in self.space_list]
        return self.generate_tests(smoke_data)

    def geodesic_vectorization_test_data(self):
        smoke_data = [
            dict(space=space, n_samples=n_samples)
            for space in self.space_list
            for n_samples in self.n_list
        ]
        return self.generate_tests(smoke_data)
