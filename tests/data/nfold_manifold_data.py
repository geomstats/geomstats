import random

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.nfold_manifold import NFoldManifold, NFoldMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.data_generation import _ManifoldTestData, _RiemannianMetricTestData


class NFoldManifoldTestData(_ManifoldTestData):
    base_list = [
        SpecialOrthogonal(2),
        Euclidean(3),
    ]
    power_list = [3, 2]
    shape_list = [(3, 2, 2), (2, 3)]
    scale_list = [[1, 2, 3], [1, 1]]

    space_args_list = list(zip(base_list, power_list))

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

    shape_list = [(power, n, n) for n, power in zip(n_list, power_list)]
    space_list = [
        NFoldManifold(base, power) for base, power in zip(base_list, power_list)
    ]
    metric_args_list = [{} for _ in shape_list]

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
        space = NFoldManifold(SpecialOrthogonal(3), 2, equip=False)
        n_samples = 4
        point = gs.stack([gs.eye(3)] * space.n_copies * n_samples)
        point = gs.reshape(point, (n_samples, *space.shape))
        tangent_vec = space.to_tangent(gs.zeros((n_samples, *space.shape)), point)
        smoke_data = [
            dict(space=space, n_samples=4, point=point, tangent_vec=tangent_vec)
        ]
        return self.generate_tests(smoke_data)

    def inner_product_scales_test_data(self):
        so3 = SpecialOrthogonal(3)
        r4 = Euclidean(4)

        point1 = so3.random_point(n_samples=2)
        vec1 = so3.random_tangent_vec(point1, n_samples=2)
        point2 = r4.random_point(n_samples=3)
        vec2 = r4.random_tangent_vec(point2, n_samples=3)
        random_data = [
            dict(
                space=NFoldManifold(so3, 2),
                scales=gs.array([1.0, 2.0]),
                point=point1,
                tangent_vec=vec1,
            ),
            dict(
                space=NFoldManifold(so3, 2),
                scales=gs.array([1.0, 2.0]),
                point=gs.stack((point1, point1)),
                tangent_vec=gs.stack((vec1, vec1)),
            ),
            dict(
                space=NFoldManifold(r4, n_copies=3),
                scales=gs.array([2.5, 2.0, 1.5]),
                point=point2,
                tangent_vec=vec2,
            ),
            dict(
                space=NFoldManifold(r4, n_copies=3),
                scales=gs.array([2.5, 2.0, 1.5]),
                point=gs.stack((point2, point2)),
                tangent_vec=gs.stack((vec2, vec2)),
            ),
        ]
        return self.generate_tests(random_data)
