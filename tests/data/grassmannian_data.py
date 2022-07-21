import random

import geomstats.backend as gs
from geomstats.geometry.grassmannian import Grassmannian, GrassmannianCanonicalMetric
from geomstats.geometry.matrices import Matrices
from tests.data_generation import _LevelSetTestData, _RiemannianMetricTestData

p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
p_yz = gs.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
p_xz = gs.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

r_y = gs.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
pi_2 = gs.pi / 2
pi_4 = gs.pi / 4


class GrassmannianTestData(_LevelSetTestData):
    n_list = random.sample(range(3, 6), 2)
    k_list = [random.sample(range(2, n), 1)[0] for n in n_list]
    space_args_list = list(zip(n_list, k_list))
    shape_list = [(n, n) for n in n_list]
    n_vecs_list = random.sample(range(1, 5), 2)
    n_points_list = random.sample(range(1, 5), 2)

    space = Grassmannian

    def belongs_test_data(self):
        smoke_data = [
            dict(n=3, p=2, point=p_xy, expected=True),
            dict(n=3, p=2, point=gs.array([p_yz, p_xz]), expected=[True, True]),
        ]
        return self.generate_tests(smoke_data)


class GrassmannianCanonicalMetricTestData(_RiemannianMetricTestData):
    n_list = random.sample(range(3, 5), 2)
    k_list = [random.sample(range(2, n), 1)[0] for n in n_list]
    metric_args_list = list(zip(n_list, k_list))
    shape_list = [(n, n) for n in n_list]
    space_list = [Grassmannian(n, p) for n, p in metric_args_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = GrassmannianCanonicalMetric

    def exp_test_data(self):
        smoke_data = [
            dict(
                n=3,
                p=2,
                tangent_vec=Matrices.bracket(pi_2 * r_y, gs.array([p_xy, p_yz])),
                base_point=gs.array([p_xy, p_yz]),
                expected=gs.array([p_yz, p_xy]),
            ),
            dict(
                n=3,
                p=2,
                tangent_vec=Matrices.bracket(
                    pi_2 * gs.array([r_y, r_z]), gs.array([p_xy, p_yz])
                ),
                base_point=gs.array([p_xy, p_yz]),
                expected=gs.array([p_yz, p_xz]),
            ),
        ]
        return self.generate_tests(smoke_data)
