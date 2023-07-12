import math
import random

import geomstats.backend as gs
from geomstats.geometry.siegel import Siegel, SiegelMetric
from tests.data_generation import (
    _ComplexOpenSetTestData,
    _ComplexRiemannianMetricTestData,
)

LN_3 = math.log(3.0)
EXP_4 = math.exp(4.0)


class SiegelTestData(_ComplexOpenSetTestData):
    smoke_space_args_list = [(2,), (3,), (4,), (5,)]
    smoke_n_points_list = [1, 2, 1, 2]
    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = [(n, n) for n in n_list]
    n_vecs_list = random.sample(range(1, 10), 2)

    Space = Siegel

    def belongs_test_data(self):
        smoke_data = [
            dict(
                n=2, mat=gs.array([[0.2 + 0.2j, -0.1j], [-0.3j, -0.1]]), expected=True
            ),
            dict(n=2, mat=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=False),
            dict(
                n=3,
                mat=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=False,
            ),
            dict(
                n=2,
                mat=gs.array(
                    [
                        [[0.1 + 0.2j, 0.1 + 0.2j], [0.0 - 0.2j, 0.5j]],
                        [[1.0 - 1j, -1.0 + 0.2j], [0.4j, 1.0 + 1.2j]],
                    ]
                ),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def projection_test_data(self):
        smoke_data = [
            dict(
                n=2,
                mat=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0 - gs.atol, 0.0], [0.0, 1.0 - gs.atol]]),
            ),
            dict(
                n=2,
                mat=gs.array([[1j, 0.0], [0.0, 1j]]),
                expected=gs.array(
                    [[(1.0 - gs.atol) * 1j, 0.0], [0.0, (1.0 - gs.atol) * 1j]]
                ),
            ),
            dict(
                n=2,
                mat=gs.array([[-1.0, 0.0], [0.0, -2.0]]),
                expected=gs.array([[-(1 - gs.atol) / 2, 0.0], [0.0, -(1 - gs.atol)]]),
            ),
            dict(
                n=2,
                mat=gs.array([[-1j, 0.0], [0.0, -2j]]),
                expected=gs.array(
                    [[-(1 - gs.atol) / 2 * 1j, 0.0], [0.0, -(1 - gs.atol) * 1j]]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)


class SiegelMetricTestData(_ComplexRiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)

    shape_list = [(n, n) for n in n_list]
    space_list = [Siegel(n, equip=False) for n in n_list]
    connection_args_list = metric_args_list = [{} for _ in shape_list]

    n_points_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = SiegelMetric

    space_2 = Siegel(2, equip=False)
    space_3 = Siegel(3, equip=False)

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                space=self.space_3,
                tangent_vec_a=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
                ),
                base_point=gs.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                expected=6,
            ),
            dict(
                space=self.space_3,
                tangent_vec_a=gs.array([[1j, 0j, 0j], [0j, 1j, 0j], [0j, 0j, 1j]]),
                tangent_vec_b=gs.array([[1j, 0j, 0j], [0j, 2j, 0j], [0j, 0j, 3j]]),
                base_point=gs.array([[0j, 0j, 0j], [0j, 0j, 0j], [0j, 0j, 0j]]),
                expected=6,
            ),
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                space=self.space_2,
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[0.0, 0.0], [0.0, 0.0]]),
                expected=gs.array(
                    [
                        [(EXP_4 - 1) / (EXP_4 + 1), 0.0],
                        [0.0, (EXP_4 - 1) / (EXP_4 + 1)],
                    ]
                ),
            ),
            dict(
                space=self.space_2,
                tangent_vec=gs.array([[2j, 0j], [0j, 2j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array(
                    [
                        [(EXP_4 - 1) / (EXP_4 + 1) * 1j, 0j],
                        [0j, (EXP_4 - 1) / (EXP_4 + 1) * 1j],
                    ]
                ),
            ),
            dict(
                space=self.space_2,
                tangent_vec=gs.array([[0j, 0j], [0j, 0j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array([[0j, 0j], [0j, 0j]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                space=self.space_2,
                point=gs.array([[0.5, 0.0], [0.0, 0.5]]),
                base_point=gs.array([[0.0, 0.0], [0.0, 0.0]]),
                expected=gs.array([[LN_3 / 2, 0.0], [0.0, LN_3 / 2]]),
            ),
            dict(
                space=self.space_2,
                point=gs.array([[0.5j, 0j], [0j, 0.5j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array([[LN_3 / 2 * 1j, 0j], [0j, LN_3 / 2 * 1j]]),
            ),
            dict(
                space=self.space_2,
                point=gs.array([[0j, 0j], [0j, 0j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array([[0j, 0j], [0j, 0j]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def sectional_curvature_test_data(self):
        smoke_data = [
            dict(
                space=self.space_2,
                tangent_vec_a=gs.array([[1.0, 0.0], [0.0, 0.0]]),
                tangent_vec_b=gs.array([[0.0, 0.0], [0.0, 1.0]]),
                base_point=gs.array([[0.0, 0.0], [0.0, 0.0]]),
                expected=0,
            ),
            dict(
                space=self.space_2,
                tangent_vec_a=gs.array([[1j, 0j], [0j, 1j]]),
                tangent_vec_b=gs.array([[-1j, 0j], [0j, -1j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=0,
            ),
            dict(
                space=self.space_2,
                tangent_vec_a=gs.array([[1.0 + 0j, 0j], [0j, 0j]]),
                tangent_vec_b=gs.array([[0j, 1j], [0j, 0j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=-1,
            ),
            dict(
                space=self.space_2,
                tangent_vec_a=gs.array([[2 + 0j, 0j], [0j, 0j]]),
                tangent_vec_b=gs.array([[0j, 2j], [0j, 0j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=-1,
            ),
            dict(
                space=self.space_2,
                tangent_vec_a=gs.array([[0.25 + 0j, 0j], [0j, 0j]]),
                tangent_vec_b=gs.array([[0j, 0.25j], [0j, 0j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=-1,
            ),
            dict(
                space=self.space_3,
                tangent_vec_a=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
                ),
                base_point=gs.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                expected=0,
            ),
            dict(
                space=self.space_3,
                tangent_vec_a=gs.array([[1j, 0j, 0j], [0j, 1j, 0j], [0j, 0j, 1j]]),
                tangent_vec_b=gs.array([[1j, 0j, 0j], [0j, 2j, 0j], [0j, 0j, 3j]]),
                base_point=gs.array([[0j, 0j, 0j], [0j, 0j, 0j], [0j, 0j, 0j]]),
                expected=0,
            ),
            dict(
                space=self.space_3,
                tangent_vec_a=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                tangent_vec_b=gs.array(
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                base_point=gs.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                expected=-1,
            ),
            dict(
                space=self.space_3,
                tangent_vec_a=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                tangent_vec_b=gs.array(
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                base_point=gs.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                expected=-1,
            ),
            dict(
                space=self.space_3,
                tangent_vec_a=gs.array([[1j, 0j, 0j], [0j, 0j, 0j], [0j, 0j, 0j]]),
                tangent_vec_b=gs.array([[0j, 0j, 1j], [0j, 0j, 0j], [0j, 0j, 0j]]),
                base_point=gs.array([[0j, 0j, 0j], [0j, 0j, 0j], [0j, 0j, 0j]]),
                expected=-1,
            ),
        ]
        return self.generate_tests(smoke_data)
