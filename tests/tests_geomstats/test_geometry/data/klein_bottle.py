import pytest

import geomstats.backend as gs

from .manifold import ManifoldTestData
from .riemannian_metric import RiemannianMetricTestData


class KleinBottleTestData(ManifoldTestData):
    skips = ("not_belongs",)

    def equivalent_test_data(self):
        data = [
            dict(
                point_a=gs.array([0.3, 0.7]),
                point_b=gs.array([2.3, 0.7]),
                expected=gs.array(True),
            ),
            dict(
                point_a=gs.array([0.45 - 2, 0.67]),
                point_b=gs.array([1.45, 1 - 0.67]),
                expected=gs.array(True),
            ),
            dict(
                point_a=gs.array([0.11, 0.12]),
                point_b=gs.array([0.11 - 1, 1 - 0.12]),
                expected=gs.array(True),
            ),
            dict(
                point_a=gs.array([0.1, 0.12]),
                point_b=gs.array([0.1 + 2 + gs.atol / 2, 0.12]),
                expected=gs.array(True),
            ),
            dict(
                point_a=gs.array([0.1, 0.12]),
                point_b=gs.array([0.1 + 2 - gs.atol / 2, 0.12]),
                expected=gs.array(True),
            ),
            dict(
                point_a=gs.array([[0.1, 0.1], [0.5, 0.4]]),
                point_b=gs.array([[1.1, -0.1], [-0.5, 0.4]]),
                expected=gs.array([True, False]),
            ),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def equivalent_vec_test_data(self):
        return self.generate_vec_data()

    def regularize_test_data(self):
        data = [
            dict(expected=gs.array([0.3, 0.7]), point=gs.array([2.3, 0.7])),
            dict(expected=gs.array([0.45, 0.67]), point=gs.array([1.45, 1 - 0.67])),
            dict(expected=gs.array([0.11, 0.12]), point=gs.array([0.11 - 1, 1 - 0.12])),
            dict(
                expected=gs.array([gs.atol / 3 + gs.atol / 2, 0.12]),
                point=gs.array([gs.atol / 3 + 2 + gs.atol / 2, 0.12]),
            ),
            dict(
                expected=gs.array([gs.atol / 3 - gs.atol / 2 + 1, 1 - 0.12]),
                point=gs.array([gs.atol / 3 + 2 - gs.atol / 2, 0.12]),
            ),
            dict(
                expected=gs.array([[0.1, 0.1], [0.5, 0.6], [0.9, 0.4]]),
                point=gs.array([[1.1, -0.1], [-0.5, 0.4], [0.9, 0.4]]),
            ),
            dict(
                expected=gs.array([[0.0, 0.0], [0.0, 0.0]]),
                point=gs.array([[1.0, 1.0], [-1.0, -1.0]]),
            ),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def regularize_correct_domain_test_data(self):
        return self.generate_random_data()

    def to_coords_test_data(self):
        extrinsic_data = [
            dict(
                point=gs.array([0.0, gs.pi / 3]),
                coords_type="extrinsic",
                expected=gs.array(
                    [0.9563500657790063, 0.558935854019387, 1.0292223461899435, 0.0]
                ),
            ),
            dict(
                point=gs.array([gs.pi / 3, 0]),
                coords_type="extrinsic",
                expected=gs.array(
                    [
                        -0.9890273165537457,
                        -0.1477327557127968,
                        0.9563500657790063,
                        0.2922234618994349,
                    ]
                ),
            ),
            dict(
                point=gs.array([gs.pi / 3, gs.pi / 3]),
                coords_type="extrinsic",
                expected=gs.array(
                    [
                        -0.8632832052624392,
                        -0.6940870584701174,
                        0.9842968584799756,
                        0.30076291706788394,
                    ]
                ),
            ),
        ]

        bottle_data = [
            dict(
                point=gs.array([0.0, gs.pi / 3]),
                coords_type="bottle",
                expected=gs.array([-0.3825400263116025, 0.0, 0.11688938475977395]),
            ),
            dict(
                point=gs.array([gs.pi / 3, 0]),
                coords_type="bottle",
                expected=gs.array([-0.5376563769184989, 0.47393340352857033, 0.0]),
            ),
            dict(
                point=gs.array([gs.pi / 3, gs.pi / 3]),
                coords_type="bottle",
                expected=gs.array(
                    [-0.5131810087955432, 0.4681561078948537, 0.1713341081735281]
                ),
            ),
        ]

        bagel_data = [
            dict(
                point=gs.array([0.0, gs.pi / 3]),
                coords_type="bagel",
                expected=gs.array([5.292223461899435, 0.0, 0.558935854019387]),
            ),
            dict(
                point=gs.array([gs.pi / 3, 0]),
                coords_type="bagel",
                expected=gs.array([4.781750328895031, 1.4611173094971743, 0.0]),
            ),
            dict(
                point=gs.array([gs.pi / 3, gs.pi / 3]),
                coords_type="bagel",
                expected=gs.array(
                    [4.584317737096075, 1.4007895722681574, -0.5959738051368074]
                ),
            ),
        ]

        data = extrinsic_data + bottle_data + bagel_data
        return self.generate_tests(data, marks=(pytest.mark.smoke))

    def to_coords_vec_test_data(self):
        data = []
        for coords_type in ["extrinsic", "bottle", "bagel"]:
            for n_reps in self.N_VEC_REPS:
                data.append(dict(n_reps=n_reps, coords_type=coords_type))

        return self.generate_tests(data)


class KleinBottleMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    def dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([0.5, 0.5]),
                point_b=gs.array([0.0, 0.0]),
                expected=gs.array(2**0.5 / 2),
            ),
            dict(
                point_a=gs.array([0.1, 0.12]),
                point_b=gs.array([0.9, 0.8]),
                expected=gs.array((0.2**2 + (0.2 - 0.12) ** 2) ** 0.5),
            ),
            dict(
                point_a=gs.array([0.2, 0.8]),
                point_b=gs.array([0.8, 0.8]),
                expected=gs.array((0.4**2 + 0.4**2) ** 0.5),
            ),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def exp_test_data(self):
        data = [
            dict(
                base_point=gs.array([0.6, 0.3]),
                tangent_vec=gs.array(
                    [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 0.2], [-0.1, 1.1]]
                ),
                expected=gs.array(
                    [[0.6, 0.7], [0.6, 0.3], [0.6, 0.3], [0.6, 0.5], [0.5, 0.4]]
                ),
            )
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def log_test_data(self):
        data = [
            dict(
                base_point=gs.array([0.6, 0.3]),
                point=gs.array([[0.6, 0.7], [0.6, 0.3], [0.6, 0.5]]),
                expected=gs.array([[0.0, 0.4], [0.0, 0.0], [0.0, 0.2]]),
            ),
            dict(
                base_point=gs.array([0.1, 0.12]),
                point=gs.array([0.9, 0.8]),
                expected=gs.array([-0.1 - 0.1, 0.2 - 0.12]),
            ),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))
