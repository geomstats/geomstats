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
