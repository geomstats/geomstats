import pytest

import geomstats.backend as gs

from ...test_geometry.data.base import VectorSpaceOpenSetTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class ExponentialDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()

    @pytest.mark.smoke
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([5.0]), expected=True),
            dict(point=gs.array([-2.0]), expected=False),
            dict(point=gs.array([[1.0], [-1.0]]), expected=gs.array([True, False])),
            dict(point=gs.array([[0.1], [10]]), expected=gs.array([True, True])),
            dict(point=gs.array([[-2.1], [-1.0]]), expected=gs.array([False, False])),
            dict(
                point=gs.ones((2, 1, 1)),
                expected=gs.ones((2, 1), dtype=bool),
            ),
        ]
        return data


class ExponentialMetricTestData(RiemannianMetricTestData):
    @pytest.mark.smoke
    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[1], [0.5], [10]]),
                point_b=gs.array([[2], [3.5], [70]]),
                expected=gs.array([0.48045301, 3.78656631, 3.78656631]),
            ),
            dict(
                point_a=gs.array([0.1]),
                point_b=gs.array([0.99]),
                expected=gs.array(5.255715612697455),
            ),
            dict(
                point_a=gs.array([0.1]),
                point_b=gs.array([0.2]),
                expected=gs.array(0.48045301),
            ),
        ]
        return data

    @pytest.mark.smoke
    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=gs.array([0.5]),
                expected=gs.array([[4.0]]),
            ),
            dict(
                base_point=gs.array([[0.5], [0.2]]),
                expected=gs.array([[[4.0]], [[25.0]]]),
            ),
        ]
        return data
