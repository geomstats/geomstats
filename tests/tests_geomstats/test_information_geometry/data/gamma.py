import math

import pytest

import geomstats.backend as gs

from ...test_geometry.data.base import VectorSpaceOpenSetTestData
from ...test_geometry.data.riemannian_metric import CoordinateRiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class GammaDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()

    @pytest.mark.smoke
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.1, -1.0]), expected=False),
            dict(point=gs.array([0.1, 1.0]), expected=True),
            dict(point=gs.array([-1.0, 0.3]), expected=False),
            dict(point=gs.array([0.1, 5]), expected=True),
        ]
        return data

    @pytest.mark.smoke
    def maximum_likelihood_fit_test_data(self):
        smoke_data = [
            dict(
                data=gs.array([1, 2, 3, 4]),
                expected=gs.array([4.26542805, 2.5]),
            ),
            dict(
                data=[[1, 2, 3, 4, 5], [1, 2, 3, 4]],
                expected=gs.array([[3.70164381, 3.0], [4.26542805, 2.5]]),
            ),
            dict(
                data=[gs.array([1, 2, 3, 4, 5]), gs.array([1, 2, 3, 4])],
                expected=gs.array([[3.70164381, 3.0], [4.26542805, 2.5]]),
            ),
            dict(
                data=[[1, 2, 3, 4]],
                expected=gs.array([4.26542805, 2.5]),
            ),
            dict(
                data=gs.array([1, 2, 3, 4]),
                expected=gs.array([4.26542805, 2.5]),
            ),
            dict(
                data=[gs.array([1, 2, 3, 4])],
                expected=gs.array([4.26542805, 2.5]),
            ),
            dict(
                data=[[1, 2, 3, 4, 5], gs.array([1, 2, 3, 4])],
                expected=gs.array([[3.70164381, 3.0], [4.26542805, 2.5]]),
            ),
        ]
        return smoke_data


class ShapeRateGammaMetricTestData(CoordinateRiemannianMetricTestData):
    @pytest.mark.smoke
    def metric_matrix_test_data(self):
        trigamma_1 = math.pi**2 / 6
        data = [
            dict(
                base_point=gs.array([2.0, 1.0]),
                expected=gs.array([[trigamma_1 - 1, -1.0], [-1.0, 2.0]]),
            ),
            dict(
                base_point=gs.array([[2.0, 1.0], [1.0, 2.0]]),
                expected=gs.array(
                    [
                        [[trigamma_1 - 1, -1.0], [-1.0, 2.0]],
                        [[trigamma_1, -0.5], [-0.5, 0.25]],
                    ]
                ),
            ),
        ]
        return data
