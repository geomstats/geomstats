import pytest
from polpo.testing.data import CaseData

import geomstats.backend as gs

from .diffeo import DiffeoTestData


class ScaledReciprocalDiffeoTestData(DiffeoTestData):
    @pytest.mark.smoke
    def diffeomorphism_test_data(self):
        data = [
            dict(base_point=gs.array([1.0, 1.0]), expected=gs.array([1.0, 1.0])),
            dict(base_point=gs.array([1.0, 2.0]), expected=gs.array([1.0, 0.5])),
        ]
        return data

    @pytest.mark.smoke
    def inverse_test_data(self):
        data = [
            dict(image_point=gs.array([1.0, 1.0]), expected=gs.array([1.0, 1.0])),
            dict(image_point=gs.array([1.0, 2.0]), expected=gs.array([1.0, 0.5])),
        ]
        return data

    @pytest.mark.smoke
    def tangent_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([2.0, 1.0]),
                base_point=gs.array([1.0, 2.0]),
                expected=gs.array([2.0, 0.75]),
            ),
            dict(
                tangent_vec=gs.array([1.0, 1.0]),
                base_point=gs.array([1.0, 1.0]),
                expected=gs.array([1.0, 0]),
            ),
        ]
        return data

    @pytest.mark.smoke
    def inverse_tangent_test_data(self):
        data = [
            dict(
                image_tangent_vec=gs.array([2.0, 1.0]),
                image_point=gs.array([1.0, 2.0]),
                expected=gs.array([2.0, 0.75]),
            ),
            dict(
                image_tangent_vec=gs.array([1.0, 1.0]),
                image_point=gs.array([1.0, 1.0]),
                expected=gs.array([1.0, 0]),
            ),
        ]
        return data


class PositiveOrthant2TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([1.0, 2.0]), expected=gs.array(True)),
            dict(point=gs.array([1.0, -2.0]), expected=gs.array(False)),
            dict(
                point=gs.array([[1.0, 2.0], [-1.0, 2.0]]),
                expected=gs.array([True, False]),
            ),
        ]
        return data

    def projection_test_data(self):
        data = [
            dict(point=gs.array([1.0, 2.0]), expected=gs.array([1.0, 2.0])),
            dict(point=gs.array([-1.0, 2.0]), expected=gs.array([gs.atol, 2.0])),
        ]
        return data

