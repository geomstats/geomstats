import pytest
from polpo.testing.data import CaseData

import geomstats.backend as gs

from .base import VectorSpaceTestData
from .lie_group import LieGroupTestData


class HeisenbergVectorsToMatricesSmokeTestData(CaseData):
    def diffeomorphism_test_data(self):
        data = [
            dict(
                base_point=gs.array([1.0, 2.0, 3.0]),
                expected=gs.array([[1.0, 1.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]),
            )
        ]
        return data

    def inverse_test_data(self):
        data = [
            dict(
                image_point=gs.array(
                    [[1.0, 1.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
                ),
                expected=gs.array([1.0, 2.0, 3.0]),
            )
        ]
        return data

    def tangent_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([1.0, 0.0, 0.0]),
                base_point=gs.array([1.0, 2.0, 3.0]),
                expected=gs.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            )
        ]
        return data

    def inverse_tangent_test_data(self):
        data = [
            dict(
                image_tangent_vec=gs.array(
                    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                image_point=gs.array(
                    [[1.0, 1.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
                ),
                expected=gs.array([1.0, 0.0, 0.0]),
            )
        ]
        return data


class HeisenbergVectorsTestData(LieGroupTestData, VectorSpaceTestData):
    @pytest.mark.smoke
    def dim_test_data(self):
        data = [dict(expected=3)]
        return data

    @pytest.mark.smoke
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([1.0, 2.0, 3.0, 4]), expected=False),
            dict(
                point=gs.array([[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]]),
                expected=[False, False],
            ),
        ]
        return data

    @pytest.mark.smoke
    def is_tangent_test_data(self):
        data = [
            dict(
                vector=gs.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
                base_point=None,
                expected=[False, False],
            )
        ]
        return data

    @pytest.mark.smoke
    def jacobian_translation_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, -10.0, 0.2], [-2.0, 100.0, 0.5]]),
                left=True,
                expected=gs.array(
                    [
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 0.5, 1.0]],
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-50.0, -1.0, 1.0]],
                    ]
                ),
            )
        ]
        return data
