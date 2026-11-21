from polpo.testing.data import CaseData

import geomstats.backend as gs

from .complex_riemannian_metric import ComplexRiemannianMetricTestData
from .mixins import ExpDomainMixinsTestData


class HPDMatrices2TestData(CaseData):
    def belongs_test_data(self):
        smoke_data = [
            dict(point=gs.array([[3.0, -1.0], [-1.0, 3.0]]), expected=True),
            dict(point=gs.array([[3j, -1.0], [-1.0, 3.0]]), expected=False),
            dict(point=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=False),
            dict(
                point=gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]]),
                expected=[True, False],
            ),
        ]
        return smoke_data

    def projection_test_data(self):
        smoke_data = [
            dict(
                point=gs.array([[1.0 + 0.0j, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0 + 0.0j, 0.0], [0.0, 1.0]]),
            ),
            dict(
                point=gs.array([[1.0 + 0.0j, 0.5j], [0.5j, 1.0 + 0.0j]]),
                expected=gs.array([[1.0 + 0.0j, 0.0], [0.0, 1.0]]),
            ),
            dict(
                point=gs.array([[-1.0 + 0.0j, 0.0], [0.0, -2.0]]),
                expected=gs.array([[gs.atol + 0.0j, 0.0], [0.0, gs.atol]]),
            ),
        ]
        return smoke_data


class HPDMatrices3TestData(CaseData):
    def belongs_test_data(self):
        smoke_data = [
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=False,
            ),
            dict(
                point=gs.array(
                    [[3.0 + 0j, 0j, 1j], [0j, 4.0 + 0j, 0j], [-0.5j, 0j, 6.0 + 0j]]
                ),
                expected=False,
            ),
        ]
        return smoke_data


class HPDEuclideanMetricTestData(
    ExpDomainMixinsTestData, ComplexRiemannianMetricTestData
):
    pass
