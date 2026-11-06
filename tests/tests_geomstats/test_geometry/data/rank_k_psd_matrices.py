from polpo.testing.data import CaseData

import geomstats.backend as gs

from .manifold import ManifoldTestData
from .mixins import ProjectionMixinsTestData


class RankKPSDMatricesTestData(ProjectionMixinsTestData, ManifoldTestData):
    pass


class Rank2PSD3MatricesTestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array(
                    [
                        [0.8369314, -0.7342977, 1.0402943],
                        [0.04035992, -0.7218659, 1.0794858],
                        [0.9032698, -0.73601735, -0.36105633],
                    ]
                ),
                expected=False,
            ),
            dict(
                point=gs.array([[1.0, 1.0, 0], [1.0, 4.0, 0], [0, 0, 0]]),
                expected=True,
            ),
        ]
        return data


class Rank2PSD2BuresWassersteinMetricTestData(CaseData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[4.0, 0.0], [0.0, 4.0]]),
            )
        ]
        return data

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[4.0, 0.0], [0.0, 4.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[2.0, 0.0], [0.0, 2.0]]),
            )
        ]
        return data

    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                point_b=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                expected=gs.array(2 + 4 - (2 * 2 * 2**0.5)),
            )
        ]
        return data


class Rank3PSD3BuresWassersteinMetricTestData(CaseData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 2.0, 4.0], [2.0, 3.0, 8.0], [4.0, 8.0, 5.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.5, 1.5]]
                ),
                expected=gs.array(4.0),
            )
        ]
        return data
