from polpo.testing.data import CaseData
from polpo.testing.geometric import GeometricCaseData

import geomstats.backend as gs

from .manifold import ManifoldTestData
from .mixins import ProjectionMixinsTestData


class NFoldManifoldTestData(ProjectionMixinsTestData, ManifoldTestData):
    pass


class NFoldManifoldSOTestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.stack([gs.eye(3) + 1.0, gs.eye(3)])[None],
                expected=gs.array(False),
            ),
            dict(
                point=gs.stack([gs.eye(3), gs.eye(3)]),
                expected=gs.array(True),
            ),
        ]
        return data


class NFoldMetricScalesTestData(GeometricCaseData):
    def inner_product_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )
