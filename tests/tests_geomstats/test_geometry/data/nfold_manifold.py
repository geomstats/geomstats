import geomstats.backend as gs
from geomstats.test.data import TestData

from .manifold import ManifoldTestData
from .mixins import ProjectionMixinsTestData
from .riemannian_metric import RiemannianMetricTestData


class NFoldManifoldTestData(ProjectionMixinsTestData, ManifoldTestData):
    pass


class NFoldMetricTestData(RiemannianMetricTestData):
    pass


class NFoldManifoldSOTestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.stack([gs.eye(3) + 1.0, gs.eye(3)])[None],
                expected=gs.array(False),
            ),
            dict(
                point=gs.array([gs.eye(3), gs.eye(3)]),
                expected=gs.array(True),
            ),
        ]
        return self.generate_tests(data)


class NFoldMetricScalesTestData(TestData):
    def inner_product_test_data(self):
        return self.generate_random_data()
