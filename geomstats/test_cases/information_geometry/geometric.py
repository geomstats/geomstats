import geomstats.backend as gs
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class GeometricDistributionsTestCase(InformationManifoldMixinTestCase, OpenSetTestCase):
    def _check_sample_belongs_to_support(self, sample, atol):
        self.assertTrue(gs.all(sample >= 0.0))
