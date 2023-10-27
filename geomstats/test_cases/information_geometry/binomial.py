import geomstats.backend as gs
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class BinomialDistributionsTestCase(InformationManifoldMixinTestCase, OpenSetTestCase):
    def _check_sample_belongs_to_support(self, sample, _atol):
        support = set(range(self.space.n_draws + 1))
        empty_set = set({int(val) for val in gs.unique(sample)}).difference(support)
        self.assertTrue(len(empty_set) == 0)
