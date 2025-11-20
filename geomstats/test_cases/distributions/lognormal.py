import pytest

import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class LogNormalTestCase(TestCase):
    @pytest.mark.random
    def test_sample_belongs(self, n_samples, atol):
        samples = self.distribution.sample(n_samples)

        res = self.distribution.space.belongs(samples, atol=atol)
        expected = gs.ones(n_samples, dtype=bool)
        self.assertAllEqual(res, expected)
