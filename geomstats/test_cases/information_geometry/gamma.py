import geomstats.backend as gs
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class GammaDistributionsTestCase(
    InformationManifoldMixinTestCase, VectorSpaceOpenSetTestCase
):
    def _check_sample_belongs_to_support(self, sample, atol):
        self.assertTrue(gs.all(sample > 0.0))

    def test_maximum_likelihood_fit(self, data, expected, atol=gs.atol):
        """Check maximum likelihood fit.

        Parameters
        ----------
        data : array-like or list[array-like]
            Samples, or list of samples, to fit.
        expected : array-like, shape=[..., 2]
            Expected fitted parameters.
        atol : float
            Absolute tolerance.
        """
        res = self.space.maximum_likelihood_fit(data)
        self.assertAllClose(res, expected, atol=atol)
