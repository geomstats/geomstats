import geomstats.backend as gs
from geomstats.test_cases.geometry.base import VectorSpaceTestCase
from geomstats.test_cases.geometry.mixins import GroupExpTestCaseMixins


class HermitianTestCase(GroupExpTestCaseMixins, VectorSpaceTestCase):
    def test_identity_belongs(self, atol=gs.atol):
        """Check identity belongs to the space.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        self.test_belongs(self.space.identity, gs.array(True), atol=atol)
