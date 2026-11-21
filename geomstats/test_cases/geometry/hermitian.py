import geomstats.backend as gs
from geomstats.test_cases.geometry.base import ComplexVectorSpaceTestCase
from geomstats.test_cases.geometry.mixins import GroupExpTestCaseMixins


class HermitianTestCase(GroupExpTestCaseMixins, ComplexVectorSpaceTestCase):
    def test_exp_random(self, tangent_vec, base_point, atol=gs.atol):
        """Check group exponential is the addition.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Point.
        atol : float
            Absolute tolerance.
        """
        self.test_exp(tangent_vec, base_point, tangent_vec + base_point, atol=atol)

    def test_identity_belongs(self, atol=gs.atol):
        """Check identity belongs to the space.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        self.test_belongs(self.space.identity, gs.array(True), atol=atol)
