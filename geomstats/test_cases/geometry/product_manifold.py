import geomstats.backend as gs
from geomstats.test_cases.geometry.manifold import ManifoldTestCase


class ProductManifoldTestCase(ManifoldTestCase):
    def test_embed_to_product_after_project_from_product(self, point, atol=gs.atol):
        """Check embedding the factors of a point recovers it.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the product manifold.
        atol : float
            Absolute tolerance.
        """
        factor_point = self.space.project_from_product(point)
        point_ = self.space.embed_to_product(factor_point)

        self.assertAllClose(point_, point, atol=atol)
