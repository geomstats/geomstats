"""Unit tests for manifolds."""

import geomstats.tests
from geomstats.geometry.manifold import Manifold


class TestManifold(geomstats.tests.TestCase):

    def test_dimension(self):
        self.assertRaises(
            TypeError, lambda: Manifold(4))
