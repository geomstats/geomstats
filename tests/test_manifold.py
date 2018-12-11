"""
Unit tests for manifolds.
"""

from geomstats.manifold import Manifold
import geomstats.backend as gs
import geomstats.tests


class TestManifoldMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.dimension = 4
        self.manifold = Manifold(self.dimension)

    def test_dimension(self):
        result = self.manifold.dimension
        expected = self.dimension
        with self.session():
            self.assertAllClose(result, expected)

    def test_belongs(self):
        point = gs.array([1., 2., 3.])
        self.assertRaises(NotImplementedError,
                          lambda: self.manifold.belongs(point))

    def test_regularize(self):
        point = gs.array([1., 2., 3.])
        result = self.manifold.regularize(point)
        expected = point
        with self.session():
            self.assertAllClose(result, expected)


if __name__ == '__main__':
    geomstats.test.main()
