# Import the tests module

import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.heisenberg import heisenberg


class TestMyManifold(geomstats.tests.TestCase):
    # Use the setUp method to define variables that stay constant
    # during all tests. For example, here we test the
    # 4-dimensional manifold of the class MyManifold.
    def setUp(self):
        self.dimension = 4
        self.another_parameter = 3
        self.manifold = MyManifold(
            dim=self.dimension, another_parameter=3)

    # Each method:
    # - needs to start with `test_`
    # - represents a unit-test, i.e. tests one and only one method
    #  or attribute of the class MyManifold.

    # The method test_dimension tests the `dim` attribute.
    def test_dimension(self):
        result = self.manifold.dim
        expected = self.dimension
        # Each test ends with the following syntax, comparing
        # the result with the expected result, using self.assertAllClose
        self.assertAllClose(result, expected)

    # The method test_belongs tests the `belongs` method.
    def test_belongs(self):
        # Arrays are defined using geomstats backend through the prefix `gs.`.
        # This allows the code to be tested simultaneously in numpy,
        # pytorch and tensorflow. `gs.` is the equivalent of numpy's `np.` and
        # most of numpy's functions are available with `gs.`.
        point = gs.array([1., 2., 3.])
        result = self.manifold.belongs(point)
        expected = False
        self.assertAllClose(result, expected) 
