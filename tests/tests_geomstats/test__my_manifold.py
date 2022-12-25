"""Template file showing unit tests for MyManifold.

MyManifold is the template manifold defined in:
geomstats/geometry/_my_manifold.py.

For additional guidelines on how to contribute to geomstats, visit:
https://geomstats.github.io/contributing.html#contributing-code-workflow

To run these tests:
- Install packages from geomstats[dev]
- In command line, run:
```nose2 tests.test__my_manifold``` to run all the tests of this file
- In command line, run:
```nose2 tests.test__my_manifold.TestMyManifold.test_dimension```
to run the test `test_dimension` only.

To run these tests using different backends (numpy orpytorch):
- Install packages from geomstats[opt]
In command line, select the backend of interest with:
```export GEOMSTATS_BACKEND=numpy```
 or ```export GEOMSTATS_BACKEND=pytorch```
 and repeat the steps from the previous paragraph.

When you submit a PR, the tests are run with the three backends, except if you
add a decorator such as `@tests.conftest.np_and_autograd_only` or
`@tests.conftest.np_and_autograd_only` etc.
"""

# Import the tests module
import geomstats.backend as gs
import tests.conftest

# Import the manifold to be tested
from geomstats.geometry._my_manifold import MyManifold


class TestMyManifold(tests.conftest.TestCase):
    """Class testing the methods of MyManifold.

    In the class TestMyManifold, each test method:
    - needs to start with `test_`
    - represents a unit-test, i.e. it tests one and only one method
    or attribute of the class MyManifold,
    - ends with the line: `self.assertAllClose(result, expected)`,
    as in the examples below.
    """

    def setup_method(self):
        """Set up unit-tests.

        Use the setUp method to define variables that remain constant
        during all tests. For example, here we test the
        4-dimensional manifold of the class MyManifold.
        """
        self.dimension = 4
        self.another_parameter = 3
        self.manifold = MyManifold(dim=self.dimension, another_parameter=3)

    def test_dimension(self):
        """Test dimension.

        The method test_dimension tests the `dim` attribute.
        """
        result = self.manifold.dim
        expected = self.dimension
        # Each test ends with the following syntax, comparing
        # the result with the expected result, using self.assertAllClose
        self.assertAllClose(result, expected)

    def test_belongs(self):
        """Test belongs.

        The method test_belongs tests the `belongs` method.

        Note that arrays are defined using geomstats backend
        through the prefix `gs.`.
        This allows the code to be tested simultaneously in numpy,
        and pytorch. `gs.` is the equivalent of numpy's `np.` and
        most of numpy's functions are available with `gs.`.
        """
        point = gs.array([1.0, 2.0, 3.0])
        result = self.manifold.belongs(point)
        expected = False
        self.assertAllClose(result, expected)

    def test_belongs_vectorization(self):
        """Test belongs with several input points.

        All functions and methods should work with several input points,
        or vectors.
        """
        point = gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = self.manifold.belongs(point)
        expected = gs.array([False, False])
        self.assertAllClose(result, expected)

    def test_is_tangent(self):
        """Test is_tangent.

        The method test_is_tangent tests the `is_tangent` method.

        Note that arrays are defined using geomstats backend
        through the prefix `gs.`.
        This allows the code to be tested simultaneously in numpy
        and pytorch. `gs.` is the equivalent of numpy's `np.` and
        most of numpy's functions are available with `gs.`.
        """
        vector = gs.array([1.0, 2.0, 3.0, 4.0])
        result = self.manifold.is_tangent(vector)
        expected = True
        self.assertAllClose(result, expected)

    def test_is_tangent_vectorization(self):
        """Test is_tangent with several input vectors

        All functions and methods should work with several input points,
        or vectors.
        """
        vector = gs.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = self.manifold.is_tangent(vector)
        expected = gs.array([True, True])
        self.assertAllClose(result, expected)
