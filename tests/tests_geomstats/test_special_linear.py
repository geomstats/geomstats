"""Unit tests for the Special Linear group."""

import warnings

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.special_linear import SpecialLinear, SpecialLinearLieAlgebra


class TestSpecialLinear(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)
        self.n = 3
        self.n_samples = 2
        self.group = SpecialLinear(n=self.n)
        self.algebra = SpecialLinearLieAlgebra(n=self.n)

        warnings.simplefilter("ignore", category=ImportWarning)

    def test_belongs(self):
        matrix = gs.eye(self.n)
        result = self.group.belongs(matrix)
        self.assertTrue(result)

        matrix = 2 * gs.eye(self.n)
        result = self.group.belongs(matrix)
        self.assertFalse(result)

        matrix = gs.eye(self.n - 1)
        result = self.group.belongs(matrix)
        self.assertFalse(result)

    def test_belongs_vectorization(self):
        mats = gs.array([gs.eye(3), 2 * gs.eye(3)])
        result = self.group.belongs(mats)
        expected = gs.array([True, False])
        self.assertAllClose(result, expected)

    def test_random_and_belongs(self):
        matrices = self.group.random_point(n_samples=4)
        result = self.group.belongs(matrices)
        self.assertTrue(gs.all(result))

    def test_projection_and_belongs(self):
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.group, shape)
        self.assertTrue(gs.all(result))

    def test_belongs_algebra(self):
        matrix = gs.array([[2., 0., 0.], [0., -1., 0.], [0., 0., -1]])
        result = self.algebra.belongs(matrix)
        self.assertTrue(result)

        matrix = gs.array([[2., 0., 0.], [0., 2., 0.], [0., 0., -1]])
        result = self.algebra.belongs(matrix)
        self.assertFalse(result)

        matrix = gs.array([[2., 0.], [0., -2.]])
        result = self.algebra.belongs(matrix)
        self.assertFalse(result)

    def test_random_and_belongs_algebra(self):
        matrices = self.algebra.random_point(n_samples=4)
        result = self.algebra.belongs(matrices)
        self.assertTrue(gs.all(result))

    def test_projection_and_belongs_algebra(self):
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.algebra, shape)
        self.assertTrue(gs.all(result))

    def test_matrix_and_basis_representation_algebra(self):
        matrix = self.algebra.random_point()
        basis_repr = self.algebra.basis_representation(matrix)
        matrix_repr = self.algebra.matrix_representation(basis_repr)
        self.assertAllClose(matrix, matrix_repr)
