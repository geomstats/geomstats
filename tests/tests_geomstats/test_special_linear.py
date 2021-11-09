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
        pass

    def test_belongs_vectorization(self):
        pass

    def test_random_and_belongs(self):
        pass

    def test_projection_and_belongs(self):
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.group, shape)
        for res in result:
            self.assertTrue(res)

    def test_belongs_algebra(self):
        pass

    def test_random_and_belongs_algebra(self):
        pass

    def test_projection_and_belongs_algebra(self):
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.algebra, shape)
        for res in result:
            self.assertTrue(res)
