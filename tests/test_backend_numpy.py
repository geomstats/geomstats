"""
Unit tests for numpy backend.
"""

import warnings

import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup


@geomstats.tests.np_only
class TestBackendNumpy(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

        self.so3_group = SpecialOrthogonalGroup(n=3)
        self.n_samples = 2

    def test_logm(self):
        point = gs.array([[2., 0., 0.],
                          [0., 3., 0.],
                          [0., 0., 4.]])
        result = gs.linalg.logm(point)
        expected = gs.array([[0.693147180, 0., 0.],
                             [0., 1.098612288, 0.],
                             [0., 0., 1.38629436]])

        self.assertAllClose(result, expected)

    def test_expm_and_logm(self):
        point = gs.array([[2., 0., 0.],
                          [0., 3., 0.],
                          [0., 0., 4.]])
        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertAllClose(result, expected)

    def test_expm_vectorization(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])

        expected = gs.array([[[7.38905609, 0., 0.],
                              [0., 20.0855369, 0.],
                              [0., 0., 54.5981500]],
                             [[2.718281828, 0., 0.],
                              [0., 148.413159, 0.],
                              [0., 0., 403.42879349]]])

        result = gs.linalg.expm(point)

        self.assertAllClose(result, expected)

    def test_logm_vectorization_diagonal(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])

        expected = gs.array([[[0.693147180, 0., 0.],
                              [0., 1.09861228866, 0.],
                              [0., 0., 1.38629436]],
                             [[0., 0., 0.],
                              [0., 1.609437912, 0.],
                              [0., 0., 1.79175946]]])

        result = gs.linalg.logm(point)

        self.assertAllClose(result, expected)

    def test_expm_and_logm_vectorization_random_rotation(self):
        point = self.so3_group.random_uniform(self.n_samples)
        point = self.so3_group.matrix_from_rotation_vector(point)

        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertAllClose(result, expected)

    def test_expm_and_logm_vectorization(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])
        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertAllClose(result, expected)

    def test_powerm_diagonal(self):
        power = .5
        point = gs.array([[1., 0., 0.],
                          [0., 4., 0.],
                          [0., 0., 9.]])
        result = gs.linalg.powerm(point, power)
        expected = gs.array([[1., 0., 0.],
                             [0., 2., 0.],
                             [0., 0., 3.]])

        self.assertAllClose(result, expected)

    def test_powerm(self):
        power = 2.4
        point = gs.array([[1., 0., 0.],
                          [0., 2.5, 1.5],
                          [0., 1.5, 2.5]])
        result = gs.linalg.powerm(point, power)
        result = gs.linalg.powerm(result, 1/power)
        expected = point

        self.assertAllClose(result, expected)

    def test_powerm_vectorized(self):
        power = 2.4
        points = gs.array([[[1., 0., 0.],
                            [0., 4., 0.],
                            [0., 0., 9.]],
                           [[1., 0., 0.],
                            [0., 2.5, 1.5],
                            [0., 1.5, 2.5]]])
        result = gs.linalg.powerm(points, power)
        result = gs.linalg.powerm(result, 1/power)
        expected = points

        self.assertAllClose(result, expected)
