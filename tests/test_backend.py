"""
Unit tests for numpy backend.
"""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


class TestBackendNumpy(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

        self.so3_group = SpecialOrthogonalGroup(n=3)
        self.n_samples = 2

    @geomstats.tests.np_only
    def test_logm(self):
        point = gs.array([[2., 0., 0.],
                          [0., 3., 0.],
                          [0., 0., 4.]])
        result = gs.linalg.logm(point)
        expected = gs.array([[0.693147180, 0., 0.],
                             [0., 1.098612288, 0.],
                             [0., 0., 1.38629436]])

        self.assertTrue(gs.allclose(result, expected))

    @geomstats.tests.np_only
    def test_expm_and_logm(self):
        point = gs.array([[2., 0., 0.],
                          [0., 3., 0.],
                          [0., 0., 4.]])
        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertTrue(gs.allclose(result, expected))

    @geomstats.tests.np_only
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

        self.assertTrue(gs.allclose(result, expected))

    @geomstats.tests.np_only
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

        self.assertTrue(gs.allclose(result, expected))

    @geomstats.tests.np_only
    def test_expm_and_logm_vectorization_random_rotation(self):
        point = self.so3_group.random_uniform(self.n_samples)
        point = self.so3_group.matrix_from_rotation_vector(point)

        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertTrue(gs.allclose(result, expected))

    @geomstats.tests.np_only
    def test_expm_and_logm_vectorization(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])
        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertTrue(gs.allclose(result, expected))

    def test_vstack(self):
        with self.session():
            tensor_1 = gs.array([[1., 2., 3.], [4., 5., 6.]])
            tensor_2 = gs.array([[7., 8., 9.]])

            result = gs.vstack([tensor_1, tensor_2])
            expected = gs.array([
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.]])
            self.assertAllClose(result, expected)

    def test_tensor_addition(self):
        with self.session():
            tensor_1 = gs.ones((1, 1))
            tensor_2 = gs.ones((0, 1))

            result = tensor_1 + tensor_2


if __name__ == '__main__':
        geomstats.test.main()
