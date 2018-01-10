"""Unit tests for sphere module."""

import geomstats.sphere as sphere

import numpy as np
import unittest


class TestSphereMethods(unittest.TestCase):
    def test_riemannian_exp_and_log(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # NB: points on the 4-dimensional sphere are 5D vectors of norm 1.
        ref_point = np.array([1., 2., 3., 4., 6.])
        ref_point = ref_point / np.linalg.norm(ref_point)
        point = np.array([0., 5., 6., 2., -1])
        point = point / np.linalg.norm(point)

        riem_log = sphere.riemannian_log(ref_point, point)
        result = sphere.riemannian_exp(ref_point, riem_log)
        expected = point

        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
        unittest.main()
