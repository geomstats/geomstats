import pytest

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.manifold import _ManifoldTestCaseMixins


class ComplexManifoldTestCase(_ManifoldTestCaseMixins, TestCase):
    @pytest.mark.type
    def test_random_point_is_complex(self, n_points):
        point = self.data_generator.random_point(n_points)

        self.assertTrue(gs.is_complex(point))

    @pytest.mark.random
    def test_random_point_imaginary_nonzero(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        res = gs.imag(gs.abs(point))
        self.assertAllClose(res, gs.zeros_like(point), atol=atol)

    @pytest.mark.type
    def test_random_tangent_vec_is_complex(self, n_points):
        point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(point)

        self.assertTrue(gs.is_complex(tangent_vec))

    @pytest.mark.random
    def test_random_tangent_vec_imaginary_nonzero(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(point)

        res = gs.imag(gs.abs(tangent_vec))
        self.assertAllClose(res, gs.zeros_like(tangent_vec), atol=atol)
