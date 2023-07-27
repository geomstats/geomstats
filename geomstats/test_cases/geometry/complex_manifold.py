import pytest

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.manifold import _ManifoldTestCaseMixins


class ComplexManifoldTestCase(_ManifoldTestCaseMixins, TestCase):
    @pytest.mark.random
    @pytest.mark.type
    def test_random_point_is_complex(self, n_points):
        point = self.data_generator.random_point(n_points)

        self.assertTrue(gs.is_complex(point))

    @pytest.mark.random
    def test_random_point_imaginary_nonzero(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        res = gs.imag(gs.abs(point))
        self.assertAllClose(res, gs.zeros_like(point), atol=atol)
