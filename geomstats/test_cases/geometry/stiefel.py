import pytest

import geomstats.backend as gs
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.test.assertions import assert_belongs, assert_is_tangent
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class StiefelTestCase(LevelSetTestCase):
    def test_to_grassmannian(self, point, expected, atol=gs.atol):
        projected = self.space.to_grassmannian(point)
        self.assertAllClose(projected, expected, atol=atol)

    def test_to_grassmannian_belongs_to_grassmannian(self, point, atol=gs.atol):
        assert_belongs(
            Grassmannian(self.space.n, self.space.p, equip=False),
            self.space.to_grassmannian(point),
            atol=atol,
        )


class StiefelStaticMethodsTestCase(TestCase):
    def test_to_grassmannian(self, point, expected, atol=gs.atol):
        projected = self.Space.to_grassmannian(point)
        self.assertAllClose(projected, expected, atol=atol)


class StiefelCanonicalMetricTestCase(RiemannianMetricTestCase):
    def test_retraction(self, tangent_vec, base_point, expected, atol=gs.atol):
        res = self.space.metric.retraction(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_lifting(self, point, base_point, expected, atol=gs.atol):
        res = self.space.metric.lifting(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_lifting_is_tangent(self, point, base_point, atol=gs.atol):
        tangent_vec = self.space.metric.lifting(point, base_point)

        assert_is_tangent(self.space, tangent_vec, base_point, atol=atol)

    def test_lifting_after_retraction(self, tangent_vec, base_point, atol=gs.atol):
        point = self.space.metric.retraction(tangent_vec, base_point)
        tangent_vec_ = self.space.metric.lifting(point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_retraction_after_lifting(self, point, base_point, atol=gs.atol):
        tangent_vec = self.space.metric.lifting(point, base_point)

        point_ = self.space.metric.retraction(tangent_vec, base_point)
        self.assertAllClose(point_, point, atol=atol)

    def test_two_sheets_error(self, point, base_point):
        if self.space.n != self.space.p:
            raise ValueError("Test only defined for n=p.")

        det_point = gs.linalg.det(point)
        det_base = gs.linalg.det(base_point)

        sign = gs.where(det_base * det_point > 0.0, -1.0, 1.0)
        first_row = sign[..., None, None] * point[..., :1, :]
        point = gs.concatenate([first_row, point[..., 1:, :]], axis=-2)

        with pytest.raises(ValueError):
            self.space.metric.log(point, base_point)
