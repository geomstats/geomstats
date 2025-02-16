import pytest

import geomstats.backend as gs
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.geometry.stiefel import Stiefel
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class StiefelTestCase(LevelSetTestCase):
    def test_to_grassmannian(self, point, expected, atol):
        projected = self.space.to_grassmannian(point)
        self.assertAllClose(projected, expected, atol=atol)

    @pytest.mark.random
    def test_to_grassmannian_belongs_to_grassmannian(self, n_points, atol):
        point = self.space.random_point(n_points)
        grass_point = self.space.to_grassmannian(point)

        grassmannian = Grassmannian(self.space.n, self.space.p, equip=False)
        res = grassmannian.belongs(grass_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)

        self.assertAllEqual(res, expected)


class StiefelStaticMethodsTestCase(TestCase):
    def test_to_grassmannian(self, point, expected, atol):
        projected = self.Space.to_grassmannian(point)
        self.assertAllClose(projected, expected, atol=atol)


class StiefelCanonicalMetricTestCase(RiemannianMetricTestCase):
    def test_retraction(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.retraction(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_lifting(self, point, base_point, expected, atol):
        res = self.space.metric.lifting(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_lifting_vec(self, n_reps, atol):
        while True:
            point = self.data_generator.random_point()
            base_point = self.data_generator.random_point()
            try:
                expected = self.space.metric.lifting(point, base_point)
                break
            except ValueError:
                # algorithm does not work for several cases
                pass

        vec_data = generate_vectorization_data(
            data=[
                dict(point=point, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_lifting_is_tangent(self, n_points, atol):
        while True:
            point = self.data_generator.random_point(n_points)
            base_point = self.data_generator.random_point(n_points)
            try:
                tangent_vec = self.space.metric.lifting(point, base_point)
                break
            except ValueError:
                # algorithm does not work for several cases
                pass

        res = self.space.is_tangent(tangent_vec, base_point, atol=atol)
        self.assertAllEqual(res, gs.ones(n_points, dtype=bool))

    @pytest.mark.random
    def test_lifting_after_retraction(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        point = self.space.metric.retraction(tangent_vec, base_point)
        tangent_vec_ = self.space.metric.lifting(point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    @pytest.mark.random
    def test_retraction_after_lifting(self, n_points, atol):
        while True:
            point = self.data_generator.random_point(n_points)
            base_point = self.data_generator.random_point(n_points)
            try:
                tangent_vec = self.space.metric.lifting(point, base_point)
                break
            except ValueError:
                # algorithm does not work for several cases
                pass

        point_ = self.space.metric.retraction(tangent_vec, base_point)
        self.assertAllClose(point_, point, atol=atol)

    @pytest.mark.random
    def test_two_sheets_error(self, n_points):
        if self.space.n != self.space.p:
            raise ValueError("Test only defined for n=p.")

        point = self.space.random_point(n_points)
        base_point = self.space.random_point(n_points)

        det_point = gs.linalg.det(point)
        det_base = gs.linalg.det(base_point)
        if n_points > 1:
            for index in range(n_points):
                if (det_base[index] * det_point[index]) > 0.0:
                    point[index][0, :] *= -1
        else:
            if (det_base * det_point) > 0.0:
                point[0, :] *= -1.0

        with pytest.raises(ValueError):
            self.space.metric.log(point, base_point)


class StiefelConnectednessTestCase(TestCase):
    def test_is_connected(self, point, expected):
        n, p = point
        connectedness = Stiefel(n=n, p=p, equip=False).is_connected
        expected_connected = " " if expected else " not "
        msg = f"St({n}, {p}) is{expected_connected}connected."
        self.assertTrue(connectedness & expected, msg=msg)
