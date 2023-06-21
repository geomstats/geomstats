import geomstats.backend as gs
from geomstats.test.data import TestData
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)

from .base import OpenSetTestData


class PoincareHalfSpaceTestData(OpenSetTestData):
    pass


class PoincareHalfSpace2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([1.5, 2.3]), expected=True),
            dict(point=gs.array([[1.5, 2.0], [2.5, -0.3]]), expected=[True, False]),
        ]
        return self.generate_tests(data)


class PoincareHalfSpaceMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False


class PoincareHalfSpaceMetric2TestData(TestData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[1.0, 2.0], [3.0, 4.0]]),
                tangent_vec_b=gs.array([[1.0, 2.0], [3.0, 4.0]]),
                base_point=gs.array([[0.0, 1.0], [0.0, 5.0]]),
                expected=gs.array([5.0, 1.0]),
            )
        ]
        return self.generate_tests(data)

    def exp_test_data(self):
        def _exp(tangent_vec, base_point):
            circle_center = (
                base_point[0] + base_point[1] * tangent_vec[1] / tangent_vec[0]
            )
            circle_radius = gs.sqrt(
                (circle_center - base_point[0]) ** 2 + base_point[1] ** 2
            )

            moebius_d = 1
            moebius_c = 1 / (2 * circle_radius)
            moebius_b = circle_center - circle_radius
            moebius_a = (circle_center + circle_radius) * moebius_c

            point_complex = base_point[0] + 1j * base_point[1]
            tangent_vec_complex = tangent_vec[0] + 1j * tangent_vec[1]

            point_moebius = (
                1j
                * (moebius_d * point_complex - moebius_b)
                / (moebius_c * point_complex - moebius_a)
            )
            tangent_vec_moebius = (
                -1j
                * tangent_vec_complex
                * (1j * moebius_c * point_moebius + moebius_d) ** 2
            )

            end_point_moebius = point_moebius * gs.exp(
                tangent_vec_moebius / point_moebius
            )
            end_point_complex = (moebius_a * 1j * end_point_moebius + moebius_b) / (
                moebius_c * 1j * end_point_moebius + moebius_d
            )
            end_point_expected = gs.hstack(
                [gs.real(end_point_complex), gs.imag(end_point_complex)]
            )
            return end_point_expected

        inputs_to_exp = [(gs.array([2.0, 1.0]), gs.array([1.0, 1.0]))]
        data = []
        for tangent_vec, base_point in inputs_to_exp:
            data.append(
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=_exp(tangent_vec, base_point),
                )
            )
        return self.generate_tests(data)
