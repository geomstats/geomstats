import math

import geomstats.backend as gs
from geomstats.test.data import TestData

from .hermitian import HermitianMetricTestData

CDTYPE = gs.get_default_cdtype()

SQRT_2 = math.sqrt(2)

MAT9_33 = gs.array([[4, 0, 1j], [0, 3, 0], [-1j, 0, 4]], dtype=CDTYPE)
MAT10_33 = gs.array([[1, 1 + 1j, -3j], [1 - 1j, 0, 4], [3j, 4, -2]], dtype=CDTYPE)
MAT11_33 = gs.array([[1, 1j, 2], [1j, -1j, 2], [4 + 2j, 0, -2]], dtype=CDTYPE)
MAT12_33 = gs.array([[1, -1j, 4 - 2j], [-1j, 1j, 0], [2, 2, -2]], dtype=CDTYPE)
MAT13_33 = gs.array([[1, 0, 3 - 1j], [0, 0, 1], [3 + 1j, 1, -2]], dtype=CDTYPE)
MAT14_33 = gs.array([[4, 0, 0.5j], [0, 3, 0], [-1j, 0, 4]], dtype=CDTYPE)


class ComplexMatrices33TestData(TestData):
    def transconjugate_test_data(self):
        data = [
            dict(
                mat=gs.stack([MAT9_33, MAT11_33]),
                expected=gs.stack([MAT9_33, MAT12_33]),
            ),
        ]
        return self.generate_tests(data)

    def is_hermitian_test_data(self):
        data = [
            dict(
                mat=gs.stack([MAT9_33, MAT10_33, MAT11_33]),
                expected=[True, True, False],
            ),
        ]
        return self.generate_tests(data)

    def is_hpd_test_data(self):
        data = [
            dict(
                mat=gs.stack([MAT9_33, MAT10_33, MAT11_33]),
                expected=[True, False, False],
            ),
        ]
        return self.generate_tests(data)

    def to_hermitian_test_data(self):
        data = [
            dict(
                mat=gs.stack([MAT10_33, MAT11_33]),
                expected=gs.stack([MAT10_33, MAT13_33]),
            ),
        ]
        return self.generate_tests(data)


class ComplexMatricesMetricTestData(HermitianMetricTestData):
    fail_for_not_implemented_errors = False

    skips = (
        "christoffels_vec",
        "cometric_matrix_vec",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
        "metric_matrix_is_spd",
        "metric_matrix_vec",
    )
