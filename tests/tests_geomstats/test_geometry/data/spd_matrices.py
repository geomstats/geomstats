import math

from polpo.testing.data import CaseData

import geomstats.backend as gs

from .matrices import MatricesMetricTestData
from .mixins import ExpDomainMixinsTestData
from .riemannian_metric import RiemannianMetricTestData


class SymMatrixLogSmokeTestData(CaseData):
    def diffeomorphism_test_data(self):
        data = [
            dict(
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[0.0, 0.0], [0.0, 0.0]]),
            )
        ]
        return data

    def tangent_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array(
                    [[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]
                ),
                expected=gs.array(
                    [
                        [1.0, 1.0, 2 * math.log(2.0)],
                        [1.0, 1.0, 2 * math.log(2.0)],
                        [2 * math.log(2.0), 2 * math.log(2.0), 1],
                    ]
                ),
            ),
            dict(
                tangent_vec=gs.array(
                    [
                        [math.exp(1.0), math.exp(1.0), math.sinh(1.0)],
                        [math.exp(1.0), math.exp(1.0), math.sinh(1.0)],
                        [math.sinh(1.0), math.sinh(1.0), 1 / math.exp(1.0)],
                    ]
                ),
                image_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
                ),
                expected=gs.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            ),
        ]
        return data

    def inverse_tangent_test_data(self):
        data = [
            dict(
                image_tangent_vec=gs.array(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
                ),
                image_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
                ),
                expected=gs.array(
                    [
                        [math.exp(1.0), math.exp(1.0), math.sinh(1.0)],
                        [math.exp(1.0), math.exp(1.0), math.sinh(1.0)],
                        [math.sinh(1.0), math.sinh(1.0), 1 / math.exp(1.0)],
                    ]
                ),
            ),
            dict(
                image_tangent_vec=gs.array(
                    [
                        [1.0, 1.0, 2 * math.log(2.0)],
                        [1.0, 1.0, 2 * math.log(2.0)],
                        [2 * math.log(2.0), 2 * math.log(2.0), 1],
                    ]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]
                ),
                expected=gs.array([[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]]),
            ),
        ]
        return data


class MatrixPower05TestData(CaseData):
    def tangent_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]]
                ),
                expected=gs.array(
                    [
                        [1.0, 1 / 3, 1 / 3],
                        [1 / 3, 0.125, 0.125],
                        [1 / 3, 0.125, 0.125],
                    ]
                ),
            )
        ]
        return data

    def inverse_tangent_test_data(self):
        data = [
            dict(
                image_tangent_vec=gs.array(
                    [
                        [1.0, 1 / 3, 1 / 3],
                        [1 / 3, 0.125, 0.125],
                        [1 / 3, 0.125, 0.125],
                    ]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]]
                ),
                expected=gs.array([[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]),
            )
        ]
        return data


class CholeskyMapSmokeTestData(CaseData):
    def diffeomorphism_test_data(self):
        data = [
            dict(
                base_point=gs.array(
                    [[[1.0, 2.0], [2.0, 5.0]], [[1.0, 0.0], [0.0, 1.0]]]
                ),
                expected=gs.array([[[1.0, 0.0], [2.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            ),
            dict(
                base_point=gs.array(
                    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
                ),
                expected=gs.array(
                    [
                        [math.sqrt(2.0), 0.0, 0.0],
                        [0.0, math.sqrt(2.0), 0.0],
                        [0.0, 0.0, math.sqrt(2.0)],
                    ]
                ),
            ),
        ]
        return data

    def inverse_test_data(self):
        data = [
            dict(
                image_point=gs.array([[1.0, 0.0], [2.0, 1.0]]),
                expected=gs.array([[1.0, 2.0], [2.0, 5.0]]),
            ),
            dict(
                image_point=gs.array(
                    [[[2.0, 1.0], [0.0, 1.0]], [[-6.0, 0.0], [5.0, 3.0]]]
                ),
                expected=gs.array(
                    [[[5.0, 1.0], [1.0, 1.0]], [[36.0, -30.0], [-30.0, 34.0]]]
                ),
            ),
        ]
        return data

    def tangent_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[1.0, 1.0], [1.0, 1.0]]),
                base_point=gs.array([[4.0, 2.0], [2.0, 5.0]]),
                expected=gs.array([[1 / 4, 0.0], [3 / 8, 1 / 16]]),
            ),
            dict(
                tangent_vec=gs.array([[1.0, 2.0], [2.0, 5.0]]),
                image_point=gs.array([[1.0, 0.0], [2.0, 2.0]]),
                expected=gs.array([[0.5, 0.0], [1.0, 0.25]]),
            ),
            dict(
                tangent_vec=gs.array(
                    [[[-4.0, 1.0], [1.0, -4.0]], [[0.0, 4.0], [4.0, -8.0]]]
                ),
                image_point=gs.array(
                    [[[2.0, 0.0], [-1.0, 2.0]], [[4.0, 0.0], [-1.0, 2.0]]]
                ),
                expected=gs.array(
                    [[[-1.0, 0.0], [0.0, -1.0]], [[0.0, 0.0], [1.0, -1.5]]]
                ),
            ),
        ]
        return data

    def inverse_tangent_test_data(self):
        data = [
            dict(
                image_tangent_vec=gs.array([[-1.0, 0.0], [2.0, -1.0]]),
                image_point=gs.array([[1.0, 0.0], [2.0, 1.0]]),
                expected=gs.array([[-2.0, 0.0], [0.0, 6.0]]),
            ),
            dict(
                image_tangent_vec=gs.array(
                    [[[-1.0, 2.0], [2.0, -1.0]], [[0.0, 4.0], [4.0, -1.0]]]
                ),
                image_point=gs.array(
                    [[[3.0, 0.0], [-1.0, 2.0]], [[4.0, 0.0], [-1.0, 4.0]]]
                ),
                expected=gs.array(
                    [
                        [[-6.0, 11.0], [11.0, -8.0]],
                        [[0.0, 32.0], [32.0, -16.0]],
                    ]
                ),
            ),
        ]
        return data


class SPDMatrices2TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[3.0, -1.0], [-1.0, 3.0]]), expected=True),
            dict(point=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=False),
            dict(
                point=gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]]),
                expected=[True, False],
            ),
        ]

        return data

    def projection_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0, 0.0], [0.0, 1.0]]),
            ),
            dict(
                point=gs.array([[-1.0, 0.0], [0.0, -2.0]]),
                expected=gs.array([[gs.atol, 0.0], [0.0, gs.atol]]),
            ),
        ]
        return data


class SPDMatrices3TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=False,
            ),
        ]
        return data


class SPDAffineMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False


class SPD2AffineMetricTestData(CaseData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[math.exp(2.0), 0.0], [0.0, math.exp(2.0)]]),
            )
        ]
        return data

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                base_point=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                expected=gs.array(
                    [[-2 * math.log(2.0), 0.0], [0.0, -2 * math.log(2.0)]]
                ),
            )
        ]
        return data


class SPD3AffineMetricPower05TestData(CaseData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]]
                ),
                expected=gs.array(713 / 144),
            )
        ]
        return data


class SPD2BuresWassersteinMetricTestData(CaseData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[4.0, 0.0], [0.0, 4.0]]),
            )
        ]
        return data

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[4.0, 0.0], [0.0, 4.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[2.0, 0.0], [0.0, 2.0]]),
            )
        ]
        return data

    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                point_b=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                expected=gs.array(2 + 4 - (2 * 2 * math.sqrt(2.0))),
            )
        ]
        return data


class SPD3BuresWassersteinMetricTestData(CaseData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 2.0, 4.0], [2.0, 3.0, 8.0], [4.0, 8.0, 5.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.5, 1.5]]
                ),
                expected=gs.array(4.0),
            )
        ]
        return data


class SPDEuclideanMetricTestData(ExpDomainMixinsTestData, MatricesMetricTestData):
    pass


class SPD2EuclideanMetricTestData(CaseData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[3.0, 0.0], [0.0, 3.0]]),
            )
        ]
        return data

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0, 0.0], [0.0, 1.0]]),
            )
        ]
        return data

    def parallel_transport_test_data(self):
        smoke_data = [
            dict(
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                direction=gs.array([[1.0, 0.0], [0.0, 0.5]]),
                expected=gs.array([[2.0, 0.0], [0.0, 2.0]]),
            )
        ]
        return smoke_data


class SPD3EuclideanMetricTestData(CaseData):
    def exp_domain_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array(
                    [[-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
                ),
                expected=gs.array([-3.0, 1.0]),
            )
        ]
        return data


class SPD3EuclideanMetricPower05TestData(CaseData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]]
                ),
                expected=gs.array(3472 / 576),
            )
        ]
        return data


class SPD2LogEuclideanMetricTestData(CaseData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[math.exp(2.0), 0.0], [0.0, math.exp(2.0)]]),
            )
        ]
        return data

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[math.log(2.0), 0.0], [0.0, math.log(2.0)]]),
            )
        ]
        return data

    def dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                point_b=gs.array([[math.exp(1.0), 0.0], [0.0, math.exp(1.0)]]),
                expected=gs.array(math.sqrt(2.0)),
            )
        ]
        return data


class SPD3LogEuclideanMetricTestData(CaseData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array(
                    [[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]
                ),
                expected=gs.array(5.0 + (4.0 * ((2 * math.log(2.0)) ** 2))),
            )
        ]
        return data
