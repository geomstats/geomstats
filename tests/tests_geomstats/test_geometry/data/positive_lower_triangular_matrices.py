import math

from polpo.testing.data import CaseData

import geomstats.backend as gs

from .base import VectorSpaceOpenSetTestData
from .lie_group import MatrixLieGroupTestData
from .riemannian_metric import RiemannianMetricTestData

EULER = gs.exp(1.0)
SQRT_2 = math.sqrt(2)


class PositiveLowerTriangularMatricesTestData(
    MatrixLieGroupTestData, VectorSpaceOpenSetTestData
):
    pass


class PositiveLowerTriangularMatrices2TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[1.0, 0.0], [-1.0, 3.0]]), expected=True),
            dict(point=gs.array([[1.0, -1.0], [-1.0, 3.0]]), expected=False),
            dict(point=gs.array([[-1.0, 0.0], [-1.0, 3.0]]), expected=False),
            dict(
                point=gs.array(
                    [
                        [[1.0, 0], [0, 1.0]],
                        [[1.0, 2.0], [2.0, 1.0]],
                        [[-1.0, 0.0], [1.0, 1.0]],
                        [[0.0, 0.0], [1.0, 1.0]],
                    ]
                ),
                expected=gs.array([True, False, False, False]),
            ),
            dict(
                point=gs.array(
                    [
                        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[-1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    ]
                ),
                expected=gs.array([False, False, False, False]),
            ),
        ]
        return data


class CholeskyMetricTestData(RiemannianMetricTestData):
    def diag_inner_product_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
            on_metric=True,
        )

    def strictly_lower_inner_product_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("tangent_vec_a", "tangent_vec_b"),
            samplers={
                "tangent_vec_a": "random_tangent_vec_at_identity",
                "tangent_vec_b": "random_tangent_vec_at_identity",
            },
            on_metric=True,
        )


class CholeskyMetric2TestData(CaseData):
    def diag_inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[1.0, 0.0], [-2.0, -1.0]]),
                tangent_vec_b=gs.array([[2.0, 0.0], [-3.0, -1.0]]),
                base_point=gs.array([[SQRT_2, 0.0], [-3.0, 1.0]]),
                expected=gs.array(2.0),
            )
        ]
        return data

    def strictly_lower_inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[1.0, 0.0], [-2.0, -1.0]]),
                tangent_vec_b=gs.array([[2.0, 0.0], [-3.0, -1.0]]),
                expected=gs.array(6.0),
            )
        ]
        return data

    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[1.0, 0.0], [-2.0, -1.0]]),
                tangent_vec_b=gs.array([[2.0, 0.0], [-3.0, -1.0]]),
                base_point=gs.array([[SQRT_2, 0.0], [-3.0, 1.0]]),
                expected=gs.array(8.0),
            ),
            dict(
                tangent_vec_a=gs.array(
                    [
                        [[3.0, 0.0], [4.0, 2.0]],
                        [[-1.0, 0.0], [2.0, -4.0]],
                    ]
                ),
                tangent_vec_b=gs.array(
                    [[[4.0, 0.0], [3.0, 3.0]], [[3.0, 0.0], [-6.0, 2.0]]]
                ),
                base_point=gs.array([[[3, 0.0], [-2.0, 6.0]], [[1, 0.0], [-1.0, 1.0]]]),
                expected=gs.array([13.5, -23.0]),
            ),
        ]
        return data

    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[-1.0, 0.0], [2.0, 3.0]]),
                base_point=gs.array([[1.0, 0.0], [2.0, 2.0]]),
                expected=gs.array([[1 / EULER, 0.0], [4.0, 2 * gs.exp(1.5)]]),
            ),
            dict(
                tangent_vec=gs.array(
                    [[[0.0, 0.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]
                ),
                base_point=gs.array(
                    [[[1.0, 0.0], [2.0, 2.0]], [[1.0, 0.0], [0.0, 2.0]]]
                ),
                expected=gs.array(
                    [
                        [[1.0, 0.0], [4.0, 2.0]],
                        [[gs.exp(1.0), 0.0], [0.0, 2.0]],
                    ]
                ),
            ),
        ]
        return data

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[EULER, 0.0], [2.0, EULER**3]]),
                base_point=gs.array([[EULER**3, 0.0], [4.0, EULER**4]]),
                expected=gs.array([[-2.0 * EULER**3, 0.0], [-2.0, -1 * EULER**4]]),
            ),
            dict(
                point=gs.array(
                    [
                        [[gs.exp(-2.0), 0.0], [0.0, gs.exp(2.0)]],
                        [[gs.exp(-3.0), 0.0], [2.0, gs.exp(3.0)]],
                    ]
                ),
                base_point=gs.array(
                    [[[1.0, 0.0], [-1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
                ),
                expected=gs.array(
                    [[[-2.0, 0.0], [1.0, 2.0]], [[-3.0, 0.0], [2.0, 3.0]]]
                ),
            ),
        ]
        return data

    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[EULER, 0.0], [2.0, EULER**3]]),
                point_b=gs.array([[EULER**3, 0.0], [4.0, EULER**4]]),
                expected=gs.array(9.0),
            ),
            dict(
                point_a=gs.array(
                    [
                        [[EULER, 0.0], [2.0, EULER**3]],
                        [[EULER, 0.0], [4.0, EULER**3]],
                    ]
                ),
                point_b=gs.array(
                    [
                        [[EULER**3, 0.0], [4.0, EULER**4]],
                        [[EULER**3, 0.0], [7.0, EULER**4]],
                    ]
                ),
                expected=gs.array([9.0, 14.0]),
            ),
        ]
        return data
