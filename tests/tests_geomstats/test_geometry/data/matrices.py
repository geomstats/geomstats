import math

import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import MatrixVectorSpaceTestData
from .riemannian_metric import RiemannianMetricTestData


class MatrixOperationsTestData(TestData):
    def equal_vec_test_data(self):
        return self.generate_vec_data()

    def equal_true_test_data(self):
        return self.generate_random_data()

    def equal_false_test_data(self):
        return self.generate_random_data()

    def mul_vec_test_data(self):
        return self.generate_vec_data()

    def mul_identity_test_data(self):
        return self.generate_random_data()

    def bracket_vec_test_data(self):
        return self.generate_vec_data()

    def bracket_bilinearity_test_data(self):
        return self.generate_random_data()

    def transpose_vec_test_data(self):
        return self.generate_vec_data()

    def transpose_sym_test_data(self):
        return self.generate_random_data()

    def diagonal_vec_test_data(self):
        return self.generate_vec_data()

    def diagonal_sum_test_data(self):
        return self.generate_random_data()

    def is_square_vec_test_data(self):
        return self.generate_vec_data()

    def is_square_true_test_data(self):
        return self.generate_random_data()

    def is_square_false_test_data(self):
        return self.generate_random_data()

    def is_property_vec_test_data(self):
        data = []
        properties = [
            "diagonal",
            "symmetric",
            "lower_triangular",
            "upper_triangular",
            "spd",
            "skew_symmetric",
        ]
        for property_name in properties:
            data += [
                dict(property_name=property_name, n_reps=n_reps)
                for n_reps in self.N_VEC_REPS
            ]
        return self.generate_tests(data)

    def is_property_true_test_data(self):
        data = []
        properties = [
            "diagonal",
            "symmetric",
            "lower_triangular",
            "upper_triangular",
            "spd",
            "skew_symmetric",
        ]
        for property_name in properties:
            data += [
                dict(property_name=property_name, n_points=n_points)
                for n_points in self.N_RANDOM_POINTS
            ]
        return self.generate_tests(data)

    def is_property_false_square_test_data(self):
        data = []
        properties = [
            "diagonal",
            "symmetric",
            "lower_triangular",
            "upper_triangular",
            "skew_symmetric",
        ]
        for property_name in properties:
            data += [
                dict(property_name=property_name, n_points=n_points)
                for n_points in self.N_RANDOM_POINTS
            ]
        return self.generate_tests(data)

    def to_property_is_property_test_data(self):
        data = []
        properties = [
            "diagonal",
            "symmetric",
            "lower_triangular",
            "upper_triangular",
            "skew_symmetric",
            "strictly_lower_triangular",
            "strictly_upper_triangular",
        ]
        for property_name in properties:
            data += [
                dict(property_name=property_name, n_points=n_points)
                for n_points in self.N_RANDOM_POINTS
            ]
        return self.generate_tests(data)

    def to_lower_triangular_diagonal_scaled_is_lower_triangular_test_data(self):
        return self.generate_random_data()

    def congruent_vec_test_data(self):
        return self.generate_vec_data()

    def frobenius_product_vec_test_data(self):
        return self.generate_vec_data()

    def trace_product_vec_test_data(self):
        return self.generate_vec_data()

    def flatten_vec_test_data(self):
        return self.generate_vec_data()

    def align_matrices_vec_test_data(self):
        return self.generate_vec_data()


class MatrixOperationsSmokeTestData(TestData):
    def mul_reduce_test_data(self):
        mats_1 = (
            gs.array([[1.0, 2.0], [3.0, 4.0]]),
            gs.array([[-1.0, 2.0], [-3.0, 4.0]]),
            gs.array([[1.0, -2.0], [3.0, -4.0]]),
        )
        mats_2 = gs.array([[[2.0], [4.0]], [[1.0], [3.0]], [[1.0], [3.0]]])
        mat_1_x_mat_2 = gs.array([[[10.0], [22.0]], [[5.0], [9.0]], [[-5.0], [-9.0]]])
        smoke_data = [
            dict(mats=mats_1, expected=gs.array([[23.0, -26.0], [51.0, -58.0]])),
            dict(mats=(gs.stack(mats_1), mats_2), expected=mat_1_x_mat_2),
        ]
        return self.generate_tests(smoke_data)

    def bracket_test_data(self):
        smoke_data = [
            dict(
                mat_a=gs.array([[1.0, 2.0], [3.0, 4.0]]),
                mat_b=gs.array([[1.0, 2.0], [3.0, 4.0]]),
                expected=gs.array([[0.0, 0.0], [0.0, 0.0]]),
            ),
            dict(
                mat_a=gs.array([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [0.0, 1.0]]]),
                mat_b=gs.array([[[2.0, 4.0], [5.0, 4.0]], [[1.0, 4.0], [5.0, 4.0]]]),
                expected=gs.array(
                    [[[-2.0, -8.0], [9.0, 2.0]], [[10.0, 6.0], [0.0, -10.0]]]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def congruent_test_data(self):
        smoke_data = [
            dict(
                mat_1=gs.array([[1.0, 0.0], [2.0, -2]]),
                mat_2=gs.array([[0.0, -2.0], [2.0, -3]]),
                expected=gs.array([[-8.0, -20.0], [-12.0, -26.0]]),
            ),
            dict(
                mat_1=gs.array([[[0.0, 1.0], [2.0, -2]], [[1.0, 0.0], [0.0, -1]]]),
                mat_2=gs.array([[[1.0, -2.0], [2.0, -3]], [[0.0, 0.0], [-1.0, -3]]]),
                expected=gs.array(
                    [
                        [[-14.0, -23.0], [-22.0, -36.0]],
                        [[0.0, 0.0], [0.0, -8.0]],
                    ]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def frobenius_product_test_data(self):
        smoke_data = [
            dict(
                mat_1=gs.array([[[1.0, -2.0], [1.0, 4.0]], [[1.0, 2.0], [0.0, -3.0]]]),
                mat_2=gs.array([[[0.0, 4.0], [2.0, 4.0]], [[1.0, -1.0], [5.0, 4.0]]]),
                expected=gs.array([10.0, -13.0]),
            ),
            dict(
                mat_1=gs.array([[5.0, 8.0], [2.0, 2.0]]),
                mat_2=gs.array([[0.0, 0.25], [0.5, 2.0]]),
                expected=gs.array(7.0),
            ),
        ]
        return self.generate_tests(smoke_data)

    def trace_product_test_data(self):
        smoke_data = [
            dict(
                mat_1=gs.array([[-2.0, 0.0], [1.0, 2.0]]),
                mat_2=gs.array([[0.0, 1.0], [2.0, -2.0]]),
                expected=gs.array(-3.0),
            ),
            dict(
                mat_1=gs.array(
                    [[[-5.0, 0.0], [-2.0, 0.0]], [[-2.0, 1.0], [-5.0, -6.0]]]
                ),
                mat_2=gs.array(
                    [[[6.0, 5.0], [-3.0, -2.0]], [[-2.0, 0.0], [4.0, -6.0]]]
                ),
                expected=gs.array([-40.0, 44.0]),
            ),
        ]
        return self.generate_tests(smoke_data)


class MatricesTestData(MatrixVectorSpaceTestData):
    def reshape_after_flatten_test_data(self):
        return self.generate_random_data()


class MatricesMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False


class MatricesMetric22TestData(TestData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[-3.0, 1.0], [-1.0, -2.0]]),
                tangent_vec_b=gs.array([[-9.0, 0.0], [4.0, 2.0]]),
                base_point=None,
                expected=gs.array(19.0),
            ),
            dict(
                tangent_vec_a=gs.array(
                    [
                        [[-1.5, 0.0], [2.0, -3.0]],
                        [[0.5, 7.0], [0.5, -2.0]],
                    ]
                ),
                tangent_vec_b=gs.array(
                    [
                        [[2.0, 0.0], [2.0, -3.0]],
                        [[-1.0, 0.0], [1.0, -2.0]],
                    ]
                ),
                base_point=None,
                expected=gs.array([10.0, 4.0]),
            ),
        ]
        return self.generate_tests(data)

    def norm_test_data(self):
        data = [
            dict(
                vector=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                base_point=None,
                expected=gs.array(math.sqrt(2)),
            ),
            dict(
                vector=gs.array([[[3.0, 0.0], [4.0, 0.0]], [[-3.0, 0.0], [-4.0, 0.0]]]),
                base_point=None,
                expected=gs.array([5.0, 5.0]),
            ),
        ]
        return self.generate_tests(data)
