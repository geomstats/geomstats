import math

import pytest
from polpo.testing.data import CaseData
from polpo.testing.geometric import GeometricCaseData

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.test.random import MatrixOperationsRandomDataGenerator

from .base import MatrixVectorSpaceTestData
from .riemannian_metric import RiemannianMetricTestData


class MatricesTestData(MatrixVectorSpaceTestData):
    def reshape_after_flatten_test_data(self):
        return self.generate_random_data(arg_names="point")


class MatricesMetricTestData(RiemannianMetricTestData):
    pass


class MatricesMetric22TestData(CaseData):
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
        return data

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
        return data


class MatrixOperationsTestData(GeometricCaseData):
    """Test data for matrix operations.

    Operations are static methods of `Matrices` (or backend functions),
    so there is no space: random matrices are sampled with a dedicated
    data generator.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_generator = MatrixOperationsRandomDataGenerator()

        self.is_properties = (
            "diagonal",
            "symmetric",
            "lower_triangular",
            "upper_triangular",
            "spd",
            "skew_symmetric",
        )

    def _get_operation(self, op_name, op_target=None):
        return getattr(Matrices, op_name, None) or getattr(gs, op_name)

    def _generate_pair_vec_data(
        self, sampler_a, sampler_b, names=("mat_a", "mat_b"), **kwargs
    ):
        name_a, name_b = names
        return self.generate_vectorization_data(
            arg_names=names,
            samplers={name_a: sampler_a, name_b: sampler_b},
            dependencies={name_b: name_a},
            vectorization_type="sym",
            **kwargs,
        )

    @pytest.mark.smoke
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
        return smoke_data

    @pytest.mark.smoke
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
        return smoke_data

    @pytest.mark.smoke
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
        return smoke_data

    @pytest.mark.smoke
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
        return smoke_data

    @pytest.mark.smoke
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
        return smoke_data

    def equal_vec_test_data(self):
        return self._generate_pair_vec_data("random_mat", "random_mat_like")

    def mul_vec_test_data(self):
        return self._generate_pair_vec_data(
            "random_mat",
            "random_transposed_mat_like",
            op_evaluator=lambda op, mat_a, mat_b: op(mat_a, mat_b),
        )

    def bracket_vec_test_data(self):
        return self._generate_pair_vec_data("random_square_mat", "random_mat_like")

    def transpose_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="mat",
            samplers={"mat": "random_mat"},
            op_evaluator=lambda op, mat: op(mat),
        )

    def diagonal_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="mat", samplers={"mat": "random_mat"}
        )

    def is_square_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="mat", samplers={"mat": "random_square_mat"}
        )

    def is_property_vec_test_data(self):
        data = []
        for property_name in self.is_properties:
            data.extend(
                self.generate_vectorization_data(
                    arg_names="mat",
                    samplers={"mat": f"random_{property_name}_mat"},
                    op_name=f"is_{property_name}",
                    property_name=property_name,
                )
            )
        return data

    def congruent_vec_test_data(self):
        return self._generate_pair_vec_data(
            "random_square_mat", "random_mat_like", names=("mat_1", "mat_2")
        )

    def frobenius_product_vec_test_data(self):
        return self._generate_pair_vec_data(
            "random_mat", "random_mat_like", names=("mat_1", "mat_2")
        )

    def trace_product_vec_test_data(self):
        return self._generate_pair_vec_data(
            "random_mat", "random_transposed_mat_like", names=("mat_1", "mat_2")
        )

    def flatten_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="mat", samplers={"mat": "random_mat"}
        )

    def align_matrices_vec_test_data(self):
        return self._generate_pair_vec_data(
            "random_mat", "random_mat_like", names=("point", "base_point")
        )

    def equal_true_test_data(self):
        return self.generate_random_data(
            arg_names="mat", samplers={"mat": "random_mat"}
        )

    def equal_false_test_data(self):
        return self.generate_random_data(
            arg_names=("mat_a", "mat_b"),
            samplers={"mat_a": "random_mat", "mat_b": "random_mat_like"},
            dependencies={"mat_b": "mat_a"},
        )

    def mul_identity_test_data(self):
        return self.generate_random_data(
            arg_names="mat", samplers={"mat": "random_mat"}
        )

    def bracket_bilinearity_test_data(self):
        return self.generate_random_data(
            arg_names=("mat_a", "mat_b", "mat_c", "coefficients"),
            samplers={
                "mat_a": "random_square_mat",
                "mat_b": "random_mat_like",
                "mat_c": "random_mat_like",
                "coefficients": "random_coefficients",
            },
            dependencies={"mat_b": "mat_a", "mat_c": "mat_a"},
        )

    def transpose_sym_test_data(self):
        return self.generate_random_data(
            arg_names="mat", samplers={"mat": "random_symmetric_mat"}
        )

    def diagonal_sum_test_data(self):
        return self.generate_random_data(
            arg_names="mat", samplers={"mat": "random_square_mat"}
        )

    def is_square_true_test_data(self):
        return self.generate_random_data(
            arg_names="mat", samplers={"mat": "random_square_mat"}
        )

    def is_square_false_test_data(self):
        return self.generate_random_data(
            arg_names="mat", samplers={"mat": "random_non_square_mat"}
        )

    def is_property_true_test_data(self):
        data = []
        for property_name in self.is_properties:
            data.extend(
                self.generate_random_data(
                    arg_names="mat",
                    samplers={"mat": f"random_{property_name}_mat"},
                    property_name=property_name,
                )
            )
        return data

    def is_property_false_square_test_data(self):
        data = []
        for property_name in self.is_properties:
            if property_name == "spd":
                continue
            data.extend(
                self.generate_random_data(
                    arg_names="mat",
                    samplers={"mat": "random_square_mat"},
                    property_name=property_name,
                )
            )
        return data

    def to_property_is_property_test_data(self):
        properties = (
            "diagonal",
            "symmetric",
            "lower_triangular",
            "upper_triangular",
            "skew_symmetric",
            "strictly_lower_triangular",
            "strictly_upper_triangular",
        )

        data = []
        for property_name in properties:
            data.extend(
                self.generate_random_data(
                    arg_names="mat",
                    samplers={"mat": "random_square_mat"},
                    property_name=property_name,
                )
            )
        return data

    def to_lower_triangular_diagonal_scaled_is_lower_triangular_test_data(self):
        return self.generate_random_data(
            arg_names="mat", samplers={"mat": "random_square_mat"}
        )
