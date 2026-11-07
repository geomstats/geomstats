import math

import pytest
from polpo.testing.geometric import GeometricCaseData

import geomstats.backend as gs
from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.test.random import ComplexMatrixOperationsRandomDataGenerator

CDTYPE = gs.get_default_cdtype()

SQRT_2 = math.sqrt(2)

MAT9_33 = gs.array([[4, 0, 1j], [0, 3, 0], [-1j, 0, 4]], dtype=CDTYPE)
MAT10_33 = gs.array([[1, 1 + 1j, -3j], [1 - 1j, 0, 4], [3j, 4, -2]], dtype=CDTYPE)
MAT11_33 = gs.array([[1, 1j, 2], [1j, -1j, 2], [4 + 2j, 0, -2]], dtype=CDTYPE)
MAT12_33 = gs.array([[1, -1j, 4 - 2j], [-1j, 1j, 0], [2, 2, -2]], dtype=CDTYPE)
MAT13_33 = gs.array([[1, 0, 3 - 1j], [0, 0, 1], [3 + 1j, 1, -2]], dtype=CDTYPE)
MAT14_33 = gs.array([[4, 0, 0.5j], [0, 3, 0], [-1j, 0, 4]], dtype=CDTYPE)


class ComplexMatrixOperationsTestData(GeometricCaseData):
    """Test data for complex matrix operations.

    Operations are class methods of `ComplexMatrices`, so there is no
    space: random matrices are sampled with a dedicated data generator.
    """

    is_properties = ("hermitian", "hpd", "skew_hermitian")
    to_properties = ("hermitian", "skew_hermitian")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_generator = ComplexMatrixOperationsRandomDataGenerator()

    def _get_operation(self, op_name, op_target=None):
        return getattr(ComplexMatrices, op_name)

    def _generate_pair_vec_data(
        self, sampler_a, sampler_b, names=("mat_1", "mat_2"), **kwargs
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
    def transconjugate_test_data(self):
        data = [
            dict(
                mat=gs.stack([MAT9_33, MAT11_33]),
                expected=gs.stack([MAT9_33, MAT12_33]),
            ),
        ]
        return data

    @pytest.mark.smoke
    def is_property_test_data(self):
        data = [
            dict(
                property_name="hermitian",
                mat=gs.stack([MAT9_33, MAT10_33, MAT11_33]),
                expected=[True, True, False],
            ),
            dict(
                property_name="hpd",
                mat=gs.stack([MAT9_33, MAT10_33, MAT11_33]),
                expected=[True, False, False],
            ),
        ]
        return data

    @pytest.mark.smoke
    def to_hermitian_test_data(self):
        data = [
            dict(
                mat=gs.stack([MAT10_33, MAT11_33]),
                expected=gs.stack([MAT10_33, MAT13_33]),
            ),
        ]
        return data


    @pytest.mark.smoke
    def congruent_test_data(self):
        data = [
            dict(
                mat_1=gs.array([[1.0, 1j], [0.0, 2.0]], dtype=CDTYPE),
                mat_2=gs.array([[1j, 0.0], [1.0, 1.0]], dtype=CDTYPE),
                expected=gs.array([[1.0, -1.0 + 1j], [-1j, 3.0 + 1j]], dtype=CDTYPE),
            ),
        ]
        return data

    @pytest.mark.smoke
    def frobenius_product_test_data(self):
        data = [
            dict(
                mat_1=gs.array([[1j, 1.0], [0.0, 2.0]], dtype=CDTYPE),
                mat_2=gs.array([[1j, 2.0], [3.0, 1j]], dtype=CDTYPE),
                expected=gs.array(3.0 + 2j, dtype=CDTYPE),
            ),
        ]
        return data

    def transconjugate_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="mat", samplers={"mat": "random_mat"}
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
        return self._generate_pair_vec_data("random_square_mat", "random_mat_like")

    def frobenius_product_vec_test_data(self):
        return self._generate_pair_vec_data("random_mat", "random_mat_like")

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
            data.extend(
                self.generate_random_data(
                    arg_names="mat",
                    samplers={"mat": "random_square_mat"},
                    property_name=property_name,
                )
            )
        return data

    def to_property_is_property_test_data(self):
        data = []
        for property_name in self.to_properties:
            data.extend(
                self.generate_random_data(
                    arg_names="mat",
                    samplers={"mat": "random_square_mat"},
                    property_name=property_name,
                )
            )
        return data
