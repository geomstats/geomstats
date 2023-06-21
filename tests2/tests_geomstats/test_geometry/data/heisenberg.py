import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import VectorSpaceTestData
from .lie_group import LieGroupTestData


class HeisenbergVectorsTestData(LieGroupTestData, VectorSpaceTestData):
    skips = ("lie_bracket_vec",)

    def upper_triangular_matrix_from_vector_vec_test_data(self):
        return self.generate_vec_data()

    def vector_from_upper_triangular_matrix_vec_test_data(self):
        return self.generate_vec_data()

    def vector_from_upper_triangular_matrix_after_upper_triangular_matrix_from_vector_test_data(
        self,
    ):
        return self.generate_random_data()

    def upper_triangular_matrix_from_vector_after_vector_from_upper_triangular_matrix_test_data(
        self,
    ):
        return self.generate_random_data()


class HeisenbergVectors3TestData(TestData):
    def dim_test_data(self):
        smoke_data = [dict(expected=3)]
        return self.generate_tests(smoke_data)

    def belongs_test_data(self):
        data = [
            dict(point=gs.array([1.0, 2.0, 3.0, 4]), expected=False),
            dict(
                point=gs.array([[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]]),
                expected=[False, False],
            ),
        ]
        return self.generate_tests(data)

    def is_tangent_test_data(self):
        data = [
            dict(
                vector=gs.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
                base_point=None,
                expected=[False, False],
            )
        ]
        return self.generate_tests(data)

    def jacobian_translation_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, -10.0, 0.2], [-2.0, 100.0, 0.5]]),
                left=True,
                expected=gs.array(
                    [
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 0.5, 1.0]],
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-50.0, -1.0, 1.0]],
                    ]
                ),
            )
        ]
        return self.generate_tests(data)
