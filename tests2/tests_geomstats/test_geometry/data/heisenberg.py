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
