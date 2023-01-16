from tests2.data.base_data import LieGroupTestData, _ProjectionMixinsTestData


class _SpecialOrthogonalMixinsTestData:
    def skew_matrix_from_vector_vec_test_data(self):
        return self.generate_vec_data()

    def vector_from_skew_matrix_vec_test_data(self):
        return self.generate_vec_data()

    def vector_from_skew_matrix_after_skew_matrix_from_vector_test_data(self):
        return self.generate_random_data()

    def skew_matrix_from_vector_after_vector_from_skew_matrix_test_data(self):
        return self.generate_random_data()

    def rotation_vector_from_matrix_vec_test_data(self):
        return self.generate_vec_data()

    def matrix_from_rotation_vector_vec_test_data(self):
        return self.generate_vec_data()

    def rotation_vector_from_matrix_after_matrix_from_rotation_vector_test_data(self):
        return self.generate_random_data()

    def matrix_from_rotation_vector_after_rotation_vector_from_matrix_test_data(self):
        return self.generate_random_data()


class SpecialOrthogonalVectorsTestData(
    _ProjectionMixinsTestData, _SpecialOrthogonalMixinsTestData, LieGroupTestData
):
    pass


class SpecialOrthogonal2VectorsTestData(SpecialOrthogonalVectorsTestData):
    skips = (
        "test_jacobian_translation_vec",
        "test_tangent_translation_map_vec",
        "test_lie_bracket_vec",
        "test_projection_belongs",
    )


class SpecialOrthogonal3VectorsTestData(SpecialOrthogonalVectorsTestData):
    skips = ("test_projection_belongs",)

    def quaternion_from_matrix_vec_test_data(self):
        return self.generate_vec_data()

    def matrix_from_quaternion_vec_test_data(self):
        return self.generate_vec_data()

    def quaternion_from_matrix_after_matrix_from_quaternion_test_data(self):
        return self.generate_random_data()

    def matrix_from_quaternion_after_quaternion_from_matrix_test_data(self):
        return self.generate_random_data()

    def quaternion_from_rotation_vector_vec_test_data(self):
        return self.generate_vec_data()

    def rotation_vector_from_quaternion_vec_test_data(self):
        return self.generate_vec_data()

    def quaternion_from_rotation_vector_after_rotation_vector_from_quaternion_test_data(
        self,
    ):
        return self.generate_random_data()

    def rotation_vector_from_quaternion_after_quaternion_from_rotation_vector_test_data(
        self,
    ):
        return self.generate_random_data()
