import pytest

from tests2.data.base_data import LieGroupTestData, _ProjectionMixinsTestData


class _SpecialOrthogonalMixinsTestData:
    def skew_matrix_from_vector_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def vector_from_skew_matrix_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def vector_from_skew_matrix_after_skew_matrix_from_vector_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def skew_matrix_from_vector_after_vector_from_skew_matrix_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def rotation_vector_from_matrix_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def matrix_from_rotation_vector_vec_test_data(self):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)

    def rotation_vector_from_matrix_after_matrix_from_rotation_vector_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)

    def matrix_from_rotation_vector_after_rotation_vector_from_matrix_test_data(self):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data)


class SpecialOrthogonalVectorsTestData(
    _ProjectionMixinsTestData, _SpecialOrthogonalMixinsTestData, LieGroupTestData
):
    def projection_belongs_test_data(self):
        return self.generate_tests([], marks=[pytest.mark.skip])


class SpecialOrthogonal2VectorsTestData(SpecialOrthogonalVectorsTestData):
    # TODO: add skips
    def jacobian_translation_vec_test_data(self):
        return self.generate_tests([], marks=[pytest.mark.skip])

    def tangent_translation_map_vec_test_data(self):
        return self.generate_tests([], marks=[pytest.mark.skip])

    def lie_bracket_vec_test_data(self):
        return self.generate_tests([], marks=[pytest.mark.skip])


class SpecialOrthogonal3VectorsTestData(SpecialOrthogonalVectorsTestData):
    pass
