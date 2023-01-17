from tests2.data.base_data import (
    LevelSetTestData,
    LieGroupTestData,
    MatrixLieGroupTestData,
    _ProjectionMixinsTestData,
)


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


class SpecialOrthogonalMatricesTestData(
    _SpecialOrthogonalMixinsTestData, MatrixLieGroupTestData, LevelSetTestData
):
    xfails = ("test_log_after_exp",)
    tolerances = {
        "projection_belongs": {"atol": 1e-5},
        "matrix_from_rotation_vector_after_rotation_vector_from_matrix": {"atol": 1e-1},
    }

    def are_antipodals_vec_test_data(self):
        return self.generate_vec_data()


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
    tolerances = {
        "rotation_vector_from_matrix_after_matrix_from_rotation_vector": {"atol": 1e-5},
        "matrix_from_rotation_vector_after_rotation_vector_from_matrix": {"atol": 1e-1},
        "quaternion_from_matrix_after_matrix_from_quaternion": {"atol": 1e-2},
        "matrix_from_quaternion_after_quaternion_from_matrix": {"atol": 1e-1},
        "tait_bryan_angles_from_matrix_after_matrix_from_tait_bryan_angles": {
            "atol": 1e-1
        },
        "matrix_from_tait_bryan_angles_after_tait_bryan_angles_from_matrix": {
            "atol": 1e-1
        },
        "tait_bryan_angles_from_quaternion_after_quaternion_from_tait_bryan_angles": {
            "atol": 1e-1
        },
        "tait_bryan_angles_from_rotation_vector_after_rotation_vector_from_tait_bryan_angles": {
            "atol": 1e-1
        },
        "quaternion_from_tait_bryan_angles_after_tait_bryan_angles_from_quaternion": {
            "atol": 1e-1
        },
        "rotation_vector_from_tait_bryan_angles_after_tait_bryan_angles_from_rotation_vector": {
            "atol": 1e-1
        },
        "log_after_exp": {"atol": 1e-1},
    }

    def _generate_tait_bryan_angles_vec_data(self, marks=()):
        data = []
        for extrinsic in [True, False]:
            for zyx in [True, False]:
                for n_reps in self.N_VEC_REPS:
                    data.append(dict(n_reps=n_reps, extrinsic=extrinsic, zyx=zyx))
        return self.generate_tests(data, marks=marks)

    def _generate_tait_bryan_angles_random_data(self, marks=()):
        data = []
        for extrinsic in [True, False]:
            for zyx in [True, False]:
                for n_points in self.N_RANDOM_POINTS:
                    data.append(dict(n_points=n_points, extrinsic=extrinsic, zyx=zyx))
        return self.generate_tests(data, marks=marks)

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

    def matrix_from_tait_bryan_angles_vec_test_data(self):
        return self._generate_tait_bryan_angles_vec_data()

    def tait_bryan_angles_from_matrix_vec_test_data(self):
        return self._generate_tait_bryan_angles_vec_data()

    def tait_bryan_angles_from_matrix_after_matrix_from_tait_bryan_angles_test_data(
        self,
    ):
        return self._generate_tait_bryan_angles_random_data()

    def matrix_from_tait_bryan_angles_after_tait_bryan_angles_from_matrix_test_data(
        self,
    ):
        return self._generate_tait_bryan_angles_random_data()

    def quaternion_from_tait_bryan_angles_vec_test_data(self):
        return self._generate_tait_bryan_angles_vec_data()

    def tait_bryan_angles_from_quaternion_vec_test_data(self):
        return self._generate_tait_bryan_angles_vec_data()

    def quaternion_from_tait_bryan_angles_after_tait_bryan_angles_from_quaternion_test_data(
        self,
    ):
        return self._generate_tait_bryan_angles_random_data()

    def tait_bryan_angles_from_quaternion_after_quaternion_from_tait_bryan_angles_test_data(
        self,
    ):
        return self._generate_tait_bryan_angles_random_data()

    def rotation_vector_from_tait_bryan_angles_vec_test_data(self):
        return self._generate_tait_bryan_angles_vec_data()

    def tait_bryan_angles_from_rotation_vector_vec_test_data(self):
        return self._generate_tait_bryan_angles_vec_data()

    def tait_bryan_angles_from_rotation_vector_after_rotation_vector_from_tait_bryan_angles_test_data(
        self,
    ):
        return self._generate_tait_bryan_angles_random_data()

    def rotation_vector_from_tait_bryan_angles_after_tait_bryan_angles_from_rotation_vector_test_data(
        self,
    ):
        return self._generate_tait_bryan_angles_random_data()
