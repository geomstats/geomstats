from .manifold import ManifoldTestData
from .mixins import GroupExpMixinsTestData


class _LieGroupMixinsTestData(GroupExpMixinsTestData):
    def compose_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_vec_test_data(self):
        return self.generate_vec_data()

    def compose_with_inverse_is_identity_test_data(self):
        return self.generate_random_data()

    def compose_with_identity_is_point_test_data(self):
        return self.generate_random_data()

    def log_vec_test_data(self):
        return self.generate_vec_data()

    def exp_after_log_test_data(self):
        return self.generate_random_data()

    def log_after_exp_test_data(self):
        return self.generate_random_data()

    def to_tangent_at_identity_belongs_to_lie_algebra_test_data(self):
        return self.generate_random_data()

    def tangent_translation_map_vec_test_data(self):
        data = []
        for inverse in [True, False]:
            for left in [True, False]:
                data.extend(
                    [
                        dict(n_reps=n_reps, left=left, inverse=inverse)
                        for n_reps in self.N_VEC_REPS
                    ]
                )
        return self.generate_tests(data)

    def lie_bracket_vec_test_data(self):
        return self.generate_vec_data()


class MatrixLieGroupTestData(_LieGroupMixinsTestData, ManifoldTestData):
    pass


class LieGroupTestData(_LieGroupMixinsTestData, ManifoldTestData):
    def jacobian_translation_vec_test_data(self):
        data = []
        for left in [True, False]:
            data.extend([dict(n_reps=n_reps, left=left) for n_reps in self.N_VEC_REPS])
        return self.generate_tests(data)

    def exp_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def log_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def exp_from_identity_after_log_from_identity_test_data(self):
        return self.generate_random_data()

    def log_from_identity_after_exp_from_identity_test_data(self):
        return self.generate_random_data()
