import random

from tests2.data.base_data import OpenSetTestData


class SPDMatricesTestData(OpenSetTestData):
    def _generate_power_vec_data(self):
        power = [random.randint(1, 4)]
        data = []
        for power_ in power:
            data.extend(
                [dict(n_reps=n_reps, power=power_) for n_reps in self.N_VEC_REPS]
            )
        return self.generate_tests(data)

    def differential_power_vec_test_data(self):
        return self._generate_power_vec_data()

    def inverse_differential_power_vec_test_data(self):
        return self._generate_power_vec_data()

    def differential_log_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_differential_log_vec_test_data(self):
        return self.generate_vec_data()

    def differential_exp_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_differential_exp_vec_test_data(self):
        return self.generate_vec_data()

    def logm_vec_test_data(self):
        return self.generate_vec_data()

    def cholesky_factor_vec_test_data(self):
        return self.generate_vec_data()

    def cholesky_factor_belongs_to_positive_lower_triangular_matrices_test_data(self):
        return self.generate_random_data()

    def differential_cholesky_factor_vec_test_data(self):
        return self.generate_vec_data()

    def differential_cholesky_factor_belongs_to_positive_lower_triangular_matrices_test_data(
        self,
    ):
        return self.generate_random_data()
