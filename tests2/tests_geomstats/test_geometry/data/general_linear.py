import random

from tests2.data.base_data import MatrixLieGroupTestData, OpenSetTestData


class GeneralLinearTestData(MatrixLieGroupTestData, OpenSetTestData):
    xfails = ("exp_after_log",)

    def orbit_vec_test_data(self):
        n_times = random.sample(range(1, 5), 1)
        data = [dict(n_reps=n_reps, n_times=n_times) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)
