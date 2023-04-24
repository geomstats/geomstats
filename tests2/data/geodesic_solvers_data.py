from geomstats.test.data import TestData


class ExpSolverComparisonTestData(TestData):
    def exp_test_data(self):
        return self.generate_random_data()

    def geodesic_ivp_test_data(self):
        return self.generate_random_data_with_time()


class LogSolverComparisonTestData(TestData):
    tolerances = {
        "log": {"atol": 1e-4},
        "geodesic_bvp": {"atol": 1e-4},
    }

    def log_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_test_data(self):
        return self.generate_random_data_with_time()


class ExpSolverTypeCheckTestData(TestData):
    def exp_test_type_data(self):
        return self.generate_random_data()

    def geodesic_ivp_type_test_data(self):
        return self.generate_random_data_with_time()


class LogSolverTypeCheckTestData(TestData):
    def log_type_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_type_test_data(self):
        return self.generate_random_data_with_time()
