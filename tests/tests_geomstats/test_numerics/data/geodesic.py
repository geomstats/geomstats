from geomstats.test.data import TestData


class ExpSolverAgainstMetricTestData(TestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def exp_test_data(self):
        return self.generate_random_data()

    def geodesic_ivp_test_data(self):
        return self.generate_random_data_with_time()


class ExpSolverComparisonTestData(TestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def exp_test_data(self):
        return self.generate_random_data()

    def geodesic_ivp_test_data(self):
        return self.generate_random_data_with_time()


class ExpSolverTestData(TestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def exp_vec_test_data(self):
        return self.generate_vec_data()

    def geodesic_ivp_vec_test_data(self):
        return self.generate_vec_data_with_time()


class LogSolverAgainstMetricTestData(TestData):
    def log_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_test_data(self):
        return self.generate_random_data_with_time()


class LogSolverComparisonTestData(TestData):
    def log_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_test_data(self):
        return self.generate_random_data_with_time()


class LogSolverTestData(TestData):
    def log_known_tangent_vec_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_vec_test_data(self):
        return self.generate_vec_data_with_time()

    def geodesic_bvp_known_geod_test_data(self):
        return self.generate_random_data_with_time()


class ExpSolverTypeCheckTestData(TestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def exp_test_type_data(self):
        return self.generate_random_data()

    def geodesic_ivp_type_test_data(self):
        return self.generate_random_data_with_time()


class LogSolverTypeCheckTestData(TestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def log_type_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_type_test_data(self):
        return self.generate_random_data_with_time()
