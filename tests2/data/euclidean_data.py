from tests2.data.base_data import RiemannianMetricTestData, VectorSpaceTestData


class EuclideanTestData(VectorSpaceTestData):
    def exp_vec_test_data(self):
        return self.generate_vec_data()

    def exp_random_test_data(self):
        return self.generate_random_data()

    def identity_belongs_test_data(self):
        return self.generate_tests([dict()])


class EuclideanMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
