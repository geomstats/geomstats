from tests2.data.base_data import OpenSetTestData, RiemannianMetricTestData


class PoincareBallTestData(OpenSetTestData):
    xfails = ("projection_belongs",)


class PoincareBallMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False

    def mobius_add_vec_test_data(self):
        return self.generate_vec_data()

    def retraction_vec_test_data(self):
        return self.generate_vec_data()
