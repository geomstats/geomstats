from .base import OpenSetTestData
from .riemannian_metric import RiemannianMetricTestData


class PoincareBallTestData(OpenSetTestData):
    xfails = ("projection_belongs",)


class PoincareBallMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    def mobius_add_vec_test_data(self):
        return self.generate_vec_data()

    def retraction_vec_test_data(self):
        return self.generate_vec_data()
