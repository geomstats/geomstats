import random

from geomstats.test.data import TestData


class ToTangentSpaceTestData(TestData):
    def fit_transform_is_tangent_test_data(self):
        return self.generate_tests([dict(n_points=random.randint(2, 10))])

    def inverse_transform_after_transform_test_data(self):
        return self.generate_tests([dict(n_points=random.randint(2, 10))])


class ToTangentSpaceNdim2TestData(ToTangentSpaceTestData):
    skips = ("fit_transform_is_tangent",)
