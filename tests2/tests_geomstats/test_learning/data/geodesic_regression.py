import random

from geomstats.test.data import TestData


class GeodesicRegressionTestData(TestData):
    fail_for_autodiff_exceptions = False

    def loss_test_data(self):
        return self.generate_tests([dict(n_samples=random.randint(2, 10))])

    def param_belongs_and_is_tangent_test_data(self):
        return self.generate_tests(
            [
                dict(
                    n_samples=random.randint(2, 10),
                )
            ]
        )

    def predict_and_score_test_data(self):
        return self.generate_tests([dict(n_samples=random.randint(2, 10), atol=0.1)])
