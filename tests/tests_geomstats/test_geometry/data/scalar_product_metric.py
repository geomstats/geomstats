import geomstats.backend as gs
from geomstats.test.data import TestData


class WrapperTestData(TestData):
    def wrap_attr_test_data(self):
        def example_fnc():
            return 2.0

        data = [dict(func=example_fnc, scale=3)]

        return self.generate_tests(data)

    def scaling_factor_test_data(self):
        data = [
            dict(func_name="dist", scale=2.0, expected=gs.sqrt(2)),
            dict(func_name="metric_matrix", scale=2.0, expected=2.0),
            dict(func_name="cometric_matrix", scale=2.0, expected=1.0 / 2.0),
            dict(
                func_name="normalize",
                scale=2.0,
                expected=1.0
                / gs.sqrt(
                    2,
                ),
            ),
        ]
        return self.generate_tests(data)

    def non_scaled_test_data(self):
        data = [dict(func_name="non_scaled", scale=2.0)]
        return self.generate_tests(data)


class InstantiationTestData(TestData):
    def scalar_metric_multiplication_test_data(self):
        data = [dict(scale=2.0)]
        return self.generate_tests(data)

    def scaling_scalar_metric_test_data(self):
        data = [dict(scale=2.0)]
        return self.generate_tests(data)
