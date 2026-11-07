from polpo.testing.data import CaseData
from polpo.testing.geometric import GeometricCaseData


class ScalarProductMetricComparisonTestData(GeometricCaseData):
    def metric_matrix_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def cometric_matrix_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def inner_product_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    def squared_norm_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "vector"))

    def norm_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "vector"))

    def normalize_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "vector"))

    def squared_dist_test_data(self):
        return self.generate_random_data(arg_names=("point_a", "point_b"))

    def dist_test_data(self):
        return self.generate_random_data(arg_names=("point_a", "point_b"))

    def injectivity_radius_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def exp_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def log_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))


class MetricScalingTestData(CaseData):
    def scalar_metric_multiplication_test_data(self):
        data = [dict(scale=2.0)]
        return data

    def scalar_metric_multiplication_not_equipped_test_data(self):
        data = [dict(scale=2.0)]
        return data

    def scalar_metric_multiplication_error_test_data(self):
        data = [dict(scale=2), dict(scale="str")]
        return data

    def scalar_metric_signature_test_data(self):
        data = [dict(scale=2.0)]
        return data

    def scale_composition_test_data(self):
        data = [dict(scale_a=2.0, scale_b=3.0)]
        return data


class ScaledMethodRegistrationTestData(CaseData):
    def register_scaled_method_invalid_test_data(self):
        data = [
            dict(func_name="dummy_func", scaling_type="unlisted"),
            dict(func_name="_private_func", scaling_type="linear"),
            dict(func_name="dist", scaling_type="linear"),
            dict(func_name="underlying_metric", scaling_type="linear"),
        ]
        return data

    def register_scaled_method_valid_test_data(self):
        data = [dict(scale=2.0)]
        return data
