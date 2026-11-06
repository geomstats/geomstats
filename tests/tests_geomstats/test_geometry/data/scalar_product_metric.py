import pytest
from polpo.testing.data import CaseData

import geomstats.backend as gs


class WrapperTestData(CaseData):
    def wrap_attr_test_data(self):
        def example_fnc():
            return 2.0

        data = [dict(func=example_fnc, scale=3)]

        return data

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
        return data

    def non_scaled_test_data(self):
        data = [dict(func_name="non_scaled", scale=2.0)]
        return data


class InstantiationTestData(CaseData):
    @pytest.mark.random
    def scalar_metric_multiplication_test_data(self):
        data = [dict(scale=2.0)]
        return data

    def scalar_metric_multiplication_error_test_data(self):
        data = [dict(scale=2), dict(scale="str")]
        return data

    def scalar_metric_signature_test_data(self):
        data = [dict(scale=2.0)]
        return data

    @pytest.mark.random
    def scaling_scalar_metric_test_data(self):
        data = [dict(scale=2.0)]
        return data


class CustomizationTestData(CaseData):
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
