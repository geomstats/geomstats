from tests2.data.base_data import FiberBundleTestData
from tests2.data.general_linear_data import GeneralLinearTestData


class GeneralLinearBuresWassersteinBundleTestData(
    FiberBundleTestData, GeneralLinearTestData
):
    xfails = (
        "align_vec",
        "exp_after_log",
        "log_after_exp",
        "log_after_align_is_horizontal",
        "horizontal_lift_is_horizontal",
    )

    skips = (
        # not implemented
        "integrability_tensor_derivative_vec",
        "integrability_tensor_vec",
    )

    ignores_if_not_autodiff = (
        "align_vec",
        "log_after_align_is_horizontal",
    )

    tolerances = {
        "align_vec": {"atol": 1e-1},
        "horizontal_lift_is_horizontal": {"atol": 1e-1},
        "log_after_exp": {"atol": 1e-1},
        "log_after_align_is_horizontal": {"atol": 1e-1},
    }
