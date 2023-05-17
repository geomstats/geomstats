from tests2.data.base_data import FiberBundleTestData


class GeneralLinearBuresWassersteinBundleTestData(FiberBundleTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
    xfails = (
        "align_vec",
        "exp_after_log",
        "log_after_exp",
        "log_after_align_is_horizontal",
        "horizontal_lift_is_horizontal",
    )

    tolerances = {
        "align_vec": {"atol": 1e-1},
        "horizontal_lift_is_horizontal": {"atol": 1e-1},
        "log_after_exp": {"atol": 1e-1},
        "log_after_align_is_horizontal": {"atol": 1e-1},
    }
