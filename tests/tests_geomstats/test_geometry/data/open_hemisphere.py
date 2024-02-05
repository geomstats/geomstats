from .pullback_metric import PullbackDiffeoMetricTestData


class OpenHemispherePullbackMetricTestData(PullbackDiffeoMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    trials = 3
