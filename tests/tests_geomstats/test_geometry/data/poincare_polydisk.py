from .nfold_manifold import NFoldManifoldTestData, NFoldMetricTestData


class PoincarePolydiskTestData(NFoldManifoldTestData):
    skips = ("not_belongs",)
    tolerances = {
        "projection_belongs": {"atol": 1e-6},
    }


class PoincarePolydiskMetricTestData(NFoldMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False
