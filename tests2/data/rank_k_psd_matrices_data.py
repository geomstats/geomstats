from tests2.data.base_data import (
    FiberBundleTestData,
    ManifoldTestData,
    _ProjectionMixinsTestData,
)
from tests2.data.full_rank_matrices_data import FullRankMatricesTestData


class RankKPSDMatricesTestData(_ProjectionMixinsTestData, ManifoldTestData):
    tolerances = {
        "to_tangent_is_tangent": {"atol": 1e-1},
    }
    xfails = ("to_tangent_is_tangent",)


class BuresWassersteinBundleTestData(FullRankMatricesTestData, FiberBundleTestData):
    skips = (
        "integrability_tensor_vec",
        "integrability_tensor_derivative_vec",
        "horizontal_lift_vec",
        "horizontal_lift_is_horizontal",
        "tangent_riemannian_submersion_after_horizontal_lift",
    )
