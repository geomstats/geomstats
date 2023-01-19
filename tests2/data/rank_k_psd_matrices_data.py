from tests2.data.base_data import ManifoldTestData, _ProjectionMixinsTestData


class RankKPSDMatricesTestData(_ProjectionMixinsTestData, ManifoldTestData):
    tolerances = {
        "to_tangent_is_tangent": {"atol": 1e-1},
    }
    xfails = ("to_tangent_is_tangent",)
