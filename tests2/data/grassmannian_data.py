import geomstats.backend as gs
from geomstats.test.data import TestData
from tests2.data.base_data import LevelSetTestData
from tests2.data.matrices_data import MatricesMetricTestData


class GrassmannianTestData(LevelSetTestData):
    pass


class Grassmannian32TestData(TestData):
    def belongs_test_data(self):
        p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        p_yz = gs.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p_xz = gs.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        data = [
            dict(point=p_xy, expected=gs.array(True)),
            dict(point=gs.stack([p_yz, p_xz]), expected=gs.array([True, True])),
        ]
        return self.generate_tests(data)


class GrassmannianCanonicalMetricTestData(MatricesMetricTestData):
    skips = (
        # not implemented
        "covariant_riemann_tensor_bianchi_identity",
        "covariant_riemann_tensor_is_interchange_symmetric",
        "covariant_riemann_tensor_is_skew_symmetric_1",
        "covariant_riemann_tensor_is_skew_symmetric_2",
        "covariant_riemann_tensor_vec",
        "curvature_vec",
        "curvature_derivative_vec",
        "directional_curvature_vec",
        "directional_curvature_derivative_vec",
        "injectivity_radius_vec",
        "ricci_tensor_vec",
        "riemann_tensor_vec",
        "scalar_curvature_vec",
        "sectional_curvature_vec",
    )
