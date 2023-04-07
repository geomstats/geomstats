import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.test.data import TestData
from tests2.data.base_data import LevelSetTestData, RiemannianMetricTestData


class StiefelTestData(LevelSetTestData):
    def to_grassmannian_vec_test_data(self):
        return self.generate_vec_data()


class StiefelStaticMethodsTestData(TestData):
    def to_grassmannian_test_data(self):
        p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        point1 = gs.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]) / gs.sqrt(2.0)
        batch_points = Matrices.mul(
            GeneralLinear.exp(gs.array([gs.pi * r_z / n for n in [2, 3, 4]])),
            point1,
        )
        data = [
            dict(point=point1, expected=p_xy),
            dict(point=batch_points, expected=gs.array([p_xy, p_xy, p_xy])),
        ]
        return self.generate_tests(data)


class StiefelCanonicalMetricTestData(RiemannianMetricTestData):
    skips = (
        # not implemented
        "christoffels_vec",
        "cometric_matrix_vec",
        "covariant_riemann_tensor_bianchi_identity",
        "covariant_riemann_tensor_is_interchange_symmetric",
        "covariant_riemann_tensor_is_skew_symmetric_1",
        "covariant_riemann_tensor_is_skew_symmetric_2",
        "covariant_riemann_tensor_vec",
        "curvature_vec",
        "curvature_derivative_vec",
        "directional_curvature_vec",
        "directional_curvature_derivative_vec",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
        "metric_matrix_is_spd",
        "metric_matrix_vec",
        "parallel_transport_transported_is_tangent",
        "parallel_transport_vec_with_direction",
        "parallel_transport_vec_with_end_point",
        "ricci_tensor_vec",
        "riemann_tensor_vec",
        "scalar_curvature_vec",
        "sectional_curvature_vec",
    )

    ignores_if_not_autodiff = ("inner_product_derivative_matrix_vec",)
