"""Invariant metric test data.

Given the slowness of these tests, I've decided to separate the data that
computes a method at identity (because the normal tests will call those
methods). Just use such data if interested in more detailed tests (notice
the tests are already defined).
"""

import geomstats.backend as gs
from geomstats.test.data import TestData

from .riemannian_metric import RiemannianMetricTestData


class _InvariantMetricAtIdentityMixinsTestData(TestData):
    def inner_product_at_identity_vec_test_data(self):
        return self.generate_vec_data()


class _InvariantMetricMixinsTestData(RiemannianMetricTestData):
    def invariance_test_data(self):
        return self.generate_random_data()


class InvariantMetricMatrixAtIdentityTestData(_InvariantMetricAtIdentityMixinsTestData):
    def connection_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def curvature_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def sectional_curvature_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def curvature_derivative_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def exp_at_identity_vec_test_data(self):
        return self.generate_vec_data()

    def log_after_exp_at_identity_test_data(self):
        return self.generate_random_data()

    def exp_after_log_at_identity_test_data(self):
        return self.generate_random_data()


class InvariantMetricMatrixTestData(_InvariantMetricMixinsTestData):
    def structure_constant_vec_test_data(self):
        return self.generate_vec_data()

    def dual_adjoint_vec_test_data(self):
        return self.generate_vec_data()

    def dual_adjoint_structure_constant_test_data(self):
        return self.generate_random_data()

    def connection_vec_test_data(self):
        return self.generate_vec_data()


class InvariantMetricMatrixSOTestData(InvariantMetricMatrixTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    _atol = 1e-3
    tolerances = {
        "dist_triangle_inequality": {"atol": _atol},
        "dist_is_symmetric": {"atol": _atol * 10},
        "squared_dist_is_symmetric": {"atol": _atol * 10},
        "exp_after_log": {"atol": _atol},
        "log_after_exp": {"atol": _atol},
        "geodesic_boundary_points": {"atol": _atol},
        "geodesic_bvp_reverse": {"atol": _atol},
        "geodesic_ivp_belongs": {"atol": _atol},
        "geodesic_bvp_belongs": {"atol": _atol},
        "parallel_transport_bvp_transported_is_tangent": {"atol": _atol},
        "parallel_transport_ivp_transported_is_tangent": {"atol": _atol},
        "parallel_transport_bvp_norm": {"atol": _atol},
        "parallel_transport_bvp_vec": {"atol": _atol},
    }
    xfails = ("geodesic_bvp_belongs", "geodesic_bvp_reverse")


class InvariantMetricMatrixSO3TestData(TestData):
    def connection_translation_map_test_data(self):
        return self.generate_random_data()

    def connection_smoke_test_data(self):
        return self.generate_tests([dict()])

    def sectional_curvature_smoke_test_data(self):
        return self.generate_tests([dict()])

    def curvature_smoke_test_data(self):
        return self.generate_tests([dict()])

    def structure_constant_smoke_test_data(self):
        return self.generate_tests([dict()])

    def curvature_derivative_at_identity_smoke_test_data(self):
        return self.generate_tests([dict()])

    def inner_product_from_vec_representation_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([1.0, 0, 2.0]),
                tangent_vec_b=gs.array([1.0, 0, 0.5]),
                expected=4.0,
            ),
            dict(
                tangent_vec_a=gs.array([[1.0, 0, 2.0], [0, 3.0, 5.0]]),
                tangent_vec_b=gs.array([1.0, 0, 0.5]),
                expected=gs.array([4.0, 5.0]),
            ),
        ]

        return self.generate_tests(data)


class InvariantMetricMatrixSETestData(InvariantMetricMatrixTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
    skip_vec = True

    _atol = 1e-2
    tolerances = {
        "dist_vec": {"atol": _atol},
        "dist_is_log_norm": {"atol": _atol},
        "dist_point_to_itself_is_zero": {"atol": _atol},
        "dist_is_symmetric": {"atol": _atol},
        "dist_triangle_inequality": {"atol": _atol},
        "exp_belongs": {"atol": _atol},
        "exp_after_log": {"atol": _atol},
        "log_after_exp": {"atol": _atol},
        "log_vec": {"atol": _atol},
        "geodesic_bvp_vec": {"atol": _atol},
        "geodesic_boundary_points": {"atol": _atol},
        "geodesic_bvp_reverse": {"atol": _atol},
        "geodesic_bvp_belongs": {"atol": _atol},
        "parallel_transport_bvp_transported_is_tangent": {"atol": _atol},
        "parallel_transport_ivp_transported_is_tangent": {"atol": _atol},
        "parallel_transport_bvp_norm": {"atol": _atol},
        "parallel_transport_bvp_vec": {"atol": _atol},
        "squared_dist_is_symmetric": {"atol": _atol},
        "squared_dist_vec": {"atol": _atol},
    }
    xfails = ("geodesic_bvp_belongs", "geodesic_bvp_reverse")


class InvariantMetricVectorAtIdentityTestData(_InvariantMetricAtIdentityMixinsTestData):
    def left_exp_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def left_log_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def left_exp_from_identity_after_left_log_from_identity_test_data(self):
        return self.generate_random_data()

    def left_log_from_identity_after_left_exp_from_identity_test_data(self):
        return self.generate_random_data()

    def exp_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def log_from_identity_vec_test_data(self):
        return self.generate_vec_data()

    def exp_from_identity_after_log_from_identity_test_data(self):
        return self.generate_random_data()

    def log_from_identity_after_exp_from_identity_test_data(self):
        return self.generate_random_data()


class InvariantMetricVectorTestData(_InvariantMetricMixinsTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
    skip_vec = True
    skips = ("invariance",)


class BiInvariantMetricTestData(RiemannianMetricTestData):
    def invariance_test_data(self):
        return self.generate_random_data()


class BiInvariantMetricVectorsSOTestData(BiInvariantMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
    skips = (
        "parallel_transport_bvp_norm",
        "parallel_transport_ivp_norm",
        "parallel_transport_bvp_transported_is_tangent",
        "parallel_transport_ivp_transported_is_tangent",
        "parallel_transport_bvp_vec",
        "parallel_transport_ivp_vec",
        "invariance",
    )


class BiInvariantMetricMatrixTestData(BiInvariantMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
