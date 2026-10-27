import math

import pytest

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.vectorization import get_batch_shape

from .connection import ConnectionComparisonTestCase, ConnectionTestCase
from .mixins import DistTestCaseMixins


class RiemannianMetricTestCase(DistTestCaseMixins, ConnectionTestCase):
    def test_metric_matrix(self, base_point, expected, atol=gs.atol):
        res = self.space.metric.metric_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_metric_matrix_is_spd(self, base_point, atol=gs.atol):
        metric_matrix = self.space.metric.metric_matrix(base_point)

        res = SPDMatrices(n=math.prod(self.space.shape)).belongs(
            metric_matrix, atol=atol
        )
        expected_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(res, expected)

    def test_cometric_matrix(self, base_point, expected, atol=gs.atol):
        res = self.space.metric.cometric_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inner_product_derivative_matrix(self, base_point, expected, atol=gs.atol):
        res = self.space.metric.inner_product_derivative_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
    ):
        # TODO: test inner_product with itself?
        res = self.space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inner_product_is_symmetric(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check inner product is symmetric.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        inner_product_ab = self.space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        inner_product_ba = self.space.metric.inner_product(
            tangent_vec_b, tangent_vec_a, base_point
        )

        self.assertAllClose(inner_product_ab, inner_product_ba, atol=atol)

    def test_inner_coproduct(
        self, cotangent_vec_a, cotangent_vec_b, base_point, expected, atol=gs.atol
    ):
        res = self.space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_hamiltonian(self, state, expected, atol=gs.atol):
        res = self.space.metric.hamiltonian(state)
        self.assertAllClose(res, expected, atol=atol)

    def test_squared_norm(self, vector, base_point, expected, atol=gs.atol):
        res = self.space.metric.squared_norm(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_norm(self, vector, base_point, expected, atol=gs.atol):
        res = self.space.metric.norm(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_norm_is_positive(self, vector, base_point, atol=gs.atol):
        norm_ = self.space.metric.norm(vector, base_point)

        res = gs.all(norm_ > -atol)
        self.assertTrue(res)

    def test_normalize(self, vector, base_point, expected, atol=gs.atol):
        res = self.space.metric.normalize(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_normalize_is_unitary(self, vec, point, atol=gs.atol):
        normalized_vec = self.space.metric.normalize(vec, point)
        res = self.space.metric.norm(normalized_vec, point)

        batch_shape = get_batch_shape(self.space.point_ndim, vec, point)
        expected = gs.ones(batch_shape)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_dist_is_log_norm(self, point_a, point_b, atol=gs.atol):
        """Check distance is norm of log.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        log_norm = self.space.metric.norm(
            self.space.metric.log(point_b, point_a), point_a
        )
        dist_ = self.space.metric.dist(point_a, point_b)
        self.assertAllClose(dist_, log_norm, atol=atol)

    def test_diameter(self, points, expected, atol=gs.atol):
        res = self.space.metric.diameter(points)
        self.assertAllClose(res, expected, atol=atol)

    def test_normal_basis(self, basis, base_point, expected, atol=gs.atol):
        res = self.space.metric.normal_basis(basis, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_covariant_riemann_tensor(self, base_point, expected, atol=gs.atol):
        res = self.space.metric.covariant_riemann_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_covariant_riemann_tensor_is_skew_symmetric_1(
        self, base_point, atol=gs.atol
    ):
        """Check covariant riemannian tensor verifies first skew symmetry.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        # TODO: add definition of first skew symmetry in docstrings
        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        skew_symmetry_1 = covariant_metric_tensor + gs.moveaxis(
            covariant_metric_tensor, [-2, -1], [-1, -2]
        )

        res = gs.all(gs.abs(skew_symmetry_1) < atol)
        self.assertTrue(res)

    def test_covariant_riemann_tensor_is_skew_symmetric_2(
        self, base_point, atol=gs.atol
    ):
        """Check covariant riemannian tensor verifies second skew symmetry.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        # TODO: add definition of second skew symmetry in docstrings
        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        skew_symmetry_2 = covariant_metric_tensor + gs.moveaxis(
            covariant_metric_tensor, [-4, -3], [-3, -4]
        )

        res = gs.all(gs.abs(skew_symmetry_2) < atol)
        self.assertTrue(res)

    def test_covariant_riemann_tensor_bianchi_identity(self, base_point, atol=gs.atol):
        """Check covariant riemannian tensor verifies Bianchi identity.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        # TODO: add Bianchi identity in docstrings
        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        bianchi_identity = (
            covariant_metric_tensor
            + gs.moveaxis(covariant_metric_tensor, [-3, -2, -1], [-2, -1, -3])
            + gs.moveaxis(covariant_metric_tensor, [-3, -2, -1], [-1, -3, -2])
        )

        res = gs.all(gs.abs(bianchi_identity) < gs.atol)
        self.assertTrue(res)

    def test_covariant_riemann_tensor_is_interchange_symmetric(
        self, base_point, atol=gs.atol
    ):
        """Check covariant riemannian tensor verifies interchange symmetry.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        interchange_symmetry = covariant_metric_tensor - gs.moveaxis(
            covariant_metric_tensor, [-4, -3, -2, -1], [-2, -1, -4, -3]
        )

        res = gs.all(gs.abs(interchange_symmetry) < atol)
        self.assertTrue(res)

    def test_sectional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
    ):
        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_scalar_curvature(self, base_point, expected, atol=gs.atol):
        res = self.space.metric.scalar_curvature(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_parallel_transport_ivp_norm(
        self, tangent_vec, base_point, direction, atol=gs.atol
    ):
        """Check parallel transported norm is preserved.

        This is for parallel transport defined by initial value problem (ivp).

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )

        end_point = self.space.metric.exp(direction, base_point)

        self.assertAllClose(
            self.space.metric.norm(transported, end_point),
            self.space.metric.norm(tangent_vec, base_point),
            atol=atol,
        )

    def test_parallel_transport_bvp_norm(
        self, tangent_vec, base_point, end_point, atol=gs.atol
    ):
        """Check parallel transported norm is preserved.

        This is for parallel transport defined by boundary value problem (bvp).

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        self.assertAllClose(
            self.space.metric.norm(transported, end_point),
            self.space.metric.norm(tangent_vec, base_point),
            atol=atol,
        )

    def test_injectivity_radius(self, base_point, expected, atol=gs.atol):
        res = self.space.metric.injectivity_radius(base_point)
        self.assertAllClose(res, expected, atol=atol)


class RiemannianMetricComparisonTestCase(ConnectionComparisonTestCase):
    def test_metric_matrix(self, base_point, atol=gs.atol):
        res = self.space.metric.metric_matrix(base_point)
        res_ = self.other_space.metric.metric_matrix(base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_cometric_matrix(self, base_point, atol=gs.atol):
        res = self.space.metric.cometric_matrix(base_point)
        res_ = self.other_space.metric.cometric_matrix(base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_inner_product_derivative_matrix(self, base_point, atol=gs.atol):
        res = self.space.metric.inner_product_derivative_matrix(base_point)
        res_ = self.other_space.metric.inner_product_derivative_matrix(base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        res = self.space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        res_ = self.other_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    def test_inner_coproduct(
        self, cotangent_vec_a, cotangent_vec_b, base_point, atol=gs.atol
    ):
        res = self.space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        res_ = self.other_space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    def test_squared_norm(self, vector, base_point, atol=gs.atol):
        res = self.space.metric.squared_norm(vector, base_point)
        res_ = self.other_space.metric.squared_norm(vector, base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_norm(self, vector, base_point, atol=gs.atol):
        res = self.space.metric.norm(vector, base_point)
        res_ = self.other_space.metric.norm(vector, base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_normalize(self, vector, base_point, atol=gs.atol):
        res = self.space.metric.normalize(vector, base_point)
        res_ = self.other_space.metric.normalize(vector, base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_squared_dist(self, point_a, point_b, atol=gs.atol):
        res = self.space.metric.squared_dist(point_a, point_b)
        res_ = self.other_space.metric.squared_dist(point_a, point_b)
        self.assertAllClose(res, res_, atol=atol)

    def test_dist(self, point_a, point_b, atol=gs.atol):
        res = self.space.metric.dist(point_a, point_b)
        res_ = self.other_space.metric.dist(point_a, point_b)
        self.assertAllClose(res, res_, atol=atol)

    def test_covariant_riemann_tensor(self, base_point, atol=gs.atol):
        res = self.space.metric.covariant_riemann_tensor(base_point)
        res_ = self.other_space.metric.covariant_riemann_tensor(base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_sectional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        res_ = self.other_space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    def test_scalar_curvature(self, base_point, atol=gs.atol):
        res = self.space.metric.scalar_curvature(base_point)
        res_ = self.other_space.metric.scalar_curvature(base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_injectivity_radius(self, base_point, atol=gs.atol):
        res = self.space.metric.injectivity_radius(base_point)
        res_ = self.other_space.metric.injectivity_radius(base_point)
        self.assertAllClose(res, res_, atol=atol)
