import math

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.assertions import assert_belongs, assert_same_result
from geomstats.vectorization import get_batch_shape

from .connection import ConnectionComparisonTestCase, ConnectionTestCase
from .mixins import DistTestCaseMixins


class RiemannianMetricTestCase(DistTestCaseMixins, ConnectionTestCase):
    def test_metric_matrix(self, base_point, expected, atol=gs.atol):
        """Check metric matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., dim, dim]
            Expected inner-product matrix.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.metric_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_cometric_matrix(self, base_point, expected, atol=gs.atol):
        """Check cometric matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., dim, dim]
            Expected inverse of the inner-product matrix.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.cometric_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inner_product_derivative_matrix(self, base_point, expected, atol=gs.atol):
        """Check derivative of the inner-product matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., dim, dim, dim]
            Expected derivative of the inner-product matrix.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.inner_product_derivative_matrix(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
    ):
        """Check inner product.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[...,]
            Expected inner product.
        atol : float
            Absolute tolerance.
        """
        # TODO: test inner_product with itself?
        res = self.space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inner_coproduct(
        self, cotangent_vec_a, cotangent_vec_b, base_point, expected, atol=gs.atol
    ):
        """Check inner coproduct.

        Parameters
        ----------
        cotangent_vec_a : array-like, shape=[..., dim]
            Cotangent vector at `base_point`.
        cotangent_vec_b : array-like, shape=[..., dim]
            Cotangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[...,]
            Expected inner coproduct.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_hamiltonian(self, state, expected, atol=gs.atol):
        """Check Hamiltonian.

        Parameters
        ----------
        state : tuple of arrays
            Position and momentum variables.
        expected : array-like, shape=[...,]
            Expected Hamiltonian energy.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.hamiltonian(state)
        self.assertAllClose(res, expected, atol=atol)

    def test_squared_norm(self, vector, base_point, expected, atol=gs.atol):
        """Check squared norm.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[...,]
            Expected squared norm.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.squared_norm(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_norm(self, vector, base_point, expected, atol=gs.atol):
        """Check norm.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[...,]
            Expected norm.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.norm(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_normalize(self, vector, base_point, expected, atol=gs.atol):
        """Check normalization.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected unit tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.normalize(vector, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_diameter(self, points, expected, atol=gs.atol):
        """Check diameter.

        Parameters
        ----------
        points : array-like, shape=[n_points, *space.shape]
            Points on the manifold.
        expected : float
            Expected distance between the two farthest points.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.diameter(points)
        self.assertAllClose(res, expected, atol=atol)

    def test_normal_basis(self, basis, base_point, expected, atol=gs.atol):
        """Check normal basis.

        Parameters
        ----------
        basis : array-like, shape=[dim, *space.shape]
            Basis of the tangent space at `base_point`.
        base_point : array-like, shape=[*space.shape]
            Point on the manifold.
        expected : array-like, shape=[dim, *space.shape]
            Expected orthonormal basis.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.normal_basis(basis, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_covariant_riemann_tensor(self, base_point, expected, atol=gs.atol):
        """Check covariant Riemann tensor.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., dim, dim, dim, dim]
            Expected covariant Riemann tensor.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.covariant_riemann_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_sectional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
    ):
        """Check sectional curvature.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[...,]
            Expected sectional curvature.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_scalar_curvature(self, base_point, expected, atol=gs.atol):
        """Check scalar curvature.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[...,]
            Expected scalar curvature.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.scalar_curvature(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_injectivity_radius(self, base_point, expected, atol=gs.atol):
        """Check injectivity radius.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[...,]
            Expected injectivity radius.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.injectivity_radius(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_metric_matrix_is_spd(self, base_point, atol=gs.atol):
        """Check metric matrix is symmetric positive definite.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        metric_matrix = self.space.metric.metric_matrix(base_point)

        assert_belongs(
            SPDMatrices(n=math.prod(self.space.shape)), metric_matrix, atol=atol
        )

    def test_inner_product_is_symmetric(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check inner product is symmetric.

        For a complex inner product, checks it is Hermitian.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        inner_product_ab = self.space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        inner_product_ba = self.space.metric.inner_product(
            tangent_vec_b, tangent_vec_a, base_point
        )

        self.assertAllClose(inner_product_ab, gs.conj(inner_product_ba), atol=atol)

    def test_norm_is_positive(self, vector, base_point, atol=gs.atol):
        """Check norm is non-negative.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        norm_ = self.space.metric.norm(vector, base_point)

        res = gs.all(norm_ > -atol)
        self.assertTrue(res)

    def test_normalize_is_unitary(self, vector, base_point, atol=gs.atol):
        """Check normalized vector has unit norm.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        normalized_vec = self.space.metric.normalize(vector, base_point)
        res = self.space.metric.norm(normalized_vec, base_point)

        batch_shape = get_batch_shape(self.space.point_ndim, vector, base_point)
        expected = gs.ones(batch_shape)
        self.assertAllClose(res, expected, atol=atol)

    def test_dist_is_log_norm(self, point_a, point_b, atol=gs.atol):
        """Check distance is norm of log.

        Parameters
        ----------
        point_a : array-like, shape=[..., *space.shape]
            Point on the manifold.
        point_b : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        log_norm = self.space.metric.norm(
            self.space.metric.log(point_b, point_a), point_a
        )
        dist_ = self.space.metric.dist(point_a, point_b)
        self.assertAllClose(dist_, log_norm, atol=atol)

    def test_covariant_riemann_tensor_is_skew_symmetric_1(
        self, base_point, atol=gs.atol
    ):
        """Check covariant Riemann tensor verifies first skew symmetry.

        That is, :math:`R_{abcd} = -R_{abdc}`.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        skew_symmetry_1 = covariant_metric_tensor + gs.moveaxis(
            covariant_metric_tensor, [-2, -1], [-1, -2]
        )

        res = gs.all(gs.abs(skew_symmetry_1) < atol)
        self.assertTrue(res)

    def test_covariant_riemann_tensor_is_skew_symmetric_2(
        self, base_point, atol=gs.atol
    ):
        """Check covariant Riemann tensor verifies second skew symmetry.

        That is, :math:`R_{abcd} = -R_{bacd}`.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        skew_symmetry_2 = covariant_metric_tensor + gs.moveaxis(
            covariant_metric_tensor, [-4, -3], [-3, -4]
        )

        res = gs.all(gs.abs(skew_symmetry_2) < atol)
        self.assertTrue(res)

    def test_covariant_riemann_tensor_bianchi_identity(self, base_point, atol=gs.atol):
        """Check covariant Riemann tensor verifies the first Bianchi identity.

        That is, :math:`R_{abcd} + R_{acdb} + R_{adbc} = 0`.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        bianchi_identity = (
            covariant_metric_tensor
            + gs.moveaxis(covariant_metric_tensor, [-3, -2, -1], [-2, -1, -3])
            + gs.moveaxis(covariant_metric_tensor, [-3, -2, -1], [-1, -3, -2])
        )

        res = gs.all(gs.abs(bianchi_identity) < atol)
        self.assertTrue(res)

    def test_covariant_riemann_tensor_is_interchange_symmetric(
        self, base_point, atol=gs.atol
    ):
        """Check covariant Riemann tensor verifies interchange symmetry.

        That is, :math:`R_{abcd} = R_{cdab}`.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        covariant_metric_tensor = self.space.metric.covariant_riemann_tensor(base_point)
        interchange_symmetry = covariant_metric_tensor - gs.moveaxis(
            covariant_metric_tensor, [-4, -3, -2, -1], [-2, -1, -4, -3]
        )

        res = gs.all(gs.abs(interchange_symmetry) < atol)
        self.assertTrue(res)

    def test_parallel_transport_ivp_norm(
        self, tangent_vec, base_point, direction, atol=gs.atol
    ):
        """Check parallel transport along a direction preserves the norm.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        direction : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` along which to transport.
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
        """Check parallel transport to an end point preserves the norm.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        end_point : array-like, shape=[..., *space.shape]
            Point on the manifold to transport to.
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


class RiemannianMetricComparisonTestCase(ConnectionComparisonTestCase):
    """Test case comparing the Riemannian metrics of two spaces.

    Each test checks a method gives the same result for `space` and
    `other_space`.
    """

    def test_metric_matrix(self, base_point, atol=gs.atol):
        """Check metric matrix is the same for both spaces.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.metric_matrix(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_cometric_matrix(self, base_point, atol=gs.atol):
        """Check cometric matrix is the same for both spaces.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.cometric_matrix(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_inner_product_derivative_matrix(self, base_point, atol=gs.atol):
        """Check derivative of the inner-product matrix is the same for both spaces.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.inner_product_derivative_matrix(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check inner product is the same for both spaces.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.inner_product(
                tangent_vec_a, tangent_vec_b, base_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_inner_coproduct(
        self, cotangent_vec_a, cotangent_vec_b, base_point, atol=gs.atol
    ):
        """Check inner coproduct is the same for both spaces.

        Parameters
        ----------
        cotangent_vec_a : array-like, shape=[..., dim]
            Cotangent vector at `base_point`.
        cotangent_vec_b : array-like, shape=[..., dim]
            Cotangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.inner_coproduct(
                cotangent_vec_a, cotangent_vec_b, base_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_squared_norm(self, vector, base_point, atol=gs.atol):
        """Check squared norm is the same for both spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.squared_norm(vector, base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_norm(self, vector, base_point, atol=gs.atol):
        """Check norm is the same for both spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.norm(vector, base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_normalize(self, vector, base_point, atol=gs.atol):
        """Check normalization is the same for both spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.normalize(vector, base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_squared_dist(self, point_a, point_b, atol=gs.atol):
        """Check squared distance is the same for both spaces.

        Parameters
        ----------
        point_a : array-like, shape=[..., *space.shape]
            Point on the manifold.
        point_b : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.squared_dist(point_a, point_b),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_dist(self, point_a, point_b, atol=gs.atol):
        """Check distance is the same for both spaces.

        Parameters
        ----------
        point_a : array-like, shape=[..., *space.shape]
            Point on the manifold.
        point_b : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.dist(point_a, point_b),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_covariant_riemann_tensor(self, base_point, atol=gs.atol):
        """Check covariant Riemann tensor is the same for both spaces.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.covariant_riemann_tensor(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_sectional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check sectional curvature is the same for both spaces.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.sectional_curvature(
                tangent_vec_a, tangent_vec_b, base_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_scalar_curvature(self, base_point, atol=gs.atol):
        """Check scalar curvature is the same for both spaces.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.scalar_curvature(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_injectivity_radius(self, base_point, atol=gs.atol):
        """Check injectivity radius is the same for both spaces.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.injectivity_radius(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )
