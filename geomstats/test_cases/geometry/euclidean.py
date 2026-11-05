import geomstats.backend as gs
from geomstats.test.assertions import assert_belongs
from geomstats.test_cases.geometry.base import VectorSpaceTestCase
from geomstats.test_cases.geometry.mixins import GroupExpTestCaseMixins
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.vectorization import get_batch_shape


class EuclideanTestCase(GroupExpTestCaseMixins, VectorSpaceTestCase):
    def test_exp_is_addition(self, tangent_vec, base_point, atol=gs.atol):
        """Check group exponential is addition.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., dim]
            Point on the Euclidean space.
        atol : float
            Absolute tolerance.
        """
        self.test_exp(tangent_vec, base_point, tangent_vec + base_point, atol)

    def test_identity_belongs(self, atol=gs.atol):
        """Check identity belongs to the Euclidean space.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        assert_belongs(self.space, self.space.identity, atol=atol)


class EuclideanMetricTestCase(RiemannianMetricTestCase):
    def test_inner_product_derivative_matrix_is_zeros(self, base_point, atol=gs.atol):
        """Check derivative of the inner-product matrix vanishes.

        The metric matrix is constant.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the Euclidean space.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.inner_product_derivative_matrix(base_point)

        batch_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.zeros(batch_shape + 3 * (self.space.dim,))
        self.assertAllClose(res, expected, atol=atol)

    def test_christoffels_are_zeros(self, base_point, atol=gs.atol):
        """Check Christoffel symbols vanish.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the Euclidean space.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.christoffels(base_point)

        batch_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.zeros(batch_shape + 3 * (self.space.dim,))
        self.assertAllClose(res, expected, atol=atol)


class CanonicalEuclideanMetricTestCase(EuclideanMetricTestCase):
    def test_cometric_matrix_is_identity(self, base_point, atol=gs.atol):
        """Check cometric matrix is the identity.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the Euclidean space.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.cometric_matrix(base_point)

        batch_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.broadcast_to(
            gs.eye(self.space.dim), batch_shape + 2 * (self.space.dim,)
        )
        self.assertAllClose(res, expected, atol=atol)
