import geomstats.backend as gs
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class _InvariantMetricTestCaseMixins(RiemannianMetricTestCase):
    def test_invariance(
        self, point_a, point_b, tangent_vec_a, tangent_vec_b, atol=gs.atol
    ):
        """Check the inner product is invariant by translation.

        Parameters
        ----------
        point_a : array-like, shape=[..., *space.shape]
            Group element where the inner product is computed.
        point_b : array-like, shape=[..., *space.shape]
            Group element by which `point_a` is translated.
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `point_a`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `point_a`.
        atol : float
            Absolute tolerance.

        Notes
        -----
        The translation is on the left if the metric is left-invariant,
        on the right otherwise.
        """
        left = self.space.metric.left

        translation_map = self.space.tangent_translation_map(
            point_b, left=left, inverse=False
        )

        if left:
            translated_point = self.space.compose(point_b, point_a)
        else:
            translated_point = self.space.compose(point_a, point_b)

        inner_prod = self.space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, point_a
        )
        inner_prod_ = self.space.metric.inner_product(
            translation_map(tangent_vec_a),
            translation_map(tangent_vec_b),
            translated_point,
        )
        self.assertAllClose(inner_prod, inner_prod_, atol=atol)


class InvariantMetricMatrixTestCase(_InvariantMetricTestCaseMixins):
    def test_structure_constant(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected, atol=gs.atol
    ):
        """Check structure constant.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        tangent_vec_c : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        expected : array-like, shape=[...]
            Expected structure constant.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.structure_constant(
            tangent_vec_a, tangent_vec_b, tangent_vec_c
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_dual_adjoint(self, tangent_vec_a, tangent_vec_b, expected, atol=gs.atol):
        """Check dual adjoint.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        expected : array-like, shape=[..., n, n]
            Expected tangent vector at the identity.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.dual_adjoint(tangent_vec_a, tangent_vec_b)
        self.assertAllClose(res, expected, atol=atol)

    def test_connection(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
    ):
        """Check Levi-Civita connection.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Point on the group.
        expected : array-like, shape=[..., n, n]
            Expected tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.connection(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_curvature_derivative_at_identity(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        expected,
        atol=gs.atol,
    ):
        """Check curvature derivative at the identity.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        tangent_vec_c : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        tangent_vec_d : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        expected : array-like, shape=[..., n, n]
            Expected tangent vector at the identity.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.curvature_derivative_at_identity(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_dual_adjoint_structure_constant(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, atol=gs.atol
    ):
        """Check dual adjoint against structure constant.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        tangent_vec_c : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        atol : float
            Absolute tolerance.

        Notes
        -----
        Checks :math:`<ad^*_a b, c> = <[a, c], b>`.
        """
        res = self.space.metric.inner_product_at_identity(
            self.space.metric.dual_adjoint(tangent_vec_a, tangent_vec_b), tangent_vec_c
        )
        expected = self.space.metric.structure_constant(
            tangent_vec_a, tangent_vec_c, tangent_vec_b
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_connection_is_invariant(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check the connection of translated vectors is the translated connection.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at the identity.
        base_point : array-like, shape=[..., n, n]
            Point on the group.
        atol : float
            Absolute tolerance.

        Notes
        -----
        The translation is on the left if the metric is left-invariant,
        on the right otherwise.
        """
        translation_map = self.space.tangent_translation_map(
            base_point, left=self.space.metric.left
        )

        res = self.space.metric.connection(
            translation_map(tangent_vec_a), translation_map(tangent_vec_b), base_point
        )
        expected = translation_map(
            self.space.metric.connection(tangent_vec_a, tangent_vec_b)
        )
        self.assertAllClose(res, expected, atol=atol)


class InvariantMetricVectorTestCase(_InvariantMetricTestCaseMixins):
    pass


class BiInvariantMetricTestCase(RiemannianMetricTestCase):
    def test_invariance(
        self, point_a, point_b, tangent_vec_a, tangent_vec_b, atol=gs.atol
    ):
        """Check the inner product is invariant by left and right translations.

        Parameters
        ----------
        point_a : array-like, shape=[..., *space.shape]
            Group element where the inner product is computed.
        point_b : array-like, shape=[..., *space.shape]
            Group element by which `point_a` is translated.
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `point_a`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `point_a`.
        atol : float
            Absolute tolerance.
        """
        inner_prod = self.space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, point_a
        )

        for left in (True, False):
            translation_map = self.space.tangent_translation_map(
                point_b, left=left, inverse=False
            )

            if left:
                translated_point = self.space.compose(point_b, point_a)
            else:
                translated_point = self.space.compose(point_a, point_b)

            inner_prod_ = self.space.metric.inner_product(
                translation_map(tangent_vec_a),
                translation_map(tangent_vec_b),
                translated_point,
            )
            self.assertAllClose(inner_prod, inner_prod_, atol=atol)
