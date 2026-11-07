import geomstats.backend as gs
from geomstats.test.assertions import assert_belongs, assert_is_tangent
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.mixins import GroupExpTestCaseMixins


class _LieGroupTestCaseMixins(GroupExpTestCaseMixins):
    def test_identity(self, expected, atol=gs.atol):
        """Check identity.

        Parameters
        ----------
        expected : array-like, shape=[*space.shape]
            Expected identity.
        atol : float
            Absolute tolerance.
        """
        self.assertAllClose(self.space.identity, expected, atol=atol)

    def test_compose(self, point_a, point_b, expected, atol=gs.atol):
        """Check composition.

        Parameters
        ----------
        point_a : array-like, shape=[..., *space.shape]
            Point on the group.
        point_b : array-like, shape=[..., *space.shape]
            Point on the group.
        expected : array-like, shape=[..., *space.shape]
            Expected composition.
        atol : float
            Absolute tolerance.
        """
        composed = self.space.compose(point_a, point_b)
        self.assertAllClose(composed, expected, atol=atol)

    def test_inverse(self, point, expected, atol=gs.atol):
        """Check inverse.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the group.
        expected : array-like, shape=[..., *space.shape]
            Expected inverse.
        atol : float
            Absolute tolerance.
        """
        inverse = self.space.inverse(point)
        self.assertAllClose(inverse, expected, atol=atol)

    def test_log(self, point, base_point, expected, atol=gs.atol):
        """Check group logarithm.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the group.
        base_point : array-like, shape=[..., *space.shape]
            Point on the group.
        expected : array-like, shape=[..., *space.shape]
            Expected tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        vec = self.space.log(point, base_point)
        self.assertAllClose(vec, expected, atol=atol)

    def test_tangent_translation_map(
        self, point, tangent_vec, expected, atol=gs.atol, left=True, inverse=False
    ):
        """Check tangent translation map.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the group defining the translation.
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector to translate.
        expected : array-like, shape=[..., *space.shape]
            Expected translated tangent vector.
        atol : float
            Absolute tolerance.
        left : bool
            Whether to use left or right translation.
            Optional, default: True.
        inverse : bool
            Whether to translate by the inverse of `point`.
            Optional, default: False.
        """
        res = self.space.tangent_translation_map(point, left=left, inverse=inverse)(
            tangent_vec
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_lie_bracket(
        self, tangent_vec_a, tangent_vec_b, expected, atol=gs.atol, base_point=None
    ):
        """Check Lie bracket.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        expected : array-like, shape=[..., *space.shape]
            Expected Lie bracket.
        atol : float
            Absolute tolerance.
        base_point : array-like, shape=[..., *space.shape]
            Point on the group.
            Optional, default: identity.
        """
        bracket = self.space.lie_bracket(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(bracket, expected, atol=atol)

    def test_compose_with_inverse_is_identity(self, point, atol=gs.atol):
        """Check composing with the inverse gives the identity.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the group.
        atol : float
            Absolute tolerance.
        """
        inverse = self.space.inverse(point)
        identity = gs.broadcast_to(self.space.identity, point.shape)

        identity_ = self.space.compose(point, inverse)
        self.assertAllClose(identity_, identity, atol=atol)

        identity_ = self.space.compose(inverse, point)
        self.assertAllClose(identity_, identity, atol=atol)

    def test_compose_with_identity_is_itself(self, point, atol=gs.atol):
        """Check composing with the identity gives the point itself.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the group.
        atol : float
            Absolute tolerance.
        """
        point_ = self.space.compose(point, self.space.identity)
        self.assertAllClose(point_, point, atol=atol)

        point_ = self.space.compose(self.space.identity, point)
        self.assertAllClose(point_, point, atol=atol)

    def test_exp_after_log(self, point, base_point, atol=gs.atol):
        """Check group exponential is a left inverse of the group logarithm.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the group.
        base_point : array-like, shape=[..., *space.shape]
            Point on the group.
        atol : float
            Absolute tolerance.
        """
        vec = self.space.log(point, base_point)
        point_ = self.space.exp(vec, base_point)

        self.assertAllClose(point_, point, atol=atol)

    def test_log_after_exp(self, tangent_vec, base_point, atol=gs.atol):
        """Check group logarithm is a left inverse of the group exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the group.
        atol : float
            Absolute tolerance.
        """
        point = self.space.exp(tangent_vec, base_point)
        tangent_vec_ = self.space.log(point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_to_tangent_at_identity_belongs_to_lie_algebra(
        self, tangent_vec, atol=gs.atol
    ):
        """Check tangent vectors at the identity belong to the Lie algebra.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        atol : float
            Absolute tolerance.
        """
        assert_belongs(self.space.lie_algebra, tangent_vec, atol=atol)

    def test_tangent_translation_map_is_tangent(
        self, tangent_vec, base_point, point, atol=gs.atol, left=True
    ):
        """Check tangent translation map gives a tangent vector.

        Translating `tangent_vec` at `base_point` by `point` gives a tangent
        vector at the composition of `point` and `base_point`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the group.
        point : array-like, shape=[..., *space.shape]
            Point on the group defining the translation.
        atol : float
            Absolute tolerance.
        left : bool
            Whether to use left or right translation.
            Optional, default: True.
        """
        translate = self.space.tangent_translation_map(point, left=left, inverse=False)

        if left:
            end_point = self.space.compose(point, base_point)
        else:
            end_point = self.space.compose(base_point, point)

        assert_is_tangent(self.space, translate(tangent_vec), end_point, atol=atol)

    def test_lie_bracket_belongs(self, tangent_vec_a, tangent_vec_b, atol=gs.atol):
        """Check Lie bracket at the identity belongs to the Lie algebra.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        atol : float
            Absolute tolerance.
        """
        bracket = self.space.lie_bracket(tangent_vec_a, tangent_vec_b)
        assert_belongs(self.space.lie_algebra, bracket, atol=atol)

    def test_lie_bracket_antisymmetry(
        self, tangent_vec_a, tangent_vec_b, atol=gs.atol
    ):
        """Check Lie bracket is antisymmetric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        atol : float
            Absolute tolerance.
        """
        bracket = self.space.lie_bracket(tangent_vec_a, tangent_vec_b)
        bracket_ = -self.space.lie_bracket(tangent_vec_b, tangent_vec_a)
        self.assertAllClose(bracket, bracket_, atol=atol)

    def test_lie_bracket_jacobi_identity(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, atol=gs.atol
    ):
        """Check Lie bracket satisfies the Jacobi identity.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        tangent_vec_c : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        atol : float
            Absolute tolerance.
        """
        lie_bracket = self.space.lie_bracket

        lhs = lie_bracket(lie_bracket(tangent_vec_a, tangent_vec_b), tangent_vec_c)
        rhs = lie_bracket(
            tangent_vec_a, lie_bracket(tangent_vec_b, tangent_vec_c)
        ) - lie_bracket(tangent_vec_b, lie_bracket(tangent_vec_a, tangent_vec_c))

        self.assertAllClose(lhs, rhs, atol=atol)

    def test_lie_bracket_bilinearity(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, atol=gs.atol
    ):
        """Check Lie bracket is bilinear.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        tangent_vec_c : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        atol : float
            Absolute tolerance.
        """
        lie_bracket = self.space.lie_bracket
        a, b = gs.random.rand(2)

        lhs = lie_bracket(a * tangent_vec_a + b * tangent_vec_b, tangent_vec_c)
        rhs = a * lie_bracket(tangent_vec_a, tangent_vec_c) + b * lie_bracket(
            tangent_vec_b, tangent_vec_c
        )
        self.assertAllClose(lhs, rhs, atol=atol)

        lhs = lie_bracket(tangent_vec_c, a * tangent_vec_a + b * tangent_vec_b)
        rhs = a * lie_bracket(tangent_vec_c, tangent_vec_a) + b * lie_bracket(
            tangent_vec_c, tangent_vec_b
        )
        self.assertAllClose(lhs, rhs, atol=atol)


class MatrixLieGroupTestCase(_LieGroupTestCaseMixins, ManifoldTestCase):
    pass


class LieGroupTestCase(_LieGroupTestCaseMixins, ManifoldTestCase):
    def test_jacobian_translation(self, point, expected, atol=gs.atol, left=True):
        """Check Jacobian of the translation.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the group defining the translation.
        expected : array-like, shape=[..., dim, dim]
            Expected Jacobian.
        atol : float
            Absolute tolerance.
        left : bool
            Whether to use left or right translation.
            Optional, default: True.
        """
        res = self.space.jacobian_translation(point, left=left)
        self.assertAllClose(res, expected, atol=atol)

    def test_exp_from_identity(self, tangent_vec, expected, atol=gs.atol):
        """Check group exponential from the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        expected : array-like, shape=[..., *space.shape]
            Expected point on the group.
        atol : float
            Absolute tolerance.
        """
        res = self.space.exp_from_identity(tangent_vec)
        self.assertAllClose(res, expected, atol=atol)

    def test_log_from_identity(self, point, expected, atol=gs.atol):
        """Check group logarithm from the identity.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the group.
        expected : array-like, shape=[..., *space.shape]
            Expected tangent vector at the identity.
        atol : float
            Absolute tolerance.
        """
        vec = self.space.log_from_identity(point)
        self.assertAllClose(vec, expected, atol=atol)

    def test_exp_from_identity_after_log_from_identity(self, point, atol=gs.atol):
        """Check exponential from identity is a left inverse of the logarithm.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the group.
        atol : float
            Absolute tolerance.
        """
        tangent_vec = self.space.log_from_identity(point)
        point_ = self.space.exp_from_identity(tangent_vec)

        self.assertAllClose(point_, point, atol=atol)

    def test_log_from_identity_after_exp_from_identity(
        self, tangent_vec, atol=gs.atol
    ):
        """Check logarithm from identity is a left inverse of the exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at the identity.
        atol : float
            Absolute tolerance.
        """
        point = self.space.exp_from_identity(tangent_vec)
        tangent_vec_ = self.space.log_from_identity(point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)
