"""Left- and right- invariant metrics that exist on Lie groups.

Lead authors: Nicolas Guigui and Nina Miolane.
"""

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.integrator import integrate
from geomstats.numerics.bvp import ScipySolveBVP
from geomstats.numerics.geodesic import ExpODESolver, LogODESolver
from geomstats.numerics.ivp import ScipySolveIVP
from geomstats.vectorization import repeat_out

EPSILON = 1e-6


class InvariantMetricMatrixExpODESolver(ExpODESolver):
    """An exp solver adapted to _InvariantMetricMatrix."""

    def exp(self, tangent_vec, base_point):
        r"""Compute Riemannian exponential of tan. vector wrt to base point.

        If :math:`\gamma` is a geodesic, then it satisfies the
        Euler-Poincare equation [Kolev]_:

        .. math::

            \dot{\gamma}(t) = (dL_{\gamma(t)}) X(t)
            \dot{X}(t) = ad^*_{X(t)}X(t)

        where :math:`ad^*` is the dual adjoint map with respect to the
        metric. For a right-invariant metric, :math:`dR` is used instead of
        :math:`dL` and :math:`ad^*` is replaced by :math:`-ad^*`. The
        exponential map is approximated by numerical integration
        of this equation, with initial conditions :math:`\dot{\gamma}(0)`
        given by the argument `tangent_vec` and :math:`\gamma(0)` by
        `base_point`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., n, n]
            Point in the group.
            Optional, defaults to identity if None.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Point in the group equal to the Riemannian exponential
            of tangent_vec at the base point.

        References
        ----------
        https://en.wikipedia.org/wiki/Runge–Kutta_methods

        .. [Kolev]   Kolev, Boris. “Lie Groups and Mechanics: An Introduction.”
                     Journal of Nonlinear Mathematical Physics 11, no. 4, 2004:
                     480–98. https://doi.org/10.2991/jnmp.2004.11.4.5.
        """
        base_point, left_angular_vel = self._space.metric._pre_exp(
            tangent_vec, base_point
        )
        return super().exp(left_angular_vel, base_point)

    def geodesic_ivp(self, tangent_vec, base_point):
        """Geodesic curve for initial value problem."""
        base_point, left_angular_vel = self._space.metric._pre_exp(
            tangent_vec, base_point
        )
        return super().geodesic_ivp(left_angular_vel, base_point)


class InvariantMetricMatrixLogODESolver(LogODESolver):
    """A log solver adapted to _InvariantMetricMatrix."""

    def log(self, point, base_point):
        """Logarithm map."""
        left_angular_vel = super().log(point, base_point)
        return self._space.to_tangent(
            self._space.tangent_translation_map(
                base_point, left=self._space.metric.left
            )(left_angular_vel),
            base_point,
        )


class _InvariantMetricMatrix(RiemannianMetric):
    """Class for invariant metrics on Matrix Lie groups.

    This class supports both left and right invariant metrics
    which exist on Lie groups.

    Parameters
    ----------
    space : LieGroup
        Group to equip with the invariant metric.
    metric_mat_at_identity : array-like, shape=[dim, dim]
        Matrix that defines the metric at identity.
        Optional, defaults to identity matrix if None.
    left : bool
        Whether to use a left or right invariant metric.
        Optional, default: True.

    References
    ----------
    .. [Milnor]    Milnor, John. “Curvatures of Left Invariant Metrics on Lie
                   Groups.” Advances in Mathematics 21, no. 3, 1976:
                   293–329. https://doi.org/10.1016/S0001-8708(76)80002-3.
    .. [Kolev]     Kolev, Boris. “Lie Groups and Mechanics: An Introduction.”
                   Journal of Nonlinear Mathematical Physics 11, no. 4, 2004:
                    480–98. https://doi.org/10.2991/jnmp.2004.11.4.5.
    .. [Gallier]   Gallier, Jean, and Jocelyn Quaintance. Differential Geometry
                   and Lie Groups: A Computational Perspective.
                   Geonger International Publishing, 2020.
                   https://doi.org/10.1007/978-3-030-46040-2.
    """

    def __init__(self, space, metric_mat_at_identity=None, left=True):
        super().__init__(space=space)
        if metric_mat_at_identity is None:
            metric_mat_at_identity = gs.eye(space.dim)
        self.metric_mat_at_identity = metric_mat_at_identity
        self.left = left

        self._instantiate_solvers()

    def _instantiate_solvers(self):
        self.log_solver = InvariantMetricMatrixLogODESolver(
            self._space, n_nodes=100, use_jac=False, integrator=ScipySolveBVP(tol=1e-8)
        )
        self.exp_solver = InvariantMetricMatrixExpODESolver(
            self._space, integrator=ScipySolveIVP(atol=1e-8, point_ndim=2)
        )

    @property
    def reshaped_metric_matrix(self):
        """Diagonal metric matrix reshaped to a symmetric matrix of size n.

        Reshape a diagonal metric matrix of size `dim x dim` into a symmetric
        matrix of size `n x n` where :math:`dim= n (n -1) / 2` is the
        dimension of the space of skew symmetric matrices. The
        non-diagonal coefficients in the output matrix correspond to the
        basis matrices of this space. The diagonal is filled with ones.
        This useful to compute a matrix inner product.

        Returns
        -------
        symmetric_matrix : array-like, shape=[n, n]
            Symmetric matrix.
        """
        if Matrices.is_diagonal(self.metric_mat_at_identity):
            metric_coeffs = gs.diagonal(self.metric_mat_at_identity)
            metric_mat = gs.abs(
                self._space.lie_algebra.matrix_representation(metric_coeffs)
            )
            return metric_mat
        raise ValueError("This is only possible for a diagonal matrix")

    def inner_product_at_identity(self, tangent_vec_a, tangent_vec_b):
        """Compute inner product at tangent space at identity.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            First tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Second tangent vector at identity.

        Returns
        -------
        inner_prod : array-like, shape=[...]
            Inner-product of the two tangent vectors.
        """
        tan_b = tangent_vec_b
        metric_mat = self.metric_mat_at_identity
        if Matrices.is_diagonal(metric_mat) and self._space.lie_algebra is not None:
            tan_b = tangent_vec_b * self.reshaped_metric_matrix
        return Matrices.frobenius_product(tangent_vec_a, tan_b)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute inner product of two vectors in tangent space at base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            First tangent vector at base_point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Second tangent vector at base_point.
        base_point : array-like, shape=[..., n, n]
            Point in the group.
            Optional, defaults to identity if None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        if base_point is None:
            return self.inner_product_at_identity(tangent_vec_a, tangent_vec_b)

        tangent_translation = self._space.tangent_translation_map(
            base_point, left=self.left, inverse=True
        )
        tangent_vec_a_at_id = tangent_translation(tangent_vec_a)
        tangent_vec_b_at_id = tangent_translation(tangent_vec_b)
        return self.inner_product_at_identity(tangent_vec_a_at_id, tangent_vec_b_at_id)

    def structure_constant(self, tangent_vec_a, tangent_vec_b, tangent_vec_c):
        r"""Compute the structure constant of the metric.

        For three tangent vectors :math:`x, y, z` at identity,
        compute :math:`<[x,y], z>`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_c : array-like, shape=[..., n, n]
            Tangent vector at identity.

        Returns
        -------
        structure_constant : array-like, shape=[...,]
        """
        bracket = Matrices.bracket(tangent_vec_a, tangent_vec_b)
        return self.inner_product_at_identity(bracket, tangent_vec_c)

    def dual_adjoint(self, tangent_vec_a, tangent_vec_b):
        r"""Compute the metric dual adjoint map.

        For two tangent vectors at identity :math:`x,y`, this corresponds to
        the vector :math:`a` such that
        :math:`\forall z, <[x,z], y > = <a, z>`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at identity.

        Returns
        -------
        ad_star : array-like, shape=[..., n, n]
            Tangent vector at identity corresponding to :math:`ad_x^*(y)`.

        References
        ----------
        .. [Kolev] Kolev, Boris. “Lie Groups and Mechanics: An Introduction.”
                   Journal of Nonlinear Mathematical Physics 11, no. 4, 2004:
                   480–98. https://doi.org/10.2991/jnmp.2004.11.4.5.

        .. [Gallier]   Gallier, Jean, and Jocelyn Quaintance. Differential
                       Geometry and Lie Groups: A Computational Perspective.
                       Geonger International Publishing, 2020.
                       https://doi.org/10.1007/978-3-030-46040-2.
        """
        basis = self.normal_basis(self._space.lie_algebra.basis)
        return -gs.einsum(
            "i...,ijk->...jk",
            gs.array(
                [
                    self.structure_constant(tan, tangent_vec_a, tangent_vec_b)
                    for tan in basis
                ]
            ),
            gs.array(basis),
        )

    def connection_at_identity(self, tangent_vec_a, tangent_vec_b):
        r"""Compute the Levi-Civita connection at identity.

        For two tangent vectors at identity :math:`x,y`, one can associate
        left (respectively right) invariant vector fields :math:`\tilde{x},
        \tilde{y}`. Then the vector :math:`(\nabla_\tilde{x}(\tilde{x}))_{
        Id}` is computed using the lie bracket and the dual adjoint map. This
        is a bilinear map that characterizes the connection [Gallier]_.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at identity.

        Returns
        -------
        nabla : array-like, shape=[..., n, n]
            Tangent vector at identity.

        References
        ----------
        .. [Gallier]   Gallier, Jean, and Jocelyn Quaintance. Differential
                       Geometry and Lie Groups: A Computational Perspective.
                       Geonger International Publishing, 2020.
                       https://doi.org/10.1007/978-3-030-46040-2.
        """
        sign = 1.0 if self.left else -1.0
        return (
            sign
            / 2
            * (
                Matrices.bracket(tangent_vec_a, tangent_vec_b)
                - self.dual_adjoint(tangent_vec_a, tangent_vec_b)
                - self.dual_adjoint(tangent_vec_b, tangent_vec_a)
            )
        )

    def connection(self, tangent_vec_a, tangent_vec_b, base_point=None):
        r"""Compute the Levi-Civita connection of invariant vector fields.

        For two tangent vectors at a base point :math:`p, x,y`, one can
        associate left (respectively right) invariant vector fields :math:
        `\tilde{x}, \tilde{y}`. Then the vector :math:`(\nabla_\tilde{x}(
        \tilde{x}))_{p}` is computed using the invariance of the connection
        and its value at identity [Gallier]_.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Point in the group.

        Returns
        -------
        nabla : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.

        References
        ----------
        .. [Gallier]   Gallier, Jean, and Jocelyn Quaintance. Differential
                       Geometry and Lie Groups: A Computational Perspective.
                       Geonger International Publishing, 2020.
                       https://doi.org/10.1007/978-3-030-46040-2.
        """
        if base_point is None:
            return self.connection_at_identity(tangent_vec_a, tangent_vec_b)
        translation_map = self._space.tangent_translation_map(
            base_point, left=self.left, inverse=True
        )
        tan_a_at_id = translation_map(tangent_vec_a)
        tan_b_at_id = translation_map(tangent_vec_b)

        translation_map = self._space.tangent_translation_map(
            base_point, left=self.left, inverse=False
        )

        value_at_id = self.connection_at_identity(tan_a_at_id, tan_b_at_id)
        return translation_map(value_at_id)

    def curvature_at_identity(self, tangent_vec_a, tangent_vec_b, tangent_vec_c):
        r"""Compute the curvature at identity.

        For three tangent vectors at identity :math:`x,y,z`,
        the curvature is defined by
        :math:`R(x, y)z = \nabla_{[x,y]}z
        - \nabla_x\nabla_y z + \nabla_y\nabla_x z`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_c : array-like, shape=[..., n, n]
            Tangent vector at identity.

        Returns
        -------
        curvature : array-like, shape=[..., n, n]
            Tangent vector at identity.
        """
        bracket = Matrices.bracket(tangent_vec_a, tangent_vec_b)
        bracket_term = self.connection_at_identity(bracket, tangent_vec_c)

        left_term = self.connection_at_identity(
            tangent_vec_a, self.connection_at_identity(tangent_vec_b, tangent_vec_c)
        )

        right_term = self.connection_at_identity(
            tangent_vec_b, self.connection_at_identity(tangent_vec_a, tangent_vec_c)
        )

        return bracket_term - left_term + right_term

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point=None):
        r"""Compute the curvature.

        For three tangent vectors at a base point :math:`x,y,z`,
        the curvature is defined by
        :math:`R(x, y)z = \nabla_{[x,y]}z
        - \nabla_x\nabla_y z + \nabla_y\nabla_x z`. It is computed using
        the invariance of the connection and its value at identity.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        base_point :  array-like, shape=[..., n, n]
            Point on the group. Optional, default is the identity.

        Returns
        -------
        curvature : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        """
        if base_point is None:
            return self.curvature_at_identity(
                tangent_vec_a, tangent_vec_b, tangent_vec_c
            )

        translation_map = self._space.tangent_translation_map(
            base_point, left=self.left, inverse=True
        )
        tan_a_at_id = translation_map(tangent_vec_a)
        tan_b_at_id = translation_map(tangent_vec_b)
        tan_c_at_id = translation_map(tangent_vec_c)

        translation_map = self._space.tangent_translation_map(
            base_point, left=self.left, inverse=False
        )
        value_at_id = self.curvature_at_identity(tan_a_at_id, tan_b_at_id, tan_c_at_id)

        return translation_map(value_at_id)

    def sectional_curvature_at_identity(self, tangent_vec_a, tangent_vec_b):
        """Compute the sectional curvature at identity.

        For two orthonormal tangent vectors at identity :math:`x,y`,
        the sectional curvature is defined by :math:`< R(x, y)x,
        y>`. Non-orthonormal vectors can be given.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at identity.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at identity.

        References
        ----------
        https://en.wikipedia.org/wiki/Sectional_curvature

        .. [Milnor]    Milnor, John. “Curvatures of Left Invariant Metrics on
                       Lie Groups.” Advances in Mathematics 21, no. 3, 1976:
                       293–329. https://doi.org/10.1016/S0001-8708(76)80002-3.
        """
        curvature = self.curvature_at_identity(
            tangent_vec_a, tangent_vec_b, tangent_vec_a
        )
        num = self.inner_product(tangent_vec_b, curvature)
        denom = (
            self.squared_norm(tangent_vec_a) * self.squared_norm(tangent_vec_a)
            - self.inner_product(tangent_vec_a, tangent_vec_b) ** 2
        )
        condition = gs.isclose(denom, 0.0)
        denom = gs.where(condition, EPSILON, denom)
        return gs.where(~condition, num / denom, 0.0)

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the sectional curvature.

        For two orthonormal tangent vectors at a base point :math:`x,y`,
        the sectional curvature is defined by :math:`<R(x, y)x,
        y>`. Non-orthonormal vectors can be given.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Point in the group. Optional, default is the identity

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at `base_point`.

        References
        ----------
        https://en.wikipedia.org/wiki/Sectional_curvature

        .. [Milnor]    Milnor, John. “Curvatures of Left Invariant Metrics on
                       Lie Groups.” Advances in Mathematics 21, no. 3, 1976:
                       293–329. https://doi.org/10.1016/S0001-8708(76)80002-3.
        """
        if base_point is None:
            return self.sectional_curvature_at_identity(tangent_vec_a, tangent_vec_b)
        translation_map = self._space.tangent_translation_map(
            base_point, inverse=True, left=self.left
        )
        tan_a_at_id = translation_map(tangent_vec_a)
        tan_b_at_id = translation_map(tangent_vec_b)
        return self.sectional_curvature_at_identity(tan_a_at_id, tan_b_at_id)

    def curvature_derivative_at_identity(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d
    ):
        r"""Compute the covariant derivative of the curvature at identity.

        For four tangent vectors at identity :math:`x, y, z, t`,
        the covariant derivative of the curvature :math:`(\nabla_x R)(y, z)t`
        is computed using Leibniz formula.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_c : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_d : array-like, shape=[..., n, n]
            Tangent vector at identity.

        Returns
        -------
        curvature_derivative : array-like, shape=[..., n, n]
            Tangent vector at identity.
        """
        first_term = self.connection_at_identity(
            tangent_vec_a,
            self.curvature_at_identity(tangent_vec_b, tangent_vec_c, tangent_vec_d),
        )

        second_term = self.curvature_at_identity(
            self.connection_at_identity(tangent_vec_a, tangent_vec_b),
            tangent_vec_c,
            tangent_vec_d,
        )

        third_term = self.curvature_at_identity(
            tangent_vec_b,
            self.connection_at_identity(tangent_vec_a, tangent_vec_c),
            tangent_vec_d,
        )

        fourth_term = self.curvature_at_identity(
            tangent_vec_b,
            tangent_vec_c,
            self.connection_at_identity(tangent_vec_a, tangent_vec_d),
        )

        return first_term - second_term - third_term - fourth_term

    def curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point=None,
    ):
        r"""Compute the covariant derivative of the curvature.

        For four tangent vectors at a base point :math:`x, y, z, t`,
        the covariant derivative of the curvature :math:`(\nabla_x R)(y, z)t`
        is computed at the base point using Leibniz formula.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_d : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Point on the group.

        Returns
        -------
        curvature_derivative : array-like, shape=[..., n, n]
            Tangent vector at identity.
        """
        if base_point is None:
            return self.curvature_derivative_at_identity(
                tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d
            )
        translation_map = self._space.tangent_translation_map(
            base_point, inverse=True, left=self.left
        )
        tan_a_at_id = translation_map(tangent_vec_a)
        tan_b_at_id = translation_map(tangent_vec_b)
        tan_c_at_id = translation_map(tangent_vec_c)
        tan_d_at_id = translation_map(tangent_vec_d)

        value_at_id = self.curvature_derivative_at_identity(
            tan_a_at_id, tan_b_at_id, tan_c_at_id, tan_d_at_id
        )
        translation_map = self._space.tangent_translation_map(
            base_point, inverse=False, left=self.left
        )

        return translation_map(value_at_id)

    def _pre_exp(self, tangent_vec, base_point):
        if base_point is None:
            base_point = self._space.identity
            left_angular_vel = tangent_vec
        else:
            left_angular_vel = self._space.tangent_translation_map(
                base_point, left=self.left, inverse=True
            )(tangent_vec)

        return base_point, self._space.to_tangent(left_angular_vel)

    def log(self, point, base_point):
        r"""Compute Riemannian logarithm of a point from a base point.

        The log is computed by solving an optimization problem.
        The cost function to be optimized is defined by:

        .. math::

            L(v) = \Vert exp_x(v) - y \Vert^2

        where :math:`x,y` are respectively `base_point` and `point`,
        an extrinsic 2-norm is used, and exp is computed by integration
        of the Euler-Poincare equation [Kolev]_.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in the group.
        base_point : array-like, shape=[..., n, n]
            Point in the group, from which to compute the log.
            Optional, default: identity.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.

        References
        ----------
        .. [Kolev]   Kolev, Boris. “Lie Groups and Mechanics: An Introduction.”
                     Journal of Nonlinear Mathematical Physics 11, no. 4, 2004:
                     480–98. https://doi.org/10.2991/jnmp.2004.11.4.5.
        """
        if hasattr(self._space, "are_antipodals") and not gs.all(
            ~self._space.are_antipodals(point, base_point)
        ):
            raise ValueError(
                "The Logarithm map is not well-defined for"
                f" antipodal matrices: {point} and {base_point}."
            )
        return self.log_solver.log(point, base_point)

    def parallel_transport(
        self,
        tangent_vec,
        base_point,
        direction=None,
        end_point=None,
        n_steps=10,
        step="rk4",
        return_endpoint=False,
    ):
        r"""Compute the parallel transport of a tangent vec along a geodesic.

        Approximate solution for the parallel transport of a tangent vector a
        along the geodesic between two points `base_point` and `end_point`
        or alternatively defined by :math:`t \mapsto exp_{(base\_point)}(
        t*direction)`. The parallel transport equation is written entirely
        in the Lie algebra and solved with an integration scheme.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.
        direction : array-like, shape=[..., n, n]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None
        end_point : array-like, shape=[..., n, n]
            Point on the manifold. Point to transport to.
            Unused if `tangent_vec_b` is given
            Optional, default: None
        n_steps : int
            Number of integration steps to take.
            Optional, default : 10.
        step : str, {'euler', 'rk2', 'rk4'}
            Scheme to use for the approximation of the solution of the ODE
            Optional, default : rk4
        return_endpoint : bool
            Whether the end-point of the geodesic should be returned.
            Optional, default : False.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n, n]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        end_point : array-like, shape=[..., n, n]
            `exp_(base_point)(tangent_vec_b)`, only returned if
            `return_endpoint` is set to `True`.

        See Also
        --------
        geomstats.integrator

        References
        ----------
        .. [GP21]  Guigui, Nicolas, and Xavier Pennec. “A Reduced Parallel
                   Transport Equation on Lie Groups with a Left-Invariant
                   Metric.” 5th conference on Geometric Science of Information,
                   Paris 2021. Springer. Lecture Notes in Computer Science.
                   https://hal.inria.fr/hal-03154318.
        """
        if direction is None:
            if end_point is not None:
                tangent_vec_b_ = self.log(end_point, base_point)
            else:
                raise ValueError(
                    "Either an end_point or a tangent_vec_b must be given to define the"
                    " geodesic along which to transport."
                )
        else:
            tangent_vec_b_ = direction

        translation_map = self._space.tangent_translation_map(
            base_point, left=self.left, inverse=True
        )
        left_angular_vel_a = self._space.to_tangent(translation_map(tangent_vec))
        left_angular_vel_b = self._space.to_tangent(translation_map(tangent_vec_b_))

        def acceleration(state, time):
            """Compute the right-hand-side of the parallel transport eq."""
            omega = state[..., 1, :, :]
            zeta = state[..., 2, :, :]
            new_state = self.geodesic_equation(state[..., :2, :, :], time)
            gam_dot = new_state[..., 0, :, :]
            omega_dot = new_state[..., 1, :, :]
            zeta_dot = -self.connection_at_identity(omega, zeta)
            return gs.stack([gam_dot, omega_dot, zeta_dot], axis=-3)

        base_point, left_angular_vel_a, left_angular_vel_b = gs.broadcast_arrays(
            base_point, left_angular_vel_a, left_angular_vel_b
        )
        initial_state = gs.stack(
            [base_point, left_angular_vel_b, left_angular_vel_a], axis=-3
        )

        flow = integrate(acceleration, initial_state, n_steps=n_steps, step=step)
        gamma = flow[-1][..., 0, :, :]
        zeta_t = flow[-1][..., 2, :, :]
        transported = self._space.tangent_translation_map(
            gamma, left=self.left, inverse=False
        )(zeta_t)

        return (transported, gamma) if return_endpoint else transported

    def geodesic_equation(self, state, _time):
        r"""Compute the geodesic ODE associated with the invariant metric.

        This is a reduced geodesic equation written entirely in the Lie
        algebra. It is known as Euler-Poincare equation [Kolev]_.

        .. math::
            \dot{\gamma}(t) = (dL_{\gamma(t)}) X(t)
            \dot{X}(t) = ad^*_{X(t)}X(t)

        Parameters
        ----------
        state : array-like, shape=[..., dim]
            Tangent vector at the position.
        _time : array-like, shape=[..., dim]
            Point on the manifold, the position at which to compute the
            geodesic ODE.

        Returns
        -------
        geodesic_ode : array-like, shape=[..., dim]
            Value of the vector field to be integrated at position.

        References
        ----------
        .. [Kolev] Kolev, Boris. “Lie Groups and Mechanics: An Introduction.”
            Journal of Nonlinear Mathematical Physics 11, no. 4, 2004:
            480–98. https://doi.org/10.2991/jnmp.2004.11.4.5.
        """
        sign = 1.0 if self.left else -1.0
        basis = self.normal_basis(self._space.lie_algebra.basis)

        point = state[..., 0, :, :]
        vector = state[..., 1, :, :]

        velocity = self._space.tangent_translation_map(point, left=self.left)(vector)
        coefficients = gs.stack(
            [
                self.structure_constant(vector, basis_vector, vector)
                for basis_vector in basis
            ],
            axis=-1,
        )
        acceleration = gs.einsum("...i,ijk->...jk", coefficients, basis)
        return gs.stack([velocity, sign * acceleration], axis=-3)


class _InvariantMetricVector(RiemannianMetric):
    """Class for invariant metrics on Lie groups represented by vectors.

    Parameters
    ----------
    space : LieGroup
        Group to equip with the invariant metric
    left : bool
        Whether to use a left or right invariant metric.
        Optional, default: True.
    """

    def __init__(self, space, left=True):
        super().__init__(space=space)
        self.metric_mat_at_identity = gs.eye(space.dim)
        self.left = left

    @staticmethod
    def inner_product_at_identity(tangent_vec_a, tangent_vec_b):
        """Compute inner product at tangent space at identity.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            First tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., dim]
            Second tangent vector at identity.

        Returns
        -------
        inner_prod : array-like, shape=[..., dim]
            Inner-product of the two tangent vectors.
        """
        return gs.dot(tangent_vec_a, tangent_vec_b)

    def metric_matrix(self, base_point=None):
        """Compute inner product matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim], optional
            Point in the group (the default is identity).

        Returns
        -------
        metric_mat : array-like, shape=[..., dim, dim]
            Metric matrix at base_point.
        """
        if base_point is None:
            return self.metric_mat_at_identity

        base_point = self._space.regularize(base_point)
        jacobian = self._space.jacobian_translation(point=base_point, left=self.left)

        inv_jacobian = gs.linalg.inv(jacobian)
        inv_jacobian_transposed = Matrices.transpose(inv_jacobian)

        return Matrices.mul(
            inv_jacobian_transposed, self.metric_mat_at_identity, inv_jacobian
        )

    def left_exp_from_identity(self, tangent_vec):
        """Compute the exponential from identity with the left-invariant metric.

        Compute Riemannian exponential of a tangent vector at the identity
        associated to the left-invariant metric.

        If the method is called by a right-invariant metric, it uses the
        left-invariant metric associated to the same inner-product matrix
        at the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at identity.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point in the group.
        """
        return self._space.regularize_tangent_vec_at_identity(tangent_vec=tangent_vec)

    def exp_from_identity(self, tangent_vec):
        """Compute Riemannian exponential of tangent vector from the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at identity.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point in the group.
        """
        if self.left:
            exp = self.left_exp_from_identity(tangent_vec)

        else:
            opp_left_exp = self.left_exp_from_identity(-tangent_vec)
            exp = self._space.inverse(opp_left_exp)

        return self._space.regularize(exp)

    def exp(self, tangent_vec, base_point=None):
        """Compute Riemannian exponential of tan. vector wrt to base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim]
            Point in the group.
            Optional, defaults to identity if None.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point in the group equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        identity = self._space.identity

        if base_point is None:
            base_point = identity
        else:
            base_point = self._space.regularize(base_point)

        if gs.allclose(base_point, identity):
            return self.exp_from_identity(tangent_vec)

        tangent_vec_at_id = self._space.tangent_translation_map(
            point=base_point, left=self.left, inverse=True
        )(tangent_vec)
        exp_from_id = self.exp_from_identity(tangent_vec_at_id)

        if self.left:
            exp = self._space.compose(base_point, exp_from_id)

        else:
            exp = self._space.compose(exp_from_id, base_point)

        return self._space.regularize(exp)

    def left_log_from_identity(self, point):
        """Compute Riemannian log of a point wrt. id of left-invar. metric.

        Compute Riemannian logarithm of a point wrt the identity associated
        to the left-invariant metric.

        If the method is called by a right-invariant metric, it uses the
        left-invariant metric associated to the same inner-product matrix
        at the identity.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the group.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the identity equal to the Riemannian logarithm
            of point at the identity.
        """
        point = self._space.regularize(point)
        return self._space.regularize_tangent_vec_at_identity(tangent_vec=point)

    def log_from_identity(self, point):
        """Compute Riemannian logarithm of a point wrt the identity.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the group.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the identity equal to the Riemannian logarithm
            of point at the identity.
        """
        point = self._space.regularize(point)
        if self.left:
            return self.left_log_from_identity(point)

        inv_point = self._space.inverse(point)
        left_log = self.left_log_from_identity(inv_point)
        return -left_log

    def log(self, point, base_point=None):
        """Compute Riemannian logarithm of a point from a base point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the group.
        base_point : array-like, shape=[..., dim], optional
            Point in the group, from which to compute the log,
            (the default is identity).

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        identity = self._space.identity

        if base_point is None:
            base_point = identity
        else:
            base_point = self._space.regularize(base_point)

        if gs.allclose(base_point, identity):
            return self.log_from_identity(point)

        point = self._space.regularize(point)

        if self.left:
            point_near_id = self._space.compose(self._space.inverse(base_point), point)

        else:
            point_near_id = self._space.compose(point, self._space.inverse(base_point))

        log_from_id = self.log_from_identity(point_near_id)
        return self._space.tangent_translation_map(base_point, left=self.left)(
            log_from_id
        )

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute inner product of two vectors in tangent space at base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            First tangent vector at base_point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Second tangent vector at base_point.
        base_point : array-like, shape=[..., n, n]
            Point in the group.
            Optional, defaults to identity if None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        if base_point is None:
            return self.inner_product_at_identity(tangent_vec_a, tangent_vec_b)

        tangent_translation = self._space.tangent_translation_map(
            base_point, left=self.left, inverse=True
        )
        tangent_vec_a_at_id = tangent_translation(tangent_vec_a)
        tangent_vec_b_at_id = tangent_translation(tangent_vec_b)
        return self.inner_product_at_identity(tangent_vec_a_at_id, tangent_vec_b_at_id)


class InvariantMetric:
    """Class for invariant metrics on Lie groups.

    This class supports both left and right invariant metrics
    which exist on Lie groups.

    If `point_type == 'vector'`, points are parameterized by the Riemannian
    logarithm for the canonical left-invariant metric.

    Parameters
    ----------
    space : LieGroup
        Group to equip with the invariant metric
    metric_mat_at_identity : array-like, shape=[dim, dim]
        Matrix that defines the metric at identity.
        Optional, defaults to identity matrix if None.
    left : bool
        Whether to use a left or right invariant metric.
        Optional, default: True.
    """

    def __new__(cls, space, metric_mat_at_identity=None, left=True):
        """Instantiate a special euclidean group.

        Select the object to instantiate depending on the point_type.
        """
        if space.point_ndim == 1:
            return _InvariantMetricVector(space, left=left)
        return _InvariantMetricMatrix(
            space,
            left=left,
            metric_mat_at_identity=metric_mat_at_identity,
        )


class BiInvariantMetric(RiemannianMetric):
    """Class for bi-invariant metrics on compact Lie groups.

    Compact Lie groups and direct products of compact Lie groups with vector
    spaces admit bi-invariant metrics [Gallier]_. Products Lie groups are not
    implemented. Other groups such as SE(3) admit bi-invariant pseudo-metrics.

    Parameters
    ----------
    space : LieGroup
        The group to equip with the bi-invariant metric

    References
    ----------
    .. [Gallier] Gallier, Jean, and Jocelyn Quaintance. Differential Geometry
        and Lie Groups: A Computational Perspective.
        Geonger International Publishing, 2020.
        https://doi.org/10.1007/978-3-030-46040-2.
    """

    def __init__(self, space):
        self._check_implemented(space)

        super().__init__(space=space)
        if self._space.point_ndim == 1:
            # keeps behavior before removing inheritance
            self.left = True

    def _check_implemented(self, space):
        # TODO (nguigs): implement it for SE(3)
        if not ("SpecialOrthogonal" in space.__str__() or "SO" in space.__str__()):
            raise ValueError("The bi-invariant metric is only implemented for SO(n)")

    def exp(self, tangent_vec, base_point=None):
        """Compute Riemannian exponential of tangent vector from the identity.

        For a bi-invariant metric, this corresponds to the group exponential.

        Parameters
        ----------
        tangent_vec :
            Tangent vector at identity.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point in the group.
            Optional, default : identity.

        Returns
        -------
        exp : array-like, shape=[..., {dim, [n, n]}]
            Point in the group.

        References
        ----------
        .. [Gallier] Gallier, Jean, and Jocelyn Quaintance. Differential
            Geometry and Lie Groups: A Computational Perspective.
            Geonger International Publishing, 2020.
            https://doi.org/10.1007/978-3-030-46040-2.
        """
        return self._space.exp(tangent_vec, base_point)

    def log(self, point, base_point=None):
        """Compute Riemannian logarithm of a point wrt the identity.

        For a bi-invariant metric this corresponds to the group logarithm.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point in the group.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point in the group.
            Optional, default : identity.

        Returns
        -------
        log : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at the identity equal to the Riemannian logarithm
            of point at the identity.

        References
        ----------
        .. [Gallier] Gallier, Jean, and Jocelyn Quaintance. Differential
            Geometry and Lie Groups: A Computational Perspective.
            Geonger International Publishing, 2020.
            https://doi.org/10.1007/978-3-030-46040-2.
        """
        log = self._space.log(point, base_point)
        return self._space.to_tangent(log, base_point)

    def inner_product_at_identity(self, tangent_vec_a, tangent_vec_b):
        """Compute inner product at tangent space at identity.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., {dim, [n, n]}]
            First tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., {dim, [n, n]}]
            Second tangent vector at identity.

        Returns
        -------
        inner_prod : array-like, shape=[...]
            Inner-product of the two tangent vectors.
        """
        if self._space.point_ndim == 1:
            return gs.dot(tangent_vec_a, tangent_vec_b)

        return Matrices.frobenius_product(tangent_vec_a, tangent_vec_b) / 2

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute inner product of two vectors in tangent space at base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            First tangent vector at base_point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Second tangent vector at base_point.
        base_point : array-like, shape=[..., n, n]
            Point in the group.
            Optional, defaults to identity if None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        if base_point is None or self._space.point_ndim == 2:
            inner_prod = self.inner_product_at_identity(tangent_vec_a, tangent_vec_b)
            if base_point is not None:
                return repeat_out(
                    self._space.point_ndim,
                    inner_prod,
                    base_point,
                    tangent_vec_a,
                    tangent_vec_b,
                )
            return inner_prod

        tangent_translation = self._space.tangent_translation_map(
            base_point, left=True, inverse=True
        )
        tangent_vec_a_at_id = tangent_translation(tangent_vec_a)
        tangent_vec_b_at_id = tangent_translation(tangent_vec_b)
        return self.inner_product_at_identity(tangent_vec_a_at_id, tangent_vec_b_at_id)

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vec along a geodesic.

        Closed-form solution for the parallel transport of a tangent vector a
        along the geodesic between the base point and an end point, or alternatively
        defined by :math:`t \mapsto exp_{(base\_point)}(t*direction)`.
        As a compact Lie group endowed with its canonical bi-invariant metric is a
        symmetric space, parallel transport is achieved by a geodesic symmetry, or
        equivalently, one step of the pole ladder scheme.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.
        direction : array-like, shape=[..., n, n]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None.
        end_point : array-like, shape=[..., n, n]
            Point on the manifold. Point to transport to.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n, n]
            Transported tangent vector at
            `end_point=exp_(base_point)(tangent_vec_b)`.
        """
        if direction is None:
            if end_point is not None:
                direction = self.log(end_point, base_point)
            else:
                raise ValueError(
                    "Either an end_point or a tangent_vec_b must be given to define the"
                    " geodesic along which to transport."
                )
        midpoint = self.exp(1.0 / 2.0 * direction, base_point)
        transposed = Matrices.transpose(tangent_vec)
        transported_vec = Matrices.mul(midpoint, transposed, midpoint)
        return (-1.0) * transported_vec

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.
        In the case of a bi-invariant metric, it does not depend on the base
        point.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(gs.pi * self._space.dim**0.5)
        return repeat_out(self._space.point_ndim, radius, base_point)
