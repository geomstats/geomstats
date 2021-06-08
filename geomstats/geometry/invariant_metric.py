"""Left- and right- invariant metrics that exist on Lie groups."""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.integrator import integrate


EPSILON = 1e-6


class _InvariantMetricMatrix(RiemannianMetric):
    """Class for invariant metrics on Matrix Lie groups.

    This class supports both left and right invariant metrics
    which exist on Lie groups.

    Parameters
    ----------
    group : LieGroup
        Group to equip with the invariant metric
    metric_mat_at_identity : array-like, shape=[dim, dim]
        Matrix that defines the metric at identity.
        Optional, defaults to identity matrix if None.
    left_or_right : str, {'left', 'right'}
        Whether to use a left or right invariant metric.
        Optional, default: 'left'.

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

    def __init__(self, group,
                 metric_mat_at_identity=None,
                 left_or_right='left', **kwargs):
        super(_InvariantMetricMatrix, self).__init__(
            dim=group.dim, default_point_type='matrix', **kwargs)

        self.group = group
        self.lie_algebra = group.lie_algebra
        if metric_mat_at_identity is None:
            metric_mat_at_identity = gs.eye(self.group.dim)

        geomstats.errors.check_parameter_accepted_values(
            left_or_right, 'left_or_right', ['left', 'right'])

        self.metric_mat_at_identity = metric_mat_at_identity
        self.left_or_right = left_or_right

    def reshape_metric_matrix(self):
        """Reshape diagonal metric matrix to a symmetric matrix of size n.

        Reshape a diagonal metric matrix of size `dim x dim` into a symmetric
        matrix of size `n x n` where :math: `dim= n (n -1) / 2` is the
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
                self.lie_algebra.matrix_representation(metric_coeffs))
            return metric_mat
        raise ValueError('This is only possible for a diagonal matrix')
    reshaped_metric_matrix = property(reshape_metric_matrix)

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
        if (Matrices.is_diagonal(metric_mat)
                and self.lie_algebra is not None):
            tan_b = tangent_vec_b * self.reshaped_metric_matrix
        inner_prod = Matrices.frobenius_product(tangent_vec_a, tan_b)
        return inner_prod

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
            return self.inner_product_at_identity(
                tangent_vec_a, tangent_vec_b)

        tangent_translation = self.group.tangent_translation_map(
            base_point, left_or_right=self.left_or_right, inverse=True)
        tangent_vec_a_at_id = tangent_translation(tangent_vec_a)
        tangent_vec_b_at_id = tangent_translation(tangent_vec_b)
        inner_prod = self.inner_product_at_identity(
            tangent_vec_a_at_id, tangent_vec_b_at_id)
        return inner_prod

    def structure_constant(self, tangent_vec_a, tangent_vec_b, tangent_vec_c):
        r"""Compute the structure constant of the metric.

        For three tangent vectors :math: `x, y, z` at identity,
        compute  :math: `<[x,y], z>`.

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

        For two tangent vectors at identity :math: `x,y`, this corresponds to
        the vector :math:`a` such that
        :math: `\forall z, <[x,z], y > = <a, z>`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at identity.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at identity.

        Returns
        -------
        ad_star : array-like, shape=[..., n, n]
            Tangent vector at identity corresponding to :math: `ad_x^*(y)`.

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
        basis = self.normal_basis(self.lie_algebra.basis)
        return - gs.einsum(
            'i...,ijk->...jk',
            gs.array([
                self.structure_constant(
                    tan, tangent_vec_a, tangent_vec_b) for tan in basis]),
            gs.array(basis))

    def connection_at_identity(self, tangent_vec_a, tangent_vec_b):
        r"""Compute the Levi-Civita connection at identity.

        For two tangent vectors at identity :math: `x,y`, one can associate
        left (respectively right) invariant vector fields :math: `\tilde{x},
        \tilde{y}`. Then the vector :math: `(\nabla_\tilde{x}(\tilde{x}))_{
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
        sign = 1. if self.left_or_right == 'left' else -1.
        return sign / 2 * (Matrices.bracket(tangent_vec_a, tangent_vec_b)
                           - self.dual_adjoint(tangent_vec_a, tangent_vec_b)
                           - self.dual_adjoint(tangent_vec_b, tangent_vec_a))

    def connection(self, tangent_vec_a, tangent_vec_b, base_point=None):
        r"""Compute the Levi-Civita connection of invariant vector fields.

        For two tangent vectors at a base point :math: `p, x,y`, one can
        associate left (respectively right) invariant vector fields :math:
        `\tilde{x}, \tilde{y}`. Then the vector :math: `(\nabla_\tilde{x}(
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
        translation_map = self.group.tangent_translation_map(
            base_point, left_or_right=self.left_or_right, inverse=True)
        tan_a_at_id = translation_map(tangent_vec_a)
        tan_b_at_id = translation_map(tangent_vec_b)

        translation_map = self.group.tangent_translation_map(
            base_point, left_or_right=self.left_or_right, inverse=False)

        value_at_id = self.connection_at_identity(tan_a_at_id, tan_b_at_id)
        return translation_map(value_at_id)

    def curvature_at_identity(
            self, tangent_vec_a, tangent_vec_b, tangent_vec_c):
        r"""Compute the curvature at identity.

        For three tangent vectors at identity :math: `x,y,z`,
        the curvature is defined by
        :math: `R(x, y)z = \nabla_{[x,y]}z
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
            tangent_vec_a, self.connection_at_identity(
                tangent_vec_b, tangent_vec_c))

        right_term = self.connection_at_identity(
            tangent_vec_b, self.connection_at_identity(
                tangent_vec_a, tangent_vec_c))

        return bracket_term - left_term + right_term

    def curvature(
            self, tangent_vec_a, tangent_vec_b, tangent_vec_c,
            base_point=None):
        r"""Compute the curvature.

        For three tangent vectors at a base point :math: `x,y,z`,
        the curvature is defined by
        :math: `R(x, y)z = \nabla_{[x,y]}z
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
                tangent_vec_a, tangent_vec_b, tangent_vec_c)

        translation_map = self.group.tangent_translation_map(
            base_point, left_or_right=self.left_or_right, inverse=True)
        tan_a_at_id = translation_map(tangent_vec_a)
        tan_b_at_id = translation_map(tangent_vec_b)
        tan_c_at_id = translation_map(tangent_vec_c)

        translation_map = self.group.tangent_translation_map(
            base_point, left_or_right=self.left_or_right, inverse=False)
        value_at_id = self.curvature_at_identity(
            tan_a_at_id, tan_b_at_id, tan_c_at_id)

        return translation_map(value_at_id)

    def sectional_curvature_at_identity(self, tangent_vec_a, tangent_vec_b):
        """Compute the sectional curvature at identity.

        For two orthonormal tangent vectors at identity :math: `x,y`,
        the sectional curvature is defined by :math: `< R(x, y)x,
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
            tangent_vec_a, tangent_vec_b, tangent_vec_a)
        num = self.inner_product(tangent_vec_b, curvature)
        denom = (
            self.squared_norm(tangent_vec_a)
            * self.squared_norm(tangent_vec_a)
            - self.inner_product(tangent_vec_a, tangent_vec_b) ** 2)
        condition = gs.isclose(denom, 0.)
        denom = gs.where(condition, EPSILON, denom)
        return gs.where(~condition, num / denom, 0.)

    def sectional_curvature(
            self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the sectional curvature.

        For two orthonormal tangent vectors at a base point :math: `x,y`,
        the sectional curvature is defined by :math: `<R(x, y)x,
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
            return self.sectional_curvature_at_identity(tangent_vec_a,
                                                        tangent_vec_b)
        translation_map = self.group.tangent_translation_map(
            base_point, inverse=True, left_or_right=self.left_or_right)
        tan_a_at_id = translation_map(tangent_vec_a)
        tan_b_at_id = translation_map(tangent_vec_b)
        return self.sectional_curvature_at_identity(tan_a_at_id, tan_b_at_id)

    def curvature_derivative_at_identity(
            self, tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d):
        r"""Compute the covariant derivative of the curvature at identity.

        For four tangent vectors at identity :math: `x, y, z, t`,
        the covariant derivative of the curvature :math: `(\nabla_x R)(y, z)t`
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
            self.curvature_at_identity(
                tangent_vec_b, tangent_vec_c, tangent_vec_d))

        second_term = self.curvature_at_identity(
            self.connection_at_identity(tangent_vec_a, tangent_vec_b),
            tangent_vec_c,
            tangent_vec_d)

        third_term = self.curvature_at_identity(
            tangent_vec_b,
            self.connection_at_identity(tangent_vec_a, tangent_vec_c),
            tangent_vec_d)

        fourth_term = self.curvature_at_identity(
            tangent_vec_b,
            tangent_vec_c,
            self.connection_at_identity(tangent_vec_a, tangent_vec_d))

        return first_term - second_term - third_term - fourth_term

    def curvature_derivative(
            self, tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d,
            base_point=None):
        r"""Compute the covariant derivative of the curvature.

        For four tangent vectors at a base point :math: `x, y, z, t`,
        the covariant derivative of the curvature :math: `(\nabla_x R)(y, z)t`
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
                tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d)
        translation_map = self.group.tangent_translation_map(
            base_point, inverse=True, left_or_right=self.left_or_right)
        tan_a_at_id = translation_map(tangent_vec_a)
        tan_b_at_id = translation_map(tangent_vec_b)
        tan_c_at_id = translation_map(tangent_vec_c)
        tan_d_at_id = translation_map(tangent_vec_d)

        value_at_id = self.curvature_derivative_at_identity(
            tan_a_at_id, tan_b_at_id, tan_c_at_id, tan_d_at_id)
        translation_map = self.group.tangent_translation_map(
            base_point, inverse=False, left_or_right=self.left_or_right)

        return translation_map(value_at_id)

    def exp(self, tangent_vec, base_point=None, n_steps=10, step='rk4',
            **kwargs):
        r"""Compute Riemannian exponential of tan. vector wrt to base point.

        If :math: `\gamma` is a geodesic, then it satisfies the
        Euler-Poincare equation [Kolev]_:
        .. math:

                        \dot{\gamma}(t) = (dL_{\gamma(t)}) X(t)
                        \dot{X}(t) = ad^*_{X(t)}X(t)

        where :math: `ad^*` is the dual adjoint map with respect to the
        metric. For a right-invariant metric, :math: `dR` is used instead of
        :math: `dL` and :math: `ad^*` is replaced by :math: `-ad^*`. The
        exponential map is approximated by numerical integration
        of this equation, with initial conditions :math: `\dot{\gamma}(0)`
        given by the argument `tangent_vec` and :math: `\gamma(0)` by
        `base_point`. A Runge-Kutta scheme of order 2 or 4 is used for
        integration.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., n, n]
            Point in the group.
            Optional, defaults to identity if None.
        n_steps : int,
            Number of integration steps.
            Optional, default : 15.
        step : str, {'euler', 'rk2', 'rk4'}
            Scheme to use in the integration.
            Optional, default : 'rk4'.

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
        group = self.group
        basis = self.normal_basis(self.lie_algebra.basis)
        sign = 1. if self.left_or_right == 'left' else -1.

        def lie_acceleration(state, _time):
            """Compute the right-hand side of the geodesic equation."""
            point, vector = state
            velocity = self.group.tangent_translation_map(
                point, left_or_right=self.left_or_right)(vector)
            coefficients = gs.array([self.structure_constant(
                vector, basis_vector, vector) for basis_vector in basis])
            acceleration = gs.einsum('i...,ijk->...jk', coefficients, basis)
            return gs.stack([velocity, sign * acceleration])

        if base_point is None:
            base_point = group.identity
            left_angular_vel = tangent_vec
        else:
            left_angular_vel = self.group.tangent_translation_map(
                base_point,
                left_or_right=self.left_or_right, inverse=True)(tangent_vec)
        if (base_point.ndim == 2 or base_point.shape[0] == 1) and \
                tangent_vec.ndim == 3:
            base_point = gs.stack([base_point] * len(tangent_vec))
            base_point = gs.reshape(base_point, tangent_vec.shape)
        initial_state = gs.stack(
            [base_point, group.to_tangent(left_angular_vel)])
        flow = integrate(
            lie_acceleration, initial_state, n_steps=n_steps, step=step)
        return flow[-1][0]

    def log(self, point, base_point, n_steps=15, step='rk4',
            verbose=False, max_iter=25, tol=1e-10):
        r"""Compute Riemannian logarithm of a point from a base point.

        The log is computed by solving an optimization problem.
        The cost function to be optimized is defined by:
        .. math:

                    L(v) = \Vert exp_x(v) - y \Vert^2

        where :math: `x,y` are respectively `base_point` and `point`,
        an extrinsic 2-norm is used, and exp is computed by integration
        of the Euler-Poincare equation [Kolev]_.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in the group.
        base_point : array-like, shape=[..., n, n]
            Point in the group, from which to compute the log.
            Optional, default: identity.
        n_steps : int,
            Number of integration steps to compute the exponential in the
            loss.
            Optional, default : 15.
        step : str, {'euler', 'rk2', 'rk4'}
            Scheme to use in the integration procedure of the exponential in
            the loss.
            Optional, default : 'rk4'.
        verbose : bool,
            Verbosity level of the optimization procedure.
            Optional. default : False.
        max_iter : int,
            Maximum of iteration of the optimization procedure.
            Optional, default : 25.
        tol : float,
            Tolerance for the stopping criterion of the optimization.
            Optional, default : 1e-10.

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
        return self.group.to_tangent(
            super(_InvariantMetricMatrix, self).log(
                point, base_point, n_steps=n_steps, step=step,
                verbose=verbose, max_iter=max_iter, tol=tol), base_point)

    def parallel_transport(
            self, tangent_vec_a, tangent_vec_b, base_point, n_steps=10,
            step='rk4', return_endpoint=False):
        r"""Compute the parallel transport of a tangent vec along a geodesic.

        Approximate solution for the parallel transport of a tangent vector a
        along the geodesic defined by :math: `t \mapsto exp_(base_point)(t*
        tangent_vec_b)`. The parallel transport equation is written entirely
        in the Lie algebra and solved with an integration scheme.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n + 1, n + 1]
            Tangent vector at base point to be transported.
        tangent_vec_b : array-like, shape=[..., n + 1, n + 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
        base_point : array-like, shape=[..., n + 1, n + 1]
            Point on the hypersphere.
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
        transported_tangent_vec: array-like, shape=[..., n + 1, n + 1]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        end_point : array-like, shape=[..., n + 1, n + 1]
            `exp_(base_point)(tangent_vec_b)`, only returned if
            `return_endpoint` is set to `True`.

        See Also
        --------
        geomstats.integrator

        References
        ----------
        [GP21]_    Guigui, Nicolas, and Xavier Pennec. “A Reduced Parallel
                   Transport Equation on Lie Groups with a Left-Invariant
                   Metric.” 5th conference on Geometric Science of Information,
                   Paris 2021. Springer. Lecture Notes in Computer Science.
                   https://hal.inria.fr/hal-03154318.
        """
        group = self.group
        translation_map = group.tangent_translation_map(
            base_point,
            left_or_right=self.left_or_right, inverse=True)
        left_angular_vel_a = group.to_tangent(translation_map(tangent_vec_a))
        left_angular_vel_b = group.to_tangent(translation_map(tangent_vec_b))

        def acceleration(state, time):
            """Compute the right-hand-side of the parallel transport eq."""
            omega, zeta = state[1:]
            gam_dot, omega_dot = self.geodesic_equation(state[:2], time)
            zeta_dot = - self.connection_at_identity(omega, zeta)
            return gs.stack([gam_dot, omega_dot, zeta_dot])

        if (base_point.ndim == 2 or base_point.shape[0] == 1) and \
                (3 in (tangent_vec_a.ndim, tangent_vec_b.ndim)):
            n_sample = tangent_vec_a.shape[0] if tangent_vec_a.ndim == 3 else\
                tangent_vec_b.shape[0]
            base_point = gs.stack([base_point] * n_sample)

        initial_state = gs.stack([
            base_point, left_angular_vel_b, left_angular_vel_a])
        flow = integrate(
            acceleration, initial_state, n_steps=n_steps, step=step)
        gamma, _, zeta_t = flow[-1]
        transported = group.tangent_translation_map(
            gamma, left_or_right=self.left_or_right, inverse=False)(zeta_t)
        return (transported, gamma) if return_endpoint else transported

    def geodesic_equation(self, state, _time):
        r"""Compute the geodesic ODE associated with the invariant metric.

        This is a reduced geodesic equation written entirely in the Lie
        algebra. It is known as Euler-Poincare equation [Kolev].
        .. math:
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
        .. [Kolev]   Kolev, Boris. “Lie Groups and Mechanics: An Introduction.”
             Journal of Nonlinear Mathematical Physics 11, no. 4, 2004:
             480–98. https://doi.org/10.2991/jnmp.2004.11.4.5.
        """
        sign = 1. if self.left_or_right == 'left' else -1.
        basis = self.normal_basis(self.lie_algebra.basis)

        point, vector = state
        velocity = self.group.tangent_translation_map(
            point, left_or_right=self.left_or_right)(vector)
        coefficients = gs.array([self.structure_constant(
            vector, basis_vector, vector) for basis_vector in basis])
        acceleration = gs.einsum('i...,ijk->...jk', coefficients, basis)
        return gs.stack([velocity, sign * acceleration])


class _InvariantMetricVector(RiemannianMetric):
    """Class for invariant metrics on Lie groups represented by vectors.

    Parameters
    ----------
    group : LieGroup
        Group to equip with the invariant metric
    left_or_right : str, {'left', 'right'}
        Whether to use a left or right invariant metric.
        Optional, default: 'left'.
    """

    def __init__(self, group, left_or_right='left'):
        super(_InvariantMetricVector, self).__init__(dim=group.dim)

        self.group = group
        self.metric_mat_at_identity = gs.eye(group.dim)
        self.left_or_right = left_or_right

        geomstats.errors.check_parameter_accepted_values(
            left_or_right, 'left_or_right', ['left', 'right'])

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
        return gs.einsum(
            '...i,...i->...', tangent_vec_a, tangent_vec_b)

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

        base_point = self.group.regularize(base_point)
        jacobian = self.group.jacobian_translation(
            point=base_point, left_or_right=self.left_or_right)

        inv_jacobian = gs.linalg.inv(jacobian)
        inv_jacobian_transposed = Matrices.transpose(inv_jacobian)

        metric_mat = Matrices.mul(
            inv_jacobian_transposed, self.metric_mat_at_identity, inv_jacobian)
        return metric_mat

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
        tangent_vec = self.group.regularize_tangent_vec_at_identity(
            tangent_vec=tangent_vec,
            metric=self)
        return tangent_vec

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
        if self.left_or_right == 'left':
            exp = self.left_exp_from_identity(tangent_vec)

        else:
            opp_left_exp = self.left_exp_from_identity(-tangent_vec)
            exp = self.group.inverse(opp_left_exp)

        exp = self.group.regularize(exp)
        return exp

    def exp(self, tangent_vec, base_point=None, **kwargs):
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
        identity = self.group.identity

        if base_point is None:
            base_point = identity
        else:
            base_point = self.group.regularize(base_point)

        if gs.allclose(base_point, identity):
            return self.exp_from_identity(tangent_vec)

        tangent_vec_at_id = self.group.tangent_translation_map(
            point=base_point,
            left_or_right=self.left_or_right,
            inverse=True)(tangent_vec)
        exp_from_id = self.exp_from_identity(tangent_vec_at_id)

        if self.left_or_right == 'left':
            exp = self.group.compose(base_point, exp_from_id)

        else:
            exp = self.group.compose(exp_from_id, base_point)

        exp = self.group.regularize(exp)
        return exp

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
        point = self.group.regularize(point)
        log = self.group.regularize_tangent_vec_at_identity(
            tangent_vec=point, metric=self)
        return log

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
        point = self.group.regularize(point)
        if self.left_or_right == 'left':
            log = self.left_log_from_identity(point)

        else:
            inv_point = self.group.inverse(point)
            left_log = self.left_log_from_identity(inv_point)
            log = - left_log

        return log

    def log(self, point, base_point=None, **kwargs):
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
        identity = self.group.identity

        if base_point is None:
            base_point = identity
        else:
            base_point = self.group.regularize(base_point)

        if gs.allclose(base_point, identity):
            return self.log_from_identity(point)

        point = self.group.regularize(point)

        if self.left_or_right == 'left':
            point_near_id = self.group.compose(
                self.group.inverse(base_point), point)

        else:
            point_near_id = self.group.compose(
                point, self.group.inverse(base_point))

        log_from_id = self.log_from_identity(point_near_id)
        log = self.group.tangent_translation_map(
            base_point, left_or_right=self.left_or_right)(log_from_id)
        return log

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
            return self.inner_product_at_identity(
                tangent_vec_a, tangent_vec_b)

        tangent_translation = self.group.tangent_translation_map(
            base_point, left_or_right=self.left_or_right, inverse=True)
        tangent_vec_a_at_id = tangent_translation(tangent_vec_a)
        tangent_vec_b_at_id = tangent_translation(tangent_vec_b)
        inner_prod = self.inner_product_at_identity(
            tangent_vec_a_at_id, tangent_vec_b_at_id)
        return inner_prod


class InvariantMetric(_InvariantMetricVector, _InvariantMetricMatrix):
    """Class for invariant metrics on Lie groups.

    This class supports both left and right invariant metrics
    which exist on Lie groups.

    If `point_type='vector'`, points are parameterized by the Riemannian
    logarithm for the canonical left-invariant metric.

    Parameters
    ----------
    group : LieGroup
        Group to equip with the invariant metric
    metric_mat_at_identity : array-like, shape=[dim, dim]
        Matrix that defines the metric at identity.
        Optional, defaults to identity matrix if None.
    left_or_right : str, {'left', 'right'}
        Whether to use a left or right invariant metric.
        Optional, default: 'left'.
    point_type : str, {'vector', 'matrix'}
        Point representation.
        Optional, default: group.default_point_type.
    """

    def __new__(
            cls, group, metric_mat_at_identity=None,
            left_or_right='left', point_type=None):
        """Instantiate a special euclidean group.

        Select the object to instantiate depending on the point_type.
        """
        if point_type is None:
            point_type = group.default_point_type
        if point_type == 'vector':
            return _InvariantMetricVector(group, left_or_right=left_or_right)
        return _InvariantMetricMatrix(
            group,
            left_or_right=left_or_right,
            metric_mat_at_identity=metric_mat_at_identity)


class BiInvariantMetric(_InvariantMetricVector):
    """Class for bi-invariant metrics on compact Lie groups.

    Compact Lie groups and direct products of compact Lie groups with vector
    spaces admit bi-invariant metrics [Gallier]_. Products Lie groups are not
    implemented. Other groups such as SE(3) admit bi-invariant pseudo-metrics.

    Parameters
    ----------
    group : LieGroup
        The group to equip with the bi-invariant metric

    References
    ----------
    .. [Gallier]   Gallier, Jean, and Jocelyn Quaintance. Differential Geometry
                   and Lie Groups: A Computational Perspective.
                   Geonger International Publishing, 2020.
                   https://doi.org/10.1007/978-3-030-46040-2.
    """

    def __init__(self, group):
        super(BiInvariantMetric, self).__init__(
            group=group)
        condition = (
            'SpecialOrthogonal' not in group.__str__()
            and 'SO' not in group.__str__()
            and 'SpecialOrthogonal3' not in group.__str__())
        # TODO (nguigs): implement it for SE(3)
        if condition:
            raise ValueError(
                'The bi-invariant metric is only implemented for SO(n)')
        self.default_point_type = group.default_point_type

    def exp(self, tangent_vec, base_point=None, **kwargs):
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
        .. [Gallier]   Gallier, Jean, and Jocelyn Quaintance. Differential
                       Geometry and Lie Groups: A Computational Perspective.
                       Geonger International Publishing, 2020.
                       https://doi.org/10.1007/978-3-030-46040-2.
        """
        return self.group.exp(tangent_vec, base_point)

    def log(self, point, base_point=None, **kwargs):
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
        .. [Gallier]   Gallier, Jean, and Jocelyn Quaintance. Differential
                       Geometry and Lie Groups: A Computational Perspective.
                       Geonger International Publishing, 2020.
                       https://doi.org/10.1007/978-3-030-46040-2.
        """
        log = self.group.log(point, base_point)
        return self.group.to_tangent(log, base_point)

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
        if self.default_point_type == 'vector':
            return super(BiInvariantMetric, self).inner_product_at_identity(
                tangent_vec_a, tangent_vec_b)
        return Matrices.frobenius_product(tangent_vec_a, tangent_vec_b)

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
        if base_point is None or self.default_point_type == 'matrix':
            return self.inner_product_at_identity(
                tangent_vec_a, tangent_vec_b)

        return super(BiInvariantMetric, self).inner_product(
            tangent_vec_a, tangent_vec_b, base_point)

    def parallel_transport(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the parallel transport of a tangent vec along a geodesic.

        Closed-form solution for the parallel transport of a tangent vector a
        along the geodesic defined by :math: `t \mapsto exp_(base_point)(t*
        tangent_vec_b)`. As a compact Lie group endowed with its
        canonical bi-invariant metric is a symmetric space, parallel
        transport is achieved by a geodesic symmetry, or equivalently, one step
         of the pole ladder scheme.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n + 1, n + 1]
            Tangent vector at base point to be transported.
        tangent_vec_b : array-like, shape=[..., n + 1, n + 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
        base_point : array-like, shape=[..., n + 1, n + 1]
            Point on the hypersphere.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n + 1, n + 1]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        midpoint = self.exp(1. / 2. * tangent_vec_b, base_point)
        transposed = Matrices.transpose(tangent_vec_a)
        transported_vec = Matrices.mul(midpoint, transposed, midpoint)
        return (-1.) * transported_vec
