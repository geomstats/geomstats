"""Kendall Pre-Shape space.

Lead authors: Elodie Maignant and Nicolas Guigui.
"""

import geomstats.backend as gs
from geomstats.geometry.base import LevelSet
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.integrator import integrate
from geomstats.vectorization import get_batch_shape, repeat_out


class PreShapeSpace(LevelSet):
    r"""Class for the Kendall pre-shape space.

    The pre-shape space is the sphere of the space of centered k-ad of
    landmarks in :math:`R^m` (for the Frobenius norm). It is endowed with the
    spherical Procrustes metric d(x, y):= arccos(tr(xy^t)).

    Points are represented by :math:`k \times m` centred matrices as in
    [Nava]_. Beware that this is not the usual convention from the literature.

    Parameters
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.

    References
    ----------
    .. [Nava]  Nava-Yazdani, E., H.-C. Hege, T. J.Sullivan, and C. von Tycowicz.
              “Geodesic Analysis in Kendall’s Shape Space with Epidemiological
              Applications.”
              Journal of Mathematical Imaging and Vision 62, no. 4 549–59.
              https://doi.org/10.1007/s10851-020-00945-w.
    """

    def __init__(self, k_landmarks, m_ambient, equip=True):
        self.k_landmarks = k_landmarks
        self.m_ambient = m_ambient

        super().__init__(
            dim=m_ambient * (k_landmarks - 1) - 1,
            equip=equip,
        )

        self._sphere = Hypersphere(dim=m_ambient * k_landmarks - 1)

        self._quotient_map = {
            (PreShapeMetric, "rotations"): (PreShapeSpaceBundle, KendallShapeMetric),
        }

    def _get_total_space_metric(self):
        return (
            self.metric._total_space.metric
            if hasattr(self.metric, "_total_space")
            else self.metric
        )

    def new(self, equip=True):
        """Create manifold with same parameters."""
        return PreShapeSpace(
            k_landmarks=self.k_landmarks, m_ambient=self.m_ambient, equip=equip
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PreShapeMetric

    def _define_embedding_space(self):
        return Matrices(self.k_landmarks, self.m_ambient)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]

        Returns
        -------
        submersion : array-like, shape=[...]
        """
        return self.embedding_space.metric.squared_norm(point) - 1.0

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., dim+1]
        point : array-like, shape=[..., dim+1]

        Returns
        -------
        tangent_submersion : array-like, shape=[...]
        """
        return self.embedding_space.metric.inner_product(vector, point)

    def projection(self, point):
        """Project a point on the pre-shape space.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point in Matrices space.

        Returns
        -------
        projected_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point projected on the pre-shape space.

        Notes
        -----
        * Requires space to be equipped.
        """
        total_space_metric = self._get_total_space_metric()

        centered_point = self.center(point)
        frob_norm = total_space_metric.norm(centered_point)
        return gs.einsum("...,...ij->...ij", 1.0 / frob_norm, centered_point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the pre-shape space from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Not used.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled on the pre-shape space.
        """
        return self.random_uniform(n_samples)

    def random_uniform(self, n_samples=1):
        """Sample in the pre-shape space from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., k_landmarks, m_ambient]
            Points sampled on the pre-shape space.

        Notes
        -----
        * Requires space to be equipped.
        """
        samples = self._sphere.random_uniform(n_samples)
        samples = gs.reshape(samples, (-1, self.k_landmarks, self.m_ambient))
        if n_samples == 1:
            samples = samples[0]
        return self.projection(samples)

    @staticmethod
    def is_centered(point, atol=gs.atol):
        """Check that landmarks are centered around 0.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point in Matrices space.
        atol :  float
            Tolerance at which to evaluate mean == 0.
            Optional, default: backend atol.

        Returns
        -------
        is_centered : array-like, shape=[...,]
            Boolean evaluating if point is centered.
        """
        mean = gs.mean(point, axis=-2)
        return gs.all(gs.isclose(mean, 0.0, atol=atol), axis=-1)

    @staticmethod
    def center(point):
        """Center landmarks around 0.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point in Matrices space.

        Returns
        -------
        centered : array-like, shape=[..., k_landmarks, m_ambient]
            Point with centered landmarks.
        """
        mean = gs.mean(point, axis=-2)
        return point - mean[..., None, :]

    def to_tangent(self, vector, base_point):
        """Project a vector to the tangent space.

        Project a vector in the embedding matrix space
        to the tangent space of the pre-shape space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., k_landmarks, m_ambient]
            Vector in Matrix space.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space defining the tangent space,
            where the vector will be projected.

        Returns
        -------
        tangent_vec : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector in the tangent space of the pre-shape space
            at the base point.

        Notes
        -----
        * Requires space to be equipped.
        """
        if not gs.all(self.is_centered(base_point)):
            raise ValueError("The base_point does not belong to the pre-shape space")

        total_space_metric = self._get_total_space_metric()

        vector = self.center(vector)
        sq_norm = Matrices.frobenius_product(base_point, base_point)
        inner_prod = total_space_metric.inner_product(base_point, vector)
        coef = inner_prod / sq_norm
        return vector - gs.einsum("...,...ij->...ij", coef, base_point)


class PreShapeSpaceBundle(FiberBundle):
    r"""Class for the Kendall pre-shape space bundle."""

    def align(self, point, base_point):
        """Align point to base_point.

        Find the optimal rotation R in SO(m) such that the base point and
        R.point are well positioned.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the manifold.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the manifold.

        Returns
        -------
        aligned : array-like, shape=[..., k_landmarks, m_ambient]
            R.point.
        """
        return Matrices.align_matrices(point, base_point)

    def vertical_projection(self, tangent_vec, base_point, return_skew=False):
        r"""Project to vertical subspace.

        Compute the vertical component of a tangent vector :math:`w` at a
        base point :math:`x` by solving the sylvester equation:

        .. math::
            Axx^T + xx^TA = wx^T - xw^T

        where `A` is skew-symmetric.
        Then `Ax` is the vertical projection of `w`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector to the pre-shape space at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space.
        return_skew : bool
            Whether to return the skew-symmetric matrix A.
            Optional, default: False

        Returns
        -------
        vertical : array-like, shape=[..., k_landmarks, m_ambient]
            Vertical component of `tangent_vec`.
        skew : array-like, shape=[..., m_ambient, m_ambient]
            Vertical component of `tangent_vec`.
        """
        transposed_point = Matrices.transpose(base_point)
        left_term = gs.matmul(transposed_point, base_point)
        alignment = gs.matmul(Matrices.transpose(tangent_vec), base_point)
        right_term = alignment - Matrices.transpose(alignment)
        skew = gs.linalg.solve_sylvester(left_term, left_term, right_term)

        vertical = -gs.matmul(base_point, skew)
        return (vertical, skew) if return_skew else vertical

    def is_horizontal(self, tangent_vec, base_point, atol=gs.atol):
        """Check whether the tangent vector is horizontal at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the manifold.
            Optional, default: none.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if tangent vector is horizontal.
        """
        product = gs.matmul(Matrices.transpose(tangent_vec), base_point)
        is_tangent = self._total_space.is_tangent(tangent_vec, base_point, atol)
        is_symmetric = Matrices.is_symmetric(product, atol)
        return gs.logical_and(is_tangent, is_symmetric)

    def integrability_tensor(self, tangent_vec_x, tangent_vec_e, base_point):
        r"""Compute the fundamental tensor A of the submersion.

        The fundamental tensor A is defined for tangent vectors of the total
        space by [ONeill]_
        :math:`A_X Y = ver\nabla^M_{hor X}(hor Y) + hor \nabla^M_{hor X}(ver Y)`
        where :math:`hor, ver` are the horizontal and vertical projections.

        For the Kendall shape space, we have the closed-form expression at
        base-point P [Pennec]_:
        :math:`A_X E = P Sylv_P(E^\top hor(X)) + F + <F,P> P` where
        :math:`F = hor(X) Sylv_P(P^\top E)` and :math:`Sylv_P(B)` is the
        unique skew-symmetric matrix :math:`\Omega` solution of
        :math:`P^\top P \Omega + \Omega P^\top P = B - B^\top`.

        Parameters
        ----------
        tangent_vec_x : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        tangent_vec_e : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point of the total space.

        Returns
        -------
        vector : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`, result of the A tensor applied to
            `tangent_vec_x` and `tangent_vec_e`.

        References
        ----------
        .. [ONeill]  O’Neill, Barrett. The Fundamental Equations of a
            Submersion, Michigan Mathematical Journal 13, no. 4
            (December 1966): 459–69. https://doi.org/10.1307/mmj/1028999604.

        .. [Pennec] Pennec, Xavier. Computing the curvature and its gradient
            in Kendall shape spaces. Unpublished.
        """
        hor_x = self.horizontal_projection(tangent_vec_x, base_point)
        p_top = Matrices.transpose(base_point)
        p_top_p = gs.matmul(p_top, base_point)

        def sylv_p(mat_b):
            """Solves Sylvester equation for vertical component."""
            return gs.linalg.solve_sylvester(
                p_top_p, p_top_p, mat_b - Matrices.transpose(mat_b)
            )

        e_top_hor_x = gs.matmul(Matrices.transpose(tangent_vec_e), hor_x)
        sylv_e_top_hor_x = sylv_p(e_top_hor_x)

        p_top_e = gs.matmul(p_top, tangent_vec_e)
        sylv_p_top_e = sylv_p(p_top_e)

        return gs.matmul(base_point, sylv_e_top_hor_x) + gs.matmul(hor_x, sylv_p_top_e)

    def integrability_tensor_derivative(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        nabla_x_y,
        tangent_vec_e,
        nabla_x_e,
        base_point,
    ):
        r"""Compute the covariant derivative of the integrability tensor A.

        The horizontal covariant derivative :math:`\nabla_X (A_Y E)` is
        necessary to compute the covariant derivative of the curvature in a
        submersion.
        The components :math:`\nabla_X (A_Y E)` and :math:`A_Y E` are
        computed here for the Kendall shape space at base-point
        :math:`P = base\_point` for horizontal vector fields fields :math:
        `X, Y` extending the values :math:`X|_P = horizontal\_vec\_x`,
        :math:`Y|_P = horizontal\_vec\_y` and a general vector field
        :math:`E` extending :math:`E|_P = tangent\_vec\_e` in a neighborhood
        of the base-point P with covariant derivatives
        :math:`\nabla_X Y |_P = nabla_x y` and
        :math:`\nabla_X E |_P = nabla_x e`.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, m_ambient]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, m_ambient]
            Horizontal tangent vector at `base_point`.
        nabla_x_y : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        tangent_vec_e : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        nabla_x_e : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point of the total space.

        Returns
        -------
        nabla_x_a_y_e : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`, result of :math:`\nabla_X^S
            (A_Y E)`.
        a_y_e : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`, result of :math:`A_Y E`.

        References
        ----------
        .. [Pennec] Pennec, Xavier. Computing the curvature and its gradient
        in Kendall shape spaces. Unpublished.
        """
        if not gs.all(self._total_space.belongs(base_point)):
            raise ValueError("The base_point does not belong to the pre-shape space")
        if not gs.all(self.is_horizontal(horizontal_vec_x, base_point)):
            raise ValueError("Tangent vector x is not horizontal")
        if not gs.all(self.is_horizontal(horizontal_vec_y, base_point)):
            raise ValueError("Tangent vector y is not horizontal")
        if not gs.all(self._total_space.is_tangent(nabla_x_y, base_point)):
            raise ValueError("Vector nabla_x_y is not tangent")
        a_x_y = self.integrability_tensor(
            horizontal_vec_x, horizontal_vec_y, base_point
        )
        if not gs.all(self.is_horizontal(nabla_x_y - a_x_y, base_point)):
            raise ValueError(
                "Tangent vector nabla_x_y is not the gradient "
                "of a horizontal distribution"
            )
        if not gs.all(self._total_space.is_tangent(tangent_vec_e, base_point)):
            raise ValueError("Tangent vector e is not tangent")
        if not gs.all(self._total_space.is_tangent(nabla_x_e, base_point)):
            raise ValueError("Vector nabla_x_e is not tangent")

        p_top = Matrices.transpose(base_point)
        p_top_p = gs.matmul(p_top, base_point)
        e_top = Matrices.transpose(tangent_vec_e)
        x_top = Matrices.transpose(horizontal_vec_x)
        y_top = Matrices.transpose(horizontal_vec_y)

        def sylv_p(mat_b):
            """Solves Sylvester equation for vertical component."""
            return gs.linalg.solve_sylvester(
                p_top_p, p_top_p, mat_b - Matrices.transpose(mat_b)
            )

        omega_ep = sylv_p(gs.matmul(p_top, tangent_vec_e))
        omega_ye = sylv_p(gs.matmul(e_top, horizontal_vec_y))
        tangent_vec_b = gs.matmul(horizontal_vec_x, omega_ye)
        tangent_vec_e_sym = tangent_vec_e - 2.0 * gs.matmul(base_point, omega_ep)

        a_y_e = gs.matmul(base_point, omega_ye) + gs.matmul(horizontal_vec_y, omega_ep)

        tmp_tangent_vec_p = (
            gs.matmul(e_top, nabla_x_y)
            - gs.matmul(y_top, nabla_x_e)
            - 2.0 * gs.matmul(p_top, tangent_vec_b)
        )

        tmp_tangent_vec_y = gs.matmul(p_top, nabla_x_e) + gs.matmul(
            x_top, tangent_vec_e_sym
        )

        scal_x_a_y_e = self._total_space.metric.inner_product(
            horizontal_vec_x, a_y_e, base_point
        )

        nabla_x_a_y_e = (
            gs.matmul(base_point, sylv_p(tmp_tangent_vec_p))
            + gs.matmul(horizontal_vec_y, sylv_p(tmp_tangent_vec_y))
            + gs.matmul(nabla_x_y, omega_ep)
            + tangent_vec_b
            + gs.einsum("...,...ij->...ij", scal_x_a_y_e, base_point)
        )

        return nabla_x_a_y_e, a_y_e

    def integrability_tensor_derivative_parallel(
        self, horizontal_vec_x, horizontal_vec_y, horizontal_vec_z, base_point
    ):
        r"""Compute derivative of the integrability tensor A (special case).

        The horizontal covariant derivative :math:`\nabla_X (A_Y Z)` of the
        integrability tensor A may be computed more efficiently in the case of
        parallel vector fields in the quotient space.
        :math:`\nabla_X (A_Y Z)` and :math:`A_Y Z` are computed here for the
        Kendall shape space with quotient-parallel vector fields :math:`X,
        Y, Z` extending the values horizontal_vec_x, horizontal_vec_y and
        horizontal_vec_z by parallel transport in a neighborhood of the
        base-space. Such vector fields verify :math:`\nabla_X^X = A_X X =
        0`, :math:`\nabla_X^Y = A_X Y` and similarly for Z.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        horizontal_vec_z : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point of the total space.

        Returns
        -------
        nabla_x_a_y_z : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`, result of
            :math:`\nabla_X (A_Y Z)` with `X = horizontal_vec_x`,
            `Y = horizontal_vec_y` and `Z = horizontal_vec_z`.
        a_y_z : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`, result of :math:`A_Y Z`
            with `Y = horizontal_vec_y` and `Z = horizontal_vec_z`.

        References
        ----------
        .. [Pennec] Pennec, Xavier. Computing the curvature and its gradient
            in Kendall shape spaces. Unpublished.
        """
        # Vectors X and Y have to be horizontal.
        if not gs.all(self._total_space.is_centered(base_point)):
            raise ValueError("The base_point does not belong to the pre-shape space")
        if not gs.all(self.is_horizontal(horizontal_vec_x, base_point)):
            raise ValueError("Tangent vector x is not horizontal")
        if not gs.all(self.is_horizontal(horizontal_vec_y, base_point)):
            raise ValueError("Tangent vector y is not horizontal")
        if not gs.all(self.is_horizontal(horizontal_vec_z, base_point)):
            raise ValueError("Tangent vector z is not horizontal")

        p_top = Matrices.transpose(base_point)
        p_top_p = gs.matmul(p_top, base_point)

        def sylv_p(mat_b):
            """Solves Sylvester equation for vertical component."""
            return gs.linalg.solve_sylvester(
                p_top_p, p_top_p, mat_b - Matrices.transpose(mat_b)
            )

        z_top = Matrices.transpose(horizontal_vec_z)
        y_top = Matrices.transpose(horizontal_vec_y)
        omega_yz = sylv_p(gs.matmul(z_top, horizontal_vec_y))
        a_y_z = gs.matmul(base_point, omega_yz)
        omega_xy = sylv_p(gs.matmul(y_top, horizontal_vec_x))
        omega_xz = sylv_p(gs.matmul(z_top, horizontal_vec_x))

        omega_yz_x = gs.matmul(horizontal_vec_x, omega_yz)
        omega_xz_y = gs.matmul(horizontal_vec_y, omega_xz)
        omega_xy_z = gs.matmul(horizontal_vec_z, omega_xy)

        tangent_vec_f = 2.0 * omega_yz_x + omega_xz_y - omega_xy_z
        omega_fp = sylv_p(gs.matmul(p_top, tangent_vec_f))
        omega_fp_p = gs.matmul(base_point, omega_fp)

        nabla_x_a_y_z = omega_yz_x - omega_fp_p

        return nabla_x_a_y_z, a_y_z

    def iterated_integrability_tensor_derivative_parallel(
        self, horizontal_vec_x, horizontal_vec_y, base_point
    ):
        r"""Compute iterated derivatives of the integrability tensor A.

        The iterated horizontal covariant derivative
        :math:`\nabla_X (A_Y A_X Y)` (where :math:`X` and :math:`Y` are
        horizontal vector fields) is a key ingredient in the computation of
        the covariant derivative of the directional curvature in a submersion.

        The components :math:`\nabla_X (A_Y A_X Y)`, :math:`A_X A_Y A_X Y`,
        :math:`\nabla_X (A_X Y)`,  and intermediate computations
        :math:`A_Y A_X Y` and :math:`A_X Y` are computed here for the
        Kendall shape space in the special case of quotient-parallel vector
        fields :math:`X, Y` extending the values horizontal_vec_x and
        horizontal_vec_y by parallel transport in a neighborhood.
        Such vector fields verify :math:`\nabla_X^X = A_X X` and
        :math:`\nabla_X^Y = A_X Y`.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point of the total space.

        Returns
        -------
        nabla_x_a_y_a_x_y : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`, result of
            :math:`\nabla_X^S (A_Y A_X Y)` with
            `X = horizontal_vec_x` and `Y = horizontal_vec_y`.
        a_x_a_y_a_x_y : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`, result of
            :math:`A_X A_Y A_X Y` with
            `X = horizontal_vec_x` and `Y = horizontal_vec_y`.
        nabla_x_a_x_y : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`, result of
            :math:`\nabla_X^S (A_X Y)` with
            `X = horizontal_vec_x` and `Y = horizontal_vec_y`.
        a_y_a_x_y : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`, result of :math:`A_Y A_X Y` with
            `X = horizontal_vec_x` and `Y = horizontal_vec_y`.
        a_x_y : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`, result of :math:`A_X Y` with
            `X = horizontal_vec_x` and `Y = horizontal_vec_y`.

        References
        ----------
        .. [Pennec] Pennec, Xavier. Computing the curvature and its gradient
            in Kendall shape spaces. Unpublished.
        """
        if not gs.all(self._total_space.is_centered(base_point)):
            raise ValueError("The base_point does not belong to the pre-shape space")
        if not gs.all(self.is_horizontal(horizontal_vec_x, base_point)):
            raise ValueError("Tangent vector x is not horizontal")
        if not gs.all(self.is_horizontal(horizontal_vec_y, base_point)):
            raise ValueError("Tangent vector y is not horizontal")

        p_top = Matrices.transpose(base_point)
        p_top_p = gs.matmul(p_top, base_point)

        def sylv_p(mat_b):
            """Solves Sylvester equation for vertical component."""
            return gs.linalg.solve_sylvester(
                p_top_p, p_top_p, mat_b - Matrices.transpose(mat_b)
            )

        y_top = Matrices.transpose(horizontal_vec_y)
        x_top = Matrices.transpose(horizontal_vec_x)
        x_y_top = gs.matmul(y_top, horizontal_vec_x)
        omega_xy = sylv_p(x_y_top)
        vertical_vec_v = gs.matmul(base_point, omega_xy)
        omega_xy_x = gs.matmul(horizontal_vec_x, omega_xy)
        omega_xy_y = gs.matmul(horizontal_vec_y, omega_xy)

        v_top = Matrices.transpose(vertical_vec_v)
        x_v_top = gs.matmul(v_top, horizontal_vec_x)
        omega_xv = sylv_p(x_v_top)
        omega_xv_p = gs.matmul(base_point, omega_xv)

        y_v_top = gs.matmul(v_top, horizontal_vec_y)
        omega_yv = sylv_p(y_v_top)
        omega_yv_p = gs.matmul(base_point, omega_yv)

        nabla_x_v = 3.0 * omega_xv_p + omega_xy_x
        a_y_a_x_y = omega_yv_p + omega_xy_y
        tmp_mat = gs.matmul(x_top, a_y_a_x_y)
        a_x_a_y_a_x_y = -gs.matmul(base_point, sylv_p(tmp_mat))

        omega_xv_y = gs.matmul(horizontal_vec_y, omega_xv)
        omega_yv_x = gs.matmul(horizontal_vec_x, omega_yv)
        omega_xy_v = gs.matmul(vertical_vec_v, omega_xy)
        norms = Matrices.frobenius_product(vertical_vec_v, vertical_vec_v)
        sq_norm_v_p = gs.einsum("...,...ij->...ij", norms, base_point)

        tmp_mat = gs.matmul(p_top, 3.0 * omega_xv_y + 2.0 * omega_yv_x) + gs.matmul(
            y_top, omega_xy_x
        )

        nabla_x_a_y_v = (
            3.0 * omega_xv_y
            + omega_yv_x
            + omega_xy_v
            - gs.matmul(base_point, sylv_p(tmp_mat))
            + sq_norm_v_p
        )

        return nabla_x_a_y_v, a_x_a_y_a_x_y, nabla_x_v, a_y_a_x_y, vertical_vec_v


class PreShapeMetric(RiemannianMetric):
    """Procrustes metric on the pre-shape space."""

    def _flatten_point(self, point):
        sphere_embedding_dim = self._space._sphere.embedding_space.dim
        return gs.reshape(point, point.shape[:-2] + (sphere_embedding_dim,))

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, m_ambient]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., k_landmarks, m_ambient]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dk_landmarks, m_ambient]
            Point on the pre-shape space.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        return self._space.embedding_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space.

        Returns
        -------
        exp : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        flat_bp = self._flatten_point(base_point)
        flat_tan = self._flatten_point(tangent_vec)
        flat_exp = self._space._sphere.metric.exp(flat_tan, flat_bp)
        return gs.reshape(flat_exp, tangent_vec.shape)

    def log(self, point, base_point):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space.

        Returns
        -------
        log : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        batch_shape = get_batch_shape(self._space.point_ndim, point, base_point)

        flat_bp = self._flatten_point(base_point)
        flat_pt = self._flatten_point(point)

        flat_log = self._space._sphere.metric.log(flat_pt, flat_bp)

        return gs.reshape(flat_log, batch_shape + self._space.shape)

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point):
        r"""Compute the curvature.

        For three tangent vectors at a base point :math:`x,y,z`,
        the curvature is defined by
        :math:`R(X, Y)Z = \nabla_{[X,Y]}Z
        - \nabla_X\nabla_Y Z + - \nabla_Y\nabla_X Z`, where :math:`\nabla`
        is the Levi-Civita connection. In the case of the hypersphere,
        we have the closed formula
        :math:`R(X,Y)Z = \langle X, Z \rangle Y - \langle Y,Z \rangle X`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        base_point :  array-like, shape=[..., k_landmarks, m_ambient]
            Point on the group. Optional, default is the identity.

        Returns
        -------
        curvature : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        """
        batch_shape = get_batch_shape(
            self._space.point_ndim,
            base_point,
            tangent_vec_a,
            tangent_vec_b,
            tangent_vec_c,
        )
        flat_a = self._flatten_point(tangent_vec_a)
        flat_b = self._flatten_point(tangent_vec_b)
        flat_c = self._flatten_point(tangent_vec_c)
        flat_bp = self._flatten_point(base_point)

        curvature = self._space._sphere.metric.curvature(
            flat_a, flat_b, flat_c, flat_bp
        )
        return gs.reshape(curvature, batch_shape + self._space.shape)

    def curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b=None,
        tangent_vec_c=None,
        tangent_vec_d=None,
        base_point=None,
    ):
        r"""Compute the covariant derivative of the curvature.

        For four vectors fields :math:`H|_P =` `tangent_vec_a`,
        :math:`X|_P =` `tangent_vec_b`, :math:`Y|_P =` `tangent_vec_c`,
        :math:`Z|_P =` `tangent_vec_d` with
        tangent vector value specified in argument at the `base_point` :math:`P`,
        the covariant derivative of the curvature
        :math:`(\nabla_H R)(X, Y) Z |_P` is computed at the `base_point` :math:`P`.
        Since the sphere is a constant curvature space this vanishes identically.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point` along which the curvature is
            derived.
        tangent_vec_b : array-like, shape=[..., k_landmarks, m_ambient]
            Unused tangent vector at `base_point` (since curvature derivative
            vanishes).
        tangent_vec_c : array-like, shape=[..., k_landmarks, m_ambient]
            Unused tangent vector at `base_point` (since curvature derivative
            vanishes).
        tangent_vec_d : array-like, shape=[..., k_landmarks, m_ambient]
            Unused tangent vector at `base_point` (since curvature derivative
            vanishes).
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Unused point on the group.

        Returns
        -------
        curvature_derivative : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        """
        batch_shape = get_batch_shape(
            self._space.point_ndim,
            tangent_vec_a,
            tangent_vec_b,
            tangent_vec_c,
            tangent_vec_d,
            base_point,
        )
        return gs.zeros(batch_shape + self._space.shape)

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        """Compute the Riemannian parallel transport of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space.
        direction : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at a base point.
            Optional, default : None.
        end_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space, to transport to. Unused if
            `tangent_vec_b` is given.
            Optional, default : None.

        Returns
        -------
        transported : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        if direction is None:
            if end_point is not None:
                direction = self.log(end_point, base_point)
            else:
                raise ValueError(
                    "Either an end_point or a tangent_vec_b must be given to define the"
                    " geodesic along which to transport."
                )

        batch_shape = get_batch_shape(
            self._space.point_ndim, tangent_vec, base_point, direction, end_point
        )

        flat_bp = self._flatten_point(base_point)
        flat_tan_a = self._flatten_point(tangent_vec)
        flat_tan_b = self._flatten_point(direction)

        flat_transport = self._space._sphere.metric.parallel_transport(
            flat_tan_a, flat_bp, flat_tan_b
        )
        return gs.reshape(flat_transport, batch_shape + self._space.shape)

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.
        In the case of the sphere, it does not depend on the base point and is
        Pi everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        radius = gs.array(gs.pi)
        return repeat_out(self._space.point_ndim, radius, base_point)


class KendallShapeMetric(QuotientMetric):
    """Quotient metric on the shape space.

    The Kendall shape space is obtained by taking the quotient of the
    pre-shape space by the space of rotations of the ambient space.
    """

    def directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point=None
    ):
        r"""Compute the covariant derivative of the directional curvature.

        For two vectors fields :math:`X|_P =` `tangent_vec_a`,
        :math:`Y|_P =` `tangent_vec_b` with tangent vector value specified in argument
        at the `base_point` :math:`P`,
        the covariant derivative (in the direction :math:`X`)
        :math:`(\nabla_X R_Y)(X) |_P = (\nabla_X R)(Y, X) Y |_P` of the
        directional curvature (in the direction :math:`Y`)
        :math:`R_Y(X) = R(Y, X) Y` is a quadratic tensor in :math:`X` and :math:`Y`
        that plays an important role in the computation of the moments of the
        empirical Fréchet mean [Pennec]_.

        In more details, let :math:`X, Y` be the horizontal lift of parallel
        vector fields extending the tangent vectors given in argument by
        parallel transport in a neighborhood of the`base_point` :math:`P` in the
        base-space. Such vector fields verify :math:`\nabla^T_X X=0` and
        :math:`\nabla^T_X Y = A_X Y` using the connection :math:`\nabla^T`
        of the total space. Then the covariant derivative of the
        directional curvature tensor is given by
        :math:`\nabla_X (R_Y(X)) = hor \nabla^T_X (R^T_Y(X)) - A_X( ver R^T_Y(X))
        - 3 (\nabla_X^T A_Y A_X Y - A_X A_Y A_X Y )`, where :math:`R^T_Y(X)`
        is the directional curvature tensor of the total space.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the group.

        Returns
        -------
        curvature_derivative : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point`.
        """
        horizontal_x = self._fiber_bundle.horizontal_projection(
            tangent_vec_a, base_point
        )
        horizontal_y = self._fiber_bundle.horizontal_projection(
            tangent_vec_b, base_point
        )
        (
            nabla_x_a_y_a_x_y,
            a_x_a_y_a_x_y,
            _,
            _,
            _,
        ) = self._fiber_bundle.iterated_integrability_tensor_derivative_parallel(
            horizontal_x, horizontal_y, base_point
        )
        return 3.0 * (nabla_x_a_y_a_x_y - a_x_a_y_a_x_y)

    def parallel_transport(
        self,
        tangent_vec,
        base_point,
        direction=None,
        end_point=None,
        n_steps=100,
        step="rk4",
    ):
        r"""Compute the parallel transport of a tangent vec along a geodesic.

        Approximation of the solution of the parallel transport of a tangent
        vector a along the geodesic between two points `base_point` and
        `end_point` or alternatively defined by
        :math:`t\mapsto exp_{(base\_point)}(t*direction)`

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Initial point of the geodesic to transport along.
        direction : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector ar `base_point`, initial velocity of the geodesic to
            transport  along.
            Optional, default: None.
        end_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point to transport to. Unused if `tangent_vec_b` is given.
            Optional, default: None.
        n_steps : int
            Number of steps to use to approximate the solution of the
            ordinary differential equation.
            Optional, default: 100.
        step : str, {'euler', 'rk2', 'rk4'}
            Scheme to use in the integration scheme.
            Optional, default: 'rk4'.

        Returns
        -------
        transported :  array-like, shape=[..., k_landmarks, m_ambient]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.

        References
        ----------
        .. [GMTP21] Guigui, Nicolas, Elodie Maignant, Alain Trouvé, and Xavier
            Pennec. “Parallel Transport on Kendall Shape Spaces.”
            5th conference on Geometric Science of Information,
            Paris 2021. Lecture Notes in Computer Science.
            Springer, 2021. https://hal.inria.fr/hal-03160677.

        See Also
        --------
        Integration module: geomstats.integrator
        """
        if direction is None:
            if end_point is not None:
                direction = self.log(end_point, base_point)
            else:
                raise ValueError(
                    "Either an end_point or a tangent_vec_b must be given to define the"
                    " geodesic along which to transport."
                )
        horizontal_a = self._fiber_bundle.horizontal_projection(tangent_vec, base_point)
        horizontal_b = self._fiber_bundle.horizontal_projection(direction, base_point)

        def force(state, time):
            gamma_t = self._total_space.metric.exp(time * horizontal_b, base_point)
            speed = self._total_space.metric.parallel_transport(
                horizontal_b, base_point, time * horizontal_b
            )
            coef = self.inner_product(speed, state, gamma_t)
            normal = gs.einsum("...,...ij->...ij", coef, gamma_t)

            align = gs.matmul(Matrices.transpose(speed), state)
            right = align - Matrices.transpose(align)
            left = gs.matmul(Matrices.transpose(gamma_t), gamma_t)
            skew_ = gs.linalg.solve_sylvester(left, left, right)
            vertical_ = -gs.matmul(gamma_t, skew_)
            return vertical_ - normal

        flow = integrate(force, horizontal_a, n_steps=n_steps, step=step)
        return flow[-1]
