"""Kendall Pre-Shape space."""

import logging

import geomstats.backend as gs
from geomstats.algebra_utils import flip_determinant
from geomstats.errors import check_tf_error
from geomstats.geometry.base import EmbeddedManifold
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.integrator import integrate


class PreShapeSpace(EmbeddedManifold, FiberBundle):
    r"""Class for the Kendall pre-shape space.

    The pre-shape space is the sphere of the space of centered k-ad of
    landmarks in :math: `R^m` (for the Frobenius norm). It is endowed with the
    spherical Procrustes metric d(x, y):= arccos(tr(xy^t)).

    Points are represented by :math: `k \times m` centred matrices as in
    [Nava]_. Beware that this is not the usual convention from the literature.

    Parameters
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.

    References
    ----------
    ..[Nava]  Nava-Yazdani, E., H.-C. Hege, T. J.Sullivan, and C. von Tycowicz.
              “Geodesic Analysis in Kendall’s Shape Space with Epidemiological
              Applications.”
              Journal of Mathematical Imaging and Vision 62, no. 4 549–59.
              https://doi.org/10.1007/s10851-020-00945-w.
    """

    def __init__(self, k_landmarks, m_ambient):
        embedding_manifold = Matrices(k_landmarks, m_ambient)
        embedding_metric = embedding_manifold.metric
        super(PreShapeSpace, self).__init__(
            dim=m_ambient * (k_landmarks - 1) - 1,
            embedding_space=embedding_manifold,
            submersion=embedding_metric.squared_norm, value=1.,
            tangent_submersion=embedding_metric.inner_product,
            ambient_metric=PreShapeMetric(k_landmarks, m_ambient))
        self.k_landmarks = k_landmarks
        self.m_ambient = m_ambient
        self.ambient_metric = PreShapeMetric(k_landmarks, m_ambient)

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
        """
        centered_point = self.center(point)
        frob_norm = self.ambient_metric.norm(centered_point)
        projected_point = gs.einsum(
            '...,...ij->...ij', 1. / frob_norm, centered_point)

        return projected_point

    def random_point(self, n_samples=1, bound=1.):
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
        """
        samples = Hypersphere(
            self.m_ambient * self.k_landmarks - 1).random_uniform(n_samples)
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
        return gs.all(gs.isclose(mean, 0., atol=atol), axis=-1)

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
        """
        if not gs.all(self.is_centered(base_point)):
            raise ValueError('The base_point does not belong to the pre-shape'
                             ' space')
        vector = self.center(vector)
        sq_norm = Matrices.frobenius_product(base_point, base_point)
        inner_prod = self.ambient_metric.inner_product(base_point, vector)
        coef = inner_prod / sq_norm
        tangent_vec = vector - gs.einsum('...,...ij->...ij', coef, base_point)

        return tangent_vec

    def vertical_projection(
            self, tangent_vec, base_point, return_skew=False):
        r"""Project to vertical subspace.

        Compute the vertical component of a tangent vector :math: `w` at a
        base point :math: `x` by solving the sylvester equation:
        .. math::
                        `Axx^T + xx^TA = wx^T - xw^T`

        where A is skew-symmetric. Then Ax is the vertical projection of w.

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

        vertical = - gs.matmul(base_point, skew)
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
        is_tangent = self.is_tangent(tangent_vec, base_point, atol)
        is_symmetric = Matrices.is_symmetric(product, atol)
        return gs.logical_and(is_tangent, is_symmetric)

    def align(self, point, base_point, **kwargs):
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
        mat = gs.matmul(Matrices.transpose(point), base_point)
        left, singular_values, right = gs.linalg.svd(mat)
        det = gs.linalg.det(mat)
        conditioning = (
            (singular_values[..., -2]
             + gs.sign(det) * singular_values[..., -1]) /
            singular_values[..., 0])
        if gs.any(conditioning < gs.atol):
            logging.warning(f'Singularity close, ill-conditioned matrix '
                            f'encountered: {conditioning}')
        if gs.any(gs.isclose(conditioning, 0.)):
            logging.warning("Alignment matrix is not unique.")
        flipped = flip_determinant(Matrices.transpose(right), det)
        return Matrices.mul(point, left, Matrices.transpose(flipped))

    def integrability_tensor(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the fundamental tensor A of the submersion.

        The fundamental tensor A is defined for tangent vectors of the total
        space by [O'Neill]_
        :math: `A_X Y = ver\nabla^M_{hor X} (hor Y)
            + hor \nabla^M_{hor X}( ver Y)`
        where :math: `hor,ver` are the horizontal and vertical projections.

        For the pre-shape space, we have closed-form expressions and the result
        does not depend on the vertical part of :math: `X`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., {ambient_dim, [n, n]}]
            Point of the total space.

        Returns
        -------
        vector : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector at `base_point`, result of the A tensor applied to
            `tangent_vec_a` and `tangent_vec_b`.

        References
        ----------
        [O'Neill]  O’Neill, Barrett. The Fundamental Equations of a Submersion,
        Michigan Mathematical Journal 13, no. 4 (December 1966): 459–69.
        https://doi.org/10.1307/mmj/1028999604.
        """
        # Only the horizontal part of a counts
        horizontal_a = self.horizontal_projection(tangent_vec_a, base_point)
        vertical_b, skew = self.vertical_projection(
            tangent_vec_b, base_point, return_skew=True)
        horizontal_b = tangent_vec_b - vertical_b

        # For the horizontal part of b
        transposed_point = Matrices.transpose(base_point)
        sigma = gs.matmul(transposed_point, base_point)
        alignment = gs.matmul(Matrices.transpose(horizontal_a), horizontal_b)
        right_term = alignment - Matrices.transpose(alignment)
        skew_hor = gs.linalg.solve_sylvester(sigma, sigma, right_term)
        vertical = - gs.matmul(base_point, skew_hor)

        # For the vertical part of b
        vert_part = -gs.matmul(horizontal_a, skew)
        tangent_vert = self.to_tangent(vert_part, base_point)
        horizontal_ = self.horizontal_projection(tangent_vert, base_point)

        return vertical + horizontal_


class PreShapeMetric(RiemannianMetric):
    """Procrustes metric on the pre-shape space.

    Parameters
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.
    """

    def __init__(self, k_landmarks, m_ambient):
        super(PreShapeMetric, self).__init__(
            dim=m_ambient * (k_landmarks - 1) - 1,
            default_point_type='matrix')

        self.embedding_metric = MatricesMetric(k_landmarks, m_ambient)
        self.sphere_metric = Hypersphere(m_ambient * k_landmarks - 1).metric

        self.k_landmarks = k_landmarks
        self.m_ambient = m_ambient

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
        inner_prod = self.embedding_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point)

        return inner_prod

    def exp(self, tangent_vec, base_point, **kwargs):
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
        flat_bp = gs.reshape(base_point, (-1, self.sphere_metric.dim + 1))
        flat_tan = gs.reshape(tangent_vec, (-1, self.sphere_metric.dim + 1))
        flat_exp = self.sphere_metric.exp(flat_tan, flat_bp)
        return gs.reshape(flat_exp, tangent_vec.shape)

    def log(self, point, base_point, **kwargs):
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
        flat_bp = gs.reshape(base_point, (-1, self.sphere_metric.dim + 1))
        flat_pt = gs.reshape(point, (-1, self.sphere_metric.dim + 1))
        flat_log = self.sphere_metric.log(flat_pt, flat_bp)
        try:
            log = gs.reshape(flat_log, base_point.shape)
        except (RuntimeError,
                check_tf_error(ValueError, 'InvalidArgumentError')):
            log = gs.reshape(flat_log, point.shape)
        return log

    def curvature(
            self, tangent_vec_a, tangent_vec_b, tangent_vec_c,
            base_point):
        r"""Compute the curvature.

        For three tangent vectors at a base point :math: `x,y,z`,
        the curvature is defined by
        :math: `R(X, Y)Z = \nabla_{[X,Y]}Z
        - \nabla_X\nabla_Y Z + - \nabla_Y\nabla_X Z`, where :math: `\nabla`
        is the Levi-Civita connection. In the case of the hypersphere,
        we have the closed formula
        :math: `R(X,Y)Z = \langle X, Z \rangle Y - \langle Y,Z \rangle X`.

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
        max_shape = base_point.shape
        for arg in [tangent_vec_a, tangent_vec_b, tangent_vec_c]:
            if arg.ndim >= 3:
                max_shape = arg.shape
        flat_shape = (-1, self.sphere_metric.dim + 1)
        flat_a = gs.reshape(tangent_vec_a, flat_shape)
        flat_b = gs.reshape(tangent_vec_b, flat_shape)
        flat_c = gs.reshape(tangent_vec_c, flat_shape)
        flat_bp = gs.reshape(base_point, flat_shape)
        curvature = self.sphere_metric.curvature(
            flat_a, flat_b, flat_c, flat_bp)
        curvature = gs.reshape(curvature, max_shape)
        return curvature

    def parallel_transport(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Riemannian parallel transport of a tangent vector.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at a base point.
        tangent_vec_b : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space.

        Returns
        -------
        transported : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        max_shape = (
            tangent_vec_a.shape if tangent_vec_a.ndim == 3
            else tangent_vec_b.shape)

        flat_bp = gs.reshape(base_point, (-1, self.sphere_metric.dim + 1))
        flat_tan_a = gs.reshape(
            tangent_vec_a, (-1, self.sphere_metric.dim + 1))
        flat_tan_b = gs.reshape(
            tangent_vec_b, (-1, self.sphere_metric.dim + 1))

        flat_transport = self.sphere_metric.parallel_transport(
            flat_tan_a, flat_tan_b, flat_bp)
        return gs.reshape(flat_transport, max_shape)


class KendallShapeMetric(QuotientMetric):
    """Quotient metric on the shape space.

    The Kendall shape space is obtained by taking the quotient of the
    pre-shape space by the space of rotations of the ambient space.

    Parameters
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.
    """

    def __init__(self, k_landmarks, m_ambient):
        bundle = PreShapeSpace(k_landmarks, m_ambient)
        super(KendallShapeMetric, self).__init__(
            fiber_bundle=bundle,
            dim=bundle.dim - int(m_ambient * (m_ambient - 1) / 2))

    def parallel_transport(
            self, tangent_vec_a, tangent_vec_b, base_point, n_steps=100,
            step='rk4'):
        r"""Compute the parallel transport of a tangent vec along a geodesic.

        Approximation of the solution of the parallel transport of a tangent
        vector a along the geodesic defined by :math: `t \mapsto exp_(
        base_point)(t* tangent_vec_b)`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k, m]
            Tangent vector at `base_point` to transport.
        tangent_vec_b : array-like, shape=[..., k, m]
            Tangent vector ar `base_point`, initial velocity of the geodesic to
            transport  along.
        base_point : array-like, shape=[..., k, m]
            Initial point of the geodesic.
        n_steps : int
            Number of steps to use to approximate the solution of the
            ordinary differential equation.
            Optional, default: 100
        step : str, {'euler', 'rk2', 'rk4'}
            Scheme to use in the integration scheme.
            Optional, default: 'rk4'.

        Returns
        -------
        transported :  array-like, shape=[..., k, m]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.

        References
        ----------
        [GMTP21]_   Guigui, Nicolas, Elodie Maignant, Alain Trouvé, and Xavier
                    Pennec. “Parallel Transport on Kendall Shape Spaces.”
                    5th conference on Geometric Science of Information,
                    Paris 2021. Lecture Notes in Computer Science.
                    Springer, 2021. https://hal.inria.fr/hal-03160677.

        See Also
        --------
        Integration module: geomstats.integrator
        """
        horizontal_a = self.fiber_bundle.horizontal_projection(
            tangent_vec_a, base_point)
        horizontal_b = self.fiber_bundle.horizontal_projection(
            tangent_vec_b, base_point)

        def force(state, time):
            gamma_t = self.ambient_metric.exp(time * horizontal_b, base_point)
            speed = self.ambient_metric.parallel_transport(
                horizontal_b, time * horizontal_b, base_point)
            coef = self.inner_product(speed, state, gamma_t)
            normal = gs.einsum('...,...ij->...ij', coef, gamma_t)

            align = gs.matmul(Matrices.transpose(speed), state)
            right = align - Matrices.transpose(align)
            left = gs.matmul(Matrices.transpose(gamma_t), gamma_t)
            skew_ = gs.linalg.solve_sylvester(left, left, right)
            vertical_ = - gs.matmul(gamma_t, skew_)
            return vertical_ - normal

        flow = integrate(force, horizontal_a, n_steps=n_steps, step=step)
        return flow[-1]
