"""Kendall Pre-Shape space."""

import logging

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.errors import check_tf_error
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6


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
    [Nava]  Nava-Yazdani, E., H.-C. Hege, T. J. Sullivan, and C. von Tycowicz.
            “Geodesic Analysis in Kendall’s Shape Space with Epidemiological
            Applications.”
            Journal of Mathematical Imaging and Vision 62, no. 4 549–59.
            https://doi.org/10.1007/s10851-020-00945-w.
    """

    def __init__(self, k_landmarks, m_ambient):
        embedding_manifold = Matrices(k_landmarks, m_ambient)
        super(PreShapeSpace, self).__init__(
            dim=m_ambient * (k_landmarks - 1) - 1,
            embedding_manifold=embedding_manifold,
            default_point_type='matrix',
            total_space=embedding_manifold)
        self.embedding_metric = self.embedding_manifold.metric
        self.k_landmarks = k_landmarks
        self.m_ambient = m_ambient
        self.metric = ProcrustesMetric(k_landmarks, m_ambient)

    def belongs(self, point, atol=TOLERANCE):
        """Test if a point belongs to the pre-shape space.

        This tests whether the point is centered and whether the point's
        Frobenius norm is 1.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point in Matrices space.
        atol : float
            Tolerance at which to evaluate norm == 1 and mean == 0.
            Optional, default: 1e-6.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the pre-shape space.
        """
        shape = point.shape[-2:] == (self.k_landmarks, self.m_ambient)
        frob_norm = self.embedding_metric.norm(point)
        diff = gs.abs(frob_norm - 1)
        is_centered = gs.logical_and(self.is_centered(point, atol), shape)
        return gs.logical_and(
            gs.less_equal(diff, atol), is_centered)

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
        frob_norm = self.embedding_metric.norm(centered_point)
        projected_point = gs.einsum(
            '...,...ij->...ij', 1. / frob_norm, centered_point)

        return projected_point

    def random_uniform(self, n_samples=1, tol=1e-6):
        """Sample in the pre-shape space from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        tol : float
            Tolerance.
            Optional, default: 1e-6.

        Returns
        -------
        samples : array-like, shape=[..., k_landmarks, m_ambient]
            Points sampled on the pre-shape space.
        """
        samples = Hypersphere(
            self.m_ambient * self.k_landmarks - 1).random_uniform(
            n_samples, tol)
        samples = gs.reshape(samples, (-1, self.k_landmarks, self.m_ambient))
        if n_samples == 1:
            samples = samples[0]
        return self.projection(samples)

    @staticmethod
    def is_centered(point, atol=TOLERANCE):
        """Check that landmarks are centered around 0.

        Parameters
        ----------
        point : array-like, shape=[..., k_landmarks, m_ambient]
            Point in Matrices space.
        atol :  float
            Tolerance at which to evaluate mean == 0.
            Optional, default: 1e-6.

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

    def to_tangent(self, vector, base_point=None):
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
        sq_norm = gs.sum(base_point ** 2, axis=(-1, -2))
        inner_prod = self.embedding_metric.inner_product(base_point, vector)
        coef = inner_prod / sq_norm
        tangent_vec = vector - gs.einsum('...,...ij->...ij', coef, base_point)

        return tangent_vec

    def is_tangent(self, vector, base_point=None, atol=TOLERANCE):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., k_landmarks, m_ambient]
            Vector.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the manifold.
            Optional, default: none.
        atol : float
            Absolute tolerance.
            Optional, default: 1e-6.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        is_centered = self.is_centered(vector, atol)
        inner_prod = self.embedding_metric.inner_product(base_point, vector)
        is_normal = gs.isclose(inner_prod, 0., atol=atol)
        return gs.logical_and(is_centered, is_normal)

    @staticmethod
    def vertical_projection(tangent_vec, base_point):
        r"""Project to vertical subspace.

        Compute the vertical component of a tangent vector :math: `w` at a
        base point :math: `x` by solving the sylvester equation:
        .. math::
                        `Axx^T + xx^TA = wx^T - xw^T`

        Then Ax is returned.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_landmarks, m_ambient]
            Tangent vector to the pre-shape space at `base_point`.
        base_point : array-like, shape=[..., k_landmarks, m_ambient]
            Point on the pre-shape space.

        Returns
        -------
        vertical : array-like, shape=[..., k_landmarks, m_ambient]
            Vertical component of `tangent_vec`.
        """
        transposed_point = Matrices.transpose(base_point)
        left_term = gs.matmul(transposed_point, base_point)
        alignment = gs.matmul(Matrices.transpose(tangent_vec), base_point)
        right_term = alignment - Matrices.transpose(alignment)
        skew = gs.linalg.solve_sylvester(left_term, left_term, right_term)

        return - gs.matmul(base_point, skew)

    def is_horizontal(self, tangent_vec, base_point, atol=TOLERANCE):
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
            Optional, default: 1e-6.

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
        if gs.any(conditioning < 5e-4):
            logging.warning(f'Singularity close, ill-conditioned matrix '
                            f'encountered: {conditioning}')
        if gs.any(gs.isclose(conditioning, 0.)):
            logging.warning("Alignment matrix is not unique.")
        if gs.any(det < 0):
            ones = gs.ones(self.m_ambient)
            reflection_vec = gs.concatenate(
                [ones[:-1], gs.array([-1.])], axis=0)
            mask = gs.cast(det < 0, gs.float32)
            sign = (mask[..., None] * reflection_vec
                    + (1. - mask)[..., None] * ones)
            j_matrix = from_vector_to_diagonal_matrix(sign)
            rotation = Matrices.mul(
                Matrices.transpose(right), j_matrix, Matrices.transpose(left))
        else:
            rotation = gs.matmul(
                Matrices.transpose(right), Matrices.transpose(left))
        return gs.matmul(point, Matrices.transpose(rotation))


class ProcrustesMetric(RiemannianMetric):
    """Procrustes metric on the pre-shape space.

    Parameters
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.
    """

    def __init__(self, k_landmarks, m_ambient):
        super(ProcrustesMetric, self).__init__(
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
        flat_bp = gs.reshape(base_point, (-1, self.sphere_metric.dim + 1))
        flat_tan = gs.reshape(tangent_vec, (-1, self.sphere_metric.dim + 1))
        flat_exp = self.sphere_metric.exp(flat_tan, flat_bp)
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
        flat_bp = gs.reshape(base_point, (-1, self.sphere_metric.dim + 1))
        flat_pt = gs.reshape(point, (-1, self.sphere_metric.dim + 1))
        flat_log = self.sphere_metric.log(flat_pt, flat_bp)
        try:
            log = gs.reshape(flat_log, base_point.shape)
        except (RuntimeError,
                check_tf_error(ValueError, 'InvalidArgumentError')):
            log = gs.reshape(flat_log, point.shape)
        return log


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
        super(KendallShapeMetric, self).__init__(
            fiber_bundle=PreShapeSpace(k_landmarks, m_ambient),
            ambient_metric=ProcrustesMetric(k_landmarks, m_ambient))
