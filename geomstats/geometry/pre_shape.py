"""Kendall Pre-Shape space."""

import logging

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from  geomstats.errors import check_tf_error
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6


class PreShapeSpace(EmbeddedManifold):
    """Class for the Kendall pre-shape space.

    The pre-shape space is the sphere of the space of centered k-ad of
    landmarks in R^m (for the Frobenius norm). It is endowed with the
    spherical Procrustes metric d(x, y):= arccos(tr(xy^t)).

    Parameters
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.
    """

    def __init__(self, k_landmarks, m_ambient):
        super(PreShapeSpace, self).__init__(
            dim=m_ambient * (k_landmarks - 1) - 1,
            embedding_manifold=Matrices(m_ambient, k_landmarks))
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
        atol
        point : array-like, shape=[..., m_ambient, k_landmarks]
            Point in Matrices space.
        atol : float
            Tolerance at which to evaluate norm == 1 and mean == 0.
            Optional, default: 1e-6.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the pre-shape space.
        """
        frob_norm = self.embedding_metric.norm(point)
        diff = gs.abs(frob_norm - 1)
        is_centered = self.is_centered(point, atol)
        return gs.logical_and(
            gs.less_equal(diff, atol), is_centered)

    def projection(self, point):
        """Project a point on the pre-shape space.

        Parameters
        ----------
        point : array-like, shape=[..., m_ambient, k_landmarks]
            Point in Matrices space.

        Returns
        -------
        projected_point : array-like, shape=[..., m_ambient, k_landmarks]
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
        samples : array-like, shape=[..., m_ambient, k_landmarks]
            Points sampled on the pre-shape space.
        """
        samples = Hypersphere(
            self.m_ambient * self.k_landmarks - 1).random_uniform(
            n_samples, tol)
        samples = gs.reshape(samples, (-1, self.m_ambient, self.k_landmarks))
        if n_samples == 1:
            samples = samples[0]
        return self.projection(samples)

    @staticmethod
    def is_centered(point, atol=TOLERANCE):
        """Check that landmarks are centered around 0.

        Parameters
        ----------
        point : array-like, shape=[..., m_ambient, k_landmarks]
            Point in Matrices space.
        atol :  float
            Tolerance at which to evaluate mean == 0.
            Optional, default: 1e-6.

        Returns
        -------
        is_centered : array-like, shape=[...,]
            Boolean evaluating if point is centered.
        """
        mean = gs.mean(point, axis=-1)
        return gs.all(gs.isclose(mean, 0., atol=atol), axis=-1)

    @staticmethod
    def center(point):
        """Center landmarks around 0.

        Parameters
        ----------
        point : array-like, shape=[..., m_ambient, k_landmarks]
            Point in Matrices space.

        Returns
        -------
        centered : array-like, shape=[..., m_ambient, k_landmarks]
            Point with centered landmarks.
        """
        mean = gs.mean(point, axis=-1)
        return Matrices.transpose(
            Matrices.transpose(point) - mean[..., None, :])

    def to_tangent(self, vector, base_point=None):
        """Project a vector to the tangent space.

        Project a vector in the embedding matrix space
        to the tangent space of the pre-shape space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., m, k]
            Vector in Matrix space.
        base_point : array-like, shape=[..., m , k]
            Point on the pre-shape space defining the tangent space,
            where the vector will be projected.

        Returns
        -------
        tangent_vec : array-like, shape=[..., m, k]
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
        vector : array-like, shape=[..., m, k]
            Vector.
        base_point : array-like, shape=[..., m, k]
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
        tangent_vec : array-like, shape=[..., m, k]
            Tangent vector to the pre-shape space at `base_point`.
        base_point : array-like, shape=[..., m, k]
            Point on the pre-shape space.

        Returns
        -------
        vertical : array-like, shape=[..., m, k]
            Vertical component of `tangent_vec`.
        """
        transposed_point = Matrices.transpose(base_point)
        left_term = gs.matmul(base_point, transposed_point)
        right_term = gs.matmul(tangent_vec, transposed_point) - gs.matmul(
            base_point, Matrices.transpose(tangent_vec))
        skew = gs.linalg.solve_sylvester(left_term, left_term, right_term)

        return gs.matmul(skew, base_point)

    @classmethod
    def horizontal_projection(cls, tangent_vec, base_point):
        r"""Project to horizontal subspace.

        Compute the horizontal component of a tangent vector :math: `w` at a
        base point :math: `x` by removing the vertical component.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., m, k]
            Tangent vector to the pre-shape space at `base_point`.
        base_point : array-like, shape=[..., m, k]
            Point on the pre-shape space.

        Returns
        -------
        horizontal : array-like, shape=[..., m, k]
            Horizontal component of `tangent_vec`.
        """
        return tangent_vec - cls.vertical_projection(tangent_vec, base_point)

    def is_horizontal(self, tangent_vec, base_point, atol=TOLERANCE):
        """Check whether the tangent vector is horizontal at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., m, k]
            Tangent vector.
        base_point : array-like, shape=[..., m, k]
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
        product = gs.matmul(tangent_vec, Matrices.transpose(base_point))
        is_tangent = self.is_tangent(tangent_vec, base_point, atol)
        is_symmetric = Matrices.is_symmetric(product, atol)
        return gs.logical_and(is_tangent, is_symmetric)

    def realign(self, point, base_point):
        """Realign point to base_point.

        Find the optimal rotation R in SO(m) such that the base point and
        R.point are well positioned.

        Parameters
        ----------
        point : array-like, shape=[..., m, k]
            Point on the manifold.
        base_point : array-like, shape=[..., m, k]
            Point on the manifold.

        Returns
        -------
        aligned : array-like, shape=[..., m, k]
            R.point.
        """
        M = gs.matmul(point, Matrices.transpose(base_point))
        U, S, Vh = gs.linalg.svd(M)
        if gs.any(gs.isclose(S[..., -1] + S[..., -2], 0.)):
            logging.warning("Alignement matrix is not unique.")
        det = gs.linalg.det(M)
        if gs.any(det < 0):
            ones = gs.ones(self.m_ambient)
            changer = gs.concatenate([ones[:-1], gs.array([-1.])], axis=0)
            mask = gs.cast(det < 0, gs.float32)
            sign = (mask[..., None] * changer[None, :]
                    + (1. - mask)[..., None] * ones[None, :])
            j_matrix = from_vector_to_diagonal_matrix(sign)
            R = Matrices.mul(
                Matrices.transpose(Vh), j_matrix, Matrices.transpose(U))
        else:
            R = gs.matmul(Matrices.transpose(Vh), Matrices.transpose(U))
        return gs.matmul(R, point)


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
            dim=m_ambient * (k_landmarks - 1) - 1)

        self.embedding_metric = MatricesMetric(m_ambient, k_landmarks)
        self.sphere_metric = Hypersphere(m_ambient * k_landmarks - 1).metric

        self.k_landmarks = k_landmarks
        self.m_ambient = m_ambient

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1], optional
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
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim + 1]
            Point on the pre-shape space.

        Returns
        -------
        exp : array-like, shape=[..., dim + 1]
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
        point : array-like, shape=[..., dim + 1]
            Point on the pre-shape space.
        base_point : array-like, shape=[..., dim + 1]
            Point on the pre-shape space.

        Returns
        -------
        log : array-like, shape=[..., dim + 1]
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
