"""Graph Space."""

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices, MatricesMetric


class GraphSpace:
    r"""Class for the Graph Space.

    Graph Space to analyse populations of labelled and unlabelled graphs.
    The space is a quotient space obtained by applying the permutation
    action of nodes to the space of adjecency matrices.

    Points are represented by :math:`n \times n` adjecency matrices.

    Parameters
    ----------
    n : int
        Number of graph nodes
    p : int
        Dimension of euclidean parameter associated to a graph.

    References
    ----------
    ..[Calissano]  Calissano, A., Feragen, A., Vantini, S.
              “Graph Space: Geodesic Principal Components for aPopulation of Network-valued Data.”
              Mox report 14, 2020.
              https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    """

    def __init__(self, n, p=None):
        self.nodes = n
        self.p = p
        self.metric = GraphSpaceMetric(n)
        self.adjmat = Matrices(self.nodes, self.nodes)

    def belongs(self, mat, atol=gs.atol):
        r"""Check if the matrix is an adjecency matrix associated to the
        graph with n nodes.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if mat is belongs to the space.
        """
        return self.adjmat.belongs(mat)

    def random_point(self, n_samples=1, bound=1.0):
        r"""Sample in Graph Space.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix to be permuted.
        permutation : array-like, shape=[..., n, n]
            List containing the index permutation of the nodes of the graph

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled in GraphSpace(n).
        """
        return self.adjmat.random_point(n_samples=n_samples, bound= bound)

    def permute(self, mat, permutation, bound=1.0):
        r"""Permutation action applied to graph observation.

        Parameters
        ----------
        mat : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample in the tangent space.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points permuted.
        """
        return np.array([mat[i][:, permutation[i]][permutation[i], :] for i in range(len(permutation))])

class GraphSpaceMetric:
    """Quotient metric on the graph space.

    Parameters
    ----------
    n : int
        Number of nodes
    """

    def __init__(self, n):
        self.total_space_metric = MatricesMetric(k_landmarks, m_ambient)
        self.nodes = n

    def dist(self, a, b):
        """Compute the distance between two points.

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
            tangent_vec_a, tangent_vec_b, base_point
        )

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
        except (RuntimeError, check_tf_error(ValueError, "InvalidArgumentError")):
            log = gs.reshape(flat_log, point.shape)
        return log

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
        max_shape = base_point.shape
        for arg in [tangent_vec_a, tangent_vec_b, tangent_vec_c]:
            if arg.ndim >= 3:
                max_shape = arg.shape
        flat_shape = (-1, self.sphere_metric.dim + 1)
        flat_a = gs.reshape(tangent_vec_a, flat_shape)
        flat_b = gs.reshape(tangent_vec_b, flat_shape)
        flat_c = gs.reshape(tangent_vec_c, flat_shape)
        flat_bp = gs.reshape(base_point, flat_shape)
        curvature = self.sphere_metric.curvature(flat_a, flat_b, flat_c, flat_bp)
        curvature = gs.reshape(curvature, max_shape)
        return curvature

    def curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b=None,
        tangent_vec_c=None,
        tangent_vec_d=None,
        base_point=None,
    ):
        r"""Compute the covariant derivative of the curvature.

        For four vectors fields :math:`H|_P = tangent_vec_a, X|_P =
        tangent_vec_b, Y|_P = tangent_vec_c, Z|_P = tangent_vec_d` with
        tangent vector value specified in argument at the base point `P`,
        the covariant derivative of the curvature
        :math:`(\nabla_H R)(X, Y) Z |_P` is computed at the base point P.
        Since the sphere is a constant curvature space this
        vanishes identically.

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
            Tangent vector at base point.
        """
        return gs.zeros_like(tangent_vec_a)

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
            tangent_vec_a.shape if tangent_vec_a.ndim == 3 else tangent_vec_b.shape
        )

        flat_bp = gs.reshape(base_point, (-1, self.sphere_metric.dim + 1))
        flat_tan_a = gs.reshape(tangent_vec_a, (-1, self.sphere_metric.dim + 1))
        flat_tan_b = gs.reshape(tangent_vec_b, (-1, self.sphere_metric.dim + 1))

        flat_transport = self.sphere_metric.parallel_transport(
            flat_tan_a, flat_tan_b, flat_bp
        )
        return gs.reshape(flat_transport, max_shape)
