"""Manifold for sets of landmarks that belong to any given manifold."""

import geomstats.backend as gs
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric import \
    ProductRiemannianMetric, RiemannianMetric


class Landmarks(ProductManifold):
    """Class for space of landmarks.

    The landmark space is a product manifold where all manifolds in the
    product are the same. The default metric is the product metric and
    is often referred to as the L2 metric.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold to which landmarks belong.
    n_landmarks : int
        Number of landmarks.
    """

    def __init__(self, ambient_manifold, n_landmarks):
        super(Landmarks, self).__init__(
            manifolds=[ambient_manifold] * n_landmarks,
            default_point_type='matrix')
        self.ambient_manifold = ambient_manifold
        self.metric = L2Metric(ambient_manifold, n_landmarks)
        self.n_landmarks = n_landmarks


class L2Metric(ProductRiemannianMetric):
    """L2 Riemannian metric on the space of landmarks.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which landmarks lie
    n_landmarks: int
            Number of landmarks.

    """

    def __init__(self, ambient_manifold, n_landmarks):
        super(L2Metric, self).__init__(
            metrics=[ambient_manifold.metric] * n_landmarks,
            default_point_type='matrix')
        self.ambient_manifold = ambient_manifold
        self.ambient_metric = ambient_manifold.metric

    def geodesic(
            self, initial_point, end_point=None, initial_tangent_vec=None):
        """Generate parameterized function for the geodesic curve.

        Geodesic curve defined by either:
        - an initial landmark set and an initial tangent vector,
        - an initial landmark set and an end landmark set.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Landmark set, initial point of the geodesic.
        end_point : array-like, shape=[..., dim]
            Landmark set, end point of the geodesic. If None,
            an initial tangent vector must be given.
            Optional, default : None
        initial_tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point, the initial speed of the geodesics.
            If None, an end point must be given and a logarithm is computed.
            Optional, default : None

        Returns
        -------
        path : callable
            Time parameterized geodesic curve.
        """
        landmarks_ndim = 2
        initial_landmarks = gs.to_ndarray(
            initial_point, to_ndim=landmarks_ndim + 1)

        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                'Specify an end landmark set or an initial tangent'
                'vector to define the geodesic.')
        if end_point is not None:
            end_landmarks = gs.to_ndarray(
                end_point, to_ndim=landmarks_ndim + 1)
            shooting_tangent_vec = self.log(
                point=end_landmarks, base_point=initial_landmarks)
            if initial_tangent_vec is not None:
                if not gs.allclose(shooting_tangent_vec, initial_tangent_vec):
                    raise RuntimeError(
                        'The shooting tangent vector is too'
                        ' far from the initial tangent vector.')
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(
            initial_tangent_vec, to_ndim=landmarks_ndim + 1)

        def landmarks_on_geodesic(t):
            t = gs.cast(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_landmarks = gs.to_ndarray(
                initial_landmarks, to_ndim=landmarks_ndim + 1)
            new_initial_tangent_vec = gs.to_ndarray(
                initial_tangent_vec, to_ndim=landmarks_ndim + 1)

            tangent_vecs = gs.einsum('il,nkm->ikm', t, new_initial_tangent_vec)

            def point_on_landmarks(tangent_vec):
                if gs.ndim(tangent_vec) < 2:
                    raise RuntimeError
                exp = self.exp(
                    tangent_vec=tangent_vec,
                    base_point=new_initial_landmarks)
                return exp

            landmarks_at_time_t = gs.vectorize(
                tangent_vecs,
                point_on_landmarks,
                signature='(i,j)->(i,j)')

            return landmarks_at_time_t

        return landmarks_on_geodesic


class KernelMetric(RiemannianMetric):
    r"""Kernel metric for the LDDMM framework on landmark spaces.

    Parameters
    ----------
    ambient_manifold : Riemannian manifold embedding the landmarks. Let's
    begin with the Euclidean space R^m.
    k_landmarks: int
        Number of landmarks.
    kernel : callable
        Kernel function to generate the space of admissible vector fields. It
        should take two points of the ambient space as inputs and output a
        scalar. A first example is the Gaussian kernel:
        .. math:

                    k(x, y) = exp(-|x-y|^2/ \sigma)
    """
    def __init__(self, ambient_manifold, k_landmarks, kernel):
        super(KernelMetric, self).__init__(
            default_point_type='matrix',
            dim=k_landmarks * ambient_manifold.dim)
        self.kernel = kernel

    def kernel_matrix(self, point_a, point_b):
        r"""Compute the kernel matrix.

        Let's first consider landmarks in a Euclidean space with the L2
        distance. This routine could later be improved using keops.

        .. math:
                    K_{i,j} = kernel(x_i, y_j)

        Where :math: `x_1, \ldots, x_k `are the landmark positions of
        `point_a` and :math: `y_1, \ldots, y_k `are the landmark positions of
        `point_b`.

        Parameters
        ----------
        point_a : landmark configuration
        point_b : landmark configuration

        Returns
        -------
        kernel_mat : [..., k_landmarks, k_landmarks]
        """
        raise NotImplementedError

    def sharp_map(self, covector, base_point):
        r"""Compute the tangent vector associated to covector.

        This is a convolution:
        .. math:
                    v(x) = \sum_{i=1}^k k(x, bp_i)m_i = K m

        where :math: `bp_1, \ldots, bp_k` are the landmark positions of
        `base_point`, and :math: `m_1, \dots, m_k` are the moment vectors of
        `covector`. This can be written as a matrix product with K the kernel
        matrix.

        Parameters
        ----------
        covector : array-like, shape=[..., k_landmarks, ambient_dim]
            Momentum vector
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Landmark configuration

        Returns
        -------
        vector_field : callable
            Velocity field associated to the covector. Takes a configuration
            as input and returns a velocity fields at this configuration.
        """
        def vector_field(position):
            raise NotImplementedError

        return vector_field

    def flat_map(self, tangent_vector, point, base_point):
        r"""Compute the covector associated to tangent_vector.

        This solves for the momentum vectors :math: `m_1, \ldots, m_k` such
        that at all :math: `x` given by `point`,
        .. math:
            v(x) = \sum_{i=1}^k k(x, bp_i)m_i = K m

        where :math: `bp_1, \ldots, bp_k` are the landmark positions of
        `base_point`. This is solved by inverting the kernel matrix.

        Parameters
        ----------
        tangent_vector : vector field evaluated at point
        point : landmark configuration
        base_point : landmark configuration at which to represent the
        momentum vectors.

        Returns
        -------
        momentum_vectors
        """
        raise NotImplementedError

    def cometric_inner_product(self, covector_1, covector_2, base_point):
        r"""Computes the inner product between to covectors.

        This is the inner product associated to the kernel matrix:
        .. math:

                <m, m'> = \sum_{i=1}^k p_i^T k(x_i, x_j) p_j

        Parameters
        ----------
        covector_1 : moment vectors at `base_point`
        covector_2 : moment vectors at `base_point`
        base_point : landmark configuration

        Returns
        -------
        inner_prod : float
        """
        raise NotImplementedError

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        r"""Computes the inner product between to tangnet vectors.

        This is the inner product associated to the kernel matrix and the
        covectors associated to the input tangent vectors. This equivalent to
        computing the L2 inner product between the whitened vector fields:
        .. math:

                <m, m'> = \sum_{i=1}^k p_i^T k(x_i, x_j) p_j =
                < K^{-1/2}v_a, K^{-1/2}v_b>_2

        Parameters
        ----------
        tangent_vec_a : moment vectors at `base_point`
        tangent_vec_b : moment vectors at `base_point`
        base_point : landmark configuration

        Returns
        -------
        inner_prod : float
        """
        raise NotImplementedError

    def hamiltonian(self, state):
        position, momentum = state
        return 1/2 * self.cometric_inner_product(
            momentum, momentum, position)

    def geodesic_equations(self, state):
        H_q, H_p = gs.autograd.elementwise_grad(self.hamiltonian)(state)
        return gs.array([H_p, - H_q])
