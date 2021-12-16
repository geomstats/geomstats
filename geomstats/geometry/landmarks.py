"""Manifold for sets of landmarks that belong to any given manifold."""

import geomstats.backend as gs
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric import ProductRiemannianMetric
from geomstats.geometry.riemannian_metric import RiemannianCometric


class Landmarks(ProductManifold):
    """Class for space of landmarks.

    The landmark space is a product manifold where all manifolds in the
    product are the same. The default metric is the product metric and
    is often referred to as the L2 metric.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold to which landmarks belong.
    k_landmarks : int
        Number of landmarks.
    """

    def __init__(self, ambient_manifold, k_landmarks):
        super(Landmarks, self).__init__(
            manifolds=[ambient_manifold] * k_landmarks, default_point_type="matrix"
        )
        self.ambient_manifold = ambient_manifold
        self.metric = L2Metric(ambient_manifold, k_landmarks)
        self.k_landmarks = k_landmarks


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
            metrics=[ambient_manifold.metric] * n_landmarks, default_point_type="matrix"
        )
        self.ambient_manifold = ambient_manifold
        self.ambient_metric = ambient_manifold.metric

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
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
        initial_landmarks = gs.to_ndarray(initial_point, to_ndim=landmarks_ndim + 1)

        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end landmark set or an initial tangent"
                "vector to define the geodesic."
            )
        if end_point is not None:
            end_landmarks = gs.to_ndarray(end_point, to_ndim=landmarks_ndim + 1)
            shooting_tangent_vec = self.log(
                point=end_landmarks, base_point=initial_landmarks
            )
            if initial_tangent_vec is not None:
                if not gs.allclose(shooting_tangent_vec, initial_tangent_vec):
                    raise RuntimeError(
                        "The shooting tangent vector is too"
                        " far from the initial tangent vector."
                    )
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(
            initial_tangent_vec, to_ndim=landmarks_ndim + 1
        )

        def landmarks_on_geodesic(t):
            t = gs.cast(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_landmarks = gs.to_ndarray(
                initial_landmarks, to_ndim=landmarks_ndim + 1
            )
            new_initial_tangent_vec = gs.to_ndarray(
                initial_tangent_vec, to_ndim=landmarks_ndim + 1
            )

            tangent_vecs = gs.einsum("il,nkm->ikm", t, new_initial_tangent_vec)

            def point_on_landmarks(tangent_vec):
                if gs.ndim(tangent_vec) < 2:
                    raise RuntimeError
                exp = self.exp(
                    tangent_vec=tangent_vec, base_point=new_initial_landmarks
                )
                return exp

            landmarks_at_time_t = gs.vectorize(
                tangent_vecs, point_on_landmarks, signature="(i,j)->(i,j)"
            )

            return landmarks_at_time_t

        return landmarks_on_geodesic


class KernelMetric(RiemannianCometric):
    r"""Kernel cometric for the LDDMM framework on landmark spaces.

    Parameters
    ----------
    k_landmarks : int
        Dimension of the Euclidean space R^n containing the landmarks.
    ambient_dimension: int
        Number of landmarks.
    kernel : callable
        Kernel function to generate the space of admissible vector fields. It
        should take two points of the ambient space as inputs and output a
        scalar. An example is the Gaussian kernel:
        .. math:

                    k(x, y) = exp(-|x-y|^2/ \sigma)
    """

    def __init__(self, k_landmarks, ambient_dimension, kernel=lambda d: gs.exp(-d)):
        super(KernelMetric, self).__init__(
            default_point_type="matrix", dim=ambient_dimension * k_landmarks
        )
        self.kernel = kernel
        self.ambient_dimension = ambient_dimension
        self.k_landmarks = k_landmarks

    def kernel_matrix(self, base_point):
        r"""Compute the kernel matrix.

        .. math:
                    K_{i,j} = kernel(x_i, x_j)

        Where :math: `x_i` are the landmarks of the base point :math: `x`

        Parameters
        ----------
        base_point : landmark configuration :math: `x`

        Returns
        -------
        kernel_mat : [..., k_landmarks, k_landmarks]
        """
        squared_dist = gs.sum((base_point[..., :, None, :] - base_point) ** 2, axis=-1)
        return self.kernel(squared_dist)

    def inner_coproduct(self, momentum_1, momentum_2, base_point):
        r"""Compute the inner coproduct between two momenta.

        This is the inner product associated to the kernel matrix:
        .. math:

                <m, m'> = \sum_{i=1}^k m_i^T k(x_i, x_j) m_j

        Parameters
        ----------
        momentum_1 : moment vectors at `base_point`
        momentum_2 : moment vectors at `base_point`
        base_point : landmark configuration

        Returns
        -------
        inner_prod : float
        """
        velocity_2 = gs.einsum(
            "...ij,...jk->...ik", self.kernel_matrix(base_point), momentum_2
        )
        inner_coproduct = gs.einsum("...ij,...ij->...", momentum_1, velocity_2)
        return inner_coproduct
