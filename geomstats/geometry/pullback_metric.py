"""Classes for the pullback metric.

Lead author: Nina Miolane.
"""
import abc
import itertools

import joblib

import geomstats.backend as gs
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric


class PullbackMetric(RiemannianMetric):
    r"""Pullback metric.

    Let :math:`f` be an immersion :math:`f: M \rightarrow N`
    of one manifold :math:`M` into the Riemannian manifold :math:`N`
    with metric :math:`g`.
    The pull-back metric :math:`f^*g` is defined on :math:`M` for a
    base point :math:`p` as:
    :math:`(f^*g)_p(u, v) = g_{f(p)}(df_p u , df_p v)
    \quad \forall u, v \in T_pM`

    Note
    ----
    The pull-back metric is currently only implemented for an
    immersion into the Euclidean space, i.e. for
    :math:`N=\mathbb{R}^n`.

    Parameters
    ----------
    dim : int
        Dimension of the underlying manifold.
    embedding_dim : int
        Dimension of the embedding Euclidean space.
    immersion : callable
        Map defining the immersion into the Euclidean space.
    """

    def __init__(
        self,
        dim,
        embedding_space,
        immersion,
        jacobian_immersion=None,
        tangent_immersion=None,
    ):
        super(PullbackMetric, self).__init__(dim=dim)
        self.embedding_space = embedding_space
        self.embedding_metric = embedding_space.metric

        self.immersion = immersion

        if jacobian_immersion is None:
            jacobian_immersion = gs.autodiff.jacobian(immersion)

        if tangent_immersion is None:

            def _tangent_immersion(v, x):
                # TODO: More general for shape
                return gs.einsum('...ij,...j->...j', jacobian_immersion(x), v)
            tangent_immersion = _tangent_immersion

        self.tangent_immersion = tangent_immersion

    def metric_matrix(self, base_point=None, n_jobs=1, **joblib_kwargs):
        r"""Metric matrix at the tangent space at a base point.

        Let :math:`f` be the immersion
        :math:`f: M \rightarrow \mathbb{R}^n` of the manifold
        :math:`M` into the Euclidean space :math:`\mathbb{R}^n`.

        The elements of the metric matrix at a base point :math:`p`
        are defined as:
        :math:`(f*g)_{ij}(p) = <df_p e_i , df_p e_j>`,
        for :math:`e_i, e_j` basis elements of :math:`M`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        immersed_base_point = self.immersion(base_point)
        jacobian_immersion = self.jacobian_immersion(base_point)
        basis_elements = gs.eye(self.dim)

        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_inner_product(i, j):
            immersed_basis_element_i = gs.matmul(jacobian_immersion, basis_elements[i])
            immersed_basis_element_j = gs.matmul(jacobian_immersion, basis_elements[j])
            return self.embedding_metric.inner_product(
                immersed_basis_element_i,
                immersed_basis_element_j,
                base_point=immersed_base_point,
            )

        pool = joblib.Parallel(n_jobs=n_jobs, **joblib_kwargs)
        out = pool(
            pickable_inner_product(i, j)
            for i, j in itertools.product(range(self.dim), range(self.dim))
        )

        metric_mat = gs.reshape(gs.array(out), (-1, self.dim, self.dim))
        return metric_mat[0] if base_point.ndim == 1 else metric_mat


class PullbackDiffeoMetric(RiemannianMetric, abc.ABC):

    def __init__(self, dim):
        super(PullbackMetric, self).__init__(dim=dim)
        self.embedding_space = self._create_embedding_space()
        self.embedding_metric = self.embedding_space.metric
        self._jacobian = None
        self._inverse_jacobian = None

    @abc.abstractmethod
    def _create_embedding_space(self):
        pass

    @abc.abstractmethod
    def diffeomorphism(self, base_point):
        pass

    @abc.abstractmethod
    def inverse_diffeomorphism(self, im_point):
        pass

    def jacobian(self, base_point):
        # Can be overwritten or never use if tangent method are
        # How it works when batched with autodiff ??
        if self._jacobian is None:
            self._jacobian = gs.autodiff.jacobian(self.diffeomorphism)
        return self._jacobian(base_point)

    def inverse_jacobian(self, im_point):
        # Can be overwritten or never use if tangent method are
        # How it works when batched with autodiff ??
        if self._inverse_jacobian is None:
            self._inverse_jacobian = gs.autodiff.jacobian(self.inverse_diffeomorphism)
        return self._inverse_jacobian(im_point)

    def tangent_diffeomorphism(self, tangent_vec, base_point):
        # Can be overwritten when close form
        # TODO: More general for shape
        return gs.matmul(self.jacobian(base_point), tangent_vec)

    def inverse_tangent_diffeomorphism(self, tangent_vec, im_point):
        # Can be overwritten when close form
        # TODO: More general for shape
        return gs.matmul(self.inverse_jacobian(im_point), tangent_vec)

    # HERE ALL THE METHOD IN THE CASE OF DIFFEO
    def exp(self, tangent_vec, base_point, **kwargs):
        """
        Compute the exponential map via diffeomorphic pullback.
        Parameters
        ----------
        tangent_vec : array-like
            Tangent vector.
        base_point : array-like
            Base point.
        Returns
        -------
        exp : array-like
            Riemannian exponential of tangent_vec at base_point.
        """
        new_base_point = self.diffeomorphism(base_point)
        new_tangent_vec = self.tangent_diffeomorphism(tangent_vec, base_point)
        new_exp = self.embedding_metric.exp(new_tangent_vec,
                                            new_base_point, **kwargs)
        exp = self.inverse_diffeomorphism(new_exp)
        return exp

    def log(self, point, base_point, **kwargs):
        """
        Compute the logarithm map via diffeomorphic pullback.
        Parameters
        ----------
        point : array-like
            Point.
        base_point : array-like
            Point.
        Returns
        -------
        log : array-like
            Logarithm of point from base_point.
        """
        new_base_point = self.diffeomorphism(base_point)
        new_point = self.diffeomorphism(point)
        new_log = self.embedding_metric.log(new_point, new_base_point,
                                            **kwargs)
        log = self.inverse_tangent_diffeomorphism(new_log, new_base_point)
        return log

    def dist(self, point_a, point_b, **kwargs):
        """
        Compute astutely the distance when the immersion is a diffeomorphism.
        Parameters
        ----------
        point_a : array-like
            Point a.
        point_b : array-like
            Point b.
        Returns
        -------
        dist : array-like
            Distance between point_a and point_b.
        """
        new_point_a = self.diffeomorphism(point_a)
        new_point_b = self.diffeomorphism(point_b)
        new_distance = self.embedding_metric.dist(new_point_a,
                                                  new_point_b, **kwargs)
        return new_distance

    def curvature(
            self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point):
        """
        Compute astutely the curvature when the immersion is a diffeomorphism.
        Parameters
        ----------
        tangent_vec_a : array-like
            Tangent vector a.
        tangent_vec_b : array-like
            Tangent vector b.
        tangent_vec_c : array-like
            Tangent vector c.
        base_point : array-like
            Base point.
        Returns
        -------
        curvature : array-like
            Curvature in directions tangent_vec a, b, c at base_point.
        """
        new_base_point = self.diffeomorphism(base_point)
        new_tangent_vec_a = self.tangent_diffeomorphism(tangent_vec_a,
                                                        base_point)
        new_tangent_vec_b = self.tangent_diffeomorphism(tangent_vec_b,
                                                        base_point)
        new_tangent_vec_c = self.tangent_diffeomorphism(tangent_vec_c,
                                                        base_point)
        new_curvature = self.embedding_metric.curvature(
            new_tangent_vec_a, new_tangent_vec_b, new_tangent_vec_c,
            new_base_point)
        curvature = self.inverse_tangent_diffeomorphism(new_curvature,
                                                        new_base_point)
        return curvature

    def parallel_transport(
            self, tangent_vec, base_point, direction=None, end_point=None
    ):
        """
        Compute astutely the parallel transport when the immersion is a diffeo.
        Parameters
        ----------
        tangent_vec : array-like
            Tangent vector.
        base_point : array-like
            Base point.
        direction : array-like
            Direction.
        end_point : array-like
            End point.
        Returns
        -------
        parallel_transport : array-like
            Parallel transport.
        """
        new_base_point = self.diffeomorphism(base_point)
        new_tangent_vec = self.tangent_diffeomorphism(tangent_vec)
        if direction is None:
            new_direction = None
        else:
            new_direction = self.tangent_diffeomorphism(direction)
        if end_point is None:
            new_end_point = None
        else:
            new_end_point = self.diffeomorphism(end_point)
        new_parallel_transport = self.embedding_metric.parallel_transport(
            new_tangent_vec, new_base_point, direction=new_direction,
            end_point=new_end_point)
        parallel_transport = self.inverse_tangent_diffeomorphism(
            new_parallel_transport, new_base_point
        )
        return parallel_transport