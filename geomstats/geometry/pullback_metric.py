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
        embedding_dim,
        immersion,
        jacobian_immersion=None,
        tangent_immersion=None,
    ):
        super(PullbackMetric, self).__init__(dim=dim)
        self.embedding_metric = EuclideanMetric(embedding_dim)
        self.immersion = immersion
        if jacobian_immersion is None:
            jacobian_immersion = gs.autodiff.jacobian(immersion)
        self.jacobian_immersion = jacobian_immersion
        if tangent_immersion is None:

            def _tangent_immersion(v, x):
                return gs.matmul(jacobian_immersion(x), v)

        self.tangent_immersion = _tangent_immersion

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
    """
    Pullback metric via a diffeomorphism.

    Parameters
    ----------
    dim : int
        Dimension.
    """

    def __init__(self, dim, shape):
        super(PullbackDiffeoMetric, self).__init__(dim=dim, shape=shape)

        # Inner variable
        self._embedding_metric = None
        self._jacobian_diffeomorphism = None
        self._inverse_jacobian_diffeomorphism = None

        self.shape_dim = int(gs.prod(gs.array(shape)))
        self.embedding_space_shape_dim = int(
            gs.prod(gs.array(self.embedding_metric.shape))
        )

    @abc.abstractmethod
    def create_embedding_metric(self):
        r"""Create the metric this metric is in diffeomorphism with.

        This instantiate the metric to use as image space of the
        diffeomorphism.

        -------
        embedding_metric : RiemannianMetric object
            The metric of the embedding space
        """

    @property
    def embedding_metric(self):
        r"""Property wrapper around the metric.

        -------
        embedding_metric : RiemannianMetric object
            The metric of the embedding space
        """
        if self._embedding_metric is None:
            self._embedding_metric = self.create_embedding_metric()
        return self._embedding_metric

    @abc.abstractmethod
    def diffeomorphism(self, base_point):
        r"""Diffeomorphism at base point.

        Let :math:`f` be the diffeomorphism
        :math:`f: M \rightarrow N` of the manifold
        :math:`M` into the manifold `N`.

        Parameters
        ----------
        base_point : array-like, shape=[..., *shape]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., *i_shape]
            Inner-product matrix.
        """

    def jacobian_diffeomorphism(self, base_point):
        r"""Jacobian of the diffeomorphism at base point.

        Let :math:`f` be the diffeomorphism
        :math:`f: M \rightarrow N` of the manifold
        :math:`M` into the manifold `N`.

        Parameters
        ----------
        base_point : array-like, shape=[..., *shape]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., *i_shape, *shape]
            Inner-product matrix.
        """
        rad = base_point.shape[: -len(self.shape)]
        base_point = gs.reshape(base_point, (-1,) + self.shape)
        # Only compute graph once
        if self._jacobian_diffeomorphism is None:
            self._jacobian_diffeomorphism = gs.autodiff.jacobian(self.diffeomorphism)

        J = self._jacobian_diffeomorphism(base_point)

        # We are [N, *shape], restore the batch dimension as first dim
        J = gs.moveaxis(
            gs.diagonal(J, axis1=0, axis2=len(self.embedding_metric.shape) + 1),
            -1,
            0,
        )
        J = gs.reshape(J, rad + self.embedding_metric.shape + self.shape)
        return J

    def tangent_diffeomorphism(self, tangent_vec, base_point):
        r"""Tangent diffeomorphism at base point.

        Let :math:`f` be the diffeomorphism
        :math:`f: M \rightarrow N` of the manifold
        :math:`M` into the manifold `N`.

        df_p is a linear map from T_pM to T_f(p)N called
        the tangent immesion

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *shape]
            Tangent vector at base point.

        base_point : array-like, shape=[..., *shape]
            Base point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., *i_shape]
            Image tangent vector at image of the base point.
        """
        base_point, tangent_vec = gs.broadcast_arrays(base_point, tangent_vec)
        rad = base_point.shape[: -len(self.shape)]

        J_flat = gs.reshape(
            self.jacobian_diffeomorphism(base_point),
            (-1, self.embedding_space_shape_dim, self.shape_dim),
        )

        tv_flat = tangent_vec.reshape(-1, self.shape_dim)

        image_tv = gs.reshape(
            gs.einsum("...ij,...j->...i", J_flat, tv_flat),
            rad + self.embedding_metric.shape,
        )

        return image_tv

    @abc.abstractmethod
    def inverse_diffeomorphism(self, image_point):
        r"""Inverse diffeomorphism at base point.

        Let :math:`f` be the diffeomorphism
        :math:`f: M \rightarrow N` of the manifold
        :math:`M` into the manifold `N`.

        :math:`f^-1: N \rightarrow M` of the manifold

        Parameters
        ----------
        image_point : array-like, shape=[..., *i_shape]
            Base point.

        Returns
        -------
        base_point : array-like, shape=[..., *shape]
            Inner-product matrix.
        """

    def inverse_jacobian_diffeomorphism(self, image_point):
        r"""Invercse Jacobian of the diffeomorphism at image point.

        Parameters
        ----------
        image_point : array-like, shape=[..., *i_shape]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., *shape, *i_shape]
            Inner-product matrix.
        """
        rad = image_point.shape[: -len(self.embedding_metric.shape)]
        image_point = gs.reshape(image_point, (-1,) + self.embedding_metric.shape)

        # Only compute graph once
        if self._inverse_jacobian_diffeomorphism is None:
            self._inverse_jacobian_diffeomorphism = gs.autodiff.jacobian(
                self.inverse_diffeomorphism
            )

        J = self._inverse_jacobian_diffeomorphism(image_point)
        J = gs.moveaxis(gs.diagonal(J, axis1=0, axis2=len(self.shape) + 1), -1, 0)
        J = gs.reshape(J, rad + self.shape + self.embedding_metric.shape)
        return J

    def inverse_tangent_diffeomorphism(self, image_tangent_vec, image_point):
        r"""Tangent diffeomorphism at base point.

        Let :math:`f` be the diffeomorphism
        :math:`f: M \rightarrow N` of the manifold
        :math:`M` into the manifold `N`.

        df_p is a linear map from T_pM to T_f(p)N called
        the tangent immesion

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *i_shape]
            Tangent vector at base point.

        image_point : array-like, shape=[..., *i_shape]
            Base point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., *shape]
            Image tangent vector at image of the base point.
        """
        image_point, image_tangent_vec = gs.broadcast_arrays(
            image_point, image_tangent_vec
        )
        rad = image_tangent_vec.shape[: -len(self.embedding_metric.shape)]

        J_flat = gs.reshape(
            self.inverse_jacobian_diffeomorphism(image_point),
            (-1, self.shape_dim, self.embedding_space_shape_dim),
        )

        itv_flat = image_tangent_vec.reshape(-1, self.embedding_space_shape_dim)

        tv = gs.reshape(
            gs.einsum("...ij,...j->...i", J_flat, itv_flat), rad + self.shape
        )
        return tv

    def metric_matrix(self, base_point=None):
        """Metric matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., *shape]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        raise NotImplementedError(
            "The computation of the pullback metric matrix"
            " is not implemented yet in general shape setting."
        )

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., *shape]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., *shape]
            Tangent vector at base point.
        base_point: array-like, shape=[..., *shape]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        return self.embedding_metric.inner_product(
            self.tangent_diffeomorphism(tangent_vec_a, base_point),
            self.tangent_diffeomorphism(tangent_vec_b, base_point),
            self.diffeomorphism(base_point),
        )

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
        new_exp = self.embedding_metric.exp(new_tangent_vec, new_base_point, **kwargs)
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
        new_log = self.embedding_metric.log(new_point, new_base_point, **kwargs)
        log = self.inverse_tangent_diffeomorphism(new_log, new_base_point)
        return log

    def dist(self, point_a, point_b, **kwargs):
        """
        Compute the distance via diffeomorphic pullback.

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
        new_distance = self.embedding_metric.dist(new_point_a, new_point_b, **kwargs)
        return new_distance

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point):
        """
        Compute the curvature via diffeomorphic pullback.

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
        new_tangent_vec_a = self.tangent_diffeomorphism(tangent_vec_a, base_point)
        new_tangent_vec_b = self.tangent_diffeomorphism(tangent_vec_b, base_point)
        new_tangent_vec_c = self.tangent_diffeomorphism(tangent_vec_c, base_point)
        new_curvature = self.embedding_metric.curvature(
            new_tangent_vec_a, new_tangent_vec_b, new_tangent_vec_c, new_base_point
        )
        curvature = self.inverse_tangent_diffeomorphism(new_curvature, new_base_point)
        return curvature

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        """
        Compute the parallel transport via diffeomorphic pullback.

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
        new_tangent_vec = self.tangent_diffeomorphism(tangent_vec, base_point)
        if direction is None:
            new_direction = None
        else:
            new_direction = self.tangent_diffeomorphism(direction, base_point)
        if end_point is None:
            new_end_point = None
        else:
            new_end_point = self.diffeomorphism(end_point)
        new_parallel_transport = self.embedding_metric.parallel_transport(
            new_tangent_vec,
            new_base_point,
            direction=new_direction,
            end_point=new_end_point,
        )
        parallel_transport = self.inverse_tangent_diffeomorphism(
            new_parallel_transport, new_base_point
        )
        return parallel_transport
