"""Classes for the pullback metric.

Lead author: Nina Miolane.
"""
import abc
import itertools
import math

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
    shape : tuple of int
        Shape of one element of the underlying manifold.
        Optional, default : None.
    """

    def __init__(self, dim, shape=None):
        super(PullbackDiffeoMetric, self).__init__(dim=dim, shape=shape)

        self._embedding_metric = None
        self._raw_jacobian_diffeomorphism = None
        self._raw_inverse_jacobian_diffeomorphism = None

        self.shape_dim = math.prod(shape)
        self.embedding_space_shape_dim = math.prod(self.embedding_metric.shape)

    @abc.abstractmethod
    def define_embedding_metric(self):
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
        self._embedding_metric = (
            self._embedding_metric or self.define_embedding_metric()
        )
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

    def raw_jacobian_diffeomorphism(self, base_point):
        r"""Raw jacobian of the diffeomorphism.

        Raw jacobian autodiff of diffeomorphism regardless of vectorization.

        Parameters
        ----------
        base_point : array-like, shape=[..., *shape]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., *i_shape, ..., *shape]
            Inner-product matrix.
        """
        if self._raw_jacobian_diffeomorphism is None:
            self._raw_jacobian_diffeomorphism = gs.autodiff.jacobian(
                self.diffeomorphism
            )
        return self._raw_jacobian_diffeomorphism(base_point)

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

        J = self.raw_jacobian_diffeomorphism(base_point)
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
        base_point, tangent_vec = gs.broadcast_to(base_point, tangent_vec.shape)
        rad = base_point.shape[: -len(self.shape)]

        J_flat = gs.reshape(
            self.jacobian_diffeomorphism(base_point),
            (-1, self.embedding_space_shape_dim, self.shape_dim),
        )
        tv_flat = gs.reshape(tangent_vec, (-1, self.shape_dim))

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

    def raw_inverse_jacobian_diffeomorphism(self, image_point):
        r"""Raw jacobian of the inverse_diffeomorphism.

        Raw jacobian autodiff of inverse_diffeomorphism regardless of vectorization.

        Parameters
        ----------
        image_point : array-like, shape=[..., *i_shape]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., *shape, ..., *i_shape]
            Inner-product matrix.
        """
        if self._raw_inverse_jacobian_diffeomorphism is None:
            self._raw_inverse_jacobian_diffeomorphism = gs.autodiff.jacobian(
                self.inverse_diffeomorphism
            )
        return self._raw_inverse_jacobian_diffeomorphism(image_point)

    def inverse_jacobian_diffeomorphism(self, image_point):
        r"""Inverse Jacobian of the diffeomorphism at image point.

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

        J = self.raw_inverse_jacobian_diffeomorphism(image_point)
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
        image_point, image_tangent_vec = gs.broadcast_to(
            image_point, image_tangent_vec.shape
        )
        rad = image_tangent_vec.shape[: -len(self.embedding_metric.shape)]

        J_flat = gs.reshape(
            self.inverse_jacobian_diffeomorphism(image_point),
            (-1, self.shape_dim, self.embedding_space_shape_dim),
        )

        itv_flat = gs.reshape(image_tangent_vec, (-1, self.embedding_space_shape_dim))

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
        """Compute the exponential map via diffeomorphic pullback.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *shape]
            Tangent vector.
        base_point : array-like, shape=[..., *shape]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., *shape]
            Riemannian exponential of tangent_vec at base_point.
        """
        image_base_point = self.diffeomorphism(base_point)
        image_tangent_vec = self.tangent_diffeomorphism(tangent_vec, base_point)
        image_exp = self.embedding_metric.exp(
            image_tangent_vec, image_base_point, **kwargs
        )
        exp = self.inverse_diffeomorphism(image_exp)
        return exp

    def log(self, point, base_point, **kwargs):
        """Compute the logarithm map via diffeomorphic pullback.

        Parameters
        ----------
        point : array-like, shape=[..., *shape]
            Point.
        base_point : array-like, shape=[..., *shape]
            Point.

        Returns
        -------
        log : array-like, shape=[..., *shape]
            Logarithm of point from base_point.
        """
        image_base_point = self.diffeomorphism(base_point)
        image_point = self.diffeomorphism(point)
        image_log = self.embedding_metric.log(image_point, image_base_point, **kwargs)
        log = self.inverse_tangent_diffeomorphism(image_log, image_base_point)
        return log

    def dist(self, point_a, point_b, **kwargs):
        """Compute the distance via diffeomorphic pullback.

        Parameters
        ----------
        point_a : array-like, shape=[..., *shape]
            Point a.
        point_b : array-like, shape=[..., *shape]
            Point b.

        Returns
        -------
        dist : array-like
            Distance between point_a and point_b.
        """
        image_point_a = self.diffeomorphism(point_a)
        image_point_b = self.diffeomorphism(point_b)
        distance = self.embedding_metric.dist(image_point_a, image_point_b, **kwargs)
        return distance

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point):
        """Compute the curvature via diffeomorphic pullback.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *shape]
            Tangent vector a.
        tangent_vec_b : array-like, shape=[..., *shape]
            Tangent vector b.
        tangent_vec_c : array-like, shape=[..., *shape]
            Tangent vector c.
        base_point : array-like, shape=[..., *shape]
            Base point.

        Returns
        -------
        curvature : array-like, shape=[..., *shape]
            Curvature in directions tangent_vec a, b, c at base_point.
        """
        image_base_point = self.diffeomorphism(base_point)
        image_tangent_vec_a = self.tangent_diffeomorphism(tangent_vec_a, base_point)
        image_tangent_vec_b = self.tangent_diffeomorphism(tangent_vec_b, base_point)
        image_tangent_vec_c = self.tangent_diffeomorphism(tangent_vec_c, base_point)
        image_curvature = self.embedding_metric.curvature(
            image_tangent_vec_a,
            image_tangent_vec_b,
            image_tangent_vec_c,
            image_base_point,
        )
        curvature = self.inverse_tangent_diffeomorphism(
            image_curvature, image_base_point
        )
        return curvature

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        """Compute the parallel transport via diffeomorphic pullback.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *shape]
            Tangent vector.
        base_point : array-like, shape=[..., *shape]
            Base point.
        direction : array-like, shape=[..., *shape]
            Direction.
        end_point : array-like, shape=[..., *shape]
            End point.

        Returns
        -------
        parallel_transport : array-like, shape=[..., *shape]
            Parallel transport.
        """
        image_base_point = self.diffeomorphism(base_point)
        image_tangent_vec = self.tangent_diffeomorphism(tangent_vec, base_point)

        if direction is None:
            image_direction = None
        else:
            image_direction = self.tangent_diffeomorphism(direction, base_point)

        if end_point is None:
            image_end_point = None
        else:
            image_end_point = self.diffeomorphism(end_point)

        image_parallel_transport = self.embedding_metric.parallel_transport(
            image_tangent_vec,
            image_base_point,
            direction=image_direction,
            end_point=image_end_point,
        )
        parallel_transport = self.inverse_tangent_diffeomorphism(
            image_parallel_transport, image_base_point
        )
        return parallel_transport
