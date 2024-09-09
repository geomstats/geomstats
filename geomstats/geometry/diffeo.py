"""Diffeomorpism implementation."""

import abc
import math

import geomstats.backend as gs
from geomstats.vectorization import get_batch_shape, repeat_out_multiple_ndim


class Diffeo:
    r"""Diffeormorphism.

    Let :math:`f` be the diffeomorphism
    :math:`f: M \rightarrow N` of the manifold
    :math:`M` into the manifold :math:`N`.
    """

    @abc.abstractmethod
    def __call__(self, base_point):
        """Diffeomorphism at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., *image_shape]
            Image point.
        """

    @abc.abstractmethod
    def inverse(self, image_point):
        r"""Inverse diffeomorphism at image point.

        :math:`f^{-1}: N \rightarrow M`

        Parameters
        ----------
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        """

    @abc.abstractmethod
    def tangent(self, tangent_vec, base_point=None, image_point=None):
        r"""Tangent diffeomorphism at base point.

        df_p is a linear map from T_pM to T_f(p)N.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image of the base point.

        Notes
        -----
        Signature choice (i.e. having the possibility to pass both base and image
        points) comes from performance considerations.
        In several methods (e.g. `PullbackDiffeoMetric.inner_product`) the need
        to call `tangent_diffeomorphism` comes after having called
        `diffeomorphism`, which means we have both `base_point` and
        `image_point` available.
        In some cases, `tangent_diffeomorphism` needs access to `base_point`
        (e.g. `SRVTransform`), while in others, it needs access to `image_point`
        (e.g. `CholeskyMap`).
        By passing both, we give the corresponding implementation the possibility
        to choose the one that is more convenient (performancewise).
        If we pass only one of the two, it has the mechanims to perform the
        necessary transformations (e.g. calling `diffeomorphism` or
        `inverse_diffeomorphism` on it).
        """

    @abc.abstractmethod
    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        r"""Inverse tangent diffeomorphism at image point.

        df^-1_p is a linear map from T_f(p)N to T_pM

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.

        Notes
        -----
        See `tangent_diffeomorphism` docstrings for signature considerations.
        """


class AutodiffDiffeo(Diffeo):
    """Diffeomorphism through autodiff."""

    def __init__(self, space_shape, image_space_shape=None):
        super().__init__()
        if image_space_shape is None:
            image_space_shape = space_shape

        self._space_shape = space_shape
        self._space_point_ndim = len(space_shape)

        self._image_space_shape = image_space_shape
        self._image_space_point_ndim = len(image_space_shape)

        self._shape_prod = math.prod(space_shape)
        self._image_shape_prod = math.prod(image_space_shape)

    def jacobian(self, base_point):
        r"""Jacobian of the diffeomorphism at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., *image_shape, *space_shape]
            Jacobian of the diffeomorphism.
        """
        return gs.autodiff.jacobian_vec(
            self.__call__, point_ndim=self._space_point_ndim
        )(base_point)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        r"""Tangent diffeomorphism at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image of the base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)

        batch_shape = get_batch_shape(self._space_point_ndim, tangent_vec, base_point)
        flat_batch_shape = (-1,) if batch_shape else ()

        j_flat = gs.reshape(
            self.jacobian(base_point),
            flat_batch_shape + (self._image_shape_prod, self._shape_prod),
        )
        tv_flat = gs.reshape(tangent_vec, flat_batch_shape + (self._shape_prod,))

        image_tv = gs.reshape(
            gs.einsum("...ij,...j->...i", j_flat, tv_flat),
            batch_shape + self._image_space_shape,
        )

        return image_tv

    def inverse_jacobian(self, image_point):
        r"""Jacobian of the inverse diffeomorphism at image point.

        Parameters
        ----------
        image_point : array-like, shape=[..., *image_shape]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., *shape, *image_shape]
            Jacobian of the inverse diffeomorphism.
        """
        return gs.autodiff.jacobian_vec(
            self.inverse, point_ndim=self._image_space_point_ndim
        )(image_point)

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        r"""Tangent diffeomorphism at image point.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Tangent vector at base point.
        image_point : array-like, shape=[..., *image_shape]
            Base point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        """
        if image_point is None:
            image_point = self(base_point)

        batch_shape = get_batch_shape(
            self._image_space_point_ndim, image_tangent_vec, image_point
        )
        flat_batch_shape = (-1,) if batch_shape else ()

        j_flat = gs.reshape(
            self.inverse_jacobian(image_point),
            flat_batch_shape + (self._shape_prod, self._image_shape_prod),
        )

        itv_flat = gs.reshape(
            image_tangent_vec, flat_batch_shape + (self._image_shape_prod,)
        )

        tv = gs.reshape(
            gs.einsum("...ij,...j->...i", j_flat, itv_flat),
            batch_shape + self._space_shape,
        )
        return tv


class ReversedDiffeo(Diffeo):
    """Reverses the direction of a diffeomorphism.

    Parameters
    ----------
    diffeo : Diffeo.
    """

    def __init__(self, diffeo):
        self.diffeo = diffeo

    def __call__(self, base_point):
        """Diffeomorphism at base point."""
        return self.diffeo.inverse(base_point)

    def inverse(self, image_point):
        """Inverse diffeomorphism at image point."""
        return self.diffeo(image_point)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point."""
        return self.diffeo.inverse_tangent(
            tangent_vec, image_point=base_point, base_point=image_point
        )

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Tangent diffeomorphism at image point."""
        return self.diffeo.tangent(
            image_tangent_vec, base_point=image_point, image_point=image_point
        )


class ComposedDiffeo(Diffeo):
    """A composed diffeomorphism.

    Parameters
    ----------
    diffeos : list[Diffeo]
        An (ordered) list of diffeomorphisms.
    """

    def __init__(self, diffeos):
        self.diffeos = diffeos

    def __call__(self, base_point):
        """Diffeomorphism at base point."""
        image_point = base_point
        for diffeo in self.diffeos:
            image_point = diffeo(image_point)

        return image_point

    def inverse(self, image_point):
        """Inverse diffeomorphism at image point."""
        base_point = image_point
        for diffeo in reversed(self.diffeos):
            base_point = diffeo.inverse(base_point)
        return base_point

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point."""
        if base_point is None:
            base_point = self.inverse(image_point)

        for diffeo in self.diffeos:
            image_point = diffeo(base_point)
            image_tangent_vec = diffeo.tangent(
                tangent_vec, base_point=base_point, image_point=image_point
            )
            tangent_vec = image_tangent_vec
            base_point = image_point

        return image_tangent_vec

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Inverse tangent diffeomorphism at image point."""
        if image_point is None:
            image_point = self(base_point)

        for diffeo in reversed(self.diffeos):
            base_point = diffeo.inverse(image_point)
            tangent_vec = diffeo.inverse_tangent(
                image_tangent_vec, image_point=image_point, base_point=base_point
            )
            image_tangent_vec = tangent_vec
            image_point = base_point

        return tangent_vec


class VectorSpaceDiffeo(Diffeo):
    """A diffeo between vector spaces."""

    def __init__(self, space_ndim, image_space_ndim=None):
        super().__init__()
        self.space_ndim = space_ndim
        self.image_space_ndim = image_space_ndim or space_ndim

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        r"""Tangent diffeomorphism at base point.

        df_p is a linear map from T_pM to T_f(p)N.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image of the base point.
        """
        out = self(tangent_vec)
        return repeat_out_multiple_ndim(
            out,
            self.space_ndim,
            (tangent_vec, base_point),
            self.image_space_ndim,
            (image_point,),
            out_ndim=self.image_space_ndim,
        )

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        r"""Inverse tangent diffeomorphism at image point.

        df^-1_p is a linear map from T_f(p)N to T_pM

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        """
        out = self.inverse(image_tangent_vec)
        return repeat_out_multiple_ndim(
            out,
            self.image_space_ndim,
            (image_tangent_vec, image_point),
            self.space_ndim,
            (base_point,),
            out_ndim=self.space_ndim,
        )


class InvolutionDiffeomorphism(Diffeo):
    """A diffeomorphism that is also an involution."""

    def inverse(self, image_point):
        r"""Inverse diffeomorphism at image point.

        :math:`f^{-1}: N \rightarrow M`

        Parameters
        ----------
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        """
        return self(image_point)

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        r"""Inverse tangent diffeomorphism at image point.

        df^-1_p is a linear map from T_f(p)N to T_pM

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        """
        return self.tangent(
            image_tangent_vec, base_point=image_point, image_point=base_point
        )
