"""Diffeomorpism implementation."""

import abc
import math

import geomstats.backend as gs
from geomstats.vectorization import get_batch_shape


class Diffeo:
    r"""Diffeormorphism.

    Let :math:`f` be the diffeomorphism
    :math:`f: M \rightarrow N` of the manifold
    :math:`M` into the manifold `N`.
    """

    @abc.abstractmethod
    def diffeomorphism(self, base_point):
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
    def inverse_diffeomorphism(self, image_point):
        r"""Inverse diffeomorphism at base point.

        :math:`f^-1: N \rightarrow M`

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
    def tangent_diffeomorphism(self, tangent_vec, base_point=None, image_point=None):
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
    def inverse_tangent_diffeomorphism(
        self, image_tangent_vec, image_point=None, base_point=None
    ):
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

    def __init__(self, space_shape, image_space_shape):
        super().__init__()
        self._space_shape = space_shape
        self._space_point_ndim = len(space_shape)

        self._image_space_shape = image_space_shape
        self._image_space_point_ndim = len(image_space_shape)

        self._shape_prod = math.prod(space_shape)
        self._image_shape_prod = math.prod(image_space_shape)

    def jacobian_diffeomorphism(self, base_point):
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
            self.diffeomorphism, point_ndim=self._space_point_ndim
        )(base_point)

    def tangent_diffeomorphism(self, tangent_vec, base_point=None, image_point=None):
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
            base_point = self.inverse_diffeomorphism(image_point)

        batch_shape = get_batch_shape(self._space_point_ndim, tangent_vec, base_point)
        flat_batch_shape = (-1,) if batch_shape else ()

        j_flat = gs.reshape(
            self.jacobian_diffeomorphism(base_point),
            flat_batch_shape + (self._image_shape_prod, self._shape_prod),
        )
        tv_flat = gs.reshape(tangent_vec, flat_batch_shape + (self._shape_prod,))

        image_tv = gs.reshape(
            gs.einsum("...ij,...j->...i", j_flat, tv_flat),
            batch_shape + self._image_space_shape,
        )

        return image_tv

    def inverse_jacobian_diffeomorphism(self, image_point):
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
            self.inverse_diffeomorphism, point_ndim=self._image_space_point_ndim
        )(image_point)

    def inverse_tangent_diffeomorphism(
        self, image_tangent_vec, image_point=None, base_point=None
    ):
        r"""Tangent diffeomorphism at base point.

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
            image_point = self.diffeomorphism(base_point)

        batch_shape = get_batch_shape(
            self._image_space_point_ndim, image_tangent_vec, image_point
        )
        flat_batch_shape = (-1,) if batch_shape else ()

        j_flat = gs.reshape(
            self.inverse_jacobian_diffeomorphism(image_point),
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

    def diffeomorphism(self, base_point):
        """Diffeomorphism at base point."""
        return self.diffeo.inverse_diffeomorphism(base_point)

    def inverse_diffeomorphism(self, image_point):
        """Inverse diffeomorphism at base point."""
        return self.diffeo.diffeomorphism(image_point)

    def tangent_diffeomorphism(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point."""
        return self.diffeo.inverse_tangent_diffeomorphism(
            tangent_vec, image_point=base_point, base_point=image_point
        )

    def inverse_tangent_diffeomorphism(
        self, image_tangent_vec, image_point=None, base_point=None
    ):
        """Tangent diffeomorphism at base point."""
        return self.diffeo.tangent_diffeomorphism(
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

    def diffeomorphism(self, base_point):
        """Diffeomorphism at base point."""
        image_point = base_point
        for diffeo in self.diffeos:
            image_point = diffeo.diffeomorphism(image_point)

        return image_point

    def inverse_diffeomorphism(self, image_point):
        """Inverse diffeomorphism at base point."""
        base_point = image_point
        for diffeo in reversed(self.diffeos):
            base_point = diffeo.inverse_diffeomorphism(base_point)
        return base_point

    def tangent_diffeomorphism(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point."""
        if base_point is None:
            base_point = self.inverse_diffeomorphism(image_point)

        for diffeo in self.diffeos:
            image_point = diffeo.diffeomorphism(base_point)
            image_tangent_vec = diffeo.tangent_diffeomorphism(
                tangent_vec, base_point=base_point, image_point=image_point
            )
            tangent_vec = image_tangent_vec
            base_point = image_point

        return image_tangent_vec

    def inverse_tangent_diffeomorphism(
        self, image_tangent_vec, image_point=None, base_point=None
    ):
        """Inverse tangent diffeomorphism at image point."""
        if image_point is None:
            image_point = self.diffeomorphism(base_point)

        for diffeo in reversed(self.diffeos):
            base_point = diffeo.inverse_diffeomorphism(image_point)
            tangent_vec = diffeo.inverse_tangent_diffeomorphism(
                image_tangent_vec, image_point=image_point, base_point=base_point
            )
            image_tangent_vec = tangent_vec
            image_point = base_point

        return tangent_vec
