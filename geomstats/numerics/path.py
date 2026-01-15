"""Discrete-path related machinery."""

import geomstats.backend as gs
from geomstats.numerics.finite_differences import forward_difference
from geomstats.numerics.interpolation import UniformUnitIntervalLinearInterpolator


class UniformlySampledPathEnergy:
    """Riemannian path energy of a uniformly-sampled path.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    """

    def __init__(self, space):
        self._space = space

    def __call__(self, path):
        """Compute Riemannian path energy.

        Parameters
        ----------
        path : array-like, shape=[..., n_times, *point_shape]
            Piecewise linear path.

        Returns
        -------
        energy : array-like, shape=[...,]
            Path energy.
        """
        return self.energy(path)

    def energy_per_time(self, path):
        """Compute Riemannian path energy per time.

        Parameters
        ----------
        path : array-like, shape=[..., n_times, *point_shape]
            Piecewise linear path.

        Returns
        -------
        energy : array-like, shape=[..., n_times - 1]
            Stepwise path energy.
        """
        time_axis = -(self._space.point_ndim + 1)
        point_ndim_slc = tuple([slice(None)] * self._space.point_ndim)

        n_time = path.shape[time_axis]
        tangent_vecs = forward_difference(path, axis=time_axis)
        return self._space.metric.squared_norm(
            tangent_vecs,
            path[(..., slice(0, -1)) + point_ndim_slc],
        ) / (2 * (n_time - 1))

    def energy(self, path):
        """Compute Riemannian path energy.

        Parameters
        ----------
        path : array-like, shape=[..., n_times, *point_shape]
            Piecewise linear path.

        Returns
        -------
        energy : array-like, shape=[...,]
            Path energy.
        """
        return gs.sum(self.energy_per_time(path), axis=-1)


class UniformlySampledDiscretePath:
    """A uniformly-sampled discrete path.

    Parameters
    ----------
    path : array-like, [..., *point_shape]
    interpolator : Interpolator1D
    """

    def __init__(self, path, interpolator=None, **interpolator_kwargs):
        if interpolator is None:
            interpolator = UniformUnitIntervalLinearInterpolator(
                path, **interpolator_kwargs
            )
        self.interpolator = interpolator

    def __call__(self, t):
        """Interpolate path.

        Parameters
        ----------
        t : array-like, shape=[n_time,]
            Interpolation time.

        Returns
        -------
        point : array-like, shape=[..., n_time, *point_shape]
        """
        if not gs.is_array(t):
            t = gs.array([t])

        if gs.ndim(t) == 0:
            t = gs.expand_dims(t, axis=0)
        return self.interpolator(t)
