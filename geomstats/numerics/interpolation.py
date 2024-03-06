"""Interpolation machinery."""

import geomstats.backend as gs


class UniformUnitIntervalLinearInterpolator:
    """A 1D linear interpolator.

    Assumes interpolation occurs in the unit interval and
    data is uniformly sampled.

    Parameters
    ----------
    data : array-like, [..., *point_shape]
    point_ndim : int
        Dimension of point.
    """

    def __init__(self, data, point_ndim=1):
        self.data = data
        self.point_ndim = point_ndim
        self._time_axis = -(point_ndim + 1)
        self._n_times = self.data.shape[self._time_axis]
        self._delta = 1 / (self._n_times - 1)

    def __call__(self, t):
        """Interpolate data.

        Parameters
        ----------
        t : array-like, shape=[n_time]
            Interpolation time.

        Returns
        -------
        point : array-like, shape=[..., n_time, *point_shape]
        """
        return self.interpolate(t)

    def _from_t_to_interval(self, t):
        """Get interval index from time.

        Parameters
        ----------
        t : array-like, shape=[n_time]
            Interpolation time.

        Returns
        -------
        interval_index : array-like, shape=[n_times]
        """
        return gs.cast(
            t // self._delta,
            dtype=gs.int32,
        )

    def interpolate(self, t):
        """Interpolate data.

        Parameters
        ----------
        t : array-like, shape=[n_time]
            Interpolation time.

        Returns
        -------
        point : array-like, shape=[..., n_time, *point_shape]
        """
        interval_index = self._from_t_to_interval(t)

        point_ndim_slc = (slice(None),) * self.point_ndim

        max_bound_reached = interval_index == self._n_times - 1

        end_index = gs.where(max_bound_reached, interval_index, interval_index + 1)

        initial_point = self.data[..., interval_index, *point_ndim_slc]
        end_point = self.data[..., end_index, *point_ndim_slc]

        diff = end_point - initial_point
        ratio = gs.mod(t, self._delta) / self._delta

        ijk = "ijk"[self.point_ndim]
        return initial_point + gs.einsum(f"t,...t{ijk}->...t{ijk}", ratio, diff)
