"""Interpolation machinery."""

from abc import ABC, abstractmethod

import geomstats.backend as gs


class Interpolator(ABC):
    """Abstract class for interpolator."""

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

    @abstractmethod
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


class _LinearInterpolator1D(Interpolator, ABC):
    def __init__(self, data, point_ndim=1):
        self.data = data
        self.point_ndim = point_ndim

        time_axis = -(point_ndim + 1)
        self._n_times = self.data.shape[time_axis]

    @abstractmethod
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

    def _get_ratio(self, t, interval_index, end_index):
        """Get ratio within interval.

        Parameters
        ----------
        t : array-like, shape=[n_time]
            Interpolation time.
        interval_index : array-like, shape=[n_times]
        end_index : array-like, shape=[n_times]

        Returns
        -------
        ratio : array-like, shape=[n_time]
            Ratio of t within interval.
        """

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
        if not gs.is_array(t):
            t = gs.array([t])

        if gs.ndim(t) == 0:
            t = gs.expand_dims(t, axis=0)

        interval_index = self._from_t_to_interval(t)

        max_bound_reached = interval_index == self._n_times - 1

        end_index = gs.where(max_bound_reached, interval_index, interval_index + 1)

        point_ndim_slc = (slice(None),) * self.point_ndim
        initial_point = self.data[..., interval_index, *point_ndim_slc]
        end_point = self.data[..., end_index, *point_ndim_slc]

        ratio = self._get_ratio(t, interval_index, end_index)

        diff = end_point - initial_point
        ijk = "ijk"[: self.point_ndim]
        return initial_point + gs.einsum(f"t,...t{ijk}->...t{ijk}", ratio, diff)


class UniformUnitIntervalLinearInterpolator(_LinearInterpolator1D):
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
        super().__init__(data, point_ndim=point_ndim)
        self._delta = 1 / (self._n_times - 1)

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

    def _get_ratio(self, t, interval_index, end_index):
        """Get ratio within interval.

        Parameters
        ----------
        t : array-like, shape=[n_time]
            Interpolation time.
        interval_index : array-like, shape=[n_times]
            Ignored.
        end_index : array-like, shape=[n_times]
            Ignored.

        Returns
        -------
        ratio : array-like, shape=[n_time]
            Ratio of t within interval.
        """
        return gs.mod(t, self._delta) / self._delta


class LinearInterpolator1D(_LinearInterpolator1D):
    """A 1D linear interpolator.

    Assumes interpolation occurs in the unit interval.

    Parameters
    ----------
    times : array-like, [n_times]
        Times. Must be sorted.
    data : array-like, [..., *point_shape]
    point_ndim : int
        Dimension of point.
    """

    def __init__(self, times, data, point_ndim=1):
        super().__init__(data, point_ndim=point_ndim)
        self.times = times
        self._delta = self.times[1:] - self.times[:-1]

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
        indices = gs.searchsorted(self.times, t) - 1
        return gs.where(indices < 0, 0, indices)

    def _get_ratio(self, t, interval_index, end_index):
        """Get ratio within interval.

        Parameters
        ----------
        t : array-like, shape=[n_time]
            Interpolation time.
        interval_index : array-like, shape=[n_times]
        end_index : array-like, shape=[n_times]

        Returns
        -------
        ratio : array-like, shape=[n_time]
            Ratio of t within interval.
        """
        delta = self._delta[interval_index]
        return (delta - (self.times[end_index] - t)) / delta
