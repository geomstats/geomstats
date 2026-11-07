"""Non-vectorized dynamic programming aligner for discrete curves."""

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import SRVTransform, insert_zeros
from geomstats.geometry.fiber_bundle import AlignerAlgorithm
from geomstats.numerics.interpolation import (
    LinearInterpolator1D,
    UniformUnitIntervalLinearInterpolator,
)
from geomstats.vectorization import check_is_batch


class DynamicProgrammingAligner(AlignerAlgorithm):
    r"""Align two curves through dynamic programming.

    Variant of
    :class:`geomstats.geometry.discrete_curves.DynamicProgrammingAligner`
    with a non-vectorized implementation.

    Find the reparametrization gamma of end_curve that minimizes the distance
    between initial_curve and end_curve reparametrized by gamma, and output
    the corresponding distance, using a dynamic programming algorithm.

    The objective can be expressed in terms of square root velocity (SRV)
    representations: it is equivalent to finding the gamma that maximizes
    the L2 scalar product between :math:`initial_{srv}` and :math:`end_{srv}@\gamma`,
    where :math:`initial_{srv}` is the SRV representation of the initial curve
    and :math:`end_{srv}@\gamma` is the SRV representation of the end curve
    reparametrized by :math:`\gamma`, i.e

    .. math::
        end_{srv}@\gamma(t) = end_{srv}(\gamma(t))\cdot|\gamma(t)|^\frac{1}{2}

    The dynamic programming algorithm assumes that for every subinterval
    :math:`\left[\frac{i}{n},\frac{i+1}{n}\right]` of :math:`\left[0,1\right]`,
    gamma is linear.

    Parameters
    ----------
    total_space : Manifold
        Total space with reparametrizations fiber bundle structure.
    n_space_grid : int
        Number of sampling points of the unit interval.
    max_slope : int
        Maximum slope allowed for a reparametrization.

    References
    ----------
    .. [WAJ2007] M. Washington, S. Anuj & H. Joshi,
        "On Shape of Plane Elastic Curves", in International Journal of Computer
        Vision. 73(3):307-324, 2007.
    """

    def __init__(self, total_space, n_space_grid=100, max_slope=6):
        k_sampling_points = total_space.k_sampling_points

        super().__init__(total_space)
        self.n_space_grid = n_space_grid
        self.max_slope = max_slope

        self._srv_transform = SRVTransform(
            self._total_space.ambient_manifold,
            k_sampling_points,
        )

    def _resample_srv_function(self, srv_function):
        """Resample SRV function of a discrete curve.

        Parameters
        ----------
        srv_function : array, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.

        Returns
        -------
        srv : array, shape=[..., n_space_grid - 1, ambient_dim]
            SRV function of the curve at the right size.

        Notes
        -----
        If `n_space_grid` < `k_sampling_points`, it subsamples the curve.
        """
        n_space_grid = self.n_space_grid
        index = gs.array(range(n_space_grid - 1))

        ratio = (self._total_space.k_sampling_points - 1) / (n_space_grid - 1)
        indices = gs.cast(gs.floor(index * ratio), dtype=int)

        return srv_function[..., indices, :]

    @staticmethod
    def _compute_restricted_integral(
        srv_1, srv_2, x_min_index, x_max_index, y_min_index, y_max_index
    ):
        r"""Compute the value of an integral over a subinterval.

        Compute n * the value of the integral of

        .. math::

            q_1(t)\cdot q_2(\gamma(t))\cdot|\gamma(t)|^\frac{1}{2}

        over :math:`\left[\x_min,x_max\right]` where gamma restricted to
        :math:`\left[\x_min,x_max\right]` is a linear.

        Parameters
        ----------
        srv_1 : array, shape=[n, ambient_dim]
            SRV function of the initial curve.
        srv_2 : array, shape=[n, ambient_dim]
            SRV function of the end curve.
        x_min_index : int
            Beginning of the subinterval.
        x_max_index : int
            End of the subinterval.
        y_min_index : int
            Value of gamma at x_min.
        y_max_index : int
            Value of gamma at x_max.

        Returns
        -------
        value : float
            Value of the integral described above.
        """
        delta_y_index = y_max_index - y_min_index
        delta_x_index = x_max_index - x_min_index

        gamma_slope = delta_y_index / delta_x_index

        x_bound_indices = list(range(x_min_index, x_max_index + 1))
        y_bounds_indices = [
            (y_index - y_min_index) / gamma_slope + x_min_index
            for y_index in range(y_min_index, y_max_index + 1)
        ]

        lower_bound = x_min_index
        x_index, y_index = 0, 0

        value = 0.0
        while x_index < delta_x_index and y_index < delta_y_index:
            upper_bound = min(
                x_bound_indices[x_index + 1], y_bounds_indices[y_index + 1]
            )

            # NB: normalization is done outside
            length = upper_bound - lower_bound
            value += length * gs.dot(
                srv_1[x_min_index + x_index], srv_2[y_min_index + y_index]
            )

            if (
                gs.abs(x_bound_indices[x_index + 1] - y_bounds_indices[y_index + 1])
                < gs.atol
            ):
                x_index += 1
                y_index += 1
            elif x_bound_indices[x_index + 1] < y_bounds_indices[y_index + 1]:
                x_index += 1
            else:
                y_index += 1
            lower_bound = upper_bound

        return gamma_slope**0.5 * value

    def _reparametrize(self, point_with_origin, gamma):
        """Reparametrize curve by gamma.

        Parameters
        ----------
        point_with_origin : array, shape=[k_sampling_points, ambient_dim]
            Discrete curve.
        gamma : array, shape=[n_subinterval]
            Parametrization of a curve.

        Returns
        -------
        new_point : array , shape=[k_sampling_points - 1, ambient_dim]
            Curve reparametrized by gamma.
        """
        n_space_grid = self.n_space_grid
        k_sampling_points = self._total_space.k_sampling_points

        x_uniform = gs.linspace(0.0, 1.0, k_sampling_points)
        point_interpolator = UniformUnitIntervalLinearInterpolator(
            point_with_origin, point_ndim=1
        )

        x_space = gs.array([gamma_elem[1] for gamma_elem in gamma]) / (n_space_grid - 1)
        y_space = gs.array([gamma_elem[0] for gamma_elem in gamma]) / (n_space_grid - 1)

        repar_inverse = LinearInterpolator1D(y_space, x_space, point_ndim=0)

        return point_interpolator(repar_inverse(x_uniform[1:]))

    def _compute_squared_dist(self, initial_srv, end_srv, grid_last):
        """Compute squared distance using algorithmic information."""
        n_space_grid = self.n_space_grid

        norm_squared_initial_srv = self._compute_restricted_integral(
            initial_srv, initial_srv, 0, n_space_grid - 1, 0, n_space_grid - 1
        ) / (n_space_grid - 1)
        norm_squared_end_srv = self._compute_restricted_integral(
            end_srv, end_srv, 0, n_space_grid - 1, 0, n_space_grid - 1
        ) / (n_space_grid - 1)

        maximum_scalar_product = grid_last / (n_space_grid - 1)
        return (
            norm_squared_initial_srv + norm_squared_end_srv - 2 * maximum_scalar_product
        )

    def _align_single(self, point, base_point, return_sdist=False):
        r"""Align point to base point, non vectorized.

        Parameters
        ----------
        point : array-like, shape=[k_sampling_points - 1, ambient_dim]
            Discrete curve to align.
        base_point : array-like, shape=[k_sampling_points - 1, ambient_dim]
            Reference discrete curve.
        return_sdist : bool
            If True, also returns squared distance.

        Returns
        -------
        aligned : array, shape=[k_sampling_points - 1, ambient_dim]
            Curve reparametrized in an optimal way with respect to reference curve.
        squared_dist : float
            Quotient distance between point and base point.
            If return_sdist is True.
        """
        max_slope = self.max_slope
        n_space_grid = self.n_space_grid

        initial_srv = self._srv_transform(base_point)
        end_srv = self._srv_transform(point)

        resampled_initial_srv = self._resample_srv_function(initial_srv)
        resampled_end_srv = self._resample_srv_function(end_srv)

        grid = {(0, 0): 0.0}
        gamma = {(0, 0): [(0, 0)]}

        for x_max_index in range(1, n_space_grid):
            max_y_max_index = max(n_space_grid, x_max_index + 1)
            for y_max_index in range(1, max_y_max_index):
                # NB: solves minimization part
                min_x_min_index = max(0, x_max_index - max_slope)
                min_y_min_index = max(0, y_max_index - max_slope)
                for x_min_index in range(min_x_min_index, x_max_index):
                    for y_min_index in range(min_y_min_index, y_max_index):
                        if (x_min_index, y_min_index) not in grid:
                            # jump borders
                            continue
                        elif (
                            gs.abs(
                                (y_max_index - y_min_index)
                                / (x_max_index - x_min_index)
                                - 1
                            )
                            < gs.atol
                            and x_max_index - x_min_index != 1
                            and y_max_index - y_max_index != 1
                        ):
                            # jump slope == 1 except last
                            continue

                        new_value = grid[
                            x_min_index, y_min_index
                        ] + self._compute_restricted_integral(
                            resampled_initial_srv,
                            resampled_end_srv,
                            x_min_index,
                            x_max_index,
                            y_min_index,
                            y_max_index,
                        )
                        if (x_max_index, y_max_index) not in grid or grid[
                            (x_max_index, y_max_index)
                        ] < new_value:
                            grid[x_max_index, y_max_index] = new_value
                            new_gamma = gamma[(x_min_index, y_min_index)].copy()
                            new_gamma.append((x_max_index, y_max_index))
                            gamma[(x_max_index, y_max_index)] = new_gamma

        last_index = n_space_grid - 1
        point_with_origin = insert_zeros(point, axis=-2)
        point_reparametrized = self._reparametrize(
            point_with_origin, gamma[last_index, last_index]
        )

        if not return_sdist:
            return point_reparametrized

        return point_reparametrized, self._compute_squared_dist(
            resampled_initial_srv,
            resampled_end_srv,
            grid[(last_index, last_index)],
        )

    def align(self, point, base_point, return_sdist=False):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve to align.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Reference discrete curve.
        return_sdist : bool
            If True, also returns squared distance.

        Returns
        -------
        aligned : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Curve reparametrized in an optimal way with respect to reference curve.
        squared_dist : array, shape=[...,]
            Quotient distance between point and base point.
            If return_sdist is True.
        """
        is_batch = check_is_batch(
            self._total_space.point_ndim,
            point,
            base_point,
        )
        if not is_batch:
            return self._align_single(point, base_point, return_sdist=return_sdist)

        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        out = [
            self._align_single(point_, base_point_, return_sdist=return_sdist)
            for point_, base_point_ in zip(point, base_point)
        ]
        if not return_sdist:
            return gs.stack(out)

        aligned = gs.stack([out_[0] for out_ in out])
        sdists = gs.stack([out_[1] for out_ in out])
        return aligned, sdists
