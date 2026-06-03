"""Base objects for varifold implementation."""

import abc

import geomstats.backend as gs


class Pairing(abc.ABC):
    """Kernel pairing."""

    def _get_kernel_prod_args(self, point_a, point_b):
        return [x for pair in zip(point_a[:-1], point_b[:-1]) for x in pair] + [
            point_b[-1]
        ]

    def __call__(self, point_a, point_b):
        """Evaluate kernel.

        Parameters
        ----------
        point_a : tuple
        point_b : tuple

        Returns
        -------
        scalar : float
        """
        kernel_args = self._get_kernel_prod_args(point_a, point_b)
        return gs.sum(self.kernel_prod(*kernel_args) * point_a[-1])
