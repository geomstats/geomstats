"""Bregman divergence on a manifold."""

import geomstats.backend as gs


class BregmanDivergence:
    """Bregman divergence on a manifold."""

    def __init__(self, space, potential_function):
        r"""Initialize the Bregman divergence.

        Parameters
        ----------
        space : Manifold
            Manifold to compute the Bregman divergence on.
        potential_function : callable
            Scalar convex twice differentiable function.
        """
        self.space = space
        self.potential_function = potential_function
        self._check_coordinates(space)

    def _check_coordinates(self, space):
        """Check the manifold is defined in intrinsic coordinates."""
        if not space.intrinsic:
            raise ValueError(
                "Space should be equipped with intrinsic coordinates for sampling"
                "points to compute divergence."
            )

    def bregman_divergence(self, basepoint, point):
        r"""Compute the Bregman divergence between two points.

        The Bregman divergence of a function F at points x and y is defined as:
        .. math::
            D_F(x, y) = F(x) - F(y) - \langle \nabla F(y), x - y \rangle

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point on manifold to compute gradient at.
        point : array-like, shape=[..., dim]
            Point to measure the Bregman divergence to.

        Returns
        -------
        bregman_div : array
            Bregman divergence.
        """
        grad_func = gs.autodiff.value_and_grad(self.potential_function)
        func_basepoint, grad_func_basepoint = grad_func(basepoint)
        bregman_div = (
            self.potential_function(point)
            - func_basepoint
            - gs.dot(grad_func_basepoint, point - basepoint)
        )
        return bregman_div
