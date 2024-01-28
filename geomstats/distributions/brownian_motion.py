"""Brownian motion defined on a manifold."""

import geomstats.backend as gs


class BrownianMotion:
    """Class to generate a realization of Brownian motion on a manifold.

    Parameters
    ----------
    space : Manifold
        Manifold to generate Brownian motion on.

    Example
    --------
    >>> import os
    >>> os.environ["GEOMSTATS_BACKEND"] = "pytorch"
    >>> import geomstats.backend as gs
    >>> from geomstats.geometry.euclidean import Euclidean
    >>> from geomstats.distributions.brownian_motion import BrownianMotion
    >>> manifold = Euclidean(dim=3)
    >>> euclidean_brownian_motion = BrownianMotion(manifold)
    >>> sample_path = euclidean_brownian_motion.sample_path(
            end_time=1,
            n_steps=50,
            initial_point=gs.array([0.0, 0.0, 0.0])
        )

    References
    ----------
    .. [H2022] Elton P. Hsu,
        "Stochastic Analysis On Manifolds",
        American Mathematical Soc. (2002): 71-99.
    """

    def __init__(self, space):
        self.space = space

    def sample_path(self, end_time, n_steps, initial_point):
        """Generate a sample path of Brownian motion.

        Parameters
        ----------
        end_time : float
        n_steps : int
        initial_point : array-like, shape=[..., dim]

        Returns
        -------
        path : array-like, shape=[..., n_steps, dim]
        """
        step_size = end_time / n_steps

        paths = [initial_point]
        for _ in range(1, n_steps):
            paths.append(self._step(step_size, paths[-1]))
        return gs.stack(paths)

    def _step(self, step_size, current_point):
        sigma = gs.linalg.sqrtm(self.space.metric.cometric_matrix(current_point))
        christoffels = self.space.metric.christoffels(current_point)
        cometric_matrix = self.space.metric.cometric_matrix(current_point)
        drift = (
            -0.5
            * gs.einsum("...klm,...lm->...k", christoffels, cometric_matrix)
            * step_size
        )
        batch_shape = current_point.shape[: -self.space.point_ndim]
        diffusion = gs.einsum(
            "...ij,...j->...i",
            sigma,
            gs.random.normal(size=batch_shape + (self.space.dim,)) * gs.sqrt(step_size),
        )

        return current_point + drift + diffusion
