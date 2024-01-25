"""Brownian motion defined on a manifold."""
import geomstats.backend as gs


class BrownianMotion:
    """Class to generate a realization of Brownian motion on a manifold.

    Parameters
    ----------
    space : Manifold obj,
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
    .. Elton P. Hsu,
        "Stochastic Analysis On Manifolds",
        American Mathematical Soc. (2002): 71-99.
    """

    def __init__(self, space):
        self._ensure_backend()
        self._check_metric(space)
        self._check_coordinates(space)
        self.space = space
        self.dim = space.dim
        self.metric = space.metric

    def _check_metric(self, space):
        if not hasattr(space, "metric"):
            raise ValueError("Manifold is not equipped with a metric.")
        else:
            if not hasattr(space.metric, "christoffels"):
                raise ValueError("Metric should have christoffels methods.")
            if not hasattr(space.metric, "cometric_matrix"):
                raise ValueError("Metric should have cometric_matrix methods.")

    def _check_coordinates(self, space):
        if space.default_coords_type != "intrinsic":
            raise ValueError(
                "Space should be equipped with intrinsic coordinates to create Brownian"
                "motion over the local parametrization."
            )

    def _ensure_backend(self):
        if not gs.__name__ == "geomstats.pytorch":
            raise ValueError("This class is only implemented for PyTorch backend.")

    def sample_path(self, end_time, n_steps, initial_point):
        """Generate a sample path of Brownian motion."""
        if self.space.metric.christoffels(initial_point).shape != tuple(
            self.space.dim for i in range(3)
        ):
            raise ValueError("Christoffels should defined intrinsically.")
        if self.space.metric.cometric_matrix(initial_point).shape != (
            self.space.dim,
            self.space.dim,
        ):
            raise ValueError("Cometric matrix should defined intrinsically.")

        path = gs.zeros((n_steps, self.dim))
        path[0] = initial_point
        step_size = end_time / n_steps
        for i in range(1, n_steps):
            path[i] = self._step(step_size, path[i - 1])

        return path

    def _step(self, step_size, current_point):
        sigma = gs.linalg.sqrtm(self.metric.cometric_matrix(current_point))
        christoffels = self.metric.christoffels(current_point)
        cometric_matrix = self.metric.cometric_matrix(current_point)
        drift = -0.5 * gs.einsum("klm,lm->k", christoffels, cometric_matrix) * step_size
        diffusion = gs.einsum(
            "ij, j->i", sigma, gs.random.normal(size=(self.dim,)) * gs.sqrt(step_size)
        )

        return current_point + drift + diffusion
