"""The open simplex."""

from scipy.stats import dirichlet

import geomstats.backend as gs
from geomstats._einsum import scalarvecmul
from geomstats.geometry.base import LevelSet
from geomstats.geometry.diffeo import Diffeo
from geomstats.geometry.positive_orthant import PositiveOrthant
from geomstats.vectorization import repeat_out


class SqrtDiffeo(Diffeo):
    """Componentwise square root.

    It maps the open simplex onto the positive orthant of the unit sphere.
    """

    @staticmethod
    def __call__(base_point):
        """Send point of the simplex to the sphere.

        The map takes the square root of each component.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the simplex.

        Returns
        -------
        image_point : array-like, shape=[..., dim]
            Point on the sphere.
        """
        return base_point ** (1 / 2)

    @staticmethod
    def inverse(image_point):
        """Send point of the sphere to the simplex.

        The map squares each component.

        Parameters
        ----------
        image_point : array-like, shape=[..., dim]
            Point on the sphere.

        Returns
        -------
        point : array-like, shape=[..., dim]
            Point on the simplex.
        """
        return image_point**2

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Send tangent vector of the simplex to tangent space of sphere.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vec to the simplex at base point.
        base_point : array-like, shape=[..., dim]
            Point of the simplex.
        image_point : array-like, shape=[..., dim]
            Point of the sphere.

        Returns
        -------
        image_tangent_vector : array-like, shape=[..., dim]
            Tangent vec to the sphere at the image of
            base point by simplex_to_sphere.
        """
        if image_point is None:
            image_point = self(base_point)

        return tangent_vec / (2 * image_point)

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Send tangent vector of the sphere to tangent space of simplex.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., dim]
            Tangent vec to the sphere at image point.
        image_point : array-like, shape=[..., dim]
            Point of the sphere.
        base_point : array-like, shape=[..., dim]
            Point of the simplex.

        Returns
        -------
        tangent_vec_simplex : array-like, shape=[..., dim]
            Tangent vec to the simplex at the image of
            base point by sphere_to_simplex.
        """
        if image_point is None:
            image_point = self(base_point)

        return 2 * image_point * image_tangent_vec


class OpenSimplex(LevelSet):
    r"""Class for the open simplex.

    This is the set of ``dim + 1``-tuples of positive reals that sum up to
    one, i.e. the level set of the sum in the positive orthant.

    Parameters
    ----------
    dim : int
        Dimension of the simplex.
    """

    def __init__(self, dim):
        self.dim = dim
        super().__init__(dim=dim, equip=False)

    def _define_embedding_space(self):
        return PositiveOrthant(dim=self.dim + 1, equip=False)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]

        Returns
        -------
        submersed_point : array-like, shape=[...]
        """
        return gs.sum(point, axis=-1) - 1.0

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
        point : Ignored.

        Returns
        -------
        submersed_vector : array-like, shape=[...]
        """
        return gs.sum(vector, axis=-1)

    def random_point(self, n_samples=1):
        """Sample points in the open simplex.

        The Dirichlet distribution with unit concentrations, i.e. the
        uniform distribution on the simplex, is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled in the open simplex.
        """
        samples = gs.from_numpy(dirichlet.rvs(gs.ones(self.dim + 1), size=n_samples))

        return samples[0] if n_samples == 1 else samples

    def projection(self, point, atol=gs.atol):
        """Project a point on the simplex.

        Negative components are replaced by zero and the point is
        renormalized by its 1-norm.

        Parameters
        ----------
        point: array-like, shape=[..., dim + 1]
            Point in embedding Euclidean space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected_point : array-like, shape=[..., dim + 1]
            Point projected on the simplex.
        """
        point_quadrant = gs.where(point < atol, atol, point)
        norm = gs.sum(point_quadrant, axis=-1)
        projected_point = scalarvecmul(1.0 / norm, point_quadrant)
        return projected_point

    def to_tangent(self, vector, base_point=None):
        """Project a vector to the tangent space.

        Project a vector in Euclidean space on the tangent space of
        the simplex at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector in Euclidean space.
        base_point : array-like, shape=[..., dim + 1]
            Point on the simplex defining the tangent space,
            where the vector will be projected.

        Returns
        -------
        vector : array-like, shape=[..., dim + 1]
            Tangent vector in the tangent space of the simplex
            at the base point.
        """
        component_mean = gs.mean(vector, axis=-1, keepdims=True)
        tangent_vec = vector - component_mean

        return repeat_out(
            self.point_ndim, tangent_vec, vector, base_point, out_shape=self.shape
        )
