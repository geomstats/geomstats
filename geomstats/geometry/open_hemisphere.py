"""Open hemisphere.

Lead author: Olivier Bisson.
"""

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.diffeo import Diffeo
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric


class OpenHemisphereToHyperboloidDiffeo(Diffeo):
    """A diffeomorphism between the open hemisphere and the hyperboloid."""

    def diffeomorphism(self, base_point):
        """Diffeomorphism at base point."""
        return_point = gs.copy(base_point)
        return_point[..., 0] = 1.0
        first_term = base_point[..., 0]
        return gs.einsum("...,...i->...i", 1 / first_term, return_point)

    def inverse_diffeomorphism(self, image_point):
        """Inverse diffeomorphism at base point."""
        return self.diffeomorphism(image_point)

    def tangent_diffeomorphism(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point."""
        if base_point is None:
            base_point = self.inverse_diffeomorphism(image_point)

        coeffs = tangent_vec[..., 0] / base_point[..., 0]
        image_tangent_vec_0 = gs.array(-tangent_vec[..., 0] / base_point[..., 0])
        image_tangent_vec_other = tangent_vec[..., 1:] - gs.einsum(
            "...,...i->...i", coeffs, base_point[..., 1:]
        )

        image_tangent_vec = gs.concatenate(
            [gs.expand_dims(image_tangent_vec_0, axis=-1), image_tangent_vec_other],
            axis=-1,
        )
        return gs.einsum("...,...i->...i", 1 / base_point[..., 0], image_tangent_vec)

    def inverse_tangent_diffeomorphism(
        self, image_tangent_vec, image_point=None, base_point=None
    ):
        """Inverse tangent diffeomorphism at image point."""
        return self.tangent_diffeomorphism(
            image_tangent_vec, base_point=image_point, image_point=base_point
        )


class OpenHemisphere(OpenSet):
    r"""Open hemisphere.

    An open set of the hypersphere where the first coordinate is always
    positive.

    ..math::

        \mathrm{HS}^{k-1}=\left\{x \in \mathbb{R}^k: \|x\|=1
        \text { and } x_0>0\right\}
    """

    def __init__(self, dim, equip=True):
        self.dim = dim
        super().__init__(
            dim=dim,
            default_coords_type="extrinsic",
            embedding_space=Hypersphere(dim, equip=equip),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return OpenHemispherePullbackMetric

    def belongs(self, point, atol=gs.atol):
        """Check if a point belongs to the open hemisphere."""
        is_on_sphere = self.embedding_space.belongs(point)
        is_on_upper_part = gs.greater(point[..., 0], 0.0)
        return gs.logical_and(is_on_sphere, is_on_upper_part)

    def projection(self, point):
        """Project a point on the open hemisphere.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in embedding hypersphere space.

        Returns
        -------
        projected_point : array-like, shape=[..., dim]
            Point projected on the open hemisphere.
        """
        proj_point = self.embedding_space.projection(point)
        proj_point[..., 0] = gs.abs(proj_point[..., 0])
        return proj_point


class OpenHemispherePullbackMetric(PullbackDiffeoMetric):
    """Pullback diffeo metric for Open Hemisphere.

    Pulls back metric from hyperboloid.
    """

    def __init__(self, space):
        image_space = Hyperboloid(dim=space.dim)
        diffeo = OpenHemisphereToHyperboloidDiffeo()
        super().__init__(space=space, diffeo=diffeo, image_space=image_space)
