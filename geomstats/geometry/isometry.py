"""Isometries of Riemannian manifolds."""

import abc

from geomstats.geometry.diffeo import Diffeo


class PointedIsometry(Diffeo, abc.ABC):
    """Isometry sending an anchor point to a reference point.

    Besides the usual methods of a diffeomorphism, it gives the differential
    at the anchor and the differential of the inverse at the reference point,
    which subclasses can override with cheaper closed forms.

    Parameters
    ----------
    anchor : array-like, shape=[..., *point_shape]
        Point sent to the reference point.
    reference : array-like, shape=[*point_shape]
        Reference point.
    """

    def __init__(self, anchor, reference):
        self.anchor = anchor
        self.reference = reference

    def tangent_at_anchor(self, tangent_vec):
        """Send a tangent vector at the anchor by the differential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at the anchor.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at the reference point.
        """
        return self.tangent(
            tangent_vec, base_point=self.anchor, image_point=self.reference
        )

    def inverse_tangent_at_reference(self, image_tangent_vec):
        """Send a tangent vector at the reference point by the inverse differential.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at the reference point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at the anchor.
        """
        return self.inverse_tangent(
            image_tangent_vec, image_point=self.reference, base_point=self.anchor
        )
