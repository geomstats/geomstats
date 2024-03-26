from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.geometry.diffeo import Diffeo


class HyperbolicDiffeo(Diffeo):
    """Diffeomorphism between hyperbolic coordinates.

    Parameters
    ----------
    from_coordinates_system : str, {'extrinsic', 'ball', 'half-space'}
        Coordinates type.
    to_coordinates_system : str, {'extrinsic', 'ball', 'half-space'}
        Coordinates type.
    """

    def __init__(self, from_coordinates_system, to_coordinates_system):
        self.from_coordinates_system = from_coordinates_system
        self.to_coordinates_system = to_coordinates_system

    def diffeomorphism(self, base_point):
        """Diffeomorphism at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., *image_shape]
            Image point.
        """
        return _Hyperbolic.change_coordinates_system(
            base_point, self.from_coordinates_system, self.to_coordinates_system
        )

    def inverse_diffeomorphism(self, image_point):
        r"""Inverse diffeomorphism at image point.

        :math:`f^{-1}: N \rightarrow M`

        Parameters
        ----------
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        """
        return _Hyperbolic.change_coordinates_system(
            image_point, self.to_coordinates_system, self.from_coordinates_system
        )

    def tangent_diffeomorphism(self, tangent_vec, base_point=None, image_point=None):
        r"""Tangent diffeomorphism at base point.

        df_p is a linear map from T_pM to T_f(p)N.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image of the base point.
        """
        if base_point is None:
            base_point = self.inverse_diffeomorphism(image_point)
        return _Hyperbolic.change_tangent_coordinates_system(
            tangent_vec,
            base_point,
            self.from_coordinates_system,
            self.to_coordinates_system,
        )

    def inverse_tangent_diffeomorphism(
        self, image_tangent_vec, image_point=None, base_point=None
    ):
        r"""Inverse tangent diffeomorphism at image point.

        df^-1_p is a linear map from T_f(p)N to T_pM

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        """
        if image_point is None:
            image_point = self.diffeomorphism(base_point)

        return _Hyperbolic.change_tangent_coordinates_system(
            image_tangent_vec,
            image_point,
            self.to_coordinates_system,
            self.from_coordinates_system,
        )
