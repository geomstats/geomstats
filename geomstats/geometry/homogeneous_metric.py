"""Metrics on homogeneous and symmetric spaces.

On a Riemannian homogeneous space, isometries act transitively, so the
geometry at any point is the geometry at a reference point moved by an
isometry. On a Riemannian symmetric space, these isometries can be chosen as
transvections, whose differentials give the parallel transport along
geodesics.
"""

import abc

from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import repeat_out


class HomogeneousMetric(RiemannianMetric, abc.ABC):
    """Metric on a Riemannian homogeneous space.

    The geometry is only written at a reference point, and is moved to any
    other base point by an isometry sending that base point to the reference
    point.

    The isometries are given by :meth:`isometry_to_reference` as
    :class:`~geomstats.geometry.isometry.PointedIsometry` objects.
    """

    @property
    @abc.abstractmethod
    def reference_point(self):
        """Reference point at which the geometry is written.

        Returns
        -------
        reference_point : array-like, shape=[*point_shape]
            Reference point.
        """

    @abc.abstractmethod
    def isometry_to_reference(self, base_point):
        """Get an isometry sending a base point to the reference point.

        Parameters
        ----------
        base_point : array-like, shape=[..., *point_shape]
            Base point.

        Returns
        -------
        isometry : geomstats.geometry.isometry.PointedIsometry
            Isometry sending the base point to the reference point.
        """

    def inner_product_at_reference(self, tangent_vec_a, tangent_vec_b):
        """Compute the inner product at the reference point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *point_shape]
            Tangent vector at the reference point.
        tangent_vec_b : array-like, shape=[..., *point_shape]
            Tangent vector at the reference point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner product.
        """
        raise NotImplementedError("Inner product at reference is not implemented.")

    def exp_at_reference(self, tangent_vec):
        """Compute the exponential map at the reference point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at the reference point.

        Returns
        -------
        exp : array-like, shape=[..., *point_shape]
            Point on the manifold.
        """
        raise NotImplementedError("Exp at reference is not implemented.")

    def log_at_reference(self, point):
        """Compute the logarithm map at the reference point.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        log : array-like, shape=[..., *point_shape]
            Tangent vector at the reference point.
        """
        raise NotImplementedError("Log at reference is not implemented.")

    def sectional_curvature_at_reference(self, tangent_vec_a, tangent_vec_b):
        """Compute the sectional curvature at the reference point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *point_shape]
            Tangent vector at the reference point.
        tangent_vec_b : array-like, shape=[..., *point_shape]
            Tangent vector at the reference point.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at the reference point.
        """
        raise NotImplementedError(
            "Sectional curvature at reference is not implemented."
        )

    def injectivity_radius_at_reference(self):
        """Compute the injectivity radius at the reference point.

        Returns
        -------
        radius : array-like, shape=[]
            Injectivity radius.
        """
        raise NotImplementedError("Injectivity radius at reference is not implemented.")

    def squared_dist_at_reference(self, point):
        """Compute the squared distance from the reference point.

        It is the squared norm of the logarithm at the reference point.
        Subclasses can override it with a cheaper closed form.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance between the reference point and the point.
        """
        log = self.log_at_reference(point)
        return self.inner_product_at_reference(log, log)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner product at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        base_point : array-like, shape=[..., *point_shape]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner product.
        """
        isometry = self.isometry_to_reference(base_point)
        return self.inner_product_at_reference(
            isometry.tangent_at_anchor(tangent_vec_a),
            isometry.tangent_at_anchor(tangent_vec_b),
        )

    def exp(self, tangent_vec, base_point):
        """Compute the exponential map at a base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        base_point : array-like, shape=[..., *point_shape]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., *point_shape]
            Point on the manifold.
        """
        isometry = self.isometry_to_reference(base_point)
        return isometry.inverse(
            self.exp_at_reference(isometry.tangent_at_anchor(tangent_vec))
        )

    def log(self, point, base_point):
        """Compute the logarithm map at a base point.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *point_shape]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        """
        isometry = self.isometry_to_reference(base_point)
        return isometry.inverse_tangent_at_reference(
            self.log_at_reference(isometry(point))
        )

    def squared_dist(self, point_a, point_b):
        """Compute the squared distance between two points.

        Isometries preserve the distance, so it is computed from the
        reference point, after sending the first point to it.

        Parameters
        ----------
        point_a : array-like, shape=[..., *point_shape]
            Point on the manifold.
        point_b : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance.
        """
        isometry = self.isometry_to_reference(point_a)
        return self.squared_dist_at_reference(isometry(point_b))

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the sectional curvature at a base point.

        Isometries preserve the sectional curvature, so it is computed at the
        reference point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        base_point : array-like, shape=[..., *point_shape]
            Base point.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at base point.
        """
        isometry = self.isometry_to_reference(base_point)
        return self.sectional_curvature_at_reference(
            isometry.tangent_at_anchor(tangent_vec_a),
            isometry.tangent_at_anchor(tangent_vec_b),
        )

    def injectivity_radius(self, base_point):
        """Compute the injectivity radius at a base point.

        Isometries preserve the injectivity radius, so it is the one at the
        reference point.

        Parameters
        ----------
        base_point : array-like, shape=[..., *point_shape]
            Base point.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        return repeat_out(
            self._space.point_ndim, self.injectivity_radius_at_reference(), base_point
        )


class RiemannianSymmetricMetric(HomogeneousMetric, abc.ABC):
    """Metric on a Riemannian symmetric space.

    The isometries returned by :meth:`isometry_to_reference` are
    transvections: their differentials give the parallel transport along the
    geodesics through their anchor.
    """

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        The parallel transport along a geodesic is the differential of the
        transvection along it. Denoting the base point by :math:`B`, the end
        point by :math:`C` and by :math:`\phi_P` the transvection sending
        :math:`P` to the reference point, this transvection is
        :math:`\phi_B^{-1} \circ \phi_Y^{-1} \circ \phi_B` with
        :math:`Y = \phi_B(C)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., *point_shape]
            Base point.
        direction : array-like, shape=[..., *point_shape]
            Tangent vector at base point, along which the parallel transport
            is computed.
        end_point : array-like, shape=[..., *point_shape]
            Point to transport to. Unused if `direction` is given.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., *point_shape]
            Transported tangent vector at `exp_(base_point)(direction)`.
        """
        if direction is not None:
            end_point = self.exp(direction, base_point)
        elif end_point is None:
            raise ValueError(
                "Either an end_point or a direction must be given to define the"
                " geodesic along which to transport."
            )

        isometry = self.isometry_to_reference(base_point)
        end_point_at_reference = isometry(end_point)
        transported_at_reference = self.isometry_to_reference(
            end_point_at_reference
        ).inverse_tangent_at_reference(isometry.tangent_at_anchor(tangent_vec))
        return isometry.inverse_tangent(
            transported_at_reference, image_point=end_point_at_reference
        )
