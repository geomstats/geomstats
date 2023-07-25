"""Core parametrizer classes for Tests."""

import math
from functools import reduce

import pytest

import geomstats.backend as gs
from tests.conftest import TestCase


def better_squeeze(array):
    """Delete possible singleton dimension on first axis."""
    if len(array) == 1:
        return gs.squeeze(array, axis=0)
    return array


def _is_isometry(
    metric,
    space,
    tangent_vec,
    trans_tangent_vec,
    base_point,
    is_tangent_atol,
    rtol,
    atol,
):
    """Check that a transformation is an isometry.

    This is an auxiliary function.

    Parameters
    ----------
    metric : RiemannianMetric
        Riemannian metric.
    tangent_vec : array-like
        Tangent vector at base point.
    trans_tangent_vec : array-like
        Transformed tangent vector at base point.
    base_point : array-like
        Point on manifold.
    is_tangent_atol: float
        Asbolute tolerance for the is_tangent function.
    rtol : float
        Relative tolerance to test this property.
    atol : float
        Absolute tolerance to test this property.
    """
    is_tangent = space.is_tangent(trans_tangent_vec, base_point, is_tangent_atol)
    is_equinormal = gs.isclose(
        metric.norm(trans_tangent_vec, base_point),
        metric.norm(tangent_vec, base_point),
        rtol,
        atol,
    )
    return gs.logical_and(is_tangent, is_equinormal)


class ManifoldTestCase(TestCase):
    @property
    def Space(self):
        return self.testing_data.Space

    def test_manifold_shape(self, space_args):
        space = self.Space(*space_args)
        point = space.random_point()

        point_shape = (1,) if point.shape == () else point.shape

        self.assertTrue(
            space.shape == point_shape,
            f"Shape is {space.shape}, but random point shape is {point_shape}",
        )

    def test_random_point_belongs(self, space_args, n_points, atol):
        """Check that a random point belongs to the manifold.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the manifold.
        n_points : array-like
            Number of random points to sample.
        atol : float
            Absolute tolerance for the belongs function.
        """
        space = self.Space(*space_args)
        random_point = space.random_point(n_points)
        result = space.belongs(random_point, atol=atol)
        if n_points > 1:
            self.assertTrue(gs.all(result))
            self.assertTrue(result.shape[0] == n_points)
        else:
            self.assertTrue(result)

    def test_projection_belongs(self, space_args, point, atol):
        """Check that a point projected on a manifold belongs to the manifold.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the manifold.
        point : array-like
            Point to be projected on the manifold.
        atol : float
            Absolute tolerance for the belongs function.
        """
        space = self.Space(*space_args)
        projection = space.projection(point)
        belongs = space.belongs(projection, atol)
        if 1 <= len(space.shape) < point.ndim and point.shape[0] > 1:
            self.assertAllEqual(belongs, gs.ones(point.shape[: -len(space.shape)]))
            self.assertEqual(belongs.shape, point.shape[: -len(space.shape)])
        else:
            self.assertTrue(belongs)

    def test_to_tangent_is_tangent(self, space_args, vector, base_point, atol):
        """Check that to_tangent returns a tangent vector.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the manifold.
        vector : array-like
            Vector to be projected on the tangent space at base_point.
        base_point : array-like
            Point on the manifold.
        atol : float
            Absolute tolerance for the is_tangent function.
        """
        space = self.Space(*space_args)
        tangent_vec = space.to_tangent(vector, base_point)
        result = space.is_tangent(tangent_vec, base_point, atol)
        if tangent_vec.ndim > len(space.shape):
            self.assertAllEqual(result, gs.ones(tangent_vec.shape[: -len(space.shape)]))
        else:
            self.assertTrue(result)

    def test_random_tangent_vec_is_tangent(
        self, space_args, n_samples, base_point, atol
    ):
        """Check that to_tangent returns a tangent vector.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the manifold.
        n_samples : int
            Vector to be projected on the tangent space at base_point.
        base_point : array-like
            Point on the manifold.
        atol : float
            Absolute tolerance for the is_tangent function.
        """
        space = self.Space(*space_args)
        tangent_vec = space.random_tangent_vec(base_point, n_samples)
        result = space.is_tangent(tangent_vec, base_point, atol)
        self.assertAllEqual(result, gs.squeeze(gs.ones(n_samples, dtype=bool)))


class OpenSetTestCase(ManifoldTestCase):
    def test_to_tangent_is_tangent_in_embedding_space(
        self, space_args, vector, base_point, atol
    ):
        """Check that tangent vectors are in ambient space's tangent space.

        This projects a vector to the tangent space of the manifold, and
        then checks that tangent vector belongs to ambient space's tangent space.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the manifold.
        vector : array-like
            Vector to be projected on the tangent space at base_point.
        base_point : array-like
            Point on the manifold.
        atol : float
            Absolute tolerance for the is_tangent function.
        """
        space = self.Space(*space_args)
        tangent_vec = space.to_tangent(vector, base_point)
        result = gs.all(space.embedding_space.is_tangent(tangent_vec, base_point, atol))
        self.assertTrue(result)


class LieGroupTestCase(ManifoldTestCase):
    def test_compose_point_with_inverse_point_is_identity(
        self, group_args, point, rtol, atol
    ):
        """Check that composition of point and inverse is identity.

        Parameters
        ----------
        group_args : tuple
            Arguments to pass to constructor of the group.
        point : array-like
            Point on the group.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        group = self.Space(*group_args)
        result = group.compose(group.inverse(point), point)
        expected = better_squeeze(gs.array([group.identity] * len(result)))
        self.assertAllClose(result, expected, rtol=rtol, atol=atol)

    def test_compose_inverse_point_with_point_is_identity(
        self, group_args, point, rtol, atol
    ):
        """Check that composition of inverse and point is identity.

        Parameters
        ----------
        group_args : tuple
            Arguments to pass to constructor of the group.
        point : array-like
            Point on the group.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        group = self.Space(*group_args)
        result = group.compose(point, group.inverse(point))
        expected = better_squeeze(gs.array([group.identity] * len(result)))
        self.assertAllClose(result, expected, rtol=rtol, atol=atol)

    def test_compose_point_with_identity_is_point(self, group_args, point, rtol, atol):
        """Check that composition of point and identity is identity.

        Parameters
        ----------
        group_args : tuple
            Arguments to pass to constructor of the group.
        point : array-like
            Point on the group.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        group = self.Space(*group_args)
        result = group.compose(
            point, better_squeeze(gs.array([group.identity] * len(point)))
        )
        self.assertAllClose(result, point, rtol=rtol, atol=atol)

    def test_compose_identity_with_point_is_point(self, group_args, point, rtol, atol):
        """Check that composition of identity and point is identity.

        Parameters
        ----------
        group_args : tuple
            Arguments to pass to constructor of the group.
        point : array-like
            Point on the group.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        group = self.Space(*group_args)
        result = group.compose(
            better_squeeze(gs.array([group.identity] * len(point))), point
        )
        self.assertAllClose(result, point, rtol=rtol, atol=atol)

    def test_log_after_exp(self, group_args, tangent_vec, base_point, rtol, atol):
        """Check that group exponential and logarithm are inverse.

        This is calling group exponential first, then group logarithm.

        Parameters
        ----------
        group_args : tuple
            Arguments to pass to constructor of the group.
        tangent_vec : array-like
            Tangent vector to the manifold at base_point.
        base_point : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        group = self.Space(*group_args)
        exp_point = group.exp(tangent_vec, base_point)
        log_vec = group.log(exp_point, base_point)
        self.assertAllClose(log_vec, tangent_vec, rtol, atol)

    def test_exp_after_log(self, group_args, point, base_point, rtol, atol):
        """Check that group exponential and logarithm are inverse.

        This is calling group logarithm first, then group exponential.

        Parameters
        ----------
        group_args : tuple
            Arguments to pass to constructor of the group.
        point : array-like
            Point on the manifold.
        base_point : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        group = self.Space(*group_args)
        log_vec = group.log(point, base_point)
        exp_point = group.exp(log_vec, base_point)
        self.assertAllClose(exp_point, point, rtol, atol)

    def test_to_tangent_at_identity_belongs_to_lie_algebra(
        self, group_args, vector, atol
    ):
        """Check that to tangent at identity is tangent in lie algebra.

        Parameters
        ----------
        group_args : tuple
            Arguments to pass to constructor of the group.
        vector : array-like
            Vector to be projected on the tangent space at base_point.
        atol : float
            Absolute tolerance for the belongs function.
        """
        group = self.Space(*group_args)
        tangent_vec = group.to_tangent(vector, group.identity)
        result = gs.all(group.lie_algebra.belongs(tangent_vec, atol))
        self.assertTrue(result)


class VectorSpaceTestCase(ManifoldTestCase):
    def test_basis_belongs(self, space_args, atol):
        """Check that basis elements belong to vector space.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the vector space.
        atol : float
            Absolute tolerance of the belongs function.
        """
        space = self.Space(*space_args)
        result = gs.all(space.belongs(space.basis, atol))
        self.assertTrue(result)

    def test_basis_cardinality(self, space_args):
        """Check that the number of basis elements is the dimension.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the vector space.
        """
        space = self.Space(*space_args)
        basis = space.basis
        self.assertAllClose(len(basis), space.dim)

    def test_random_point_is_tangent(self, space_args, n_points, atol):
        """Check that the random point is tangent.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the vector space.
        n_points : array-like
            Number of random points to sample.
        atol : float
            Absolute tolerance for the is_tangent function.
        """
        space = self.Space(*space_args)
        points = space.random_point(n_points)
        base_point = space.random_point()
        result = space.is_tangent(points, base_point, atol)
        self.assertTrue(gs.all(result))

    def test_to_tangent_is_projection(self, space_args, vector, base_point, rtol, atol):
        """Check that to_tangent is same as projection.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the vector space.
        vector : array-like
            Vector to be projected on the tangent space at base_point.
        base_point : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space = self.Space(*space_args)
        result = space.to_tangent(vector, base_point)
        expected = space.projection(vector)
        self.assertAllClose(result, expected, rtol=rtol, atol=atol)


class MatrixLieAlgebraTestCase(VectorSpaceTestCase):
    def test_matrix_representation_after_basis_representation(
        self, algebra_args, matrix_rep, rtol, atol
    ):
        """Check that changing coordinate system twice gives back the point.

        A point written in basis representation is converted to matrix
        representation and back.

        Parameters
        ----------
        algebra_args : tuple
            Arguments to pass to constructor of the Lie algebra.
        matrix_rep : array-like
            Point on the Lie algebra given in its matrix representation.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        algebra = self.Space(*algebra_args)
        basis_rep = algebra.basis_representation(matrix_rep)
        result = algebra.matrix_representation(basis_rep)
        self.assertAllClose(result, matrix_rep, rtol, atol)

    def test_basis_representation_after_matrix_representation(
        self,
        algebra_args,
        basis_rep,
        rtol,
        atol,
    ):
        """Check that changing coordinate system twice gives back the point.

        A point written in matrix representation is converted to basis
        representation and back.

        Parameters
        ----------
        algebra_args : tuple
            Arguments to pass to constructor of the Lie algebra.
        basis_rep : array-like
            Point on the Lie algebra given in its basis representation.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        algebra = self.Space(*algebra_args)
        mat_rep = algebra.matrix_representation(basis_rep)
        result = algebra.basis_representation(mat_rep)
        self.assertAllClose(result, basis_rep, rtol, atol)


class LevelSetTestCase(ManifoldTestCase):
    def test_intrinsic_after_extrinsic(self, space_args, point_extrinsic, rtol, atol):
        """Check that changing coordinate system twice gives back the point.

        A point written in extrinsic coordinates is converted to
        intrinsic coordinates and back.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the manifold.
        point_extrinsic : array-like
            Point on the manifold in extrinsic coordinates.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space = self.Space(*space_args)
        point_intrinsic = space.extrinsic_to_intrinsic_coords(point_extrinsic)
        result = space.intrinsic_to_extrinsic_coords(point_intrinsic)
        expected = point_extrinsic
        self.assertAllClose(result, expected, rtol, atol)

    def test_extrinsic_after_intrinsic(self, space_args, point_intrinsic, rtol, atol):
        """Check that changing coordinate system twice gives back the point.

        A point written in intrinsic coordinates is converted to
        extrinsic coordinates and back.

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the manifold.
        point_intrinsic : array-like
            Point on the manifold in intrinsic coordinates.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space = self.Space(*space_args)
        point_extrinsic = space.intrinsic_to_extrinsic_coords(point_intrinsic)
        result = space.extrinsic_to_intrinsic_coords(point_extrinsic)
        expected = point_intrinsic

        self.assertAllClose(result, expected, rtol, atol)


class FiberBundleTestCase(TestCase):
    def test_is_horizontal_after_horizontal_projection(
        self, space_args, tangent_vec, base_point, rtol, atol
    ):
        total_space = self.TotalSpace(*space_args)
        bundle = self.Bundle(total_space)
        horizontal = bundle.horizontal_projection(tangent_vec, base_point)
        result = bundle.is_horizontal(horizontal, base_point, atol)
        self.assertTrue(gs.all(result))

    def test_is_vertical_after_vertical_projection(
        self, space_args, tangent_vec, base_point, rtol, atol
    ):
        total_space = self.TotalSpace(*space_args)
        bundle = self.Bundle(total_space)
        vertical = bundle.vertical_projection(tangent_vec, base_point)
        result = bundle.is_vertical(vertical, base_point, atol)
        self.assertTrue(gs.all(result))

    def test_is_horizontal_after_log_after_align(
        self, space_args, base_point, point, rtol, atol
    ):
        total_space = self.TotalSpace(*space_args)
        bundle = self.Bundle(total_space)
        aligned = bundle.align(point, base_point)
        log = total_space.metric.log(aligned, base_point)
        result = bundle.is_horizontal(log, base_point)
        self.assertTrue(gs.all(result))

    def test_riemannian_submersion_after_lift(self, space_args, base_point, rtol, atol):
        total_space = self.TotalSpace(*space_args)
        bundle = self.Bundle(total_space)
        lift = bundle.lift(base_point)
        result = bundle.riemannian_submersion(lift)
        self.assertAllClose(result, base_point, rtol, atol)

    def test_is_tangent_after_tangent_riemannian_submersion(
        self, space_args, base_cls, tangent_vec, base_point
    ):
        total_space = self.TotalSpace(*space_args)
        bundle = self.Bundle(total_space)
        projected = bundle.tangent_riemannian_submersion(tangent_vec, base_point)
        projected_pt = bundle.riemannian_submersion(base_point)
        result = base_cls(*space_args).is_tangent(projected, projected_pt)
        expected = (
            True
            if tangent_vec.shape == bundle.total_space.shape
            else gs.array([True] * len(tangent_vec))
        )
        self.assertAllEqual(result, expected)


class ProductManifoldTestCase(ManifoldTestCase):
    def test_product_dimension_is_sum_of_dimensions(self, space_args):
        """Check the dimension of the product manifold.

        Check that the dimension of the product manifold is the sum of
        the dimensions of its manifolds.

        For M = M1 x ... x Mn, we check that:
        dim(M) = dim(M1) + ... + dim(Mn)

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the product manifold.
        """
        spaces_list = space_args[0]
        result = self.Space(*space_args).dim
        expected = sum(space.dim for space in spaces_list)
        self.assertAllClose(result, expected)


class NFoldManifoldTestCase(ManifoldTestCase):
    def test_dimension_is_dim_multiplied_by_n_copies(self, space_args):
        """Check the dimension of the product manifold.

        Check that the dimension of the n-fold manifold is the multiplication of
        the dimension of one of the copies times the number of copies

        For M = M0^n, we check that:
        dim(M) = n *  dim(M0)

        Parameters
        ----------
        space_args : tuple
            Arguments to pass to constructor of the n-fold manifold.
        """
        base_manifold = space_args[0]
        n_copies = space_args[1]
        result = self.Space(*space_args).dim
        expected = base_manifold.dim * n_copies
        self.assertAllClose(result, expected)


class ConnectionTestCase(TestCase):
    @property
    def Metric(self):
        return self.testing_data.Metric

    def test_exp_shape(self, space, connection_args, tangent_vec, base_point, expected):
        """Check that exp returns an array of the expected shape.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        tangent_vec : array-like
            Tangent vector at base point.
        base_point : array-like
            Point on the manifold.
        expected : tuple
            Expected shape for the result of the exp function.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        exp = space.metric.exp(tangent_vec, base_point)
        result = gs.shape(exp)
        self.assertAllClose(result, expected)

    def test_log_shape(self, space, connection_args, point, base_point, expected):
        """Check that log returns an array of the expected shape.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        point : array-like
            Point on the manifold.
        base_point : array-like
            Point on the manifold.
        expected : tuple
            Expected shape for the result of the log function.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        log = space.metric.log(point, base_point)
        result = gs.shape(log)
        self.assertAllClose(result, expected)

    def test_exp_belongs(self, connection_args, space, tangent_vec, base_point, atol):
        """Check that the connection exponential gives a point on the manifold.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        space : Manifold
            Manifold where connection is defined.
        tangent_vec : array-like
            Tangent vector at base point.
        base_point : array-like
            Point on the manifold.
        atol : float
            Absolute tolerance for the belongs function.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        exp = space.metric.exp(tangent_vec, base_point)
        result = gs.all(space.belongs(exp, atol))
        self.assertTrue(result)

    def test_log_is_tangent(self, connection_args, space, point, base_point, atol):
        """Check that the connection logarithm gives a tangent vector.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        space : Manifold
            Manifold where connection is defined.
        point : array-like
            Point on the manifold.
        base_point : array-like
            Point on the manifold.
        atol : float
            Absolute tolerance for the is_tangent function.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        log = space.metric.log(point, base_point)
        result = gs.all(space.is_tangent(log, base_point, atol))
        self.assertTrue(result)

    def test_geodesic_ivp_belongs(
        self,
        connection_args,
        space,
        n_points,
        initial_point,
        initial_tangent_vec,
        atol,
    ):
        """Check that connection geodesics belong to manifold.

        This is for geodesics defined by the initial value problem (ivp).

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        space : Manifold
            Manifold where connection is defined.
        n_points : int
            Number of points on the geodesics.
        initial_point : array-like
            Point on the manifold.
        initial_tangent_vec : array-like
            Tangent vector at base point.
        atol : float
            Absolute tolerance for the belongs function.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        geodesic = space.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )

        t = gs.linspace(start=0.0, stop=1.0, num=n_points)
        points = geodesic(t)

        result = space.belongs(points, atol)
        expected = gs.array(n_points * [True])

        self.assertAllEqual(result, expected)

    def test_geodesic_bvp_belongs(
        self,
        connection_args,
        space,
        n_points,
        initial_point,
        end_point,
        atol,
    ):
        """Check that connection geodesics belong to manifold.

        This is for geodesics defined by the boundary value problem (bvp).

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        space : Manifold
            Manifold where connection is defined.
        n_points : int
            Number of points on the geodesics.
        initial_point : array-like
            Point on the manifold.
        end_point : array-like
            Point on the manifold.
        atol : float
            Absolute tolerance for the belongs function.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        geodesic = space.metric.geodesic(
            initial_point=initial_point, end_point=end_point
        )

        t = gs.linspace(start=0.0, stop=1.0, num=n_points)
        points = geodesic(t)

        result = space.belongs(points, atol)
        expected = gs.array(n_points * [True])

        self.assertAllEqual(result, expected)

    def test_exp_after_log(self, space, connection_args, point, base_point, rtol, atol):
        """Check that connection logarithm and exponential are inverse.

        This is calling connection logarithm first, then connection exponential.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        point : array-like
            Point on the manifold.
        base_point : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        log = space.metric.log(point, base_point=base_point)
        result = space.metric.exp(tangent_vec=log, base_point=base_point)
        self.assertAllClose(result, point, rtol=rtol, atol=atol)

    def test_log_after_exp(
        self, space, connection_args, tangent_vec, base_point, rtol, atol
    ):
        """Check that connection exponential and logarithm are inverse.

        This is calling connection exponential first, then connection logarithm.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        tangent_vec : array-like
            Tangent vector to the manifold at base_point.
        base_point : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        exp = space.metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = space.metric.log(exp, base_point=base_point)
        self.assertAllClose(result, tangent_vec, rtol=rtol, atol=atol)

    def test_exp_ladder_parallel_transport(
        self,
        space,
        connection_args,
        direction,
        tangent_vec,
        base_point,
        scheme,
        n_rungs,
        alpha,
        rtol,
        atol,
    ):
        """Check that end point of ladder parallel transport matches exponential.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        direction : array-like
            Tangent vector to the manifold at base_point.
        tangent_vec : array-like
            Tangent vector to the manifold at base_point.
        base_point : array-like
            Point on the manifold.
        scheme : str, {'pole', 'schild'}
            The scheme to use for the construction of the ladder at each step.
        n_rungs : int
            Number of steps of the ladder.
        alpha : float
            Exponent for the scaling of the vector to transport.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        ladder = space.metric.ladder_parallel_transport(
            tangent_vec,
            base_point,
            direction,
            n_rungs=n_rungs,
            scheme=scheme,
            alpha=alpha,
        )

        result = ladder["end_point"]
        expected = space.metric.exp(direction, base_point)

        self.assertAllClose(result, expected, rtol=rtol, atol=atol)

    def test_exp_geodesic_ivp(
        self, space, connection_args, n_points, tangent_vec, base_point, rtol, atol
    ):
        """Check that end point of geodesic matches exponential.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        n_points : int
            Number of points on the geodesic.
        tangent_vec : array-like
            Tangent vector to the manifold at base_point.
        base_point : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        geodesic = space.metric.geodesic(
            initial_point=base_point, initial_tangent_vec=tangent_vec
        )
        t = (
            gs.linspace(start=0.0, stop=1.0, num=n_points)
            if n_points > 1
            else gs.ones([1])
        )
        points = geodesic(t)
        multiple_inputs = tangent_vec.ndim > len(space.shape)

        result = (
            points[:, -1]
            if multiple_inputs
            else (points[-1] if n_points > 1 else points)
        )
        expected = space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(expected, result, rtol=rtol, atol=atol)

    def test_riemann_tensor_shape(self, space, connection_args, base_point, expected):
        """Check that riemann_tensor returns an array of the expected shape.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        base_point : array-like
            Point on the manifold.
        expected : tuple
            Expected shape for the result of the riemann_tensor function.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        tensor = space.metric.riemann_tensor(base_point)
        result = gs.shape(tensor)
        self.assertAllClose(result, expected)

    def test_ricci_tensor_shape(self, space, connection_args, base_point, expected):
        """Check that ricci_tensor returns an array of the expected shape.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        base_point : array-like
            Point on the manifold.
        expected : tuple
            Expected shape for the result of the ricci_tensor function.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        tensor = space.metric.ricci_tensor(base_point)
        result = gs.shape(tensor)
        self.assertAllClose(result, expected)

    def test_scalar_curvature_shape(self, space, connection_args, base_point, expected):
        """Check that scalar_curvature returns an array of the expected shape.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        base_point : array-like
            Point on the manifold.
        expected : tuple
            Expected shape for the result of the ricci_tensor function.
        """
        space.equip_with_metric(self.Metric, **connection_args)

        tensor = space.metric.scalar_curvature(base_point)
        result = gs.shape(tensor)
        self.assertAllClose(result, expected)


class RiemannianMetricTestCase(ConnectionTestCase):
    def test_dist_is_symmetric(self, space, metric_args, point_a, point_b, rtol, atol):
        """Check that the geodesic distance is symmetric.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        point_a : array-like
            Point on the manifold.
        point_b : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        dist_a_b = space.metric.dist(point_a, point_b)
        dist_b_a = space.metric.dist(point_b, point_a)
        self.assertAllClose(dist_a_b, dist_b_a, rtol=rtol, atol=atol)

    def test_dist_is_positive(self, space, metric_args, point_a, point_b, atol):
        """Check that the geodesic distance is positive.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        point_a : array-like
            Point on the manifold.
        point_b : array-like
            Point on the manifold.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        sd_a_b = space.metric.dist(point_a, point_b)
        result = gs.all(sd_a_b > -1 * atol)
        self.assertTrue(result)

    def test_squared_dist_is_symmetric(
        self, space, metric_args, point_a, point_b, rtol, atol
    ):
        """Check that the squared geodesic distance is symmetric.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        point_a : array-like
            Point on the manifold.
        point_b : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        sd_a_b = space.metric.squared_dist(point_a, point_b)
        sd_b_a = space.metric.squared_dist(point_b, point_a)
        self.assertAllClose(sd_a_b, sd_b_a, rtol=rtol, atol=atol)

    def test_squared_dist_is_positive(self, space, metric_args, point_a, point_b, atol):
        """Check that the squared geodesic distance is positive.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        point_a : array-like
            Point on the manifold.
        point_b : array-like
            Point on the manifold.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        sd_a_b = space.metric.dist(point_a, point_b)
        result = gs.all(sd_a_b > -1 * atol)
        self.assertTrue(result)

    def test_inner_product_is_symmetric(
        self, space, metric_args, tangent_vec_a, tangent_vec_b, base_point, rtol, atol
    ):
        """Check that the inner product is symmetric.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        tangent_vec_a : array-like
            Tangent vector to the manifold at base_point.
        tangent_vec_b : array-like
            Tangent vector to the manifold at base_point.
        base_point : array-like
            Point on manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        ip_a_b = space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        ip_b_a = space.metric.inner_product(tangent_vec_b, tangent_vec_a, base_point)
        self.assertAllClose(ip_a_b, ip_b_a, rtol, atol)

    def test_parallel_transport_ivp_is_isometry(
        self,
        metric_args,
        space,
        tangent_vec,
        base_point,
        direction,
        is_tangent_atol,
        rtol,
        atol,
    ):
        """Check that parallel transport is an isometry.

        This is for parallel transport defined by initial value problem (ivp).

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        space : Manifold
            Manifold where metric is defined.
        tangent_vec : array-like
            Tangent vector at base point, to be transported.
        base_point : array-like
            Point on manifold.
        direction : array-like
            Tangent vector at base point, along which to transport.
        is_tangent_atol: float
            Asbolute tolerance for the is_tangent function.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        end_point = space.metric.exp(direction, base_point)

        transported = space.metric.parallel_transport(
            tangent_vec, base_point, direction
        )
        result = _is_isometry(
            space.metric,
            space,
            tangent_vec,
            transported,
            end_point,
            is_tangent_atol,
            rtol,
            atol,
        )
        expected = gs.array(len(result) * [True])
        self.assertAllEqual(result, expected)

    def test_parallel_transport_bvp_is_isometry(
        self,
        metric_args,
        space,
        tangent_vec,
        base_point,
        end_point,
        is_tangent_atol,
        rtol,
        atol,
    ):
        """Check that parallel transport is an isometry.

        This is for parallel transport defined by boundary value problem (ivp).

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        space : Manifold
            Manifold where metric is defined.
        tangent_vec : array-like
            Tangent vector at base point, to be transported.
        base_point : array-like
            Point on manifold.
        end_point : array-like
            Point on manifold.
        is_tangent_atol: float
            Asbolute tolerance for the is_tangent function.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        transported = space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )
        result = _is_isometry(
            space.metric,
            space,
            tangent_vec,
            transported,
            end_point,
            is_tangent_atol,
            rtol,
            atol,
        )
        expected = gs.array(len(result) * [True])
        self.assertAllEqual(result, expected)

    def test_dist_is_norm_of_log(
        self, space, metric_args, point_a, point_b, rtol, atol
    ):
        """Check that distance is norm of log.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        point_a : array-like
            Point on the manifold.
        point_b : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        log = space.metric.norm(space.metric.log(point_a, point_b), point_b)
        dist = space.metric.dist(point_a, point_b)
        self.assertAllClose(log, dist, rtol, atol)

    def test_dist_point_to_itself_is_zero(self, space, metric_args, point, rtol, atol):
        """Check that distance is norm of log.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        point : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        dist = space.metric.dist(point, point)
        expected = gs.zeros_like(dist)
        self.assertAllClose(dist, expected, rtol, atol)

    def test_triangle_inequality_of_dist(
        self, space, metric_args, point_a, point_b, point_c, atol
    ):
        """Check that distance satisfies traingle inequality.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        point_a : array-like
            Point on the manifold.
        point_b : array-like
            Point on the manifold.
        point_c : array-like
            Point on the manifold.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        dist_ab = space.metric.dist(point_a, point_b)
        dist_bc = space.metric.dist(point_b, point_c)
        dist_ac = space.metric.dist(point_a, point_c)
        result = gs.all(dist_ab + dist_bc + atol >= dist_ac)
        self.assertTrue(result)

    def test_covariant_riemann_tensor_is_skew_symmetric_1(
        self, space, metric_args, base_point
    ):
        """Check that the covariant riemannian tensor verifies first skew symmetry.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        base_point : array-like
            Point on the manifold.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        covariant_metric_tensor = space.metric.covariant_riemann_tensor(base_point)
        skew_symmetry_1 = covariant_metric_tensor + gs.moveaxis(
            covariant_metric_tensor, [-2, -1], [-1, -2]
        )
        result = gs.all(gs.abs(skew_symmetry_1) < gs.atol)
        self.assertAllClose(result, gs.array(True))

    def test_covariant_riemann_tensor_is_skew_symmetric_2(
        self, space, metric_args, base_point
    ):
        """Check that the covariant riemannian tensor verifies second skew symmetry.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        base_point : array-like
            Point on the manifold.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        covariant_metric_tensor = space.metric.covariant_riemann_tensor(base_point)
        skew_symmetry_2 = covariant_metric_tensor + gs.moveaxis(
            covariant_metric_tensor, [-4, -3], [-3, -4]
        )
        result = gs.all(gs.abs(skew_symmetry_2) < gs.atol)
        self.assertAllClose(result, gs.array(True))

    def test_covariant_riemann_tensor_bianchi_identity(
        self, space, metric_args, base_point
    ):
        """Check that the covariant riemannian tensor verifies the Bianchi identity.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        base_point : array-like
            Point on the manifold.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        covariant_metric_tensor = space.metric.covariant_riemann_tensor(base_point)
        bianchi_identity = (
            covariant_metric_tensor
            + gs.moveaxis(covariant_metric_tensor, [-3, -2, -1], [-2, -1, -3])
            + gs.moveaxis(covariant_metric_tensor, [-3, -2, -1], [-1, -3, -2])
        )
        result = gs.all(gs.abs(bianchi_identity) < gs.atol)
        self.assertAllClose(result, gs.array(True))

    def test_covariant_riemann_tensor_is_interchange_symmetric(
        self, space, metric_args, base_point
    ):
        """Check that the covariant riemannian tensor verifies interchange symmetry.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        base_point : array-like
            Point on the manifold.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        covariant_metric_tensor = space.metric.covariant_riemann_tensor(base_point)
        interchange_symmetry = covariant_metric_tensor - gs.moveaxis(
            covariant_metric_tensor, [-4, -3, -2, -1], [-2, -1, -4, -3]
        )
        result = gs.all(gs.abs(interchange_symmetry) < gs.atol)
        self.assertAllClose(result, gs.array(True))

    def test_sectional_curvature_shape(
        self, space, metric_args, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        """Check that scalar_curvature returns an array of the expected shape.

        Parameters
        ----------
        connection_args : tuple
            Arguments to pass to constructor of the connection.
        base_point : array-like
            Point on the manifold.
        expected : tuple
            Expected shape for the result of the ricci_tensor function.
        """
        space.equip_with_metric(self.Metric, **metric_args)

        sectional = space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        result = sectional.shape
        self.assertAllClose(result, expected)


class ComplexRiemannianMetricTestCase(RiemannianMetricTestCase):
    def test_inner_product_is_hermitian(
        self, space, metric_args, tangent_vec_a, tangent_vec_b, base_point, rtol, atol
    ):
        """Check that the inner product is Hermitian.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        tangent_vec_a : array-like
            Tangent vector to the manifold at base_point.
        tangent_vec_b : array-like
            Tangent vector to the manifold at base_point.
        base_point : array-like
            Point on manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)
        ip_a_b = space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        ip_b_a = space.metric.inner_product(tangent_vec_b, tangent_vec_a, base_point)
        self.assertAllClose(ip_a_b, gs.conj(ip_b_a), rtol, atol)

    def test_inner_product_is_complex(
        self, space, metric_args, tangent_vec_a, tangent_vec_b, base_point
    ):
        space.equip_with_metric(self.Metric, **metric_args)
        result = space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertTrue(gs.is_complex(result))

    def test_dist_is_real(self, space, metric_args, point_a, point_b):
        space.equip_with_metric(self.Metric, **metric_args)
        result = space.metric.dist(point_a, point_b)
        self.assertTrue(not gs.is_complex(result))

    def test_log_is_complex(self, space, metric_args, point, base_point):
        space.equip_with_metric(self.Metric, **metric_args)
        result = space.metric.log(point, base_point)
        self.assertTrue(gs.is_complex(result))

    def test_exp_is_complex(self, space, metric_args, tangent_vec, base_point):
        space.equip_with_metric(self.Metric, **metric_args)
        result = space.metric.exp(tangent_vec, base_point)
        self.assertTrue(gs.is_complex(result))


class ProductRiemannianMetricTestCase(RiemannianMetricTestCase):
    def test_innerproduct_is_sum_of_innerproducts(
        self, metric_args, tangent_vec_a, tangent_vec_b, base_point, rtol, atol
    ):
        """Check the inner-product of the product metric.

        Check that the inner-product of two tangent vectors on the product
        manifold is the sum of the inner-products on each of the manifolds.

        For M = M1 x ... x Mn, equipped with the product Riemannian metric
        g = (g1, ...., gn) and tangent vectors u = (u1, ..., un) and v = (v1, ..., vn),
        we check that:
        <u, v>_g = <u1, v1>_g1 + ... + <un, vn>_gn

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        tangent_vec_a : array-like
            Point on the manifold.
        tangent_vec_b : array-like
            Point on the manifold.
        base_point : array-like
            Point on the manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        metric = self.Metric(*metric_args)
        metrics_list = metric_args[0]
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        expected = sum(
            metric.inner_product(tangent_vec_a[i], tangent_vec_b[i], base_point[i])
            for i, metric in enumerate(metrics_list)
        )
        self.assertAllClose(result, expected, rtol, atol)

    def test_metric_matrix_is_block_diagonal(self, metric_args, base_point):
        """Check that the metric matrix has the correct block diagonal form.

        Check that the metric matrix of the product metric has a block diagonal
        form, where each block is the metric matrix of one manifold.

        For M = M1 x ... x Mn equipped with the product Riemannian metric
        g = (g1, ...., gn), we check that the matrix of g is formed by
        the metric matrices of g1, ..., gn in this order, arranged in a
        block diagonal.

        Parameters
        ----------
        ...
        """
        metric = self.Metric(*metric_args)
        result = metric.metric_matrix(base_point)
        individual_metric_matrices = [metric.matrix for metric in metric_args[0]]
        expected = reduce(gs.kron, individual_metric_matrices)
        self.assertAllClose(result, expected)


class NFoldMetricTestCase(RiemannianMetricTestCase):
    def test_innerproduct_is_sum_of_innerproducts(
        self, metric_args, tangent_vec_a, tangent_vec_b, base_point, rtol, atol
    ):
        """Check the inner-product of the product metric induced by the n-fold metric.

        Check that the inner-product of two tangent vectors on the n-fold
        manifold is the sum of the inner-products on each of the manifolds.

        For M = M0^n, equipped with the n-fold Riemannian metric
        g = (g0, ...., g0) and tangent vectors u = (u1, ..., un) and v = (v1, ..., vn),
        we check that:
        <u, v>_g = <u1, v1>_g0 + ... + <un, vn>_g0

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        tangent_vec_a : array-like, shape=[n_copies, *base_shape]
            Tangent vector on the n-fold manifold.
        tangent_vec_b : array-like, shape=[n_copies, *base_shape]
            Tangent vector on the n-fold manifold.
        base_point : array-like, shape=[n_copies, *base_shape]
            Point on the n-fold manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        metric = self.Metric(*metric_args)
        base_metric = metric_args[0].base_manifold.metric
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        expected = sum(
            base_metric.inner_product(
                base_tangent_vec_a, base_tangent_vec_b, base_base_point
            )
            for base_tangent_vec_a, base_tangent_vec_b, base_base_point in zip(
                tangent_vec_a, tangent_vec_b, base_point
            )
        )
        self.assertAllClose(result, expected, rtol, atol)


class QuotientMetricTestCase(RiemannianMetricTestCase):
    def test_dist_is_smaller_than_bundle_dist(self, metric_args, space, n_points, atol):
        """Check that the quotient distance is smaller than the distance in the bundle.

        Check that the quotient metric distance between two points on the quotient
        is smaller than the fiber bundle distance between the two lifted points.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        space : Manifold
            Space to be equipped with metric.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)
        bundle = space.metric.fiber_bundle

        point_a = space.random_point(n_points)
        point_b = space.random_point(n_points)

        quotient_distance = space.metric.dist(point_a, point_b)
        bundle_distance = bundle.total_space.metric.dist(point_a, point_b)
        result = gs.all(gs.abs(bundle_distance - quotient_distance) > atol)
        self.assertTrue(result)

    def test_log_is_horizontal(self, metric_args, space, n_points, atol):
        """Check the quotient log is a horizontal tangent vector.

        Check that the quotient metric logarithm gives a tangent vector
        that is horizontal for the bundle defining the quotient.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        space : Manifold
            Space to be equipped with metric.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)
        bundle = space.metric.fiber_bundle

        point = space.random_point(n_points)
        base_point = space.random_point(n_points)

        log = space.metric.log(point, base_point)

        result = gs.all(bundle.is_horizontal(log, base_point, atol=atol))
        self.assertTrue(result)


class PullbackMetricTestCase(RiemannianMetricTestCase):
    def test_innerproduct_is_embedding_innerproduct(
        self, metric_args, tangent_vec_a, tangent_vec_b, base_point, rtol, atol
    ):
        """Check that the inner-product correspond to the embedding inner-product.

        Check that the formula defining the pullback-metric inner product is
        verified, i.e.:
        <u, v>_p = g_{f(p)}(df_p u , df_p v)
        for p a point on the manifold, f the immersion defining the pullback metric
        and df_p the differential of f at p.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        tangent_vec_a : array-like
            Tangent vector to the manifold at base_point.
        tangent_vec_b : array-like
            Tangent vector to the manifold at base_point.
        base_point : array-like
            Point on manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        metric = self.Metric(*metric_args)
        immersion = metric.immersion
        differential_immersion = metric.tangent_immersion
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        expected = metric.embedding_metric(
            differential_immersion(tangent_vec_a, base_point),
            differential_immersion(tangent_vec_b, base_point),
            immersion(base_point),
        )
        self.assertAllClose(result, expected, rtol, atol)


class PullbackDiffeoMetricTestCase(TestCase):
    @property
    def Metric(self):
        return self.testing_data.Metric

    def test_diffeomorphism_is_reciprocal(self, space, metric_args, point, rtol, atol):
        """Check that the diffeomorphism and its inverse coincide.

        Check implementation of diffeomorphism and reciprocal does agree.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        point : array-like
            Point on manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)
        point_bis = space.metric.inverse_diffeomorphism(
            space.metric.diffeomorphism(point)
        )
        self.assertAllClose(point_bis, point, rtol, atol)

    def test_tangent_diffeomorphism_is_reciprocal(
        self, space, metric_args, point, tangent_vector, rtol, atol
    ):
        """Check that the diffeomorphism differential and its inverse coincide.

        Check implementation of diffeomorphism and reciprocal differential does agree.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        point : array-like
            Point on manifold.
        tangent_vector : array-like
            Tangent vector to the manifold at point.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        space.equip_with_metric(self.Metric, **metric_args)
        image_point = space.metric.diffeomorphism(point)

        tangent_vector_bis = space.metric.inverse_tangent_diffeomorphism(
            space.metric.tangent_diffeomorphism(tangent_vector, point), image_point
        )

        self.assertAllClose(tangent_vector_bis, tangent_vector, rtol, atol)

    def test_matrix_innerproduct_and_embedded_innerproduct_coincide(
        self, space, metric_args, tangent_vec_a, tangent_vec_b, base_point, rtol, atol
    ):
        """Check that the inner-product embedded and with metric matrix coincide.

        Check that the formula defining the pullback-metric inner product is
        verified, i.e.:
        <u, v>_p = g_{f(p)}(df_p u , df_p v)
        for p a point on the manifold, f the immersion defining the pullback metric
        and df_p the differential of f at p.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        tangent_vec_a : array-like
            Tangent vector to the manifold at base_point.
        tangent_vec_b : array-like
            Tangent vector to the manifold at base_point.
        base_point : array-like
            Point on manifold.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        # Not yet implemented due to need for local basis implementation

    def test_dist_vectorization(self, space, n_samples):
        space.equip_with_metric()
        point_a = space.random_point()
        point_a = gs.broadcast_to(point_a, (n_samples,) + point_a.shape)
        point_b = space.random_point()
        point_b = gs.broadcast_to(point_b, (n_samples,) + point_b.shape)

        results = space.metric.dist(point_a, point_b)
        result = results[0]
        expected = gs.broadcast_to(result, n_samples)

        self.assertAllClose(results, expected)

    def test_geodesic(self, space):
        space.equip_with_metric()
        point_a = space.random_point()
        point_b = space.random_point()

        point = space.metric.geodesic(point_a, point_b)(1 / 2)
        result = math.prod(point.shape)
        expected = math.prod(space.shape)
        self.assertAllClose(result, expected)

    def test_geodesic_vectorization(self, space, n_samples):
        space.equip_with_metric()
        point_a = space.random_point()
        point_b = space.random_point()
        times = gs.broadcast_to(1 / 2, n_samples)

        results = space.metric.geodesic(point_a, point_b)(times)
        result = results[0]
        expected = gs.broadcast_to(result, (n_samples,) + result.shape)

        self.assertAllClose(results, expected)


class InvariantMetricTestCase(RiemannianMetricTestCase):
    @pytest.mark.skip(reason="unknown reason")
    def test_exp_at_identity_of_lie_algebra_belongs(
        self, metric_args, group, lie_algebra_point, atol
    ):
        """Check that exp of a lie algebra element is in group.

        Check that the exp at identity of a lie algebra element
        belongs to the group.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        group : LieGroup
            Lie Group on which invariant metric is defined.
        lie_algebra_point : array-like
            Point on lie algebra.
        atol : float
            Absolute tolerance for the belongs function.
        """
        group.equip_with_metric(self.Metric, **metric_args)
        exp = group.metric.exp(lie_algebra_point, group.identity)
        result = gs.all(group.belongs(exp, atol))
        self.assertTrue(result)

    def test_log_at_identity_belongs_to_lie_algebra(
        self, metric_args, group, point, atol
    ):
        """Check that log belongs to lie algebra.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        group : LieGroup
            Lie Group on which invariant metric is defined.
        point : array-like
            Point on group.
        base_point : array-like
            Point on group.
        atol : float
            Absolute tolerance for the belongs function.
        """
        group.equip_with_metric(self.Metric, **metric_args)
        log = group.metric.log(point, group.identity)
        result = gs.all(group.lie_algebra.belongs(log, atol))
        self.assertTrue(result)

    def test_exp_after_log_at_identity(self, metric_args, group, point, rtol, atol):
        """Check that exp and log at identity are inverse.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        group : LieGroup
            Lie Group on which invariant metric is defined.
        point : array-like
            Point on group.
        base_point : array-like
            Point on group.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        group.equip_with_metric(self.Metric, **metric_args)
        log = group.metric.log(point, group.identity)
        result = group.metric.exp(log, group.identity)
        self.assertAllClose(result, point, rtol, atol)

    def test_log_after_exp_at_identity(
        self, metric_args, group, tangent_vec, rtol, atol
    ):
        """Check that log and exp at identity are inverse.

        Parameters
        ----------
        metric_args : tuple
            Arguments to pass to constructor of the metric.
        group : LieGroup
            Lie Group on which invariant metric is defined.
        point : array-like
            Point on group.
        tangent_vec : array-like
            Tangent vector on group.
        rtol : float
            Relative tolerance to test this property.
        atol : float
            Absolute tolerance to test this property.
        """
        group.equip_with_metric(self.Metric, **metric_args)
        exp = group.metric.exp(tangent_vec, group.identity)
        result = group.metric.log(exp, group.identity)
        self.assertAllClose(result, tangent_vec, rtol, atol)
