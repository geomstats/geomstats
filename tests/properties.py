"""Test properties of differential geometry."""

import geomstats.backend as gs


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


class ManifoldProperties:
    def projection_shape_and_belongs(self, space_args, data, expected, belongs_atol):
        space = self.space(*space_args)
        belongs = space.belongs(space.projection(gs.array(data)), belongs_atol)
        self.assertAllClose(gs.all(belongs), gs.array(True))
        self.assertAllClose(gs.shape(belongs), expected)

    def to_tangent_shape_and_is_tangent(self, space_args, data, expected):
        space = self.space(*space_args)
        tangent = space.to_tangent(gs.array(data))
        self.assertAllClose(gs.all(space.is_tangent(tangent)), gs.array(True))
        self.assertAllclose(gs.shape(tangent), expected)


class OpenSetProperties(ManifoldProperties):
    def to_tangent_belongs_ambient_space(self, space_args, data, belongs_atol):
        space = self.space(*space_args)
        result = gs.all(space.ambient_space.belongs(gs.array(data), belongs_atol))
        self.asertAllClose(result, gs.array(True))


class LieGroupProperties:
    def exp_log_composition(self, group_args, tangent_vec, base_point, rtol, atol):
        group = self.group(*group_args)
        exp_point = group.exp_from_identity(gs.array(tangent_vec), gs.array(base_point))
        log_vec = group.log_from_idenity(exp_point, gs.array(base_point))
        self.assertAllClose(log_vec, gs.array(tangent_vec), rtol, atol)

    def log_exp_composition(self, group_args, point, base_point, rtol, atol):
        group = self.group(*group_args)
        log_vec = group.log_from_identity(gs.array(point), gs.array(base_point))
        exp_point = group.exp_from_identity(log_vec, gs.array(base_point))
        self.assertAllClose(exp_point, gs.array(point), rtol, atol)


class VectorSpaceProperties:
    def basis_belongs_and_basis_cardinality(self, algebra_args, belongs_atol):
        algebra = self.algebra(*algebra_args)
        basis = algebra.belongs(algebra.basis, belongs_atol)
        self.assertAllClose(len(basis), algebra.dim)
        self.assertAllClose(gs.all(basis), gs.array(True))


class LieAlgebraProperties(VectorSpaceProperties):
    def basis_representation_matrix_represenation_composition(
        self, algebra_args, mat_rep, rtol, atol
    ):
        algebra = self.algebra(*algebra_args)
        basis_rep = algebra.basis_representation(gs.array(mat_rep))
        result = algebra.matrix_representation(basis_rep)
        self.assertAllClose(result, gs.array(mat_rep), rtol, atol)

    def matrix_representation_basis_represenation_composition(
        self, algebra_args, basis_rep, atol, rtol
    ):
        algebra = self.algebra(*algebra_args)
        mat_rep = algebra.matrix_representation(basis_rep)
        result = algebra.basis_representation(mat_rep)
        self.assertAllClose(result, gs.array(basis_rep), rtol, atol)


class LevelSetProperties(ManifoldProperties):
    def extrinsic_then_intrinsic(self, space_args, point, rtol, atol):
        space = self.space(*space_args)
        point_intrinsic = space.extrinsic_to_intrinsic_coords(point)
        result = space.intrinsic_to_extrinsic_coords(point_intrinsic)
        expected = point

        self.assertAllClose(result, expected, rtol, atol)

    def intrinsic_then_extrinsic(self, space_args, point, rtol, atol):
        space = self.space(*space_args)
        point_extrinsic = space.intrinsic_to_extrinsic_coords(point)
        result = space.extrinsic_to_intrinsic_coords(point_extrinsic)
        expected = point

        self.assertAllClose(result, expected, rtol, atol)


class ConnectionProperties:
    def exp_belongs(
        self, connection_args, space, tangent_vec, base_point, belongs_atol
    ):
        connection = self.connection(*connection_args)
        exp = connection.exp(gs.array(tangent_vec), gs.array(base_point))
        result = gs.all(space.belongs(exp, belongs_atol))
        self.assertAllClose(result, gs.array(True))

    def log_is_tangent(
        self, connection_args, space, base_point, point, is_tangent_atol
    ):
        connection = self.connection(*connection_args)
        log = connection.log(gs.array(base_point), gs.array(point))
        result = gs.all(space.is_tangent(log, gs.array(base_point), is_tangent_atol))
        self.assertAllClose(result, gs.array(True))

    def geodesic_ivp_belongs(
        self,
        connection_args,
        space,
        n_points,
        initial_point,
        initial_tangent_vec,
        belongs_atol,
    ):
        connection = self.connection(*connection_args)
        geodesic = connection.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )

        t = gs.linspace(start=0.0, stop=1.0, num=n_points)
        points = geodesic(t)

        result = space.belongs(points, belongs_atol)
        expected = gs.array(n_points * [True])

        self.assertAllClose(result, expected)

    def geodesic_bvp_belongs(
        self, connection_args, space, n_points, initial_point, end_point, belongs_atol
    ):
        connection = self.connection(*connection_args)

        geodesic = connection.geodesic(initial_point=initial_point, end_point=end_point)

        t = gs.linspace(start=0.0, stop=1.0, num=n_points)
        points = geodesic(t)

        result = space.belongs(points, belongs_atol)
        expected = gs.array(n_points * [True])

        self.assertAllClose(result, expected)

    def log_exp_composition(self, connection_args, point, base_point, rtol, atol):
        connection = self.connection(*connection_args)
        log = connection.log(gs.array(point), base_point=gs.array(base_point))
        result = connection.exp(tangent_vec=log, base_point=gs.array(base_point))
        self.assertAllClose(result, point, rtol=rtol, atol=atol)

    def exp_log_composition(self, connection_args, tangent_vec, base_point, rtol, atol):
        connection = self.connection(*connection_args)
        exp = connection.exp(tangent_vec=tangent_vec, base_point=gs.array(base_point))
        result = connection.log(exp, base_point=gs.array(base_point))
        self.assertAllClose(result, tangent_vec, rtol=rtol, atol=atol)

    def exp_ladder_parallel_transport(
        self,
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
        connection = self.connection(*connection_args)

        ladder = connection.ladder_parallel_transport(
            tangent_vec,
            base_point,
            direction,
            n_rungs=n_rungs,
            scheme=scheme,
            alpha=alpha,
        )

        result = ladder["end_point"]
        expected = connection.exp(direction, base_point)

        self.assertAllClose(result, expected, rtol=rtol, atol=atol)

    def exp_geodesic_ivp(
        self, connection_args, n_points, tangent_vec, base_point, rtol, atol
    ):
        connection = self.connection(*connection_args)
        geodesic = connection.geodesic(
            initial_point=base_point, initial_tangent_vec=tangent_vec
        )
        t = gs.linspace(start=0.0, stop=1.0, num=n_points)
        points = geodesic(t)
        result = points[:, -1]
        expected = connection.exp(tangent_vec, base_point)
        self.assertAllClose(expected, result, rtol=rtol, atol=atol)


class RiemannianMetricProperties(ConnectionProperties):
    def squared_dist_is_symmetric(self, metric_args, point_a, point_b, rtol, atol):
        metric = self.metric(*metric_args)
        sd_a_b = metric.squared_dist(gs.array(point_a), gs.array(point_b))
        sd_b_a = metric.squared_dist(gs.array(point_b), gs.array(point_a))
        self.assertAllClose(sd_a_b, sd_b_a, rtol=rtol, atol=atol)

    def parallel_transport_ivp_is_isometry(
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
        metric = self.metric(*metric_args)

        end_point = metric.exp(direction, base_point)

        transported = metric.parallel_transport(tangent_vec, base_point, direction)
        result = _is_isometry(
            metric,
            space,
            tangent_vec,
            transported,
            end_point,
            is_tangent_atol,
            rtol,
            atol,
        )
        expected = gs.array(len(result) * [True])
        self.assertAllClose(result, expected)

    def parallel_transport_bvp_is_isometry(
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
        metric = self.metric(*metric_args)

        end_point = metric.exp(direction, base_point)

        transported = metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )
        result = self._is_isometry(
            metric,
            space,
            tangent_vec,
            transported,
            end_point,
            is_tangent_atol,
            rtol,
            atol,
        )
        expected = gs.array(len(result) * [True])
        self.assertAllClose(result, expected)
