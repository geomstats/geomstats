"""Test properties of differential geometry."""

from conftest import Parametrizer

import geomstats.backend as gs


class ConnectionParametrizer(Parametrizer):
    def __new__(cls, name, bases, attrs):
        def test_exp_belongs(self, connection_args, space, tangent_vec, base_point):
            connection = self.cls(*connection_args)
            exp = connection.exp(gs.array(tangent_vec), gs.array(base_point))
            self.assertAllClose(gs.all(space.belongs(exp)), True)

        def test_log_is_tangent(self, connection_args, space, base_point, point):
            connection = self.cls(*connection_args)
            log = connection.log(gs.array(base_point), gs.array(point))
            self.assertAllClose(
                gs.all(space.is_tangent(log, gs.array(base_point))), True
            )

        def test_geodesic_ivp_belongs(
            self, connection_args, space, n_points, initial_point, initial_tangent_vec
        ):
            connection = self.cls(*connection_args)
            geodesic = connection.geodesic(
                initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
            )

            t = gs.linspace(start=0.0, stop=1.0, num=n_points)
            points = geodesic(t)

            result = space.belongs(points)
            expected = gs.array(n_points * [True])

            self.assertAllClose(result, expected)

        def test_geodesic_bvp_belongs(
            self, connection_args, space, n_points, initial_point, end_point
        ):
            connection = self.cls(*connection_args)

            geodesic = connection.geodesic(
                initial_point=initial_point, end_point=end_point
            )

            t = gs.linspace(start=0.0, stop=1.0, num=n_points)
            points = geodesic(t)

            result = self.space.belongs(points)
            expected = gs.array(n_points * [True])

            self.assertAllClose(result, expected)
            raise NotImplementedError

        def test_log_exp_composition(
            self, connection_args, point, base_point, rtol, atol
        ):
            connection = self.cls(*connection_args)
            log = connection.log(gs.array(point), base_point=gs.array(base_point))
            result = connection.exp(tangent_vec=log, base_point=gs.array(base_point))
            self.assertAllClose(result, point, rtol=rtol, atol=atol)

        def test_exp_log_composition(
            self, connection_args, tangent_vec, base_point, rtol, atol
        ):
            connection = self.cls(*connection_args)
            exp = connection.exp(
                tangent_vec=tangent_vec, base_point=gs.array(base_point)
            )
            result = connection.log(exp, base_point=gs.array(base_point))
            self.assertAllClose(result, tangent_vec, rtol=rtol, atol=atol)

        def test_exp_ladder_parallel_transport(
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
            connection = self.cls(*connection_args)

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

            self.assertAllClose(result, expected)

        def test_exp_geodesic_ivp(
            self, connection_args, n_points, tangent_vec, base_point, rtol, atol
        ):
            connection = self.cls(*connection_args)
            geodesic = connection.geodesic(
                initial_point=base_point, initial_tangent_vec=tangent_vec
            )
            t = gs.linspace(start=0.0, stop=1.0, num=n_points)
            points = geodesic(t)
            result = points[:, -1]
            expected = connection.exp(tangent_vec, base_point)
            self.assertAllClose(expected, result, rtol=rtol, atol=atol)

        attrs[test_exp_belongs.__name__] = test_exp_belongs
        attrs[test_log_is_tangent.__name__] = test_log_is_tangent
        attrs[test_geodesic_ivp_belongs.__name__] = test_geodesic_ivp_belongs
        attrs[test_geodesic_bvp_belongs.__name__] = test_geodesic_bvp_belongs
        attrs[test_log_exp_composition.__name__] = test_log_exp_composition
        attrs[test_exp_log_composition.__name__] = test_exp_log_composition
        attrs[test_exp_geodesic_ivp.__name__] = test_exp_geodesic_ivp
        attrs[
            test_exp_ladder_parallel_transport.__name__
        ] = test_exp_ladder_parallel_transport

        return super(ConnectionParametrizer, cls).__new__(cls, name, bases, attrs)


class RiemannianMetricParametrizer(ConnectionParametrizer):
    def __new__(cls, name, bases, attrs):
        def test_squared_dist_is_symmetric(
            self, metric_args, point_a, point_b, rtol, atol
        ):
            metric = self.cls(*metric_args)
            sd_a_b = metric.squared_dist(gs.array(point_a), gs.array(point_b))
            sd_b_a = metric.squared_dist(gs.array(point_b), gs.array(point_a))
            self.assertAllClose(sd_a_b, sd_b_a, rtol=rtol, atol=atol)

        attrs[test_squared_dist_is_symmetric.__name__] = test_squared_dist_is_symmetric

        def _is_isometry(self, metric_args, space, tan_a, trans_a, endpoint):
            metric = self.cls(*metric_args)

            is_tangent = space.is_tangent(trans_a, endpoint)
            is_equinormal = gs.isclose(
                metric.norm(trans_a, endpoint), metric.norm(tan_a, endpoint)
            )
            return gs.logical_and(is_tangent, is_equinormal)

        def test_parallel_transport_ivp_is_isometry(
            self, metric_args, space, tangent_vec, base_point, direction, rtol, atol
        ):
            metric = self.cls(*metric_args)

            end_point = metric.exp(direction, base_point)

            transported = metric.parallel_transport(tangent_vec, base_point, direction)
            result = _is_isometry(tangent_vec, transported, end_point)
            expected = gs.array(len(result) * [True])
            self.assertAllClose(result, expected, rtol=rtol, atol=atol)

        def test_parallel_transport_bvp_is_isometry(
            self, metric_args, tangent_vec, base_point, direction, rtol, atol
        ):
            metric = self.cls(*metric_args)

            end_point = metric.exp(direction, base_point)

            transported = metric.parallel_transport(
                tangent_vec, base_point, end_point=end_point
            )
            result = _is_isometry(tangent_vec, transported, end_point)
            expected = gs.array(len(result) * [True])
            self.assertAllClose(result, expected, rtol=rtol, atol=atol)

        attrs[test_squared_dist_is_symmetric.__name__] = test_squared_dist_is_symmetric
        attrs[
            test_parallel_transport_ivp_is_isometry.__name__
        ] = test_parallel_transport_ivp_is_isometry
        attrs[
            test_parallel_transport_bvp_is_isometry.__name__
        ] = test_parallel_transport_bvp_is_isometry

        return super(RiemannianMetricParametrizer, cls).__new__(cls, name, bases, attrs)
