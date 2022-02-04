"""Test properties of differential geometry."""

from conftest import Parametrizer

import geomstats.backend as gs


class ConnectionParametrizer(Parametrizer):
    def __new__(cls, name, bases, attrs):
        def test_exp_belongs(self, metric_args, space, tangent_vec, base_point):
            metric = self.cls(*metric_args)
            exp = metric.exp(gs.array(tangent_vec), gs.array(base_point))
            self.assertAllClose(gs.all(space.belongs(exp)), True)

        def test_log_is_tangent(self, metric_args, space, base_point, point):
            metric = self.cls(*metric_args)
            log = metric.log(gs.array(base_point), gs.array(point))
            self.assertAllClose(
                gs.all(space.is_tangent(log, gs.array(base_point))), True
            )

        def test_geodesic_belongs(self, metric_args, space, tangent_vec, base_point):
            # self.assertAllClose(gs.all(space.belongs(points)), True)
            raise NotImplementedError

        def test_log_exp_composition(self, metric_args, point, base_point, rtol, atol):
            metric = self.cls(*metric_args)
            log = metric.log(gs.array(point), base_point=gs.array(base_point))
            result = metric.exp(tangent_vec=log, base_point=gs.array(base_point))
            self.assertAllClose(result, point, rtol=rtol, atol=atol)

        def test_exp_log_composition(self, metric_args, point, base_point, rtol, atol):
            metric = self.cls(*metric_args)
            log = metric.log(gs.array(point), base_point=gs.array(base_point))
            result = metric.exp(tangent_vec=log, base_point=gs.array(base_point))
            self.assertAllClose(result, point, rtol=rtol, atol=atol)

        def test_exp_geodesic(self, metric_args, point, base_point, rtol, atol):
            raise NotImplementedError

        def test_exp_parallel_transport(
            self, metric_args, point, base_point, rtol, atol
        ):
            raise NotImplementedError

        attrs[test_exp_belongs.__name__] = test_exp_belongs
        attrs[test_log_is_tangent.__name__] = test_log_is_tangent
        attrs[test_geodesic_belongs.__name__] = test_geodesic_belongs
        attrs[test_log_exp_composition.__name__] = test_log_exp_composition
        attrs[test_exp_log_composition.__name__] = test_exp_log_composition
        attrs[test_exp_geodesic.__name__] = test_exp_geodesic
        attrs[test_exp_parallel_transport.__name__] = test_exp_parallel_transport

        return super(ConnectionParametrizer, cls).__new__(cls, name, bases, attrs)


class RiemannianMetricParametrizer(Parametrizer):
    def __new__(cls, name, bases, attrs):
        def test_squared_dist_is_symmetric(
            self, metric_args, point_a, point_b, rtol, atol
        ):
            metric = self.cls(*metric_args)
            sd_a_b = metric.squared_dist(gs.array(point_a), gs.array(point_b))
            sd_b_a = metric.squared_dist(gs.array(point_b), gs.array(point_a))
            self.assertAllClose(sd_a_b, sd_b_a, rtol=rtol, atol=atol)

        attrs[test_squared_dist_is_symmetric.__name__] = test_squared_dist_is_symmetric

        return super(RiemannianMetricParametrizer, cls).__new__(cls, name, bases, attrs)
