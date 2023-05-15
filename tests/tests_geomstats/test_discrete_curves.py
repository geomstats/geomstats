"""Unit tests for parameterized manifolds."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.discrete_curves import (
    DiscreteCurves,
    L2CurvesMetric,
    SRVMetric,
    SRVShapeBundle,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from tests.conftest import Parametrizer
from tests.data.discrete_curves_data import (
    ClosedDiscreteCurvesTestData,
    DiscreteCurvesTestData,
    ElasticMetricTestData,
    L2CurvesMetricTestData,
    SRVMetricTestData,
    SRVQuotientMetricTestData,
    SRVShapeBundleTestData,
)
from tests.geometry_test_cases import (
    ManifoldTestCase,
    RiemannianMetricTestCase,
    TestCase,
)

s2 = Hypersphere(dim=2)
r2 = Euclidean(dim=2)
r3 = Euclidean(dim=3)


class TestDiscreteCurves(ManifoldTestCase, metaclass=Parametrizer):
    skip_test_projection_belongs = True
    skip_test_random_tangent_vec_is_tangent = True

    testing_data = DiscreteCurvesTestData()


class TestClosedDiscreteCurves(ManifoldTestCase, metaclass=Parametrizer):
    skip_test_random_tangent_vec_is_tangent = True
    skip_test_to_tangent_is_tangent = True

    testing_data = ClosedDiscreteCurvesTestData()

    @tests.conftest.np_and_autograd_only
    def test_projection_closed_curves(self, ambient_manifold, curve):
        planar_closed_curve = self.Space(ambient_manifold)
        proj = planar_closed_curve.projection(curve)
        expected = proj
        result = planar_closed_curve.projection(proj)
        self.assertAllClose(result, expected)

        result = proj[-1, :]
        expected = proj[0, :]
        self.assertAllClose(result, expected, rtol=10 * gs.rtol)


class TestL2CurvesMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_belongs = True
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = L2CurvesMetricTestData()

    def test_l2_metric_geodesic(
        self, space, curve_a, curve_b, times, k_sampling_points
    ):
        """Test the geodesic method of L2LandmarksMetric."""
        space.equip_with_metric(self.Metric)

        curves_ab = space.metric.geodesic(curve_a, curve_b)
        curves_ab = curves_ab(times)

        result = curves_ab
        expected = []
        for k in range(k_sampling_points):
            geod = space.ambient_manifold.metric.geodesic(
                initial_point=curve_a[k, :], end_point=curve_b[k, :]
            )
            expected.append(geod(times))
        expected = gs.stack(expected, axis=1)
        self.assertAllClose(result, expected)


class TestSRVMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SRVMetricTestData()

    def test_srv_transform_and_srv_transform_inverse(self, space, rtol, atol):
        """Test that srv and its inverse are inverse."""
        space.equip_with_metric(self.Metric)

        curve = space.random_point(n_samples=2)

        srv = space.metric.f_transform(curve)
        srv_inverse = space.metric.f_transform_inverse(srv, curve[:, 0])

        result = srv.shape
        expected = (curve.shape[0], curve.shape[1] - 1, 3)
        self.assertAllClose(result, expected)

        result = srv_inverse
        expected = curve
        self.assertAllClose(result, expected, rtol, atol)

    def test_diffeomorphism_and_inverse_diffeomorphism(self, space, rtol, atol):
        """Test that srv and its inverse are inverse."""
        space.equip_with_metric(self.Metric)
        curve = space.random_point(n_samples=2)

        image = space.metric.diffeomorphism(curve)
        inverse_image = space.metric.inverse_diffeomorphism(image)

        result = inverse_image.shape
        expected = (curve.shape[0], curve.shape[1], 3)
        self.assertAllClose(result, expected)

        result = inverse_image
        expected = curve
        self.assertAllClose(result, expected, rtol, atol)

    @tests.conftest.np_and_autograd_only
    def test_tangent_diffeomorphism(self, space, n_curves, curve_fun_a):
        """Test differential of square root velocity transform.
        Check that its value at (curve, tangent_vec) coincides
        with the derivative at zero of the square root velocity
        transform of a path of curves starting at curve with
        initial derivative tangent_vec.
        """
        space.equip_with_metric(self.Metric)

        sampling_times = gs.linspace(0.0, 1.0, space.k_sampling_points)
        curve_a = curve_fun_a(sampling_times)
        tangent_vec = gs.transpose(
            gs.tile(
                gs.linspace(1.0, 2.0, space.k_sampling_points),
                (space.ambient_manifold.dim, 1),
            )
        )
        result = space.metric.tangent_diffeomorphism(tangent_vec, curve_a)

        times = gs.linspace(0.0, 1.0, n_curves)
        path_of_curves = curve_a + gs.einsum("i,jk->ijk", times, tangent_vec)
        srv_path = space.metric.f_transform(path_of_curves)
        expected = n_curves * (srv_path[1] - srv_path[0])
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)

    @tests.conftest.np_and_autograd_only
    def test_inverse_tangent_diffeomorphism(self, space, curve_a):
        """Test inverse of differential of square root velocity transform.
        Check that it is the inverse of tangent_diffeomorphism.
        """
        space.equip_with_metric(self.Metric)

        tangent_vec = gs.transpose(
            gs.tile(
                gs.linspace(0.0, 1.0, space.k_sampling_points),
                (space.ambient_manifold.dim, 1),
            )
        )
        srv = space.metric.diffeomorphism(curve_a)
        d_srv = space.metric.tangent_diffeomorphism(tangent_vec, curve_a)
        result = space.metric.inverse_tangent_diffeomorphism(d_srv, srv)
        expected = tangent_vec
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)

    @tests.conftest.np_and_autograd_only
    def test_tangent_diffeomorphism_and_inverse(self, space, curve, tangent_vec):
        """Test inverse of differential of square root velocity transform.
        Check that it is the inverse of tangent_diffeomorphism.
        """
        space.equip_with_metric(self.Metric)

        srv = space.metric.diffeomorphism(curve)
        d_srv = space.metric.tangent_diffeomorphism(tangent_vec, curve)
        result = space.metric.inverse_tangent_diffeomorphism(d_srv, srv)
        expected = tangent_vec
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)

    def test_tangent_diffeomorphism_vectorization(self, space, curve_a, curve_b):
        """Test differential of square root velocity transform.
        Check vectorization.
        """
        space.equip_with_metric(self.Metric)

        curves = gs.stack((curve_a, curve_b))
        tangent_vecs = gs.random.rand(
            2, space.k_sampling_points, space.ambient_manifold.dim
        )

        result = space.metric.tangent_diffeomorphism(tangent_vecs, curves)

        res_a = space.metric.tangent_diffeomorphism(tangent_vecs[0], curve_a)
        res_b = space.metric.tangent_diffeomorphism(tangent_vecs[1], curve_b)
        expected = gs.stack([res_a, res_b])
        self.assertAllClose(result, expected)

    def test_srv_inner_product(self, space, curve, vec_a, vec_b, n_vecs):
        space.equip_with_metric(self.Metric)

        vecs_a = gs.tile(vec_a, (n_vecs, 1, 1))
        vecs_b = gs.tile(vec_b, (n_vecs, 1, 1))
        result = space.metric.inner_product(vecs_a, vecs_b, curve)

        srv = space.metric.f_transform(curve)
        tangent_srv_vecs_a = space.metric.tangent_diffeomorphism(vec_a, curve)
        tangent_srv_vecs_b = space.metric.tangent_diffeomorphism(vec_b, curve)
        expected = space.metric.embedding_space.metric.inner_product(
            tangent_srv_vecs_a, tangent_srv_vecs_b, srv
        )
        self.assertAllClose(result, expected)

        expected = []
        for i in range(n_vecs):
            expected.append(space.metric.inner_product(vecs_a[i], vecs_b[i], curve))
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)

    def test_srv_inner_product_elastic(self, space, curve, vec_a, vec_b):
        """Test inner product of SRVMetric.
        Check that the pullback metric gives an elastic metric
        with parameters a=1, b=1/2.
        """
        space.equip_with_metric(self.Metric)
        k_sampling_points = space.k_sampling_points

        result = space.metric.inner_product(vec_a, vec_b, curve)

        d_vec_a = (k_sampling_points - 1) * (vec_a[1:, :] - vec_a[:-1, :])
        d_vec_b = (k_sampling_points - 1) * (vec_b[1:, :] - vec_b[:-1, :])
        velocity_vec = (k_sampling_points - 1) * (curve[1:, :] - curve[:-1, :])
        velocity_norm = r3.metric.norm(velocity_vec)
        unit_velocity_vec = gs.einsum("ij,i->ij", velocity_vec, 1 / velocity_norm)
        a_param = 1
        b_param = 1 / 2
        integrand = (
            a_param**2 * gs.sum(d_vec_a * d_vec_b, axis=1)
            - (a_param**2 - b_param**2)
            * gs.sum(d_vec_a * unit_velocity_vec, axis=1)
            * gs.sum(d_vec_b * unit_velocity_vec, axis=1)
        ) / velocity_norm
        expected = gs.sum(integrand) / (k_sampling_points - 1)
        self.assertAllClose(result, expected)

    def test_srv_inner_product_and_dist(self, space, curve_a, curve_b):
        """Test that norm of log and dist coincide
        for curves with same / different starting points, and for
        the translation invariant / non invariant SRV metric.
        """
        space.equip_with_metric(self.Metric)
        log = space.metric.log(point=curve_b, base_point=curve_a)
        result = space.metric.norm(vector=log, base_point=curve_a)
        expected = space.metric.dist(curve_a, curve_b)
        self.assertAllClose(result, expected)

    def test_space_derivative(
        self,
        space,
        n_points,
        n_discretized_curves,
    ):
        """Test space derivative.
        Check result on an example and vectorization.
        """
        space.equip_with_metric(self.Metric)

        n_points = 3
        dim = space.ambient_manifold.dim

        curve = gs.random.rand(n_points, dim)
        result = space.metric.space_derivative(curve)
        delta = 1 / n_points
        d_curve_1 = (curve[1] - curve[0]) / delta
        d_curve_2 = (curve[2] - curve[0]) / (2 * delta)
        d_curve_3 = (curve[2] - curve[1]) / delta
        expected = gs.squeeze(
            gs.vstack(
                (
                    gs.to_ndarray(d_curve_1, 2),
                    gs.to_ndarray(d_curve_2, 2),
                    gs.to_ndarray(d_curve_3, 2),
                )
            )
        )
        self.assertAllClose(result, expected)

        path_of_curves = gs.random.rand(
            n_discretized_curves, space.k_sampling_points, dim
        )
        result = space.metric.space_derivative(path_of_curves)
        expected = []
        for i in range(n_discretized_curves):
            expected.append(space.metric.space_derivative(path_of_curves[i]))
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)

    def test_srv_metric_pointwise_inner_products(
        self, space, curves_ab, curves_bc, n_discretized_curves
    ):
        space.equip_with_metric(self.Metric)
        k_sampling_points = space.k_sampling_points
        srv_metric_r3 = space.metric

        l2_metric_s2 = L2CurvesMetric(
            DiscreteCurves(
                ambient_manifold=s2, k_sampling_points=space.k_sampling_points
            )
        )

        tangent_vecs = l2_metric_s2.log(point=curves_bc, base_point=curves_ab)
        result = srv_metric_r3.embedding_space.metric.pointwise_inner_products(
            tangent_vec_a=tangent_vecs, tangent_vec_b=tangent_vecs, base_point=curves_ab
        )
        expected_shape = (n_discretized_curves, k_sampling_points)
        self.assertAllClose(gs.shape(result), expected_shape)

        result = srv_metric_r3.embedding_space.metric.pointwise_inner_products(
            tangent_vec_a=tangent_vecs[0],
            tangent_vec_b=tangent_vecs[0],
            base_point=curves_ab[0],
        )
        expected_shape = (k_sampling_points,)
        self.assertAllClose(gs.shape(result), expected_shape)

    def test_srv_transform_and_inverse(self, space, curves):
        """Test of SRVT and its inverse.
        N.B: Here curves_ab are seen as curves in R3 and not S2.
        """
        space.equip_with_metric(self.Metric)
        srv_curves = space.metric.f_transform(curves)
        starting_points = curves[:, 0, :]
        result = space.metric.f_transform_inverse(srv_curves, starting_points)
        expected = curves

        self.assertAllClose(result, expected)


class TestElasticMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_shape = True
    skip_test_log_shape = True
    skip_test_exp_geodesic_ivp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_after_log = True
    skip_test_exp_belongs = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_inner_product_is_symmetric = True
    skip_test_log_after_exp = True
    skip_test_log_is_tangent = True
    skip_test_dist_is_norm_of_log = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_triangle_inequality_of_dist = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = ElasticMetricTestData()

    def test_cartesian_to_polar_and_polar_to_cartesian(
        self, space, a, b, n_samples, rtol, atol
    ):
        """Test conversion to polar coordinate"""
        space.equip_with_metric(self.Metric, a=a, b=b)

        curve = space.random_point(n_samples=n_samples)
        polar_curve = space.metric._cartesian_to_polar(curve)
        result = space.metric._polar_to_cartesian(polar_curve)

        self.assertAllClose(result, curve, rtol=rtol, atol=atol)

    def test_f_transform_and_srv_transform(self, space, n_samples, rtol, atol):
        """Test that the f transform coincides with the SRVF.

        This is valid for a f_transform with a=1, b=1/2.
        """
        curve = space.random_point(n_samples)

        space.equip_with_metric(self.Metric, a=1.0, b=0.5)
        result = space.metric.f_transform(curve)

        space.equip_with_metric(SRVMetric)
        expected = space.metric.f_transform(curve)

        self.assertAllClose(result, expected, rtol, atol)

    def test_f_transform_inverse_and_srv_transform_inverse(
        self, space, curve, rtol, atol
    ):
        """Test that the f transform coincides with the SRVF

        This is valid for a f transform with a=1, b=1/2.
        """
        space.equip_with_metric(self.Metric, a=1, b=0.5)

        starting_point = curve[0]
        fake_transformed_curve = curve[1:, :]

        result = space.metric.f_transform_inverse(
            fake_transformed_curve, starting_point
        )

        space.equip_with_metric(SRVMetric)
        expected = space.metric.f_transform_inverse(
            fake_transformed_curve, starting_point
        )
        self.assertAllClose(result, expected, rtol, atol)

    def test_f_transform_and_f_transform_inverse(self, space, a, b, curve, rtol, atol):
        """Test that the inverse is right."""
        space.equip_with_metric(self.Metric, a=a, b=b)

        f = space.metric.f_transform(curve)
        f_inverse = space.metric.f_transform_inverse(f, curve[0])

        result = f.shape
        expected = (curve.shape[0] - 1, 2)
        self.assertAllClose(result, expected)

        result = f_inverse
        expected = curve
        self.assertAllClose(result, expected, rtol, atol)

    def test_f_transform_and_diffeomorphism(self, space, a, b, n_samples, rtol, atol):
        """Test that f_transform coincides with
        diffeomorphism.
        """
        space.equip_with_metric(self.Metric, a=a, b=b)

        curves = space.random_point(n_samples=n_samples)

        result = space.metric.f_transform(curves)
        expected = space.metric.diffeomorphism(curves)

        self.assertAllClose(result, expected, rtol, atol)

    def test_f_transform_inverse_and_inverse_diffeomorphism(
        self, space, a, b, curve, rtol, atol
    ):
        """Test that the f transform inverse coincides
        with the inverse diffeomorphism when starting at 0.
        """
        space.equip_with_metric(self.Metric, a=a, b=b)

        starting_point = gs.zeros(gs.shape(curve[..., 0, :]))
        fake_transformed_curve = curve[1:, :]

        result = space.metric.inverse_diffeomorphism(fake_transformed_curve)
        expected = space.metric.f_transform_inverse(
            fake_transformed_curve, starting_point
        )

        self.assertAllClose(result, expected, rtol, atol)


class TestSRVShapeBundle(TestCase, metaclass=Parametrizer):
    testing_data = SRVShapeBundleTestData()

    def test_horizontal_and_vertical_projections(
        self, times, n_discretized_curves, curve_a, curve_b
    ):
        """Test horizontal and vertical projections.
        Check that horizontal and vertical parts of any tangent
        vector are othogonal with respect to the SRVMetric inner
        product, and check vectorization.
        """
        total_space = DiscreteCurves(r3, equip=True)
        srv_metric_r3 = total_space.metric
        srv_shape_bundle_r3 = SRVShapeBundle(total_space)

        geod = srv_metric_r3.geodesic(initial_point=curve_a, end_point=curve_b)
        geod = geod(times)
        tangent_vec = n_discretized_curves * (geod[1, :, :] - geod[0, :, :])
        tangent_vec_hor = srv_shape_bundle_r3.horizontal_projection(
            tangent_vec, curve_a
        )
        tangent_vec_ver = srv_shape_bundle_r3.vertical_projection(tangent_vec, curve_a)
        result = srv_metric_r3.inner_product(tangent_vec_hor, tangent_vec_ver, curve_a)
        expected = 0.0
        self.assertAllClose(result, expected, atol=1e-4)

        tangent_vecs = n_discretized_curves * (geod[1:] - geod[:-1])
        _, result = srv_shape_bundle_r3.vertical_projection(
            tangent_vecs, geod[:-1], return_norm=True
        )
        expected = []
        for i in range(n_discretized_curves - 1):
            _, res = srv_shape_bundle_r3.vertical_projection(
                tangent_vecs[i], geod[i], return_norm=True
            )
            expected.append(res)
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)

    def test_horizontal_geodesic(
            self, k_sampling_points, curve_a, n_times, type_method):
        """Test horizontal geodesic.
        Check that the time derivative of the geodesic is
        horizontal at all time.
        """
        curve_b = gs.transpose(
            gs.stack(
                (
                    gs.zeros(k_sampling_points),
                    gs.zeros(k_sampling_points),
                    gs.linspace(1.0, 0.5, k_sampling_points),
                )
            )
        )

        total_space = DiscreteCurves(r3, equip=True)
        srv_shape_bundle_r3 = SRVShapeBundle(total_space)

        method = type_method[0]
        threshold = type_method[1]

        horizontal_geod_fun = srv_shape_bundle_r3.horizontal_geodesic(
            curve_a, curve_b, method=method)
        times = gs.linspace(0.0, 1.0, n_times)
        horizontal_geod = horizontal_geod_fun(times)
        velocity_vec = n_times * (horizontal_geod[1:] - horizontal_geod[:-1])
        _, vertical_norms = srv_shape_bundle_r3.vertical_projection(
            velocity_vec, horizontal_geod[:-1], return_norm=True
        )
        result = gs.sum(vertical_norms**2, axis=1) ** (1 / 2)
        expected = gs.zeros(n_times - 1)
        self.assertAllClose(result, expected, atol=threshold)


class TestSRVQuotientMetric(TestCase, metaclass=Parametrizer):
    testing_data = SRVQuotientMetricTestData()

    def test_dist(self, sampling_times, curve_fun_a, curve_a, k_sampling_points):
        """Test quotient distance.

        Check that the quotient distance is the same as the distance
        between the end points of the horizontal geodesic.
        """
        space = DiscreteCurves(ambient_manifold=r3)
        space.equip_with_group_action("reparametrizations")

        space.equip_with_quotient_structure()

        curve_a_resampled = curve_fun_a(sampling_times**2)
        curve_b = gs.transpose(
            gs.stack(
                (
                    gs.zeros(k_sampling_points),
                    gs.zeros(k_sampling_points),
                    gs.linspace(1.0, 0.5, k_sampling_points),
                )
            )
        )
        srv_quotient_metric_r3 = space.quotient.metric
        result = srv_quotient_metric_r3.dist(curve_a_resampled, curve_b)
        expected = srv_quotient_metric_r3.dist(curve_a, curve_b)
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)

    def test_horizontal_geodesic(self, k_sampling_points, curve_a, n_times):
        """Test horizontal geodesic.
        Check that the time derivative of the geodesic is
        horizontal at all time.
        """
        curve_b = gs.transpose(
            gs.stack(
                (
                    gs.zeros(k_sampling_points),
                    gs.zeros(k_sampling_points),
                    gs.linspace(1.0, 0.5, k_sampling_points),
                )
            )
        )

        space = DiscreteCurves(ambient_manifold=r3)
        space.equip_with_group_action("reparametrizations")
        space.equip_with_quotient_structure()

        srv_quotient_metric_r3 = space.quotient.metric
        srv_shape_bundle_r3 = space.fiber_bundle

        horizontal_geod_fun = srv_quotient_metric_r3.\
            horizontal_geodesic(curve_a, curve_b)
        total_space = DiscreteCurves(r3, equip=True)
        srv_shape_bundle_r3 = SRVShapeBundle(total_space)

        method = type[0]
        threshold = type[1]

        horizontal_geod_fun = srv_shape_bundle_r3.horizontal_geodesic(
            curve_a, curve_b, method=method)
        times = gs.linspace(0.0, 1.0, n_times)
        horizontal_geod = horizontal_geod_fun(times)
        velocity_vec = n_times * (horizontal_geod[1:] - horizontal_geod[:-1])
        _, vertical_norms = srv_shape_bundle_r3.vertical_projection(
            velocity_vec, horizontal_geod[:-1], return_norm=True
        )
        result = gs.sum(vertical_norms**2, axis=1) ** (1 / 2)
        expected = gs.zeros(n_times - 1)
        self.assertAllClose(result, expected, atol=threshold)
