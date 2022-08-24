"""Unit tests for parameterized manifolds."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.discrete_curves import (
    DiscreteCurves,
    ElasticMetric,
    L2CurvesMetric,
    QuotientSRVMetric,
    SRVMetric,
    SRVShapeBundle,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.tests import tf_backend
from tests.conftest import Parametrizer
from tests.data.discrete_curves_data import (
    ClosedDiscreteCurvesTestData,
    DiscreteCurvesTestData,
    ElasticMetricTestData,
    L2CurvesMetricTestData,
    QuotientSRVMetricTestData,
    SRVMetricTestData,
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
    skip_test_projection_belongs = tf_backend()
    skip_test_random_tangent_vec_is_tangent = True
    skip_test_to_tangent_is_tangent = True

    testing_data = ClosedDiscreteCurvesTestData()

    @geomstats.tests.np_and_autograd_only
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
    skip_test_dist_is_norm_of_log = tf_backend()
    skip_test_dist_is_symmetric = tf_backend()
    skip_test_squared_dist_is_symmetric = tf_backend()
    skip_test_inner_product_is_symmetric = tf_backend()

    testing_data = L2CurvesMetricTestData()

    def test_l2_metric_geodesic(
        self, ambient_manifold, curve_a, curve_b, times, k_sampling_points
    ):
        """Test the geodesic method of L2LandmarksMetric."""
        l2_metric_s2 = self.Metric(ambient_manifold=s2)
        curves_ab = l2_metric_s2.geodesic(curve_a, curve_b)
        curves_ab = curves_ab(times)

        result = curves_ab
        expected = []
        for k in range(k_sampling_points):
            geod = l2_metric_s2.ambient_metric.geodesic(
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
    skip_test_exp_after_log = tf_backend()
    skip_test_exp_belongs = tf_backend()
    skip_test_exp_ladder_parallel_transport = tf_backend()
    skip_test_inner_product_is_symmetric = tf_backend()
    skip_test_log_after_exp = tf_backend()
    skip_test_log_is_tangent = tf_backend()

    testing_data = SRVMetricTestData()

    def test_srv_inner_product(self, curve_a, curve_b, curve_c, times):
        l2_metric_s2 = L2CurvesMetric(ambient_manifold=s2)
        srv_metric_r3 = SRVMetric(ambient_manifold=r3)
        curves_ab = l2_metric_s2.geodesic(curve_a, curve_b)
        curves_bc = l2_metric_s2.geodesic(curve_b, curve_c)
        curves_ab = curves_ab(times)
        curves_bc = curves_bc(times)
        srvs_ab = srv_metric_r3.srv_transform(curves_ab)
        srvs_bc = srv_metric_r3.srv_transform(curves_bc)

        result = srv_metric_r3.l2_curves_metric.inner_product(srvs_ab, srvs_bc)
        products = srvs_ab * srvs_bc
        expected = [gs.sum(product) for product in products]
        expected = gs.array(expected) / (srvs_ab.shape[-2] + 1)
        self.assertAllClose(result, expected)

        result = result.shape
        expected = [srvs_ab.shape[0]]
        self.assertAllClose(result, expected)

    def test_srv_norm(self, curve_a, curve_b, times):
        l2_metric_s2 = L2CurvesMetric(ambient_manifold=s2)
        srv_metric_r3 = SRVMetric(ambient_manifold=r3)
        curves_ab = l2_metric_s2.geodesic(curve_a, curve_b)
        curves_ab = curves_ab(times)
        srvs_ab = srv_metric_r3.srv_transform(curves_ab)

        result = srv_metric_r3.l2_curves_metric.norm(srvs_ab)
        products = srvs_ab * srvs_ab
        sums = [gs.sum(product) for product in products]
        squared_norm = gs.array(sums) / (srvs_ab.shape[-2] + 1)
        expected = gs.sqrt(squared_norm)
        self.assertAllClose(result, expected)

        result = result.shape
        expected = [srvs_ab.shape[0]]
        self.assertAllClose(result, expected)

    def test_srv_transform_and_srv_transform_inverse(self, rtol, atol):
        """Test that srv and its inverse are inverse."""
        metric = SRVMetric(ambient_manifold=r3)
        curve = DiscreteCurves(r3).random_point(n_samples=2)

        srv = metric.srv_transform(curve)
        srv_inverse = metric.srv_transform_inverse(srv, curve[:, 0])

        result = srv.shape
        expected = (curve.shape[0], curve.shape[1] - 1, 3)
        self.assertAllClose(result, expected)

        result = srv_inverse
        expected = curve
        self.assertAllClose(result, expected, rtol, atol)

    @geomstats.tests.np_and_autograd_only
    def test_aux_differential_srv_transform(
        self, dim, k_sampling_points, n_curves, curve_fun_a
    ):
        """Test differential of square root velocity transform.
        Check that its value at (curve, tangent_vec) coincides
        with the derivative at zero of the square root velocity
        transform of a path of curves starting at curve with
        initial derivative tangent_vec.
        """
        srv_metric_r3 = SRVMetric(r3)
        sampling_times = gs.linspace(0.0, 1.0, k_sampling_points)
        curve_a = curve_fun_a(sampling_times)
        tangent_vec = gs.transpose(
            gs.tile(gs.linspace(1.0, 2.0, k_sampling_points), (dim, 1))
        )
        result = srv_metric_r3.aux_differential_srv_transform(tangent_vec, curve_a)

        times = gs.linspace(0.0, 1.0, n_curves)
        path_of_curves = curve_a + gs.einsum("i,jk->ijk", times, tangent_vec)
        srv_path = srv_metric_r3.srv_transform(path_of_curves)
        expected = n_curves * (srv_path[1] - srv_path[0])
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)

    @geomstats.tests.np_and_autograd_only
    def test_aux_differential_srv_transform_inverse(
        self, dim, k_sampling_points, curve_a
    ):
        """Test inverse of differential of square root velocity transform.
        Check that it is the inverse of aux_differential_srv_transform.
        """
        tangent_vec = gs.transpose(
            gs.tile(gs.linspace(0.0, 1.0, k_sampling_points), (dim, 1))
        )
        srv_metric_r3 = SRVMetric(r3)
        d_srv = srv_metric_r3.aux_differential_srv_transform(tangent_vec, curve_a)
        result = srv_metric_r3.aux_differential_srv_transform_inverse(d_srv, curve_a)
        expected = tangent_vec
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)

    def test_aux_differential_srv_transform_vectorization(
        self, dim, k_sampling_points, curve_a, curve_b
    ):
        """Test differential of square root velocity transform.
        Check vectorization.
        """
        dim = 3
        curves = gs.stack((curve_a, curve_b))
        tangent_vecs = gs.random.rand(2, k_sampling_points, dim)
        srv_metric_r3 = SRVMetric(r3)
        result = srv_metric_r3.aux_differential_srv_transform(tangent_vecs, curves)

        res_a = srv_metric_r3.aux_differential_srv_transform(tangent_vecs[0], curve_a)
        res_b = srv_metric_r3.aux_differential_srv_transform(tangent_vecs[1], curve_b)
        expected = gs.stack([res_a, res_b])
        self.assertAllClose(result, expected)

    def test_srv_inner_product_elastic(self, dim, k_sampling_points, curve_a):
        """Test inner product of SRVMetric.
        Check that the pullback metric gives an elastic metric
        with parameters a=1, b=1/2.
        """
        tangent_vec_a = gs.random.rand(k_sampling_points, dim)
        tangent_vec_b = gs.random.rand(k_sampling_points, dim)
        r3 = Euclidean(dim)
        srv_metric_r3 = SRVMetric(r3)
        result = srv_metric_r3.inner_product(tangent_vec_a, tangent_vec_b, curve_a)

        d_vec_a = (k_sampling_points - 1) * (
            tangent_vec_a[1:, :] - tangent_vec_a[:-1, :]
        )
        d_vec_b = (k_sampling_points - 1) * (
            tangent_vec_b[1:, :] - tangent_vec_b[:-1, :]
        )
        velocity_vec = (k_sampling_points - 1) * (curve_a[1:, :] - curve_a[:-1, :])
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
        expected = gs.sum(integrand) / k_sampling_points
        self.assertAllClose(result, expected)

    def test_srv_inner_product_and_dist(self, dim, curve_a, curve_b):
        """Test that norm of log and dist coincide
        for curves with same / different starting points, and for
        the translation invariant / non invariant SRV metric.
        """
        r3 = Euclidean(dim=dim)
        curve_b_transl = curve_b + gs.array([1.0, 0.0, 0.0])
        curve_b = [curve_b, curve_b_transl]
        translation_invariant = [True, False]
        for curve in curve_b:
            for param in translation_invariant:
                srv_metric = SRVMetric(ambient_manifold=r3, translation_invariant=param)
                log = srv_metric.log(point=curve, base_point=curve_a)
                result = srv_metric.norm(vector=log, base_point=curve_a)
                expected = srv_metric.dist(curve_a, curve)
                self.assertAllClose(result, expected)

    def test_srv_inner_product_vectorization(
        self, dim, k_sampling_points, curve_a, curve_b
    ):
        """Test inner product of SRVMetric.
        Check vectorization.
        """
        curves = gs.stack((curve_a, curve_b))
        tangent_vecs_1 = gs.random.rand(2, k_sampling_points, dim)
        tangent_vecs_2 = gs.random.rand(2, k_sampling_points, dim)
        srv_metric_r3 = SRVMetric(r3)
        result = srv_metric_r3.inner_product(tangent_vecs_1, tangent_vecs_2, curves)

        res_a = srv_metric_r3.inner_product(
            tangent_vecs_1[0], tangent_vecs_2[0], curve_a
        )
        res_b = srv_metric_r3.inner_product(
            tangent_vecs_1[1], tangent_vecs_2[1], curve_b
        )
        expected = gs.stack((res_a, res_b))
        self.assertAllClose(result, expected)

    def test_space_derivative(
        self, dim, n_points, n_discretized_curves, k_sampling_points
    ):
        """Test space derivative.
        Check result on an example and vectorization.
        """
        n_points = 3
        dim = 3
        srv_metric_r3 = SRVMetric(Euclidean(dim))
        curve = gs.random.rand(n_points, dim)
        result = srv_metric_r3.space_derivative(curve)
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

        path_of_curves = gs.random.rand(n_discretized_curves, k_sampling_points, dim)
        result = srv_metric_r3.space_derivative(path_of_curves)
        expected = []
        for i in range(n_discretized_curves):
            expected.append(srv_metric_r3.space_derivative(path_of_curves[i]))
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)

    def test_srv_metric_pointwise_inner_products(
        self, times, curve_a, curve_b, curve_c, n_discretized_curves, k_sampling_points
    ):
        l2_metric_s2 = L2CurvesMetric(ambient_manifold=s2)
        srv_metric_r3 = SRVMetric(ambient_manifold=r3)
        curves_ab = l2_metric_s2.geodesic(curve_a, curve_b)
        curves_bc = l2_metric_s2.geodesic(curve_b, curve_c)
        curves_ab = curves_ab(times)
        curves_bc = curves_bc(times)

        tangent_vecs = l2_metric_s2.log(point=curves_bc, base_point=curves_ab)
        result = srv_metric_r3.l2_curves_metric.pointwise_inner_products(
            tangent_vec_a=tangent_vecs, tangent_vec_b=tangent_vecs, base_curve=curves_ab
        )
        expected_shape = (n_discretized_curves, k_sampling_points)
        self.assertAllClose(gs.shape(result), expected_shape)

        result = srv_metric_r3.l2_curves_metric.pointwise_inner_products(
            tangent_vec_a=tangent_vecs[0],
            tangent_vec_b=tangent_vecs[0],
            base_curve=curves_ab[0],
        )
        expected_shape = (k_sampling_points,)
        self.assertAllClose(gs.shape(result), expected_shape)

    def test_srv_transform_and_inverse(self, times, curve_a, curve_b):
        """Test of SRVT and its inverse.
        N.B: Here curves_ab are seen as curves in R3 and not S2.
        """
        l2_metric_s2 = L2CurvesMetric(ambient_manifold=s2)
        srv_metric_r3 = SRVMetric(ambient_manifold=r3)
        curves_ab = l2_metric_s2.geodesic(curve_a, curve_b)
        curves_ab = curves_ab(times)

        curves = curves_ab
        srv_curves = srv_metric_r3.srv_transform(curves)
        starting_points = curves[:, 0, :]
        result = srv_metric_r3.srv_transform_inverse(srv_curves, starting_points)
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
    skip_test_squared_dist_is_positive = tf_backend()
    skip_test_dist_is_positive = tf_backend()
    skip_test_dist_is_norm_of_log = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_triangle_inequality_of_dist = True

    testing_data = ElasticMetricTestData()

    def test_cartesian_to_polar_and_polar_to_cartesian(self, a, b, rtol, atol):
        """Test conversion to polar coordinate"""
        curves_space = DiscreteCurves(ambient_manifold=r2)
        el_metric = ElasticMetric(a=a, b=b)
        curve = curves_space.random_point()
        polar_curve = el_metric.cartesian_to_polar(curve)
        result = el_metric.polar_to_cartesian(polar_curve)

        self.assertAllClose(result, curve, rtol, atol)

    def test_cartesian_to_polar_and_polar_to_cartesian_vectorization(
        self, a, b, rtol, atol
    ):
        """Test conversion to polar coordinate"""
        curves_space = DiscreteCurves(ambient_manifold=r2)
        el_metric = ElasticMetric(a=a, b=b)
        curve = curves_space.random_point(n_samples=3)
        polar_curve = el_metric.cartesian_to_polar(curve)
        result = el_metric.polar_to_cartesian(polar_curve)

        self.assertAllClose(result, curve, rtol=rtol, atol=atol)

    def test_f_transform_and_srv_transform(self, curve, rtol, atol):
        """Test that the f transform coincides with the SRVF

        This is valid for a f transform with a=1, b=1/2.
        """
        curves_space = DiscreteCurves(ambient_manifold=r2)
        el_metric = ElasticMetric(a=1, b=0.5)

        result = el_metric.f_transform(curve)
        expected = curves_space.srv_metric.srv_transform(curve)
        self.assertAllClose(result, expected, rtol, atol)

    def test_f_transform_inverse_and_srv_transform_inverse(self, curve, rtol, atol):
        """Test that the f transform coincides with the SRVF

        This is valid for a f transform with a=1, b=1/2.
        """
        curves_space = DiscreteCurves(ambient_manifold=r2)

        el_metric = ElasticMetric(a=1, b=0.5)
        starting_point = curve[0]
        fake_transformed_curve = curve[1:, :]

        result = el_metric.f_transform_inverse(fake_transformed_curve, starting_point)
        expected = curves_space.srv_metric.srv_transform_inverse(
            fake_transformed_curve, starting_point
        )
        self.assertAllClose(result, expected, rtol, atol)

    def test_f_transform_and_srv_transform_vectorization(self, rtol, atol):
        """Test that the f transform coincides with the SRVF.

        This is valid for a f_transform with a=1, b=1/2.
        """
        curves_space = DiscreteCurves(ambient_manifold=r2)
        el_metric = ElasticMetric(a=1, b=0.5)

        curves = curves_space.random_point(n_samples=2)

        result = el_metric.f_transform(curves)
        expected = curves_space.srv_metric.srv_transform(curves)
        self.assertAllClose(result, expected, rtol, atol)

    def test_f_transform_and_inverse(self, a, b, rtol, atol):
        """Test that the inverse is right."""
        curves_space = DiscreteCurves(ambient_manifold=r2)
        el_metric = ElasticMetric(a=a, b=b)
        curve = curves_space.random_point()

        f = el_metric.f_transform(curve)
        f_inverse = el_metric.f_transform_inverse(f, curve[0])

        result = f.shape
        expected = (curve.shape[0] - 1, 2)
        self.assertAllClose(result, expected)

        result = f_inverse
        expected = curve
        self.assertAllClose(result, expected, rtol, atol)


class TestSRVShapeBundle(TestCase, metaclass=Parametrizer):
    testing_data = SRVShapeBundleTestData()

    @geomstats.tests.np_autograd_and_torch_only
    def test_horizontal_and_vertical_projections(
        self, times, n_discretized_curves, curve_a, curve_b
    ):
        """Test horizontal and vertical projections.
        Check that horizontal and vertical parts of any tangent
        vector are othogonal with respect to the SRVMetric inner
        product, and check vectorization.
        """
        srv_metric_r3 = SRVMetric(r3)
        srv_shape_bundle_r3 = SRVShapeBundle(r3)
        geod = srv_metric_r3.geodesic(initial_curve=curve_a, end_curve=curve_b)
        geod = geod(times)
        tangent_vec = n_discretized_curves * (geod[1, :, :] - geod[0, :, :])
        tangent_vec_hor = srv_shape_bundle_r3.horizontal_projection(
            tangent_vec, curve_a
        )
        tangent_vec_ver = srv_shape_bundle_r3.vertical_projection(tangent_vec, curve_a)
        print(tangent_vec_hor.shape)
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

    @geomstats.tests.np_autograd_and_torch_only
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
        srv_shape_bundle_r3 = SRVShapeBundle(r3)
        horizontal_geod_fun = srv_shape_bundle_r3.horizontal_geodesic(curve_a, curve_b)
        times = gs.linspace(0.0, 1.0, n_times)
        horizontal_geod = horizontal_geod_fun(times)
        velocity_vec = n_times * (horizontal_geod[1:] - horizontal_geod[:-1])
        _, vertical_norms = srv_shape_bundle_r3.vertical_projection(
            velocity_vec, horizontal_geod[:-1], return_norm=True
        )
        result = gs.sum(vertical_norms**2, axis=1) ** (1 / 2)
        expected = gs.zeros(n_times - 1)
        self.assertAllClose(result, expected, atol=1e-3)


class TestQuotientSRVMetric(TestCase, metaclass=Parametrizer):
    testing_data = QuotientSRVMetricTestData()

    @geomstats.tests.np_autograd_and_torch_only
    def test_dist(self, sampling_times, curve_fun_a, curve_a, k_sampling_points):
        """Test quotient distance.
        Check that the quotient distance is the same as the distance
        between the end points of the horizontal geodesic.
        """
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
        quotient_srv_metric_r3 = QuotientSRVMetric(ambient_manifold=r3)
        result = quotient_srv_metric_r3.dist(curve_a_resampled, curve_b)
        expected = quotient_srv_metric_r3.dist(curve_a, curve_b)
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)
