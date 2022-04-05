"""Unit tests for parameterized manifolds."""

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
import geomstats.tests
from geomstats.geometry.discrete_curves import (
    ClosedDiscreteCurves,
    DiscreteCurves,
    ElasticMetric,
    L2CurvesMetric,
    SRVMetric,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere


class TestDiscreteCurves(geomstats.tests.TestCase):
    def setup_method(self):
        s2 = Hypersphere(dim=2)
        r2 = Euclidean(dim=2)
        r3 = s2.embedding_space

        initial_point = [0.0, 0.0, 1.0]
        initial_tangent_vec_a = [1.0, 0.0, 0.0]
        initial_tangent_vec_b = [0.0, 1.0, 0.0]
        initial_tangent_vec_c = [-1.0, 0.0, 0.0]

        curve_fun_a = s2.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec_a
        )
        curve_fun_b = s2.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec_b
        )
        curve_fun_c = s2.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec_c
        )
        self.curve_fun_a = curve_fun_a

        self.n_sampling_points = 10
        self.sampling_times = gs.linspace(0.0, 1.0, self.n_sampling_points)
        self.curve_a = curve_fun_a(self.sampling_times)
        self.curve_b = curve_fun_b(self.sampling_times)
        self.curve_c = curve_fun_c(self.sampling_times)

        self.space_curves_in_euclidean_3d = DiscreteCurves(ambient_manifold=r3)
        self.space_curves_in_sphere_2d = DiscreteCurves(ambient_manifold=s2)
        self.space_closed_curves_in_euclidean_2d = ClosedDiscreteCurves(
            ambient_manifold=r2
        )

        self.l2_metric_s2 = L2CurvesMetric(ambient_manifold=s2)
        self.l2_metric_r3 = L2CurvesMetric(ambient_manifold=r3)
        self.srv_metric_r3 = (
            self.space_curves_in_euclidean_3d.square_root_velocity_metric
        )
        self.quotient_srv_metric_r3 = (
            self.space_curves_in_euclidean_3d.quotient_square_root_velocity_metric
        )
        self.a = 1
        self.b = 1
        self.elastic_metric = ElasticMetric(self.a, self.b)

        self.n_discretized_curves = 5
        self.times = gs.linspace(0.0, 1.0, self.n_discretized_curves)
        gs.random.seed(1234)

    def test_belongs(self):
        result = self.space_curves_in_sphere_2d.belongs(self.curve_a)
        self.assertTrue(result)

        curve_ab = [self.curve_a[:-1], self.curve_b]
        result = self.space_curves_in_sphere_2d.belongs(curve_ab)
        self.assertTrue(gs.all(result))

        curve_ab = gs.array([self.curve_a, self.curve_b])
        result = self.space_curves_in_sphere_2d.belongs(curve_ab)
        self.assertTrue(gs.all(result))

    @geomstats.tests.np_autograd_and_torch_only
    def test_l2_metric_log_and_squared_norm_and_dist(self):
        """Test that squared norm of logarithm is squared dist."""
        tangent_vec = self.l2_metric_s2.log(point=self.curve_b, base_point=self.curve_a)
        log_ab = tangent_vec
        result = self.l2_metric_s2.squared_norm(vector=log_ab, base_point=self.curve_a)
        expected = self.l2_metric_s2.dist(self.curve_a, self.curve_b) ** 2

        self.assertAllClose(result, expected)

    def test_l2_metric_log_and_exp(self):
        """Test that exp and log are inverse maps."""
        tangent_vec = self.l2_metric_s2.log(point=self.curve_b, base_point=self.curve_a)
        result = self.l2_metric_s2.exp(tangent_vec=tangent_vec, base_point=self.curve_a)
        expected = self.curve_b

        self.assertAllClose(result, expected)

    def test_l2_metric_inner_product_vectorization(self):
        """Test the vectorization inner_product."""
        n_samples = self.n_discretized_curves
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        tangent_vecs = self.l2_metric_s2.log(point=curves_bc, base_point=curves_ab)

        result = self.l2_metric_s2.inner_product(tangent_vecs, tangent_vecs, curves_ab)

        self.assertAllClose(gs.shape(result), (n_samples,))

    def test_l2_metric_dist_vectorization(self):
        """Test the vectorization of dist."""
        n_samples = self.n_discretized_curves
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        result = self.l2_metric_s2.dist(curves_ab, curves_bc)
        self.assertAllClose(gs.shape(result), (n_samples,))

    def test_l2_metric_exp_vectorization(self):
        """Test the vectorization of exp."""
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        tangent_vecs = self.l2_metric_s2.log(point=curves_bc, base_point=curves_ab)

        result = self.l2_metric_s2.exp(tangent_vec=tangent_vecs, base_point=curves_ab)
        self.assertAllClose(gs.shape(result), gs.shape(curves_ab))

    def test_l2_metric_log_vectorization(self):
        """Test the vectorization of log."""
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        tangent_vecs = self.l2_metric_s2.log(point=curves_bc, base_point=curves_ab)

        result = tangent_vecs
        self.assertAllClose(gs.shape(result), gs.shape(curves_ab))

    def test_l2_metric_geodesic(self):
        """Test the geodesic method of L2Metric."""
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_ab = curves_ab(self.times)

        result = curves_ab
        expected = []
        for k in range(self.n_sampling_points):
            geod = self.l2_metric_s2.ambient_metric.geodesic(
                initial_point=self.curve_a[k, :], end_point=self.curve_b[k, :]
            )
            expected.append(geod(self.times))
        expected = gs.stack(expected, axis=1)
        self.assertAllClose(result, expected)

    def test_srv_metric_pointwise_inner_products(self):
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        tangent_vecs = self.l2_metric_s2.log(point=curves_bc, base_point=curves_ab)
        result = self.srv_metric_r3.l2_metric.pointwise_inner_products(
            tangent_vec_a=tangent_vecs, tangent_vec_b=tangent_vecs, base_curve=curves_ab
        )
        expected_shape = (self.n_discretized_curves, self.n_sampling_points)
        self.assertAllClose(gs.shape(result), expected_shape)

        result = self.srv_metric_r3.l2_metric.pointwise_inner_products(
            tangent_vec_a=tangent_vecs[0],
            tangent_vec_b=tangent_vecs[0],
            base_curve=curves_ab[0],
        )
        expected_shape = (self.n_sampling_points,)
        self.assertAllClose(gs.shape(result), expected_shape)

    def test_srv_transform_and_inverse(self):
        """Test of SRVT and its inverse.

        N.B: Here curves_ab are seen as curves in R3 and not S2.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_ab = curves_ab(self.times)

        curves = curves_ab
        srv_curves = self.srv_metric_r3.srv_transform(curves)
        starting_points = curves[:, 0, :]
        result = self.srv_metric_r3.srv_transform_inverse(srv_curves, starting_points)
        expected = curves

        self.assertAllClose(result, expected)

    def test_srv_metric_exp_and_log(self):
        """Test that exp and log are inverse maps and vectorized.

        N.B: Here curves_ab and curves_bc are seen as curves in R3 and not S2.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        log = self.srv_metric_r3.log(point=curves_bc, base_point=curves_ab)
        result = self.srv_metric_r3.exp(tangent_vec=log, base_point=curves_ab)
        expected = curves_bc

        self.assertAllClose(gs.squeeze(result), expected)

    def test_srv_metric_geodesic(self):
        """Test that the geodesic between two curves in a Euclidean space.

        for the srv metric is the L2 geodesic betweeen the curves srvs.
        N.B: Here curve_a and curve_b are seen as curves in R3 and not S2.
        """
        geod = self.srv_metric_r3.geodesic(
            initial_curve=self.curve_a, end_curve=self.curve_b
        )
        result = geod(self.times)

        srv_a = self.srv_metric_r3.srv_transform(self.curve_a)
        srv_b = self.srv_metric_r3.srv_transform(self.curve_b)
        geod_srv = self.l2_metric_r3.geodesic(initial_point=srv_a, end_point=srv_b)
        geod_srv = geod_srv(self.times)

        starting_points = self.srv_metric_r3.ambient_metric.geodesic(
            initial_point=self.curve_a[0, :], end_point=self.curve_b[0, :]
        )
        starting_points = starting_points(self.times)

        expected = self.srv_metric_r3.srv_transform_inverse(geod_srv, starting_points)

        self.assertAllClose(result, expected)

    def test_srv_metric_dist_and_geod(self):
        """Test that the length of the geodesic gives the distance.

        N.B: Here curve_a and curve_b are seen as curves in R3 and not S2.
        """
        geod = self.srv_metric_r3.geodesic(
            initial_curve=self.curve_a, end_curve=self.curve_b
        )
        geod = geod(self.times)
        srv = self.srv_metric_r3.srv_transform(geod)
        srv_derivative = self.n_discretized_curves * (srv[1:, :] - srv[:-1, :])
        norms = self.srv_metric_r3.l2_metric.norm(srv_derivative)
        result = gs.sum(norms, 0) / self.n_discretized_curves

        expected = self.srv_metric_r3.dist(self.curve_a, self.curve_b)
        self.assertAllClose(result, expected)

    def test_random_and_belongs(self):
        random = self.space_curves_in_sphere_2d.random_point()
        result = self.space_curves_in_sphere_2d.belongs(random)
        self.assertTrue(result)
        self.assertAllClose(random.shape, (10, 3))

        random = self.space_curves_in_sphere_2d.random_point(2)
        result = self.space_curves_in_sphere_2d.belongs(random)
        self.assertTrue(gs.all(result))

    def test_is_tangent_to_tangent(self):
        point = self.space_curves_in_sphere_2d.random_point()
        vector = self.space_curves_in_sphere_2d.random_point()
        tangent_vec = self.space_curves_in_sphere_2d.to_tangent(vector, point)
        result = self.space_curves_in_sphere_2d.is_tangent(tangent_vec, point)
        self.assertTrue(result)

        point = self.space_curves_in_sphere_2d.random_point(2)
        vector = self.space_curves_in_sphere_2d.random_point(2)
        tangent_vec = self.space_curves_in_sphere_2d.to_tangent(vector, point)
        result = self.space_curves_in_sphere_2d.is_tangent(tangent_vec, point)
        self.assertTrue(gs.all(result))

    @geomstats.tests.np_and_autograd_only
    def test_projection_closed_curves(self):
        """Test that projecting the projection returns projection.

        Also test that the projection is a closed curve.
        """
        planar_closed_curves = self.space_closed_curves_in_euclidean_2d

        cells, _, _ = data_utils.load_cells()
        curves = [cell[:-10] for cell in cells[:5]]

        for curve in curves:
            proj = planar_closed_curves.project(curve)
            expected = proj
            result = planar_closed_curves.project(proj)
            self.assertAllClose(result, expected)

            result = proj[-1, :]
            expected = proj[0, :]
            self.assertAllClose(result, expected, rtol=10 * gs.rtol)

    def test_srv_inner_product(self):
        """Test that srv_inner_product works as expected.

        Also test that the resulting shape is right.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)
        srvs_ab = self.srv_metric_r3.srv_transform(curves_ab)
        srvs_bc = self.srv_metric_r3.srv_transform(curves_bc)

        result = self.srv_metric_r3.l2_metric.inner_product(srvs_ab, srvs_bc)
        products = srvs_ab * srvs_bc
        expected = [gs.sum(product) for product in products]
        expected = gs.array(expected) / (srvs_ab.shape[-2] + 1)
        self.assertAllClose(result, expected)

        result = result.shape
        expected = [srvs_ab.shape[0]]
        self.assertAllClose(result, expected)

    def test_srv_norm(self):
        """Test that srv_norm works as expected.

        Also test that the resulting shape is right.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_ab = curves_ab(self.times)
        srvs_ab = self.srv_metric_r3.srv_transform(curves_ab)

        result = self.srv_metric_r3.l2_metric.norm(srvs_ab)
        products = srvs_ab * srvs_ab
        sums = [gs.sum(product) for product in products]
        squared_norm = gs.array(sums) / (srvs_ab.shape[-2] + 1)
        expected = gs.sqrt(squared_norm)
        self.assertAllClose(result, expected)

        result = result.shape
        expected = [srvs_ab.shape[0]]
        self.assertAllClose(result, expected)

    def test_f_transform(self):
        """Test that the f transform coincides with the SRVF.

        With the parameters: a=1, b=1/2.
        """
        r2 = Euclidean(dim=2)
        elastic_metric = ElasticMetric(a=1.0, b=0.5)
        curves_r2 = DiscreteCurves(ambient_manifold=r2)
        curve_a_projected = gs.stack((self.curve_a[:, 0], self.curve_a[:, 2]), axis=-1)

        result = elastic_metric.f_transform(curve_a_projected)
        expected = gs.squeeze(
            curves_r2.square_root_velocity_metric.srv_transform(curve_a_projected)
        )
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_f_transform_and_inverse(self):
        """Test that the inverse is right."""
        cells, _, _ = data_utils.load_cells()
        curve = cells[0]
        metric = self.elastic_metric
        f = metric.f_transform(curve)
        f_inverse = metric.f_transform_inverse(f, curve[0])

        result = f.shape
        expected = (curve.shape[0] - 1, 2)
        self.assertAllClose(result, expected)

        result = f_inverse
        expected = curve
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_elastic_dist(self):
        """Test shape and positivity."""
        cells, _, _ = data_utils.load_cells()
        curve_1, curve_2 = cells[0][:10], cells[1][:10]
        metric = self.elastic_metric
        dist = metric.dist(curve_1, curve_2)

        result = dist.shape
        expected = ()
        self.assertAllClose(result, expected)

        result = dist > 0
        self.assertTrue(result)

    @geomstats.tests.np_autograd_and_torch_only
    def test_elastic_and_srv_dist(self):
        """Test that SRV dist and elastic dist coincide.

        For a=1 and b=1/2.
        """
        r2 = Euclidean(dim=2)
        elastic_metric = ElasticMetric(a=1.0, b=0.5)
        curves_r2 = DiscreteCurves(ambient_manifold=r2)
        curve_a_projected = gs.stack((self.curve_a[:, 0], self.curve_a[:, 2]), axis=-1)
        curve_b_projected = gs.stack((self.curve_b[:, 0], self.curve_b[:, 2]), axis=-1)
        result = elastic_metric.dist(curve_a_projected, curve_b_projected)
        expected = curves_r2.square_root_velocity_metric.dist(
            curve_a_projected, curve_b_projected
        )
        print(result / expected)
        self.assertAllClose(result, expected)

    def test_cartesian_to_polar_and_inverse(self):
        """Test that going back to cartesian works."""
        cells, _, _ = data_utils.load_cells()
        curve = cells[0]

        metric = self.elastic_metric
        norms, args = metric.cartesian_to_polar(curve)

        result = metric.polar_to_cartesian(norms, args)
        expected = curve
        self.assertAllClose(result, expected, rtol=10000 * gs.rtol)

    @geomstats.tests.np_and_autograd_only
    def test_aux_differential_srv_transform(self):
        """Test differential of square root velocity transform.

        Check that its value at (curve, tangent_vec) coincides
        with the derivative at zero of the square root velocity
        transform of a path of curves starting at curve with
        initial derivative tangent_vec.
        """
        dim = 3
        n_sampling_points = 2000
        sampling_times = gs.linspace(0.0, 1.0, n_sampling_points)
        curve_a = self.curve_fun_a(sampling_times)
        tangent_vec = gs.transpose(
            gs.tile(gs.linspace(1.0, 2.0, n_sampling_points), (dim, 1))
        )
        result = self.srv_metric_r3.aux_differential_srv_transform(tangent_vec, curve_a)

        n_curves = 2000
        times = gs.linspace(0.0, 1.0, n_curves)
        path_of_curves = curve_a + gs.einsum("i,jk->ijk", times, tangent_vec)
        srv_path = self.srv_metric_r3.srv_transform(path_of_curves)
        expected = n_curves * (srv_path[1] - srv_path[0])
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)

    @geomstats.tests.np_and_autograd_only
    def test_aux_differential_srv_transform_inverse(self):
        """Test inverse of differential of square root velocity transform.

        Check that it is the inverse of aux_differential_srv_transform.
        """
        dim = 3
        tangent_vec = gs.transpose(
            gs.tile(gs.linspace(0.0, 1.0, self.n_sampling_points), (dim, 1))
        )
        d_srv = self.srv_metric_r3.aux_differential_srv_transform(
            tangent_vec, self.curve_a
        )
        result = self.srv_metric_r3.aux_differential_srv_transform_inverse(
            d_srv, self.curve_a
        )
        expected = tangent_vec
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)

    def test_aux_differential_srv_transform_vectorization(self):
        """Test differential of square root velocity transform.

        Check vectorization.
        """
        dim = 3
        curves = gs.stack((self.curve_a, self.curve_b))
        tangent_vecs = gs.random.rand(2, self.n_sampling_points, dim)
        result = self.srv_metric_r3.aux_differential_srv_transform(tangent_vecs, curves)

        res_a = self.srv_metric_r3.aux_differential_srv_transform(
            tangent_vecs[0], self.curve_a
        )
        res_b = self.srv_metric_r3.aux_differential_srv_transform(
            tangent_vecs[1], self.curve_b
        )
        expected = gs.stack([res_a, res_b])
        self.assertAllClose(result, expected)

    def test_srv_inner_product_elastic(self):
        """Test inner product of SRVMetric.

        Check that the pullback metric gives an elastic metric
        with parameters a=1, b=1/2.
        """
        tangent_vec_a = gs.random.rand(self.n_sampling_points, 3)
        tangent_vec_b = gs.random.rand(self.n_sampling_points, 3)
        result = self.srv_metric_r3.inner_product(
            tangent_vec_a, tangent_vec_b, self.curve_a
        )

        r3 = Euclidean(3)
        d_vec_a = (self.n_sampling_points - 1) * (
            tangent_vec_a[1:, :] - tangent_vec_a[:-1, :]
        )
        d_vec_b = (self.n_sampling_points - 1) * (
            tangent_vec_b[1:, :] - tangent_vec_b[:-1, :]
        )
        velocity_vec = (self.n_sampling_points - 1) * (
            self.curve_a[1:, :] - self.curve_a[:-1, :]
        )
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
        expected = gs.sum(integrand) / self.n_sampling_points
        self.assertAllClose(result, expected)

    def test_srv_inner_product_and_dist(self):
        """Test that norm of log and dist coincide

        for curves with same / different starting points, and for
        the translation invariant / non invariant SRV metric.
        """
        r3 = Euclidean(dim=3)
        curve_b_transl = self.curve_b + gs.array([1.0, 0.0, 0.0])
        curve_b = [self.curve_b, curve_b_transl]
        translation_invariant = [True, False]
        for curve in curve_b:
            for param in translation_invariant:
                srv_metric = SRVMetric(ambient_manifold=r3, translation_invariant=param)
                log = srv_metric.log(point=curve, base_point=self.curve_a)
                result = srv_metric.norm(vector=log, base_point=self.curve_a)
                expected = srv_metric.dist(self.curve_a, curve)
                self.assertAllClose(result, expected)

    def test_srv_inner_product_vectorization(self):
        """Test inner product of SRVMetric.

        Check vectorization.
        """
        dim = 3
        curves = gs.stack((self.curve_a, self.curve_b))
        tangent_vecs_1 = gs.random.rand(2, self.n_sampling_points, dim)
        tangent_vecs_2 = gs.random.rand(2, self.n_sampling_points, dim)
        result = self.srv_metric_r3.inner_product(
            tangent_vecs_1, tangent_vecs_2, curves
        )

        res_a = self.srv_metric_r3.inner_product(
            tangent_vecs_1[0], tangent_vecs_2[0], self.curve_a
        )
        res_b = self.srv_metric_r3.inner_product(
            tangent_vecs_1[1], tangent_vecs_2[1], self.curve_b
        )
        expected = gs.stack((res_a, res_b))
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_split_horizontal_vertical(self):
        """Test split horizontal vertical.

        Check that horizontal and vertical parts of any tangent
        vector are othogonal with respect to the SRVMetric inner
        product, and check vectorization.
        """
        geod = self.srv_metric_r3.geodesic(
            initial_curve=self.curve_a, end_curve=self.curve_b
        )
        geod = geod(self.times)
        tangent_vec = self.n_discretized_curves * (geod[1, :, :] - geod[0, :, :])
        (
            tangent_vec_hor,
            tangent_vec_ver,
            _,
        ) = self.quotient_srv_metric_r3.split_horizontal_vertical(
            tangent_vec, self.curve_a
        )
        result = self.srv_metric_r3.inner_product(
            tangent_vec_hor, tangent_vec_ver, self.curve_a
        )
        expected = 0.0
        self.assertAllClose(result, expected, atol=1e-4)

        tangent_vecs = self.n_discretized_curves * (geod[1:] - geod[:-1])
        _, _, result = self.quotient_srv_metric_r3.split_horizontal_vertical(
            tangent_vecs, geod[:-1]
        )
        expected = []
        for i in range(self.n_discretized_curves - 1):
            _, _, res = self.quotient_srv_metric_r3.split_horizontal_vertical(
                tangent_vecs[i], geod[i]
            )
            expected.append(res)
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)

    def test_space_derivative(self):
        """Test space derivative.

        Check result on an example and vectorization.
        """
        n_points = 3
        dim = 3
        curve = gs.random.rand(n_points, dim)
        result = self.srv_metric_r3.space_derivative(curve)
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
            self.n_discretized_curves, self.n_sampling_points, dim
        )
        result = self.srv_metric_r3.space_derivative(path_of_curves)
        expected = []
        for i in range(self.n_discretized_curves):
            expected.append(self.srv_metric_r3.space_derivative(path_of_curves[i]))
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_horizontal_geodesic(self):
        """Test horizontal geodesic.

        Check that the time derivative of the geodesic is
        horizontal at all time.
        """
        curve_b = gs.transpose(
            gs.stack(
                (
                    gs.zeros(self.n_sampling_points),
                    gs.zeros(self.n_sampling_points),
                    gs.linspace(1.0, 0.5, self.n_sampling_points),
                )
            )
        )
        horizontal_geod_fun = self.quotient_srv_metric_r3.horizontal_geodesic(
            self.curve_a, curve_b
        )
        n_times = 20
        times = gs.linspace(0.0, 1.0, n_times)
        horizontal_geod = horizontal_geod_fun(times)
        velocity_vec = n_times * (horizontal_geod[1:] - horizontal_geod[:-1])
        _, _, vertical_norms = self.quotient_srv_metric_r3.split_horizontal_vertical(
            velocity_vec, horizontal_geod[:-1]
        )
        result = gs.sum(vertical_norms**2, axis=1) ** (1 / 2)
        expected = gs.zeros(n_times - 1)
        self.assertAllClose(result, expected, atol=1e-3)

    @geomstats.tests.np_autograd_and_torch_only
    def test_quotient_dist(self):
        """Test quotient distance.

        Check that the quotient distance is the same as the distance
        between the end points of the horizontal geodesic.
        """
        curve_a_resampled = self.curve_fun_a(self.sampling_times**2)
        curve_b = gs.transpose(
            gs.stack(
                (
                    gs.zeros(self.n_sampling_points),
                    gs.zeros(self.n_sampling_points),
                    gs.linspace(1.0, 0.5, self.n_sampling_points),
                )
            )
        )
        result = self.quotient_srv_metric_r3.dist(curve_a_resampled, curve_b)
        expected = self.quotient_srv_metric_r3.dist(self.curve_a, curve_b)
        self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)
