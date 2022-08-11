import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from tests.data_generation import ConnectionTestData


def _euc_metric_matrix(base_point):
    """Return matrix of Euclidean inner-product."""
    dim = base_point.shape[-1]
    return gs.eye(dim)


def _sphere_metric_matrix(base_point):
    """Return sphere's metric in spherical coordinates."""
    theta = base_point[..., 0]
    mat = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** 2]])
    return mat


class RiemannianMetricTestData(ConnectionTestData):

    dim = 2
    euc = Euclidean(dim=dim)
    sphere = Hypersphere(dim=dim)
    euc_metric = EuclideanMetric(dim=dim)
    sphere_metric = HypersphereMetric(dim=dim)

    new_euc_metric = RiemannianMetric(dim=dim)
    new_euc_metric.metric_matrix = _euc_metric_matrix

    new_sphere_metric = RiemannianMetric(dim=dim)
    new_sphere_metric.metric_matrix = _sphere_metric_matrix

    new_euc_metric = new_euc_metric
    new_sphere_metric = new_sphere_metric

    def cometric_matrix_test_data(self):
        random_data = [
            dict(
                metric=self.euc_metric,
                base_point=self.euc.random_point(),
                expected=gs.eye(self.dim),
            )
        ]
        return self.generate_tests(random_data)

    def inner_coproduct_test_data(self):
        base_point = gs.array([0.0, 0.0, 1.0])
        cotangent_vec_a = self.sphere.to_tangent(gs.array([1.0, 2.0, 0.0]), base_point)
        cotangent_vec_b = self.sphere.to_tangent(gs.array([1.0, 3.0, 0.0]), base_point)

        smoke_data = [
            dict(
                metric=self.euc_metric,
                cotangent_vec_a=gs.array([1.0, 2.0]),
                cotangent_vec_b=gs.array([1.0, 2.0]),
                base_point=self.euc.random_point(),
                expected=5.0,
            ),
            dict(
                metric=self.sphere_metric,
                cotangent_vec_a=cotangent_vec_a,
                cotangent_vec_b=cotangent_vec_b,
                base_point=base_point,
                expected=7.0,
            ),
        ]
        return self.generate_tests(smoke_data)

    def hamiltonian_test_data(self):

        smoke_data = [
            dict(
                metric=self.euc_metric,
                state=(gs.array([1.0, 2.0]), gs.array([1.0, 2.0])),
                expected=2.5,
            )
        ]
        smoke_data += [
            dict(
                metric=self.sphere_metric,
                state=(gs.array([0.0, 0.0, 1.0]), gs.array([1.0, 2.0, 1.0])),
                expected=3.0,
            )
        ]
        return self.generate_tests(smoke_data)

    def inner_product_derivative_matrix_test_data(self):
        base_point = self.euc.random_point()
        random_data = [
            dict(
                metric=self.new_euc_metric,
                base_point=base_point,
                expected=gs.zeros((self.dim,) * 3),
            )
        ]
        random_data += [
            dict(
                metric=self.euc_metric,
                base_point=base_point,
                expected=gs.zeros((self.dim,) * 3),
            )
        ]
        return self.generate_tests([], random_data)

    def inner_product_test_data(self):
        base_point = self.euc.random_point()
        tangent_vec_a = self.euc.random_point()
        tangent_vec_b = self.euc.random_point()
        random_data = [
            dict(
                metric=self.euc_metric,
                tangent_vec_a=tangent_vec_a,
                tangent_vec_b=tangent_vec_b,
                base_point=base_point,
                expected=gs.dot(tangent_vec_a, tangent_vec_b),
            )
        ]

        smoke_data = [
            dict(
                metric=self.new_sphere_metric,
                tangent_vec_a=gs.array([0.3, 0.4]),
                tangent_vec_b=gs.array([0.1, -0.5]),
                base_point=gs.array([gs.pi / 3.0, gs.pi / 5.0]),
                expected=-0.12,
            )
        ]
        return self.generate_tests(smoke_data, random_data)

    def normalize_test_data(self):
        n_points = 10
        single_point = self.euc.random_point()
        single_vector = self.euc.random_point()
        multiple_points = self.euc.random_point(n_points)
        multiple_vectors = self.euc.random_point(n_points)
        random_data = [
            dict(
                metric=self.euc_metric,
                tangent_vec=single_vector,
                point=single_point,
                expected=1,
                atol=1e-5,
            ),
            dict(
                metric=self.euc_metric,
                tangent_vec=multiple_vectors,
                point=single_point,
                expected=gs.ones(n_points),
                atol=1e-5,
            ),
            dict(
                metric=self.euc_metric,
                tangent_vec=multiple_vectors,
                point=multiple_points,
                expected=gs.ones(n_points),
                atol=1e-5,
            ),
        ]
        return self.generate_tests([], random_data)

    def random_unit_tangent_vec_test_data(self):
        single_point = self.euc.random_point()
        n_points = 10
        multiple_points = self.euc.random_point(n_points)
        n_vectors = 4
        random_data = [
            dict(
                metric=self.euc_metric,
                point=single_point,
                n_vectors=1,
                expected=1,
                atol=1e-5,
            ),
            dict(
                metric=self.euc_metric,
                point=multiple_points,
                n_vectors=1,
                expected=gs.ones(n_points),
                atol=1e-5,
            ),
            dict(
                metric=self.euc_metric,
                point=single_point,
                n_vectors=n_vectors,
                expected=gs.ones(n_vectors),
                atol=1e-5,
            ),
        ]
        return self.generate_tests([], random_data)

    def christoffels_test_data(self):
        base_point = gs.array([gs.pi / 10.0, gs.pi / 9.0])
        gs.array([gs.pi / 10.0, gs.pi / 9.0])
        smoke_data = []
        random_data = []
        smoke_data = [
            dict(
                metric=self.new_sphere_metric,
                base_point=gs.array([gs.pi / 10.0, gs.pi / 9.0]),
                expected=self.sphere_metric.christoffels(base_point),
            )
        ]
        random_data += [
            dict(
                metric=self.new_euc_metric,
                base_point=self.euc.random_point(),
                expected=gs.zeros((self.dim,) * 3),
            )
        ]
        random_data += [
            dict(
                metric=self.euc_metric,
                base_point=self.euc.random_point(),
                expected=gs.zeros((self.dim,) * 3),
            )
        ]

        return self.generate_tests(smoke_data, random_data)

    def exp_test_data(self):
        base_point = gs.array([gs.pi / 10.0, gs.pi / 9.0])
        tangent_vec = gs.array([gs.pi / 2.0, 0.0])
        expected = gs.array([gs.pi / 10.0 + gs.pi / 2.0, gs.pi / 9.0])

        euc_base_point = self.euc.random_point()
        euc_tangent_vec = self.euc.random_point()
        euc_expected = euc_base_point + euc_tangent_vec

        smoke_data = [
            dict(
                metric=self.new_sphere_metric,
                tangent_vec=tangent_vec,
                base_point=base_point,
                expected=expected,
            )
        ]
        random_data = [
            dict(
                metric=self.new_euc_metric,
                tangent_vec=euc_tangent_vec,
                base_point=euc_base_point,
                expected=euc_expected,
            )
        ]
        return self.generate_tests(smoke_data, random_data)

    def log_test_data(self):
        base_point = self.euc.random_point()
        point = self.euc.random_point()
        expected = point - base_point
        random_data = [
            dict(
                metric=self.new_euc_metric,
                point=point,
                base_point=base_point,
                expected=expected,
            )
        ]
        return self.generate_tests([], random_data)
