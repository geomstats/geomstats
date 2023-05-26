import geomstats.backend as gs
from geomstats.geometry.complex_riemannian_metric import ComplexRiemannianMetric
from geomstats.geometry.hermitian import Hermitian, HermitianMetric
from tests.data_generation import _RiemannianMetricTestData

CDTYPE = gs.get_default_cdtype()


def _herm_metric_matrix(base_point):
    """Return matrix of Hermitian inner-product."""
    dim = base_point.shape[-1]
    return gs.eye(dim, dtype=CDTYPE)


class ComplexRiemannianMetricTestData(_RiemannianMetricTestData):
    dim = 2
    herm = Hermitian(dim=dim)

    complex_riem_metric = ComplexRiemannianMetric(herm)
    complex_riem_metric.metric_matrix = _herm_metric_matrix

    metric_args_list = [(2,)]
    connection_args_list = metric_args_list = [{}]
    space_list = [herm]

    n_points_a_list = [2]
    n_points_b_list = [1]
    n_points_list = [2]
    shape_list = [(2,)]
    n_tangent_vecs_list = [2]
    Metric = HermitianMetric

    def cometric_matrix_test_data(self):
        random_data = [
            dict(
                metric=self.herm.metric,
                base_point=self.herm.random_point(),
                expected=gs.eye(self.dim, dtype=CDTYPE),
            )
        ]
        return self.generate_tests(random_data)

    def hamiltonian_test_data(self):
        smoke_data = [
            dict(
                metric=self.herm.metric,
                state=(
                    gs.array([1.0, 2.0], dtype=CDTYPE),
                    gs.array([1.0, 2.0], dtype=CDTYPE),
                ),
                expected=2.5,
            )
        ]
        return self.generate_tests(smoke_data)

    def inner_product_derivative_matrix_test_data(self):
        base_point = self.herm.random_point()
        random_data = [
            dict(
                metric=self.herm.metric,
                base_point=base_point,
                expected=gs.zeros((self.dim,) * 3),
            )
        ]
        return self.generate_tests([], random_data)

    def inner_product_test_data(self):
        base_point = self.herm.random_point()
        tangent_vec_a = self.herm.random_point()
        tangent_vec_b = self.herm.random_point()
        random_data = [
            dict(
                metric=self.herm.metric,
                tangent_vec_a=tangent_vec_a,
                tangent_vec_b=tangent_vec_b,
                base_point=base_point,
                expected=gs.dot(gs.conj(tangent_vec_a), tangent_vec_b),
            )
        ]

        return self.generate_tests(random_data)

    def normalize_test_data(self):
        n_points = 10
        single_point = self.herm.random_point()
        single_vector = self.herm.random_point()
        multiple_points = self.herm.random_point(n_points)
        multiple_vectors = self.herm.random_point(n_points)
        random_data = [
            dict(
                metric=self.herm.metric,
                tangent_vec=single_vector,
                point=single_point,
                expected=1,
                atol=1e-5,
            ),
            dict(
                metric=self.herm.metric,
                tangent_vec=multiple_vectors,
                point=single_point,
                expected=gs.ones(n_points),
                atol=1e-5,
            ),
            dict(
                metric=self.herm.metric,
                tangent_vec=multiple_vectors,
                point=multiple_points,
                expected=gs.ones(n_points),
                atol=1e-5,
            ),
        ]
        return self.generate_tests([], random_data)

    def random_unit_tangent_vec_test_data(self):
        single_point = self.herm.random_point()
        n_points = 10
        multiple_points = self.herm.random_point(n_points)
        n_vectors = 4
        random_data = [
            dict(
                metric=self.herm.metric,
                point=single_point,
                n_vectors=1,
                expected=1,
                atol=1e-5,
            ),
            dict(
                metric=self.herm.metric,
                point=multiple_points,
                n_vectors=1,
                expected=gs.ones(n_points),
                atol=1e-5,
            ),
            dict(
                metric=self.herm.metric,
                point=single_point,
                n_vectors=n_vectors,
                expected=gs.ones(n_vectors),
                atol=1e-5,
            ),
        ]
        return self.generate_tests([], random_data)

    def christoffels_test_data(self):
        random_data = []

        random_data += [
            dict(
                metric=self.complex_riem_metric,
                base_point=self.herm.random_point(),
                expected=gs.zeros((self.dim,) * 3),
            )
        ]
        random_data += [
            dict(
                metric=self.herm.metric,
                base_point=self.herm.random_point(),
                expected=gs.zeros((self.dim,) * 3),
            )
        ]

        return self.generate_tests(random_data)

    def exp_test_data(self):
        herm_base_point = self.herm.random_point()
        herm_tangent_vec = self.herm.random_point()
        herm_expected = herm_base_point + herm_tangent_vec

        random_data = [
            dict(
                metric=self.complex_riem_metric,
                tangent_vec=herm_tangent_vec,
                base_point=herm_base_point,
                expected=herm_expected,
            )
        ]
        return self.generate_tests(random_data)

    def log_test_data(self):
        base_point = self.herm.random_point()
        point = self.herm.random_point()
        expected = point - base_point
        random_data = [
            dict(
                metric=self.complex_riem_metric,
                point=point,
                base_point=base_point,
                expected=expected,
            )
        ]
        return self.generate_tests([], random_data)
