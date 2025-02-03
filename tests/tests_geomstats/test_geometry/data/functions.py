import geomstats.backend as gs
from geomstats.test.data import TestData

from .manifold import ManifoldTestData
from .riemannian_metric import RiemannianMetricTestData


def gaussian(x, mu, sig):
    a = (x - mu) ** 2 / (2 * (sig**2))
    b = 1 / (sig * (gs.sqrt(2 * gs.pi)))
    f = b * gs.exp(-a)
    l2_norm = gs.sqrt(gs.trapezoid(f**2, x))
    f_sinf = f / l2_norm

    return gs.expand_dims(f_sinf, axis=0)


class HilbertSphereTestData(ManifoldTestData):
    skips = ("not_belongs",)


class HilbertSphereSmokeTestData(TestData):
    def belongs_test_data(self):
        domain = gs.linspace(0, 1, num=50)
        points = gs.squeeze(
            gs.stack([gaussian(domain, a, 0.1) for a in gs.linspace(0.2, 0.8, 5)])
        )

        data = [
            dict(point=gaussian(domain, 0.2, 0.1), expected=True),
            dict(point=gs.sin(gs.linspace(-gs.pi, gs.pi, 50)), expected=False),
            dict(point=points, expected=gs.ones(points.shape[0], dtype=bool)),
        ]

        return self.generate_tests(data)


class HilbertSphereMetricTestData(RiemannianMetricTestData):
    trials = 5
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    skips = (
        "exp_after_log",
        "exp_belongs",
        "geodesic_boundary_points",
        "geodesic_bvp_belongs",
        "geodesic_bvp_reverse",
        "geodesic_ivp_belongs",
        "log_after_exp",
        "log_is_tangent",
    )
