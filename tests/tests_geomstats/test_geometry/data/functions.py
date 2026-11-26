from polpo.testing.data import CaseData

import geomstats.backend as gs


def gaussian(x, mu, sig):
    a = (x - mu) ** 2 / (2 * (sig**2))
    b = 1 / (sig * (gs.sqrt(2 * gs.pi)))
    f = b * gs.exp(-a)
    l2_norm = gs.sqrt(gs.trapezoid(f**2, x))
    f_sinf = f / l2_norm

    return gs.expand_dims(f_sinf, axis=0)


class HilbertSphereSmokeTestData(CaseData):
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

        return data
