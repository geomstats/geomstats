import random

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces, ElasticMetric
from tests.data_generation import _ManifoldTestData, _RiemannianMetricTestData

vertices, faces = data_utils.load_cube()
vertices = gs.array(vertices, dtype=gs.float64)
faces = gs.array(faces)


class DiscreteSurfacesTestData(_ManifoldTestData):
    space_args_list = [(faces,)]
    shape_list = [(8, 3)]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = DiscreteSurfaces

    def vertex_areas_test_data(self):
        smoke_data = [dict(faces=faces, point=vertices)]

        return self.generate_tests(smoke_data)

    def normals_test_data(self):
        smoke_data = [dict(faces=faces, point=vertices)]

        return self.generate_tests(smoke_data)

    def surface_one_forms_test_data(self):
        smoke_data = [dict(faces=faces, point=vertices)]

        return self.generate_tests(smoke_data)

    def faces_area_test_data(self):
        smoke_data = [dict(faces=faces, point=vertices)]

        return self.generate_tests(smoke_data)

    def surface_metric_matrices_test_data(self):
        smoke_data = [dict(faces=faces, point=vertices)]

        return self.generate_tests(smoke_data)

    def laplacian_test_data(self):
        smoke_data = [
            dict(
                faces=faces,
                point=vertices,
                tangent_vec=tangent_vec,
                expected=gs.zeros_like(vertices),
            )
            for tangent_vec in [gs.zeros_like(vertices), gs.ones_like(vertices)]
        ]

        return self.generate_tests(smoke_data)


class ElasticMetricTestData(_RiemannianMetricTestData):

    n_samples_list = [1]
    a0_list = [1, 5]
    a1_list = [1, 5]
    b1_list = [1, 5]
    c1_list = [1, 5]
    d1_list = [1, 5]
    a2_list = [1, 5]

    connection_args_list = metric_args_list = [
        {
            "a0": a0,
            "a1": a1,
            "b1": b1,
            "c1": c1,
            "d1": d1,
            "a2": a2,
        }
        for a0, a1, b1, c1, d1, a2 in zip(
            a0_list, a1_list, b1_list, c1_list, d1_list, a2_list
        )
    ]

    shape_list = [(8, 3)]
    space_list = [DiscreteSurfaces(faces)]
    n_points_list = [1]
    n_tangent_vecs_list = [1]
    n_points_a_list = [1]
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = ElasticMetric

    def path_energy_per_time_is_positive_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteSurfaces(faces=faces),
                a0=1,
                a1=1,
                b1=1,
                c1=1,
                d1=1,
                a2=1,
                path=gs.array([vertices, vertices, vertices]),
                atol=1e-6,
            )
        ]
        return self.generate_tests(smoke_data, [])

    def path_energy_is_positive_test_data(self):
        smoke_data = [
            dict(
                space=DiscreteSurfaces(faces=faces),
                a0=1,
                a1=1,
                b1=1,
                c1=1,
                d1=1,
                a2=1,
                path=gs.array([vertices, vertices, vertices]),
                atol=1e-6,
            )
        ]
        return self.generate_tests(smoke_data, [])
