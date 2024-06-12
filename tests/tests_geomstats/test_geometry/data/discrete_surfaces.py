import pytest

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats._mesh import Surface
from geomstats.test.data import TestData
from geomstats.vectorization import repeat_point

from .manifold import ManifoldTestData


class SurfaceTestData(TestData):
    def face_areas_test_data(self):
        vertices = gs.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0)])
        faces = gs.array([[0, 1, 2]])

        data = [dict(point=Surface(vertices, faces), expected=gs.array([[1.0]]))]

        return self.generate_tests(data)

    def face_normals_test_data(self):
        vertices = gs.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0)])
        faces = gs.array([[0, 1, 2]])

        data = [
            dict(point=Surface(vertices, faces), expected=gs.array([[0.0, 0.0, 1.0]]))
        ]

        return self.generate_tests(data)

    def face_centroids_test_data(self):
        vertices = gs.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0)])
        faces = gs.array([[0, 1, 2]])

        data = [
            dict(
                point=Surface(vertices, faces),
                expected=gs.array([[2.0 / 3.0, 2.0 / 3.0, 0.0]]),
            )
        ]

        return self.generate_tests(data)


class DiscreteSurfacesTestData(ManifoldTestData):
    skips = (
        "random_tangent_vec_shape",
        "to_tangent_vec",
    )


class DiscreteSurfacesSmokeTestData(TestData):
    vertices, _ = data_utils.load_cube()
    vertices = gs.array(vertices, dtype=gs.float64)

    def vertex_areas_test_data(self):
        number_of_contact_faces = gs.array([3, 5, 5, 5, 5, 5, 3, 5])
        triangle_area = 0.5 * 2 * 2
        expected = 2 * (number_of_contact_faces * triangle_area) / 3

        data = [
            dict(point=self.vertices, expected=expected),
            dict(
                point=repeat_point(
                    self.vertices,
                ),
                expected=repeat_point(expected),
            ),
        ]

        return self.generate_tests(data)

    def normals_test_data(self):
        expected = cube_normals = gs.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, -2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, 0.0, -2.0],
                [0.0, 0.0, -2.0],
            ]
        )
        expected = cube_normals
        data = [
            dict(point=self.vertices, expected=expected),
            dict(
                point=repeat_point(
                    self.vertices,
                ),
                expected=repeat_point(expected),
            ),
        ]
        return self.generate_tests(data)

    def surface_one_forms_prod_test_data(self):
        expected = gs.array([0.0, 4.0])
        data = [
            dict(point=self.vertices, expected=expected),
            dict(
                point=repeat_point(
                    self.vertices,
                ),
                expected=repeat_point(expected),
            ),
        ]
        return self.generate_tests(data)

    def faces_area_test_data(self):
        expected = gs.array([4.0] * 12)
        data = [
            dict(point=self.vertices, expected=expected),
            dict(
                point=repeat_point(
                    self.vertices,
                ),
                expected=repeat_point(expected),
            ),
        ]
        return self.generate_tests(data)


class ElasticMetricTestData(TestData):
    N_RANDOM_POINTS = [1]

    tolerances = {"exp_after_log": {"atol": 1e-1}}

    def exp_after_log_test_data(self):
        return self.generate_random_data(marks=(pytest.mark.slow, pytest.mark.xfail))

    def inner_product_vec_test_data(self):
        return self.generate_vec_data()

    def inner_product_is_symmetric_test_data(self):
        return self.generate_random_data()


class QuotientElasticMetricTestData(TestData):
    N_RANDOM_POINTS = [1]
    trials = 1

    def log_runs_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_runs_test_data(self):
        return self.generate_random_data()
