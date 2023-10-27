import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.test.data import TestData
from geomstats.vectorization import repeat_point

from .manifold import ManifoldTestData
from .riemannian_metric import RiemannianMetricTestData


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


class ElasticMetricTestData(RiemannianMetricTestData):
    pass


class ElasticMetricSmokeTestData(TestData):
    vertices, _ = data_utils.load_cube()
    vertices = gs.array(vertices, dtype=gs.float64)

    def path_energy_is_positive_test_data(self):
        data = [
            dict(path=gs.array([self.vertices, self.vertices, self.vertices])),
        ]
        return self.generate_tests(data)

    def path_energy_per_time_is_positive_test_data(self):
        data = [
            dict(path=gs.array([self.vertices, self.vertices, self.vertices])),
        ]
        return self.generate_tests(data)
