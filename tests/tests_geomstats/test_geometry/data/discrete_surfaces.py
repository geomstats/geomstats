import pytest
from polpo.testing.data import CaseData
from polpo.testing.geometric import GeometricCaseData

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats._mesh import Surface
from geomstats.vectorization import repeat_point

from .manifold import ManifoldTestData


class SurfaceTestData(CaseData):
    def face_areas_test_data(self):
        vertices = gs.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0)])
        faces = gs.array([[0, 1, 2]])

        data = [dict(point=Surface(vertices, faces), expected=gs.array([[1.0]]))]

        return data

    def face_normals_test_data(self):
        vertices = gs.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0)])
        faces = gs.array([[0, 1, 2]])

        data = [
            dict(point=Surface(vertices, faces), expected=gs.array([[0.0, 0.0, 1.0]]))
        ]

        return data

    def face_centroids_test_data(self):
        vertices = gs.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0)])
        faces = gs.array([[0, 1, 2]])

        data = [
            dict(
                point=Surface(vertices, faces),
                expected=gs.array([[2.0 / 3.0, 2.0 / 3.0, 0.0]]),
            )
        ]

        return data


class DiscreteCubesTestData(ManifoldTestData):
    """Discrete surfaces data.

    Smoke data is computed on the cube with sides of length 2 centered at the
    origin, meshed with 12 triangles (2 triangles per face).
    """

    vertices, _ = data_utils.load_cube()
    vertices = gs.array(vertices, dtype=gs.float64)

    @pytest.mark.smoke
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

        return data

    @pytest.mark.smoke
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
        return data

    @pytest.mark.smoke
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
        return data

    @pytest.mark.smoke
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
        return data


class ElasticMetricTestData(GeometricCaseData):
    def inner_product_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
            on_metric=True,
        )

    def inner_product_is_symmetric_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
        )

    def exp_after_log_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))


class QuotientElasticMetricTestData(GeometricCaseData):
    def log_runs_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))

    def geodesic_bvp_runs_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))
