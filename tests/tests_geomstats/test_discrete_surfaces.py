"""Unit tests for discrete_surfaces modules."""

import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.discrete_surfaces_data import DiscreteSurfacesTestData
from tests.geometry_test_cases import ManifoldTestCase


def _test_manifold_shape(test_cls, space_args):
    space = test_cls.Space(*space_args)
    point = space.random_point()

    msg = f"Shape is {space.shape}, but random point shape is {point.shape}"
    test_cls.assertTrue(space.shape == point.shape, msg)

    if space.metric is None:
        return

    msg = (
        f"Space shape is {space.shape}, "
        f"whereas space metric shape is {space.metric.shape}",
    )

    if space.metric.shape[0] is None:
        test_cls.assertTrue(len(space.shape) == len(space.metric.shape), msg)
        test_cls.assertTrue(space.shape[1:] == space.metric.shape[1:], msg)
    else:
        test_cls.assertTrue(space.shape == space.metric.shape, msg)


class TestDiscreteSurfaces(ManifoldTestCase, metaclass=Parametrizer):

    testing_data = DiscreteSurfacesTestData()

    def test_manifold_shape(self, space_args):
        return _test_manifold_shape(self, space_args)

    def test_vertex_areas(self, faces, point):
        """Test vertex_areas.

        Vertex area is the area of all of the triangles who are in contact
        with a specific vertex, according to the formula:
        vertex_areas = 2 * sum_incident_areas / 3.0

        We test this on a space whose initializing
        point is a cube, and we test the function on
        a cube with sides of length 2 centered at the origin.

        The cube is meshed with triangles, so each face should
        have area 2.
        """
        number_of_contact_faces = gs.array([3, 5, 5, 5, 5, 5, 3, 5])
        triangle_area = 0.5 * 2 * 2
        expected = 2 * (number_of_contact_faces * triangle_area) / 3
        space = self.Space(faces)

        result = space.vertex_areas(point)
        assert result.shape == (8,)
        assert expected.shape == (8,)
        assert gs.allclose(result, expected), result

        point = gs.array([point, point])
        expected = gs.array([expected, expected])
        result = space.vertex_areas(point)
        assert point.shape == (2, 8, 3)
        assert result.shape == (2, 8), result.shape
        assert gs.allclose(result, expected), result

    def test_normals(self, faces, point):
        """Test normals.

        We test this on a space whose initializing
        point is a cube, and we test the function on
        a cube with sides of length 2 centered at the origin.
        The cube is meshed with 12 triangles (2 triangles
        per face.)

        Recall that the magnitude of each normal vector is equal to
        the area of the face it is normal to.

        We compare the abs value of each normal vector array because:
        note that the "normals" variable here calculates the normals
        as pointing out of the surface, but the way that normals()
        was constructed makes it so that the normal vector could be
        pointing into the surface or out of the surface, (so it could
        either be positive or negative). Because of this, we make all
        of the normal vectors to the cube positive for testing
        purposes.
        """
        space = self.Space(faces=faces)
        cube_normals = gs.array(
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

        result = space.normals(point)
        are_close = [
            (gs.allclose(res, exp) or gs.allclose(res, -exp))
            for res, exp in zip(result, expected)
        ]

        assert gs.all(are_close)

        point = gs.array([point, point])
        result = space.normals(point)
        are_close_0 = [
            (gs.allclose(res, exp) or gs.allclose(res, -exp))
            for res, exp in zip(result[0], expected)
        ]
        are_close_1 = [
            (gs.allclose(res, exp) or gs.allclose(res, -exp))
            for res, exp in zip(result[1], expected)
        ]
        assert gs.all(gs.array([are_close_0, are_close_1]))

    def test_surface_one_forms(self, faces, point):
        """Test surface one forms."""
        space = self.Space(faces=faces)

        result = space.surface_one_forms(point=point)
        assert result.shape == (space.n_faces, 2, 3), result.shape

        first_vec = result[:, 0, :]
        second_vec = result[:, 1, :]
        inner_prods = gs.einsum("ni,ni->n", first_vec, second_vec)
        result = [prod in [0.0, 4.0] for prod in inner_prods]
        assert gs.all(result)

        singleton_point = gs.expand_dims(point, axis=0)
        print("singleton_point.shape", singleton_point.shape)
        result = space.surface_one_forms(point=singleton_point)
        assert result.shape == (1, space.n_faces, 2, 3)

        point = gs.array([point, point])
        result = space.surface_one_forms(point=point)
        assert result.shape == (2, space.n_faces, 2, 3)

        first_vec = result[:, :, 0, :]
        second_vec = result[:, :, 1, :]
        inner_prods = gs.einsum("mni,mni->mn", first_vec, second_vec)
        result = []
        for inner_prod in inner_prods:
            result.append([prod in [0.0, 4.0] for prod in inner_prod])
        assert gs.all(result)

    def test_faces_area(self, faces, point):
        """Test faces area."""
        space = self.Space(faces=faces)

        result = space.face_areas(point=point)
        expected = gs.array([4.0] * 12)
        assert result.shape == (space.n_faces,), result.shape
        assert gs.allclose(result, expected), result

        point = gs.array([point, point])
        result = space.face_areas(point=point)
        expected = gs.array([expected, expected])
        assert result.shape == (2, space.n_faces), result.shape
        assert gs.allclose(result, expected), result

    def test_surface_metric_matrices(self, faces, point):
        """Test surface metric matrices."""
        space = self.Space(faces=faces)
        result = space.surface_metric_matrices(point=point)
        assert result.shape == (
            space.n_faces,
            2,
            2,
        ), result.shape

        point = gs.array([point, point])
        result = space.surface_metric_matrices(point=point)
        assert result.shape == (2, space.n_faces, 2, 2)
