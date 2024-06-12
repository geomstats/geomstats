import math

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.vectorization import get_n_points


class SurfacesLocalRandomDataGenerator(RandomDataGenerator):
    def __init__(self, space, point, amplitude=2.0):
        super().__init__(space, amplitude)
        self.point = point

    def _get_deformation(self, n_points):
        dim = self.point.shape[-1]
        dof = self.point.shape[0] - 1

        batch_shape = () if n_points == 1 else (n_points,)

        rand_shape = batch_shape + (dof, dim)

        return (
            gs.concatenate(
                [
                    gs.zeros(batch_shape + (1, dim)),
                    gs.reshape(
                        gs.random.rand(math.prod(rand_shape)),
                        rand_shape,
                    ),
                ],
                axis=-2,
            )
            / self.amplitude
        )

    def random_point(self, n_points=1):
        return self.point + self._get_deformation(n_points)

    def random_tangent_vec(self, point):
        n_points = 1 if point.ndim == 2 else point.shape[0]
        return self._get_deformation(n_points)


class SurfaceTestCase(TestCase):
    def test_face_areas(self, point, expected, atol):
        self.assertAllClose(point.face_areas, expected, atol=atol)

    def test_face_normals(self, point, expected, atol):
        self.assertAllClose(point.face_normals, expected, atol=atol)

    def test_face_centroids(self, point, expected, atol):
        self.assertAllClose(point.face_centroids, expected, atol=atol)


class DiscreteSurfacesTestCase(ManifoldTestCase):
    def test_vertex_areas(self, point, expected, atol):
        """Test vertex_areas.

        Vertex area is the area of all of the triangles who are in contact
        with a specific vertex, according to the formula:
        vertex_areas = 2 * sum_incident_areas / 3.0
        """
        res = self.space.vertex_areas(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_normals(self, point, expected, atol):
        """Test normals.

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

        def _check_normal(normal, normal_):
            self.assertTrue(
                gs.any(
                    [
                        gs.allclose(normal, normal_, atol=atol),
                        gs.allclose(normal, -normal_, atol=atol),
                    ]
                )
            )

        res = self.space.normals(point)

        n_points = get_n_points(self.space.point_ndim, point)
        if n_points == 1:
            res = gs.expand_dims(res, axis=0)
            expected = gs.expand_dims(expected, axis=0)

        for res_, expected_ in zip(res, expected):
            for normal, normal_ in zip(res_, expected_):
                _check_normal(normal, normal_)

    def test_surface_one_forms_prod(self, point, expected):
        """Test surface one forms."""
        res = self.space.surface_one_forms(point)

        first_vec = res[..., :, 0, :]
        second_vec = res[..., :, 1, :]
        inner_prods = gs.einsum("...ni,...ni->...n", first_vec, second_vec)

        if gs.ndim(inner_prods) == 1:
            inner_prods = gs.expand_dims(inner_prods, axis=0)
            expected = gs.expand_dims(expected, axis=0)

        for inner_prods_, expected_ in zip(inner_prods, expected):
            self.assertTrue(gs.all([prod in expected_ for prod in inner_prods_]))

    def test_faces_area(self, point, expected, atol):
        """Test faces area."""
        res = self.space.face_areas(point=point)
        self.assertAllClose(res, expected, atol=atol)
